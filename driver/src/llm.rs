//! GGUF Qwen2 inference: GPU Q8 linear layers, CPU normalization/RoPE/attention.
#[path="frontier_core.rs"] mod frontier_core;
#[path="frontier_gguf.rs"] mod frontier_gguf;
#[path="frontier_storage.rs"] mod frontier_storage;
#[path="frontier_uefi.rs"] mod frontier_uefi;
#[path="frontier_gpu.rs"] mod frontier_gpu;
#[path="frontier_forward.rs"] mod frontier_forward;
#[path="frontier_cache.rs"] mod frontier_cache;
#[path="frontier_cpu_pool.rs"] mod frontier_cpu_pool;
#[path="frontier_cpu_math.rs"] mod frontier_cpu_math;
#[path="frontier_cpu_expf.rs"] mod frontier_cpu_expf;
#[path="frontier_cpu_axpy.rs"] mod frontier_cpu_axpy;
#[path="frontier_backend.rs"] mod frontier_backend;
#[path="frontier_inference.rs"] mod frontier_inference;
#[path="frontier_batch.rs"] mod frontier_batch;
#[path="frontier_copy.rs"] mod frontier_copy;
#[path="frontier_dispatch.rs"] mod frontier_dispatch;
#[path="frontier_window.rs"] mod frontier_window;
#[path="gpu_graph.rs"] mod gpu_graph;
#[path="gpu_prefill.rs"] mod gpu_prefill;
#[path="gpu_q4.rs"] mod gpu_q4;
const Q4_WEIGHTS:bool=false;
fn tensor_q4(name:&str)->bool {Q4_WEIGHTS&&(name.ends_with("ffn_gate.weight")||name.ends_with("ffn_up.weight")||name.ends_with("ffn_down.weight"))}
use alloc::{format, string::String, vec, vec::Vec};
use core::{ffi::c_void, ptr::read_volatile, sync::atomic::{fence,Ordering}};
use uefi::{boot,Identify};
use crate::{gsp::{DmaAllocator,DmaRegion},m4::field};

fn event(log:&mut Vec<String>,s:String) {uefi::println!("{}",s);log.push(s);}
fn read_model(dma:&mut crate::RbioDma,log:&mut Vec<String>)->Option<DmaRegion> {
    let handles=boot::locate_handle_buffer(boot::SearchType::ByProtocol(&crate::BlockIoProtocol::GUID)).ok()?;
    for &handle in handles.iter() {
        let mut bio=match unsafe {boot::open_protocol::<crate::BlockIoProtocol>(boot::OpenProtocolParams {
            handle,agent:boot::image_handle(),controller:None},boot::OpenProtocolAttributes::GetProtocol)} {Ok(p)=>p,Err(_)=>continue};
        let media=unsafe{bio.media.as_ref()?};let bs=media.block_size as usize;let mid=media.media_id;
        if media.media_present==0 || bs==0 || bs>4096 {continue;}
        let mut first=crate::AlignedProbeBlock([0;4096]);
        if unsafe{(bio.read_blocks)(&mut *bio,mid,0,bs,first.0.as_mut_ptr() as *mut c_void)}!=crate::RawStatus::SUCCESS || &first.0[..4]!=b"GGUF" {continue;}
        let media=unsafe{bio.media.as_ref()?};let bytes=(media.last_block+1) as usize*bs;
        if !(1_800_000_000..2_000_000_000).contains(&bytes) {continue;}
        event(log,format!("  M4 GGUF loading {} bytes from read-only model disk",bytes));
        let region=dma.alloc_pages((bytes+4095)/4096)?;
        for off in (0..bytes).step_by(4*1024*1024) {
            let amount=(bytes-off).min(4*1024*1024);
            if unsafe{(bio.read_blocks)(&mut *bio,mid,(off/bs) as u64,amount,region.host.add(off) as *mut c_void)}!=crate::RawStatus::SUCCESS {return None;}
        }
        event(log,format!("  M4 GGUF loaded: host=0x{:x} bus=0x{:x} size={}",region.host as usize,region.bus,bytes));
        return Some(region);
    }None
}

struct Reader<'a>{bytes:&'a[u8],pos:usize}
impl<'a> Reader<'a> {
    fn take(&mut self,n:usize)->&'a[u8] {let s=&self.bytes[self.pos..self.pos+n];self.pos+=n;s}
    fn u32(&mut self)->u32 {u32::from_le_bytes(self.take(4).try_into().unwrap())}
    fn u64(&mut self)->u64 {u64::from_le_bytes(self.take(8).try_into().unwrap())}
    fn string(&mut self)->&'a str {let n=self.u64() as usize;core::str::from_utf8(self.take(n)).unwrap()}
    fn skip(&mut self,t:u32) {match t {
        0|1|7=>{self.take(1);},2|3=>{self.take(2);},4|5|6=>{self.take(4);},10|11|12=>{self.take(8);},
        8=>{self.string();},9=>{let t=self.u32();let n=self.u64();for _ in 0..n {self.skip(t);}},_=>panic!("GGUF unknown type")}}
}
#[derive(Clone,Copy)]
struct Tensor<'a>{name:&'a str,n:usize,m:usize,typ:u32,off:usize}
struct Model<'a>{bytes:&'a[u8],tensors:Vec<Tensor<'a>>,tokens:Vec<&'a str>,data:usize}
impl<'a> Model<'a> {
    fn parse(bytes:&'a[u8])->Self {
        let mut r=Reader{bytes,pos:4};assert_eq!(r.u32(),3);let nt=r.u64();let nk=r.u64();
        let mut tokens=Vec::new();
        for _ in 0..nk {let k=r.string();let t=r.u32();
            if k=="tokenizer.ggml.tokens" {assert_eq!(t,9);assert_eq!(r.u32(),8);let n=r.u64();
                for _ in 0..n {tokens.push(r.string());}}
            else {r.skip(t);}
        }
        let mut tensors=Vec::new();
        for _ in 0..nt {let name=r.string();let nd=r.u32();assert!(nd==1||nd==2);
            let n=r.u64() as usize;let m=if nd==2 {r.u64() as usize} else {1};
            let typ=r.u32();let off=r.u64() as usize;tensors.push(Tensor{name,n,m,typ,off});}
        Self{bytes,tensors,tokens,data:(r.pos+31)&!31}
    }
    fn tensor(&self,name:&str)->Tensor<'a> {*self.tensors.iter().find(|t|t.name==name).expect("required tensor")}
    fn f32(&self,t:Tensor,i:usize)->f32 {let off=self.data+t.off+i*4;assert_eq!(t.typ,0);f32::from_le_bytes(self.bytes[off..off+4].try_into().unwrap())}
    fn q8(&self,t:Tensor,row:usize,col:usize)->f32 {
        assert_eq!(t.typ,8);let p=self.data+t.off+(row*t.n/32+col/32)*34;
        let h=u16::from_le_bytes(self.bytes[p..p+2].try_into().unwrap());
        half(h)*(self.bytes[p+2+col%32] as i8 as f32)
    }
    fn text(&self,id:usize)->String {
        let mut bytes=Vec::new();
        for c in self.tokens[id].chars() {
            let code=c as u32;
            if code<256 {bytes.push(code as u8);} else {
                let mut idx=0u32;let mut found=false;
                for b in 0..256u32 {if !(33..=126).contains(&b) && !(161..=172).contains(&b) && !(174..=255).contains(&b) {
                    if 256+idx==code {bytes.push(b as u8);found=true;break;}idx+=1;
                }}
                if !found {let mut buf=[0u8;4];bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());}
            }
        }String::from_utf8_lossy(&bytes).into_owned()
    }
}
fn half(h:u16)->f32 {
    let sign=((h as u32)&0x8000)<<16;let exp=(h>>10)&31;let frac=(h&1023) as u32;
    if exp==0 {let v=frac as f32*(1.0/16777216.0);if sign!=0 {-v}else{v}}
    else if exp==31 {f32::from_bits(sign|0x7f800000|(frac<<13))}
    else {f32::from_bits(sign|(((exp as u32)+112)<<23)|(frac<<13))}
}

#[derive(Clone,Copy)]
struct Job {off:usize,pa:u64,va:u64,out:usize}
struct Gpu<'a,R:Fn(u64)->u32,W:Fn(u64,u32)>{rd:&'a R,wr:&'a W,bar:u64,data:u64,userd:u64,token:u32,
    window:frontier_window::Window,
    work:DmaRegion,q:[u32;64],put:u32,stamp:u32,kernels:u64,model_data:usize,vram_weights:Vec<(usize,u64)>,swizzled_weights:Vec<(usize,u64)>,jobs:Vec<Job>,submits:u64,graph:Vec<gpu_graph::GraphJob>,profile:[u64;4],prefill:Vec<gpu_graph::GraphJob>,batch_build:bool}
impl<R:Fn(u64)->u32,W:Fn(u64,u32)> Gpu<'_,R,W> {
    fn time_ns(&self)->u64 {
        loop {let hi=(self.rd)(self.bar+0x9410);let lo=(self.rd)(self.bar+0x9400);
            if hi==(self.rd)(self.bar+0x9410) {return ((hi as u64)<<32)|lo as u64;}}
    }
    fn ap(&self,p:u64)->u64 {self.window.address(self.bar,p,self.wr)}
    fn write(&self,p:u64,v:u32) {(self.wr)(self.ap(p),v);}
    fn write64(&self,p:u64,v:u64) {self.write(p,v as u32);self.write(p+4,(v>>32) as u32);}
    fn map(&self,va:u64,region:&DmaRegion,tables:u64) {
        self.map_memory(va,region.bus,region.size,tables,0xd);
    }
    fn map_memory(&self,va:u64,physical:u64,size:usize,tables:u64,flags:u64) {
        let pdb=self.data-0x5000;let chunks=(size+0x1fffff)/0x200000;
        let pd0s=(chunks+255)/256;
        for d in 0..pd0s*1024 {self.write(tables+d as u64*4,0);}
        for i in 0..pd0s {self.write64(pdb+0x2000+(((va>>29) as usize+i)&511) as u64*8,((tables+i as u64*4096)>>4)|2);}
        for chunk in 0..chunks {
            let pt=tables+pd0s as u64*4096+chunk as u64*4096;
            self.write64(tables+(chunk/256) as u64*4096+(chunk%256) as u64*16,0);
            self.write64(tables+(chunk/256) as u64*4096+(chunk%256) as u64*16+8,(pt>>4)|2);
            for i in 0..512usize {let off=chunk*0x200000+i*4096;
                self.write64(pt+i as u64*8,if off<size {((physical+off as u64)>>4)|flags} else {0});}
        }
    }
    fn copy_buffer(&mut self,src:u64,dst:u64,bytes:usize)->bool {
        if bytes==0 || bytes%4!=0 || bytes/4>u32::MAX as usize {return false;}
        let mut q=self.q;self.stamp+=1;
        field(&mut q,384,415,((bytes/4+127)/128)as u64);
        field(&mut q,648,656,8);field(&mut q,832,863,self.stamp as u64);
        field(&mut q,1536,1567,0x04002000);field(&mut q,256,287,0x04002000>>8);
        field(&mut q,1641,1649,2);
        for(i,v)in q.iter().enumerate(){self.write(self.data+0x800+i as u64*4,*v);}
        self.write64(self.data+0x900+0x160,src);self.write64(self.data+0x900+0x168,dst);
        self.write(self.data+0x900+0x170,(bytes/4)as u32);
        let words=[0x20012000|(0x1698>>2),0x1011,0x20012000|(0x2b4>>2),0x40008,0x20012000|(0x2c0>>2),9];
        for(i,v)in words.iter().enumerate(){self.write(self.data+0x200+i as u64*4,*v);}
        self.write(self.data+(self.put%32)as u64*8,0x4000200);
        self.write(self.data+(self.put%32)as u64*8+4,(words.len()as u32)<<10);
        fence(Ordering::SeqCst);let _=(self.rd)(self.ap(self.data+(self.put%32)as u64*8+4));
        self.put=(self.put+1)%32;self.write(self.userd+0x8c,self.put);
        let _=(self.rd)(self.ap(self.userd+0x8c));(self.wr)(self.bar+0xbb0090,self.token);
        for _ in 0..100000 {if(self.rd)(self.ap(self.data+0xf10))==self.stamp {fence(Ordering::SeqCst);return true;}boot::stall(10);}
        false
    }
    fn upload_weights(&mut self,model:&Model,fb:([u64;4],u32),log:&mut Vec<String>)->bool {
        const PHYS:u64=0x10000000;const VA:u64=0x200000000;const TABLES:u64=0x0a000000;
        let mut spans=Vec::new();let mut allocated=0usize;let mut bytes=0usize;
        for t in &model.tensors {if t.typ==8 && t.name!="token_embd.weight" {
            let size=t.n/32*t.m*34;
            if t.n%32!=0 || model.data+t.off+size>model.bytes.len() {return false;}
            spans.push((t.off,allocated,size));bytes+=size;allocated+=(size+4095)&!4095;
        }}
        let chunks=(allocated+0x1fffff)/0x200000;let table_bytes=((chunks+255)/256+chunks)*4096;
        // Exclusive arena in the runtime-reported free FB[1], beyond all existing
        // scratch/context/table allocations. Never enter RM-reserved or WPR memory.
        if spans.len()!=LAYERS*7+1 || fb.0[2]!=0 || fb.1!=0 || fb.0[0]>TABLES ||
            PHYS+allocated as u64>fb.0[1].saturating_add(1) || PHYS+allocated as u64>0x2f0920000 ||
            table_bytes>0x800000 {event(log,String::from("  M4 VRAM arena validation FAILED"));return false;}
        event(log,format!("  M4 VRAM_ARENA tensors={} weight_bytes={} allocation_bytes={} PA=0x{:x} end=0x{:x} VA=0x{:x} tables=0x{:x} table_bytes={} free_FB=[0x{:x}..0x{:x}]",
            spans.len(),bytes,allocated,PHYS,PHYS+allocated as u64,VA,TABLES,table_bytes,fb.0[0],fb.0[1]));
        self.map_memory(VA,PHYS,allocated,TABLES,1);
        let code=include_bytes!("../kernels/copy_u32.sass");
        for(i,b)in code.chunks_exact(4).enumerate(){self.write(self.data+0x6000+i as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
        let pdb=self.data-0x5000;fence(Ordering::SeqCst);
        (self.wr)(self.bar+0xb830a0,(pdb>>8)as u32);(self.wr)(self.bar+0xb830a4,(pdb>>40)as u32);(self.wr)(self.bar+0xb830b0,0x80000007);
        let mut flushed=false;for _ in 0..20000 {if(self.rd)(self.bar+0xb830b0)&0x80000000==0 {flushed=true;break;}boot::stall(10);}
        if !flushed {return false;}
        let start=self.time_ns();
        for &(off,dest,size) in &spans {
            if !self.copy_buffer(0x40000000+model.data as u64+off as u64,VA+dest as u64,size) {
                event(log,format!("  M4 VRAM_UPLOAD TIMEOUT offset={} bytes={}",off,size));return false;}
        }
        let upload_ns=self.time_ns().wrapping_sub(start);
        event(log,format!("  M4 VRAM_UPLOAD completed=true tensors={} bytes={} elapsed_ns={}",spans.len(),bytes,upload_ns));
        // Full GPU readback in bounded chunks verifies every uploaded byte before
        // inference. Host embedding/norm data and the original GGUF stay unchanged.
        let start=self.time_ns();let mut checked=0usize;
        for &(off,dest,size) in &spans {for chunk in (0..size).step_by(0x100000) {
            let amount=(size-chunk).min(0x100000);
            if !self.copy_buffer(VA+dest as u64+chunk as u64,0x100000000,amount) {return false;}
            let seen=unsafe{core::slice::from_raw_parts(self.work.host,amount)};
            let expected=&model.bytes[model.data+off+chunk..model.data+off+chunk+amount];
            if seen!=expected {event(log,format!("  M4 VRAM_VERIFY MISMATCH offset={} chunk={}",off,chunk));return false;}
            checked+=amount;
        }}
        let verify_ns=self.time_ns().wrapping_sub(start);
        self.vram_weights=spans.iter().map(|&(off,dest,_)|(off,VA+dest as u64)).collect();
        event(log,format!("  *** M4 VRAM_VERIFY PASS=true checked_bytes={} elapsed_ns={} ***",checked,verify_ns));true
    }
    fn init_jobs(&mut self,model:&Model,log:&mut Vec<String>) {
        const PA:u64=0x0c000000;const VA:u64=0xc0000000;
        let size=(self.vram_weights.len()*0x400+4095)&!4095;
        self.map_memory(VA,PA,size,0x0b000000,1);
        let prefix:Vec<u32>=(0..0x160/4).map(|i|(self.rd)(self.ap(self.data+0x900+i*4))).collect();
        for(i,&(off,weight_va))in self.vram_weights.iter().enumerate(){
            let t=*model.tensors.iter().find(|t|t.off==off&&t.typ==8).unwrap();
            let out=0x20000+if t.name.ends_with("attn_k.weight")||t.name.ends_with("ffn_gate.weight"){0x10000}else if t.name.ends_with("attn_v.weight"){0x20000}else{0};
            let pa=PA+i as u64*0x400;let va=VA+i as u64*0x400;let mut q=self.q;
            field(&mut q,384,415,((t.m+3)/4)as u64);field(&mut q,1024,1055,va+0x100);
            field(&mut q,768,799,va+0x3f0);self.write(pa+0x3f0,0);
            for(d,v)in q.iter().enumerate(){self.write(pa+d as u64*4,*v);}
            for(d,v)in prefix.iter().enumerate(){self.write(pa+0x100+d as u64*4,*v);}
            self.write64(pa+0x260,weight_va);self.write64(pa+0x268,0x100000000);
            self.write64(pa+0x270,0x100000000+out as u64);self.write(pa+0x278,t.n as u32);self.write(pa+0x27c,t.m as u32);
            let pb=[0x20012000|(0x1698>>2),0x1011,0x20012000|(0x2b4>>2),(va>>8)as u32,0x20012000|(0x2c0>>2),9];
            for(d,v)in pb.iter().enumerate(){self.write(pa+0x300+d as u64*4,*v);}
            self.jobs.push(Job{off,pa,va,out});
        }
        let pdb=self.data-0x5000;fence(Ordering::SeqCst);(self.wr)(self.bar+0xb830a0,(pdb>>8)as u32);
        (self.wr)(self.bar+0xb830a4,(pdb>>40)as u32);(self.wr)(self.bar+0xb830b0,0x80000007);
        for _ in 0..20000 {if(self.rd)(self.bar+0xb830b0)&0x80000000==0{break;}boot::stall(10);}
        event(log,format!("  M4 PREBUILT jobs={} stride=0x400 QKV_batch=3 gate_up_batch=2",self.jobs.len()));
    }
    fn batch(&mut self,tensors:&[Tensor],x:&[f32],ys:&mut[&mut[f32]])->bool {
        if tensors.len()!=ys.len()||tensors.is_empty()||tensors.len()>3{return false;}
        unsafe{core::ptr::copy_nonoverlapping(x.as_ptr(),self.work.host as *mut f32,x.len());}
        let mut selected=[Job{off:0,pa:0,va:0,out:0};3];let first_stamp=self.stamp+1;
        for(i,t)in tensors.iter().enumerate(){
            if t.n!=x.len()||t.m!=ys[i].len(){return false;}
            let Some(&job)=self.jobs.iter().find(|j|j.off==t.off)else{return false;};selected[i]=job;
            self.stamp+=1;self.write(job.pa+0x68,self.stamp);
            self.write(self.data+(self.put%32)as u64*8,(job.va+0x300)as u32);
            self.write(self.data+(self.put%32)as u64*8+4,6<<10);self.put=(self.put+1)%32;
        }
        fence(Ordering::SeqCst);let _=(self.rd)(self.ap(selected[tensors.len()-1].pa+0x68));
        self.write(self.userd+0x8c,self.put);let _=(self.rd)(self.ap(self.userd+0x8c));
        (self.wr)(self.bar+0xbb0090,self.token);self.submits+=1;
        let mut done=false;for _ in 0..100000 {
            if selected[..tensors.len()].iter().enumerate().all(|(i,j)|(self.rd)(self.ap(j.pa+0x3f0))==first_stamp+i as u32){done=true;break;}boot::stall(10);}
        if !done{return false;}fence(Ordering::SeqCst);
        for(i,y)in ys.iter_mut().enumerate(){for(j,v)in y.iter_mut().enumerate(){*v=unsafe{read_volatile((self.work.host.add(selected[i].out)as *const f32).add(j))};}}
        self.kernels+=tensors.len()as u64;true
    }
    fn matvec(&mut self,t:Tensor,x:&[f32],y:&mut[f32])->bool {
        if !self.jobs.is_empty(){return self.batch(&[t],x,&mut[y]);}
        if t.typ!=8||t.n!=x.len()||t.m!=y.len()||t.n%32!=0||t.n>16384||t.m>160000 {return false;}
        unsafe {core::ptr::copy_nonoverlapping(x.as_ptr(),self.work.host as *mut f32,x.len());}
        self.stamp+=1;
        field(&mut self.q,384,415,((t.m+3)/4) as u64);
        field(&mut self.q,832,863,self.stamp as u64);
        for (i,v) in self.q.iter().enumerate() {self.write(self.data+0x800+i as u64*4,*v);}
        let weight_va=if self.vram_weights.is_empty() {0x40000000+self.model_data as u64+t.off as u64}
            else {match self.vram_weights.iter().find(|&&(off,_)|off==t.off) {Some(&(_,va))=>va,None=>return false,}};
        self.write64(self.data+0x900+0x160,weight_va);
        self.write64(self.data+0x900+0x168,0x100000000);
        self.write64(self.data+0x900+0x170,0x100020000);
        self.write(self.data+0x900+0x178,t.n as u32);self.write(self.data+0x900+0x17c,t.m as u32);
        let words=[0x20012000|(0x1698>>2),0x1011,0x20012000|(0x2b4>>2),0x40008,0x20012000|(0x2c0>>2),9];
        for(i,v)in words.iter().enumerate(){self.write(self.data+0x200+i as u64*4,*v);}
        self.write(self.data+(self.put%32) as u64*8,0x4000200);
        self.write(self.data+(self.put%32) as u64*8+4,(words.len() as u32)<<10);
        fence(Ordering::SeqCst);let _=(self.rd)(self.ap(self.data+(self.put%32) as u64*8+4));
        self.put=(self.put+1)%32;self.write(self.userd+0x8c,self.put);
        let _=(self.rd)(self.ap(self.userd+0x8c));(self.wr)(self.bar+0xbb0090,self.token);
        let mut done=false;
        for _ in 0..100000 {if(self.rd)(self.ap(self.data+0xf10))==self.stamp {done=true;break;}boot::stall(10);}
        if !done {return false;}
        fence(Ordering::SeqCst);
        for(i,v)in y.iter_mut().enumerate(){*v=unsafe{read_volatile((self.work.host.add(0x20000) as *const f32).add(i))};}
        self.kernels+=1;true
    }
}

pub fn run<R:Fn(u64)->u32,W:Fn(u64,u32)>(rd:&R,wr:&W,bar:u64,data:u64,userd:u64,token:u32,
    dma:&mut crate::RbioDma,template:&[u32;64],fb:([u64;4],u32),log:&mut Vec<String>) {
    if let Some(media)=frontier_uefi::FullMedia::discover() {
        frontier_inference::run(rd,wr,bar,data,userd,token,dma,template,fb,media,log);
        return;
    }
    // Header-only fixture gate remains available; cannot claim inference.
    if frontier_uefi::header_probe(rd,bar,log) {
        frontier_gpu::probe(rd,wr,bar,data,userd,token,dma,template,fb,log);
    }
    return;
    let Some(region)=read_model(dma,log) else {event(log,String::from("  M4 GGUF load FAILED"));return;};
    let model=Model::parse(unsafe{core::slice::from_raw_parts(region.host,region.size)});
    event(log,format!("  M4 GGUF parsed: tensors={} vocab={} data_offset={}",model.tensors.len(),model.tokens.len(),model.data));
    let Some(work)=dma.alloc_pages(256) else {return;};
    let saved=rd(bar+0x1700);
    let mut gpu=Gpu{rd,wr,bar,data,userd,token,window:frontier_window::Window::new(),work,q:*template,put:2,stamp:0x4e000000,kernels:0,model_data:model.data,vram_weights:Vec::new(),swizzled_weights:Vec::new(),jobs:Vec::new(),submits:0,graph:Vec::new(),profile:[0;4],prefill:Vec::new(),batch_build:false};
    gpu.map(0x40000000,&region,0x09000000);gpu.map(0x100000000,&gpu.work,0x09800000);
    let code=include_bytes!("../kernels/matvec_q8.sass");
    for(i,b)in code.chunks_exact(4).enumerate(){gpu.write(data+0x5000+i as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
    field(&mut gpu.q,592,607,128);field(&mut gpu.q,648,656,34);field(&mut gpu.q,642,642,0);
    field(&mut gpu.q,1075,1087,0x180/16);field(&mut gpu.q,1641,1649,(code.len()/256)as u64);
    fence(Ordering::SeqCst);wr(bar+0xb830a0,((data-0x5000)>>8)as u32);wr(bar+0xb830a4,0);wr(bar+0xb830b0,0x80000007);
    for _ in 0..20000 {if rd(bar+0xb830b0)&0x80000000==0 {break;}boot::stall(10);}
    if !gpu.upload_weights(&model,fb,log) {wr(bar+0x1700,saved);return;}
    gpu.init_jobs(&model,log);
    let t=model.tensor("blk.0.attn_q.weight");let x:Vec<f32>=(0..t.n).map(|i|((i%17)as f32-8.0)/16.0).collect();
    let mut y=vec![0f32;t.m];let ok=gpu.matvec(t,&x,&mut y);
    let mut maxerr=0f32;
    for row in [0,1,17,127,DIM-1] {let mut expect=0f32;for col in 0..t.n {expect+=model.q8(t,row,col)*x[col];}
        let err=libm::fabsf(y[row]-expect);maxerr=maxerr.max(err);
        event(log,format!("  M4 Q8_DOT row={} GPU={} CPU={} abs_error={}",row,y[row],expect,err));}
    let pass=ok&&maxerr<0.002&&y.iter().all(|x|x.is_finite());
    event(log,format!("  *** M4 GGUF_GPU_MATVEC PASS={} completed={} max_abs_error={} kernels={} ***",pass,ok,maxerr,gpu.kernels));
    if pass && gpu.init_graph(&model,fb,log) && gpu.init_prefill(&model,fb,log) {generate(&model,&mut gpu,log);}
    wr(bar+0x1700,saved);
}

const DIM:usize=1536;
const HIDDEN:usize=8960;
const KV:usize=256;
const HEAD:usize=128;
const SEQ:usize=1024;
const KV_KEYS:u64=0xa00000000;
const KV_VALUES:u64=KV_KEYS+0x2000000;
const LAYERS:usize=28;

fn norm(model:&Model,t:Tensor,x:&[f32],y:&mut[f32]) {
    let sum=x.iter().map(|v|v*v).sum::<f32>();
    let scale=1.0/libm::sqrtf(sum/x.len() as f32+1.0e-6);
    for i in 0..x.len(){y[i]=x[i]*scale*model.f32(t,i);}
}
fn rope(x:&mut[f32],pos:usize) {
    for i in 0..HEAD/2 {
        let angle=pos as f32*libm::powf(1_000_000.0,-((2*i)as f32)/HEAD as f32);
        let(s,c)=libm::sincosf(angle);
        for head in 0..x.len()/HEAD {
            let a=head*HEAD+i;let b=a+HEAD/2;let v0=x[a];let v1=x[b];
            x[a]=v0*c-v1*s;x[b]=v0*s+v1*c;
        }
    }
}
struct State {keys:Vec<f32>,values:Vec<f32>}
fn forward<R:Fn(u64)->u32,W:Fn(u64,u32)>(model:&Model,gpu:&mut Gpu<R,W>,state:&mut State,
    token:usize,pos:usize,output:bool,log:&mut Vec<String>)->Option<usize> {
    if !gpu.graph.is_empty(){return gpu.forward_gpu(model,token,pos,output,log);}
    let emb=model.tensor("token_embd.weight");
    let mut x=vec![0f32;DIM];for i in 0..DIM{x[i]=model.q8(emb,token,i);}
    let mut xn=vec![0f32;DIM];let mut q=vec![0f32;DIM];let mut k=vec![0f32;KV];let mut v=vec![0f32;KV];
    let mut att=vec![0f32;DIM];let mut tmp=vec![0f32;DIM];
    let mut up=vec![0f32;HIDDEN];let mut gate=vec![0f32;HIDDEN];
    for layer in 0..LAYERS {
        let t=|suffix:&str|model.tensor(&format!("blk.{}.{}",layer,suffix));
        norm(model,t("attn_norm.weight"),&x,&mut xn);
        if !gpu.batch(&[t("attn_q.weight"),t("attn_k.weight"),t("attn_v.weight")],&xn,&mut[&mut q,&mut k,&mut v]){event(log,format!("  M4 LLM QKV timeout pos={} layer={}",pos,layer));return None;}
        for (suffix,dest) in [("attn_q.bias",&mut q),("attn_k.bias",&mut k),("attn_v.bias",&mut v)] {
            let bias=t(suffix);for i in 0..dest.len(){dest[i]+=model.f32(bias,i);}
        }
        rope(&mut q,pos);rope(&mut k,pos);
        let off=(layer*SEQ+pos)*KV;
        state.keys[off..off+KV].copy_from_slice(&k);state.values[off..off+KV].copy_from_slice(&v);
        att.fill(0.0);
        for head in 0..DIM/HEAD {
            let kvhead=head/(DIM/KV);let mut scores=[0f32;SEQ];let mut max=f32::NEG_INFINITY;
            for time in 0..=pos {
                let koff=(layer*SEQ+time)*KV+kvhead*HEAD;let mut sum=0f32;
                for i in 0..HEAD{sum+=q[head*HEAD+i]*state.keys[koff+i];}
                scores[time]=sum/libm::sqrtf(HEAD as f32);max=max.max(scores[time]);
            }
            let mut denom=0f32;for time in 0..=pos{scores[time]=libm::expf(scores[time]-max);denom+=scores[time];}
            for time in 0..=pos{let prob=scores[time]/denom;let voff=(layer*SEQ+time)*KV+kvhead*HEAD;
                for i in 0..HEAD{att[head*HEAD+i]+=prob*state.values[voff+i];}}
        }
        if !gpu.matvec(t("attn_output.weight"),&att,&mut tmp){event(log,format!("  M4 LLM GPU TIMEOUT pos={} layer={} attn_output",pos,layer));return None;}
        for i in 0..DIM{x[i]+=tmp[i];}
        norm(model,t("ffn_norm.weight"),&x,&mut xn);
        if !gpu.batch(&[t("ffn_up.weight"),t("ffn_gate.weight")],&xn,&mut[&mut up,&mut gate]){
            event(log,format!("  M4 LLM GPU TIMEOUT pos={} layer={} ffn_up/gate",pos,layer));return None;}
        for i in 0..HIDDEN{gate[i]=gate[i]/(1.0+libm::expf(-gate[i]))*up[i];}
        if !gpu.matvec(t("ffn_down.weight"),&gate,&mut tmp){event(log,format!("  M4 LLM GPU TIMEOUT pos={} layer={} ffn_down",pos,layer));return None;}
        for i in 0..DIM{x[i]+=tmp[i];}
        if !x.iter().all(|v|v.is_finite()){event(log,format!("  M4 LLM NONFINITE pos={} layer={}",pos,layer));return None;}
    }
    if !output{return Some(0);}
    norm(model,model.tensor("output_norm.weight"),&x,&mut xn);
    let out=model.tensor("output.weight");let mut logits=vec![0f32;out.m];
    if !gpu.matvec(out,&xn,&mut logits){event(log,format!("  M4 LLM GPU TIMEOUT pos={} output",pos));return None;}
    let mut next=0;let mut best=f32::NEG_INFINITY;
    for (id,value) in logits.iter().enumerate(){if !value.is_finite(){event(log,String::from("  M4 LLM nonfinite logits"));return None;}
        if *value>best{next=id;best=*value;}}
    // No serial printing or token formatting inside timed inference.
    Some(next)
}
fn generate<R:Fn(u64)->u32,W:Fn(u64,u32)>(model:&Model,gpu:&mut Gpu<R,W>,log:&mut Vec<String>) {
    generate_one(model,gpu,include_bytes!("../models/prompt_sentence.tokens"),include_str!("../models/prompt_sentence.txt"),"sentence",log);
    generate_one(model,gpu,include_bytes!("../models/prompt_riddle.tokens"),include_str!("../models/prompt_riddle.txt"),"riddle",log);
    generate_one(model,gpu,include_bytes!("../models/prompt_scale.tokens"),include_str!("../models/prompt_scale.txt"),"scale",log);
}
fn generate_one<R:Fn(u64)->u32,W:Fn(u64,u32)>(model:&Model,gpu:&mut Gpu<R,W>,packed:&[u8],prompt_text:&str,label:&str,log:&mut Vec<String>) {
    let count=u32::from_le_bytes(packed[..4].try_into().unwrap()) as usize;
    let prompt:Vec<usize>=packed[4..].chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap()) as usize).collect();
    let max_new=if label=="scale"{512}else{160};
    assert_eq!(count,prompt.len());assert!(count+max_new<SEQ);
    assert_eq!(model.tensor("token_embd.weight").n,DIM);assert_eq!(model.tokens.len(),151936);
    event(log,format!("  M4 LLM PROMPT {}: {}",label,prompt_text));
    event(log,format!("  M4 MODEL_VARIANT linear_weight_bits={} embedding_bits=8 KV_bits=32 source_GGUF_unchanged=true",if Q4_WEIGHTS{"MIXED_Q4_FFN_Q8_OTHER"}else{"8"}));
    event(log,format!("  M4 LLM backend=UEFI GPU_GRAPH={} embedding={} greedy={}",!gpu.graph.is_empty(),if gpu.graph.is_empty(){"CPU"}else{"GPU_Q8_SYSROW"},if gpu.graph.is_empty(){"CPU"}else{"GPU_FP32_STABLE"}));
    let kv_len=if gpu.graph.is_empty(){LAYERS*SEQ*KV}else{0};
    let mut state=State{keys:vec![0f32;kv_len],values:vec![0f32;kv_len]};
    let check=gpu.time_ns();boot::stall(10000);let delta=gpu.time_ns().wrapping_sub(check);
    event(log,format!("  M4 PTIMER_SANITY requested_us=10000 elapsed_ns={}",delta));
    if !(8_000_000..100_000_000).contains(&delta) {event(log,String::from("  M4 BENCH invalid PTIMER scale; abort"));return;}
    // Build a reusable fixed-shape prompt graph and warm it once, outside timing.
    // The measured interval includes staging all token ids and one GPU submission.
    gpu.prepare_prefill(model,count,log);
    let warm_start=gpu.time_ns();let mut warm_reps=0;let(mut previous,mut stable,mut shortest,mut longest)=(0u64,0usize,u64::MAX,0u64);
    loop{let start=gpu.time_ns();if gpu.run_prefill(model,&prompt,log).is_none(){return;}let ns=gpu.time_ns()-start;warm_reps+=1;
        shortest=shortest.min(ns);longest=longest.max(ns);
        if previous>0&&ns.abs_diff(previous)*100<=previous*5{stable+=1;}else{stable=0;}previous=ns;
        let elapsed=gpu.time_ns()-warm_start;if (elapsed>=2_000_000_000&&stable>=4)||elapsed>=10_000_000_000{break;}}
    event(log,format!("  M4 PREFILL_WARMUP prompt={} repetitions={} elapsed_ns={} min_ns={} max_ns={} stable={} criterion=2s_plus_5_runs_within_5pct",label,warm_reps,gpu.time_ns()-warm_start,shortest,longest,stable>=4));
    let mut stamps=Vec::with_capacity(count+max_new);
    let mut output=String::with_capacity(2048);let mut ended=false;let mut generated=0;
    let profile_pre=gpu.profile;let prefill_kernels=gpu.kernels;let prefill_start=gpu.time_ns();
    let Some(mut next)=gpu.run_prefill(model,&prompt,log)else{return;};
    let prefill_end=gpu.time_ns();let profile_dec=gpu.profile;let decode_kernels=gpu.kernels;let decode_start=gpu.time_ns();
    for i in 0..max_new {
        if next==151645||next==151643{ended=true;break;}
        let start=gpu.time_ns();
        output.push_str(&model.text(next));generated+=1;
        let Some(n)=forward(model,gpu,&mut state,next,count+i,true,log)else{return;};next=n;
        stamps.push(("decode",count+i,start,gpu.time_ns()));
    }
    let decode_end=gpu.time_ns();
    let profile_end=gpu.profile;
    if next==151645||next==151643{ended=true;}
    if !gpu.graph.is_empty()&&!gpu.verify_gpu_greedy(model,next,log){event(log,String::from("  M4 GPU_GREEDY_CPU_CHECK FAILED"));return;}
    event(log,format!("  M4 BENCH_BATCH phase=prefill prompt={} tokens={} start_ns={} end_ns={} graph_prepare_excluded=true warmup=FULL_PROMPT",label,count,prefill_start,prefill_end));
    // Emit timestamps only after both measured phases have completed.
    for (phase,pos,start,end) in stamps {event(log,format!("  M4 BENCH_TOKEN prompt={} phase={} pos={} start_ns={} end_ns={} elapsed_ns={}",label,phase,pos,start,end,end.wrapping_sub(start)));}
    for (phase,n,start,end,kernels) in [("prefill",count,prefill_start,prefill_end,decode_kernels-prefill_kernels),
        ("decode",generated,decode_start,decode_end,gpu.kernels-decode_kernels)] {
        let ns=end.wrapping_sub(start);event(log,format!("  *** M4 BENCH prompt={} phase={} tokens={} elapsed_ns={} ms_per_token={:.6} tokens_per_sec={:.6} kernels={} weights={} linear_weight_bits={} prefill_mode=BATCHED ***",
            label,phase,n,ns,ns as f64/n as f64/1e6,n as f64*1e9/ns as f64,kernels,if gpu.vram_weights.is_empty(){"SYSMEM"}else{"VRAM"},if Q4_WEIGHTS{"MIXED_Q4_FFN_Q8_OTHER"}else{"8"}));
    }
    event(log,format!("  *** M4 LLM OUTPUT {} tokens={} eos={} gpu_kernels={}: {} ***",label,generated,ended,gpu.kernels,output));
    event(log,format!("  M4 GENERATION_LIMIT prompt={} limit={} reached={} max_context={} final_position={}",label,max_new,!ended&&generated==max_new,SEQ,count+generated));
    for(phase,n,a,b)in [("prefill",count,profile_pre,profile_dec),("decode",generated,profile_dec,profile_end)] {
        event(log,format!("  M4 PROFILE prompt={} phase={} tokens={} embed_cpu_ns={} embed_copy_ns={} graph_submit_wait_ns={} greedy_cpu_ns={}",label,phase,n,b[0]-a[0],b[1]-a[1],b[2]-a[2],b[3]-a[3]));
    }
    event(log,String::from("  M4 LLM semantic sentence/riddle verification: inspect actual output above; no hard-coded success label."));
}
