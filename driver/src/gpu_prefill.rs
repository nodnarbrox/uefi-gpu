//! Prompt-matrix execution; the existing decode graph and FP32 KV layout remain.
use super::*;
const B:u64=0x600000000;
const X:u64=B;const XN:u64=B+0x200000;const Q:u64=B+0x400000;
const K:u64=B+0x600000;const V:u64=B+0x700000;const ATT:u64=B+0x800000;
const TMP:u64=B+0xa00000;const GATE:u64=B+0x1000000;const UP:u64=B+0x1a00000;
const XQ:u64=B+0x2400000;
const D:u64=0x400000000;
impl<R:Fn(u64)->u32,W:Fn(u64,u32)> Gpu<'_,R,W> {
    fn p_add(&mut self,id:usize,grid:usize,args:&[u64],tail:&[u32]) {
        let kernels:[(&[u8],u64);8]=[
            (include_bytes!("../kernels/prefill_gemm.sass"),40),
            (include_bytes!("../kernels/prefill_resnorm.sass"),39),
            (include_bytes!("../kernels/prefill_embedding.sass"),13),
            (include_bytes!("../kernels/prefill_rope.sass"),22),
            (include_bytes!("../kernels/prefill_attention.sass"),40),
            (include_bytes!("../kernels/prefill_gemm_q4.sass"),56),
            (include_bytes!("../kernels/prefill_repack.sass"),18),
            (include_bytes!("../kernels/prefill_quant.sass"),21)];
        self.graph_add(0xe0400000+id as u64*0x10000,kernels[id].1,kernels[id].0.len(),grid,args,tail,0,false);
    }
    fn p_quant(&mut self,input:u64,cols:usize,batch:usize) {
        self.p_add(7,(cols*batch+127)/128,&[input,XQ],&[cols as u32,batch as u32]);
    }
    fn p_gemm(&mut self,model:&Model,name:&str,n:usize,output:u64) {
        let t=model.tensor(name);assert!(t.n%32==0&&t.m%16==0&&n%8==0);
        assert!(!tensor_q4(name));
        let w=self.swizzled_weights.iter().find(|&&(off,_)|off==t.off).unwrap().1;
        self.p_add(if tensor_q4(name){5}else{0},t.m/16*((n+31)/32),&[w,XQ,output],&[t.n as u32,t.m as u32,n as u32]);
    }
    fn p_finish(&mut self,decode:Vec<gpu_graph::GraphJob>) {
        let last=self.graph.last().unwrap().pa+0x64;let word=(self.rd)(self.ap(last));self.write(last,word|(1<<23));
        assert!(self.graph.len()*0x400<0x400000&&self.graph.len()*32<0x100000);
        self.prefill=core::mem::replace(&mut self.graph,decode);self.batch_build=false;
    }
    fn p_submit(&mut self,log:&mut Vec<String>)->bool {
        let last=*self.prefill.last().unwrap();self.stamp+=1;self.write(last.pa+0x68,self.stamp);
        self.write(self.data+self.put as u64*8,0xe0600000);self.write(self.data+self.put as u64*8+4,(self.prefill.len()as u32*8)<<10);
        self.put=(self.put+1)%32;fence(Ordering::SeqCst);let _=(self.rd)(self.ap(last.pa+0x68));
        self.write(self.userd+0x8c,self.put);let _=(self.rd)(self.ap(self.userd+0x8c));(self.wr)(self.bar+0xbb0090,self.token);
        self.submits+=1;
        for _ in 0..500000 {if(self.rd)(self.ap(last.pa+0x3f0))==self.stamp{fence(Ordering::SeqCst);self.kernels+=self.prefill.len()as u64;return true;}boot::stall(10);}
        event(log,format!("  M4 PREFILL TIMEOUT jobs={} sem=0x{:x} want=0x{:x}",self.prefill.len(),(self.rd)(self.ap(last.pa+0x3f0)),self.stamp));false
    }
    pub(super) fn init_prefill(&mut self,model:&Model,fb:([u64;4],u32),log:&mut Vec<String>)->bool {
        if fb.0[2]!=0||fb.1!=0||fb.0[0]>0x0d000000||fb.0[1]<0x93ffffff{return false;}
        self.map_memory(0xe0000000,0x0d000000,0x800000,0x0d800000,1);
        self.map_memory(B,0x90000000,0x4000000,0x0f000000,1);
        let codes:[&[u8];8]=[include_bytes!("../kernels/prefill_gemm.sass"),include_bytes!("../kernels/prefill_resnorm.sass"),include_bytes!("../kernels/prefill_embedding.sass"),include_bytes!("../kernels/prefill_rope.sass"),include_bytes!("../kernels/prefill_attention.sass"),include_bytes!("../kernels/prefill_gemm_q4.sass"),include_bytes!("../kernels/prefill_repack.sass"),include_bytes!("../kernels/prefill_quant.sass")];
        for(i,code)in codes.iter().enumerate(){assert!(code.len()<0x10000);for(j,b)in code.chunks_exact(4).enumerate(){self.write(0x0d400000+i as u64*0x10000+j as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}}
        let c2=include_bytes!("../kernels/prefill_attention.c2");assert_eq!(c2.len(),16);
        for d in 0..4{self.write(0x0d7f0000+d*4,0);}for(j,b)in c2.chunks_exact(4).enumerate(){self.write(0x0d7f0000+j as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
        let pdb=self.data-0x5000;fence(Ordering::SeqCst);(self.wr)(self.bar+0xb830a0,(pdb>>8)as u32);(self.wr)(self.bar+0xb830a4,(pdb>>40)as u32);(self.wr)(self.bar+0xb830b0,0x80000007);
        let mut ok=false;for _ in 0..20000{if(self.rd)(self.bar+0xb830b0)&0x80000000==0{ok=true;break;}boot::stall(10);}if !ok{return false;}
        event(log,String::from("  M4 PREFILL_ARENA PA=[0x90000000..0x94000000) commands=[0x0d000000..0x0d800000) KV=EXISTING_FP32"));
        self.init_swizzle(model,fb,log)&&self.check_prefill_gemm(model,log)
    }
    fn init_swizzle(&mut self,model:&Model,fb:([u64;4],u32),log:&mut Vec<String>)->bool {
        const PA:u64=0x180000000;const VA:u64=0x1000000000;const PT:u64=0x0f400000;
        let mut spans=Vec::new();let mut allocated=0usize;
        for &(off,src) in &self.vram_weights {let t=model.tensors.iter().find(|t|t.off==off&&t.typ==8).unwrap();
            if t.name=="output.weight" {continue;}assert!(t.m%16==0&&t.n%32==0);
            let tiles=t.m/16*(t.n/32);spans.push((off,src,allocated,tiles,t.n/32));allocated+=(tiles*576+4095)&!4095;
        }
        let chunks=(allocated+0x1fffff)/0x200000;let tb=((chunks+255)/256+chunks)*4096;
        if spans.len()!=LAYERS*7||fb.0[2]!=0||fb.1!=0||fb.0[0]>PT||PA+allocated as u64>fb.0[1].saturating_add(1)||PA+allocated as u64>0x200000000||PT+tb as u64>0x10000000{return false;}
        self.map_memory(VA,PA,allocated,PT,1);let pdb=self.data-0x5000;fence(Ordering::SeqCst);
        (self.wr)(self.bar+0xb830a0,(pdb>>8)as u32);(self.wr)(self.bar+0xb830a4,(pdb>>40)as u32);(self.wr)(self.bar+0xb830b0,0x80000007);
        let mut flushed=false;for _ in 0..20000{if(self.rd)(self.bar+0xb830b0)&0x80000000==0{flushed=true;break;}boot::stall(10);}if !flushed{return false;}
        event(log,format!("  M4 Q8_SWIZZLE_ARENA LOSSLESS=true tensors={} bytes={} PA=0x{:x} end=0x{:x}",spans.len(),allocated,PA,PA+allocated as u64));
        let decode=core::mem::take(&mut self.graph);self.batch_build=true;
        for &(_,src,dest,tiles,blocks) in &spans {self.p_add(6,tiles,&[src,VA+dest as u64],&[blocks as u32]);}
        self.p_finish(decode);if !self.p_submit(log){return false;}
        let mut checked=0usize;let start=self.time_ns();
        for (ti,&(off,_,dest,tiles,blocks)) in spans.iter().enumerate(){
            for first in (0..tiles).step_by(1800){let count=(tiles-first).min(1800);
                if !self.copy_buffer(VA+(dest+first*576)as u64,0x100000000,count*576){return false;}
                let seen=unsafe{core::slice::from_raw_parts(self.work.host as *const u32,count*144)};
                for z in 0..count {let block=first+z;let row=block/blocks*16;let b=block%blocks;
                    for i in 0..128 {let plane=i/32;let lane=i%32;let r=row+lane/4+(plane%2)*8;
                        let src=model.data+off+(r*blocks+b)*34+2+(plane/2)*16+(lane%4)*4;
                        if seen[z*144+i]!=u32::from_le_bytes(model.bytes[src..src+4].try_into().unwrap()){return false;}
                    }
                    for i in 0..16 {let src=model.data+off+((row+i)*blocks+b)*34;
                        if seen[z*144+128+i]!=half(u16::from_le_bytes(model.bytes[src..src+2].try_into().unwrap())).to_bits(){return false;}
                    }
                }
                checked+=count*576;
            }
            if ti%32==0{event(log,format!("  M4 Q8_SWIZZLE_VERIFY progress={}/{}",ti+1,spans.len()));}
        }
        self.swizzled_weights=spans.iter().map(|&(off,_,dest,_,_)|(off,VA+dest as u64)).collect();
        event(log,format!("  *** M4 Q8_SWIZZLE_VERIFY PASS=true checked_bytes={} elapsed_ns={} every_quant_and_scale=true ***",checked,self.time_ns()-start));true
    }
    fn check_prefill_gemm(&mut self,model:&Model,log:&mut Vec<String>)->bool {
        const VECTORS:usize=32;
        let name=if Q4_WEIGHTS{"blk.0.ffn_gate.weight"}else{"blk.0.attn_q.weight"};let t=model.tensor(name);let cols=t.n;let blocks=cols/32;
        assert!(VECTORS*cols*4<=self.work.size&&8*t.m*4+0x20000<=self.work.size);
        let x:Vec<f32>=(0..VECTORS*cols).map(|i|(((i%cols*7+i/cols*11)%37)as f32-18.0)/16.0).collect();
        for(i,&v)in x.iter().enumerate(){unsafe{core::ptr::write_volatile((self.work.host as *mut f32).add(i),v);}}
        let decode=core::mem::take(&mut self.graph);self.batch_build=true;
        self.p_quant(0x100000000,cols,VECTORS);self.p_gemm(model,name,VECTORS,Q);self.p_finish(decode);
        if !self.p_submit(log){return false;}
        let mut scales=vec![0f32;VECTORS*blocks];let mut quant=vec![0i8;VECTORS*cols];
        for b in 0..VECTORS*blocks{let mut a=0f32;for j in 0..32{a=a.max(libm::fabsf(x[b*32+j]));}scales[b]=a/127.0;
            for j in 0..32{quant[b*32+j]=libm::rintf(x[b*32+j]*(if a>0.0{127.0/a}else{0.0})).clamp(-127.0,127.0)as i8;}}
        let mut maxerr=0f32;let mut checked=0;
        for token in 0..VECTORS{
            if token%8==0&&!self.copy_buffer(Q+(token*t.m*4)as u64,0x100020000,8*t.m*4){return false;}
            for row in [0,1,7,8,15,16,31,32,63,64,127,t.m-1]{let mut expect=0f32;
            for b in 0..blocks{let off=model.data+t.off+(row*blocks+b)*34;let mut dot=0i32;
                let d=if tensor_q4(name){let q4=gpu_q4::block_ref(&model.bytes[off..off+34]);
                    for j in 0..32{let w=((q4[2+j%16]>>(if j<16{0}else{4}))&15)as i32-8;dot+=w*(quant[token*cols+b*32+j]as i32);}
                    half(u16::from_le_bytes([q4[0],q4[1]]))
                }else{for j in 0..32{dot+=(model.bytes[off+2+j]as i8 as i32)*(quant[token*cols+b*32+j]as i32);}half(u16::from_le_bytes(model.bytes[off..off+2].try_into().unwrap()))};
                expect+=dot as f32*(d*scales[token*blocks+b]);}
            let seen=unsafe{read_volatile((self.work.host.add(0x20000)as *const f32).add((token%8)*t.m+row))};
            if !seen.is_finite(){return false;}maxerr=maxerr.max(libm::fabsf(seen-expect));checked+=1;
        }}
        let pass=maxerr<0.003;event(log,format!("  *** M4 PREFILL_GEMM_CPU_CHECK vectors={} distinct=true samples={} max_abs_error={} PASS={} ***",VECTORS,checked,maxerr,pass));pass
    }
    pub(super) fn prepare_prefill(&mut self,model:&Model,count:usize,log:&mut Vec<String>) {
        let n=(count+7)&!7;assert!(count>0&&n<=SEQ&&n<=256);
        let decode=core::mem::take(&mut self.graph);self.batch_build=true;
        let emb=model.tensor("token_embd.weight");self.p_add(2,n*12,&[0x40000000+model.data as u64+emb.off as u64,0x100010000,X],&[]);
        for layer in 0..LAYERS{let meta=D+0x1000000+layer as u64*0x10000;let keys=KV_KEYS+(layer*SEQ*KV*4)as u64;let values=KV_VALUES+(layer*SEQ*KV*4)as u64;
            let name=|s:&str|format!("blk.{}.{}",layer,s);
            self.p_add(1,n,&[X,TMP,meta,XN],&[if layer==0{0}else{1}]);self.p_quant(XN,DIM,n);
            self.p_gemm(model,&name("attn_q.weight"),n,Q);self.p_gemm(model,&name("attn_k.weight"),n,K);self.p_gemm(model,&name("attn_v.weight"),n,V);
            self.p_add(3,n*6,&[Q,K,V,meta+0x4000,meta+0x6000,meta+0x7000,D+0x1200000,keys,values],&[]);
            self.p_add(4,n*12,&[Q,keys,values,ATT],&[]);
            self.p_quant(ATT,DIM,n);self.p_gemm(model,&name("attn_output.weight"),n,TMP);
            self.p_add(1,n,&[X,TMP,meta+0x2000,XN],&[1]);self.p_quant(XN,DIM,n);
            self.p_gemm(model,&name("ffn_gate.weight"),n,GATE);self.p_gemm(model,&name("ffn_up.weight"),n,UP);
            self.graph_add(0xc0460000,18,include_bytes!("../kernels/silu_mul.sass").len(),(n*HIDDEN+127)/128,&[GATE,UP],&[(n*HIDDEN)as u32],0,false);
            self.p_quant(GATE,HIDDEN,n);self.p_gemm(model,&name("ffn_down.weight"),n,TMP);
        }
        // Only the final real prompt token needs logits. Hand its residual state
        // to the existing decode output head; KV already contains the full prefix.
        for(src,dst)in [(X,D),(TMP,D+0x60000)] {self.graph_add(0x04002000,8,include_bytes!("../kernels/copy_u32.sass").len(),12,&[src+((count-1)*DIM*4)as u64,dst],&[DIM as u32],0,false);}
        self.graph_add(0xc0470000,37,include_bytes!("../kernels/fused_resnorm.sass").len(),1,&[D,D+0x60000,D+0x1300000,D+0x10000],&[1],0,false);
        self.graph_add(0xc04a0000,19,include_bytes!("../kernels/quant_act.sass").len(),12,&[D+0x10000,D+0xa0000],&[DIM as u32],0,false);
        let t=model.tensor("output.weight");let w=self.vram_weights.iter().find(|&&(off,_)|off==t.off).unwrap().1;
        let(code,regs,bytes)=if tensor_q4(t.name){(0xc0520000,30,include_bytes!("../kernels/linear_q4.sass").len())}else{(0xc04b0000,38,include_bytes!("../kernels/linear_dp4a.sass").len())};
        self.graph_add(code,regs,bytes,t.m,&[w,D+0xa0000,D+0x1400000],&[t.n as u32,t.m as u32],0,false);
        let tiles=(model.tokens.len()+1023)/1024;
        self.graph_add(0xc04e0000,33,include_bytes!("../kernels/greedy_tiles.sass").len(),tiles,&[D+0x1400000,D+0xb0000],&[model.tokens.len()as u32],0,false);
        self.graph_add(0xc04f0000,31,include_bytes!("../kernels/greedy_finish.sass").len(),1,&[D+0xb0000,0x100018000],&[tiles as u32],0,false);
        self.p_finish(decode);event(log,format!("  M4 PREFILL_READY tokens={} padded={} jobs={} GEMM=IMMA_M16N8K32 CTA_TILE=16x32 LOSSLESS_SWIZZLE=true FP32_KV=true",count,n,self.prefill.len()));
    }
    pub(super) fn run_prefill(&mut self,model:&Model,prompt:&[usize],log:&mut Vec<String>)->Option<usize>{
        let n=(prompt.len()+7)&!7;
        for i in 0..n{unsafe{core::ptr::write_volatile((self.work.host.add(0x10000)as *mut u32).add(i),prompt.get(i).copied().unwrap_or(prompt[0])as u32);}}
        fence(Ordering::SeqCst);if !self.p_submit(log){return None;}
        let next=unsafe{read_volatile(self.work.host.add(0x18004)as *const u32)}as usize;let bad=unsafe{read_volatile(self.work.host.add(0x18008)as *const u32)};
        if bad!=0||next>=model.tokens.len(){event(log,format!("  M4 PREFILL_GREEDY invalid id={} bad={}",next,bad));return None;}Some(next)
    }
}
