//! Bounded SSD -> CPU LRU -> VRAM LRU backend for the correctness-first graph.
//! All linears use the real GPU kernel. Nonlinear/state operations remain CPU.
//! Every transfer and consuming kernel is fenced before cache leases release.
use super::*;
use super::frontier_cache::{Cache,Key,PackedCache,PackedLease};
use super::frontier_forward::{Weights,Error,LinearRequest};
use super::frontier_gguf::{ModelIndex,Tensor};
use super::frontier_uefi::FullMedia;
use super::frontier_core as fc;
use alloc::collections::BTreeMap;
type Result<T>=core::result::Result<T,Error>;
const SLOT:usize=2*1024*1024;
const CPU_SLOTS:usize=22000;
const CPU_BUDGET:usize=16*1024*1024*1024;
const GPU_SLOTS:usize=2560;
const ALWAYS_VA:u64=0x200000000;
const CACHE_VA:u64=0x400000000;
const SCRATCH_VA:u64=0xe0000000;
const WORK_VA:u64=0x100000000;
const STAGING:usize=0x80000;
const CHUNK:usize=0x80000;
const MATVEC:&[u8]=include_bytes!("../kernels/frontier_matvec.sass");
const MATVEC_C2:&[u8]=include_bytes!("../kernels/frontier_matvec.c2");
const GDN:&[u8]=include_bytes!("../kernels/frontier_gdn.sass");
const STATE_VA:u64=0x600000000;

pub struct Backend<'g,'r,'m,R:Fn(u64)->u32,W:Fn(u64,u32)> {
    gpu:&'g mut Gpu<'r,R,W>,model:&'m ModelIndex<'m>,media:FullMedia,
    tensors:Vec<(usize,Tensor<'m>)>,resident:Vec<Option<u64>>,
    cpu:Cache,vram:PackedCache,cpu_data:super::frontier_cpu_pool::Pool,vectors:BTreeMap<usize,Vec<f32>>,
    log:&'g mut Vec<String>,pub linears:u64,pub uploaded:u64,timing:[u64;6],op_profile:Vec<(u64,u64,usize)>,quiet:bool,
    graph_profile:[(u64,u64,u64);6],profile_anomalies:u64,
    seen_experts:BTreeMap<Key,usize>,
    dispatch_groups:[u64;2],
}
impl<'g,'r,'m,R:Fn(u64)->u32,W:Fn(u64,u32)> Backend<'g,'r,'m,R,W> {
    pub fn new(gpu:&'g mut Gpu<'r,R,W>,model:&'m ModelIndex<'m>,media:FullMedia,
        fb:([u64;4],u32),log:&'g mut Vec<String>)->Result<Self> {
        let tensors:Vec<_>=model.shards.iter().enumerate().flat_map(|(i,s)|s.tensors.iter().map(move|t|(i,*t))).collect();
        let mut resident=vec![None;tensors.len()];let mut used=0usize;
        for (i,(_,t)) in tensors.iter().enumerate() {
            if t.name=="per_layer_token_embd.weight"||t.name.ends_with("_exps.weight"){continue;}
            resident[i]=Some(ALWAYS_VA+used as u64);
            used=used.checked_add((t.bytes as usize+4095)&!4095).ok_or(Error::Shape)?;
        }
        setup(gpu,used,fb,log)?;
        let n_tensors=tensors.len();
        let mut this=Self{gpu,model,media,tensors,resident,cpu:Cache::new(CPU_SLOTS,SLOT).map_err(|_|Error::Backend)?,
            vram:PackedCache::new(GPU_SLOTS*SLOT,1024*1024).map_err(|_|Error::Backend)?,
            cpu_data:super::frontier_cpu_pool::Pool::new(CPU_SLOTS,SLOT,CPU_BUDGET).map_err(|_|Error::Backend)?,
            vectors:BTreeMap::new(),log,linears:0,uploaded:0,timing:[0;6],op_profile:vec![(0,0,0);n_tensors],quiet:false,
            graph_profile:[(0,0,0);6],profile_anomalies:0,seen_experts:BTreeMap::new(),dispatch_groups:[0;2]};
        let mut buffer=vec![0u8;CHUNK];let start=this.gpu.time_ns();
        for i in 0..this.tensors.len(){if let Some(va)=this.resident[i] {
            let (disk,t)=this.tensors[i];
            for off in (0..t.bytes as usize).step_by(CHUNK){let n=(t.bytes as usize-off).min(CHUNK);
                this.media.read(disk,t.offset+off as u64,&mut buffer[..n]).map_err(|_|Error::Backend)?;
                this.upload(va+off as u64,&buffer[..n],true)?;
            }
            if i%40==0{event(this.log,format!("  FRONTIER_UPLOAD tensor={} bytes_verified={} elapsed_ns={}",t.name,this.uploaded,this.gpu.time_ns()-start));}
        }}
        event(this.log,format!("  FRONTIER_UPLOAD complete=true bytes_verified={} elapsed_ns={} inference=false",this.uploaded,this.gpu.time_ns()-start));
        this.timing=[0;6]; // Forward-only phases; exclude one-time residency setup.
        Ok(this)
    }
    fn find(&self,name:&str)->Result<usize>{self.tensors.iter().position(|(_,t)|t.name==name).ok_or(Error::Backend)}
    pub fn emit(&mut self,s:String){event(self.log,s);}
    pub fn begin_quiet(&mut self){
        self.gpu.window.reset_counts();
        self.quiet=true;self.linears=0;self.uploaded=0;self.timing.fill(0);self.op_profile.fill((0,0,0));
        self.graph_profile.fill((0,0,0));self.profile_anomalies=0;
        self.seen_experts.clear();
        self.dispatch_groups.fill(0);
        super::frontier_cpu_math::reset_counts();
        super::frontier_cpu_expf::reset();
        super::frontier_cpu_axpy::reset();
        self.vram.hits=0;self.vram.misses=0;self.vram.evictions=0;
        self.cpu.hits=0;self.cpu.misses=0;self.cpu.evictions=0;
        self.media.stats=super::frontier_storage::Stats::default();
    }
    fn upload(&mut self,dest:u64,bytes:&[u8],verify:bool)->Result<()> {
        let start=self.gpu.time_ns();
        for off in (0..bytes.len()).step_by(CHUNK){let n=(bytes.len()-off).min(CHUNK);let padded=(n+3)&!3;
            unsafe{core::ptr::write_bytes(self.gpu.work.host.add(STAGING),0,padded);
                core::ptr::copy_nonoverlapping(bytes[off..].as_ptr(),self.gpu.work.host.add(STAGING),n);}
            fence(Ordering::SeqCst);
            if !self.gpu.frontier_copy_buffer(WORK_VA+STAGING as u64,dest+off as u64,padded){return Err(Error::Backend);}
            if verify {
                unsafe{core::ptr::write_bytes(self.gpu.work.host.add(STAGING),0xa5,padded);}
                fence(Ordering::SeqCst);
                if !self.gpu.frontier_copy_buffer(dest+off as u64,WORK_VA+STAGING as u64,padded){return Err(Error::Backend);}
                let seen=unsafe{core::slice::from_raw_parts(self.gpu.work.host.add(STAGING),n)};
                if seen!=&bytes[off..off+n]{return Err(Error::Backend);}
            }
            self.uploaded+=n as u64;
        }self.timing[1]+=self.gpu.time_ns()-start;Ok(())
    }
    fn expert(&mut self,id:usize,expert:usize)->Result<(u64,PackedLease)> {
        let (disk,t)=self.tensors[id];let (offset,size)=t.expert(expert as u32).map_err(|_|Error::Shape)?;
        let key=Key{tensor:id,expert};
        if self.quiet{self.seen_experts.insert(key,size as usize);}
        let(vlease,hit)=self.vram.acquire(key,size as usize).map_err(|_|Error::Backend)?;
        let va=CACHE_VA+vlease.offset()as u64;
        if !hit {
            let (clease,chit)=self.cpu.acquire(key,size as usize).map_err(|_|Error::Backend)?;
            if !chit {
                let start=self.gpu.time_ns();
                let buffer=match self.cpu_data.prepare(clease.slot,size as usize){
                    Ok(v)=>v,Err(error)=>{
                        event(self.log,format!("  FRONTIER_CPU_CACHE allocation_failed={:?} requested={} budget={} correctness=false",error,size,CPU_BUDGET));
                        let _=self.cpu.abort(clease);let _=self.vram.abort(vlease);return Err(Error::Backend);
                    }
                };
                if self.media.read(disk,offset,buffer).is_err(){
                    let _=self.cpu.abort(clease);let _=self.vram.abort(vlease);return Err(Error::Backend);}
                self.timing[0]+=self.gpu.time_ns()-start;
                self.cpu.commit(clease).map_err(|_|Error::Backend)?;
            }
            // Temporarily move ownership, not duplicate an expert buffer.
            let bytes=self.cpu_data.take(clease.slot).map_err(|_|Error::Backend)?;
            let uploaded=self.upload(va,bytes.as_ref(),true);
            self.cpu_data.put(bytes).map_err(|_|Error::Backend)?;
            self.cpu.release(clease).map_err(|_|Error::Backend)?;
            if uploaded.is_err(){let _=self.vram.abort(vlease);return Err(Error::Backend);}
            self.vram.commit(vlease).map_err(|_|Error::Backend)?;
        }
        Ok((va,vlease))
    }
    fn cpu_read(&mut self,id:usize,offset:u64,bytes:usize,values:usize)->Result<Vec<f32>> {
        let start=self.gpu.time_ns();
        let(disk,t)=self.tensors[id];let mut raw=vec![0;bytes];
        self.media.read(disk,offset,&mut raw).map_err(|_|Error::Backend)?;
        let mut result=vec![0.;values];fc::dequantize(t.kind,&raw,&mut result).map_err(|_|Error::Backend)?;
        self.timing[5]+=self.gpu.time_ns()-start;Ok(result)
    }
    pub fn report(&mut self) {
        let(calls,fallback)=super::frontier_cpu_axpy::counts();
        event(self.log,format!("  FRONTIER_CPU_AXPY fast_calls={} fallback_calls={} separate_FP32_mul_add=true",calls,fallback));
        event(self.log,format!("  FRONTIER_HYBRID_DISPATCH small_QMD_groups={} large_multi_groups={} policy=only_11_or_23 independent_jobs_only=true",
            self.dispatch_groups[0],self.dispatch_groups[1]));
        let(enabled,calls,fallback)=super::frontier_cpu_math::counts();
        event(self.log,format!("  FRONTIER_CPU_MATH SSE2_norm_enabled={} calls={} fallback_calls={} rust_softfloat_ABI_unchanged=true",enabled,calls,fallback));
        let(calls,fallback)=super::frontier_cpu_math::dot_counts();
        event(self.log,format!("  FRONTIER_CPU_DOT enabled={} calls={} fallback_calls={} sequential_FP32=true",enabled,calls,fallback));
        let(fast,special)=super::frontier_cpu_expf::counts();
        event(self.log,format!("  FRONTIER_CPU_ACTIVATION fast_calls={} exact_sigmoid_silu=true",super::frontier_cpu_expf::activation_count()));
        event(self.log,format!("  FRONTIER_CPU_EXP fast_calls={} slow_special_cases={} same_libm_polynomial=true",fast,special));
        event(self.log,format!("  FRONTIER_EXPERT_WORKSET distinct_matrices={} raw_bytes={} includes_GPU_hits=true includes_CPU_hits=true",
            self.seen_experts.len(),self.seen_experts.values().sum::<usize>()));
        for (name,&(inclusive,other,calls)) in ["HC_mix","attention_and_residual","FFN_and_residual","PLE","embedding","output_head"].iter().zip(&self.graph_profile){
            event(self.log,format!("  FRONTIER_GRAPH_PROFILE stage={} inclusive_ns={} outside_backend_timers_ns={} calls={} anomalies={}",
                name,inclusive,other,calls,self.profile_anomalies));
        }
        let(calls,writes)=self.gpu.window.counts();
        event(self.log,format!("  FRONTIER_MMIO aperture_calls={} bank_writes={} selector_cached=true",calls,writes));
        let mut ranked:Vec<usize>=(0..self.op_profile.len()).filter(|&i|self.op_profile[i].1!=0).collect();
        ranked.sort_unstable_by(|a,b|self.op_profile[*b].0.cmp(&self.op_profile[*a].0));
        for i in ranked.into_iter().take(12){let(ns,calls,jobs)=self.op_profile[i];
            event(self.log,format!("  FRONTIER_OP_TIME group_label={} calls={} max_jobs_per_call={} launch_wait_ns={}",self.tensors[i].1.name,calls,jobs,ns));}
        event(self.log,format!("  FRONTIER_CACHE_BYTES GPU_payload={} GPU_arena={} CPU_payload={} CPU_cap={}",
            self.vram.resident_payload_bytes(),GPU_SLOTS*SLOT,self.cpu.resident_payload_bytes(),CPU_BUDGET));
        event(self.log,format!("  FRONTIER_CPU_ALLOCATION vec_capacity_bytes={} hard_byte_budget={} directory_slots={} allocator_metadata_excluded=true",
            self.cpu_data.used(),CPU_BUDGET,CPU_SLOTS));
        event(self.log,format!("  FRONTIER_CACHE linears={} GPU_hit={} GPU_miss={} GPU_evictions={} CPU_hit={} CPU_miss={} CPU_evictions={} SSD_requests={} SSD_requested={} SSD_transferred={} SSD_calls={} failures={}",
            self.linears,self.vram.hits,self.vram.misses,self.vram.evictions,self.cpu.hits,self.cpu.misses,self.cpu.evictions,
            self.media.stats.requests,self.media.stats.requested,self.media.stats.transferred,self.media.stats.calls,self.media.stats.failures));
        event(self.log,format!("  FRONTIER_TIMING expert_read_ns={} upload_verify_ns={} linear_launch_wait_ns={} output_readback_ns={} recurrent_exchange_ns={} cpu_rows_vectors_ns={} setup_excluded=true tracing_enabled={}",
            self.timing[0],self.timing[1],self.timing[2],self.timing[3],self.timing[4],self.timing[5],!self.quiet));
    }
}
impl<R:Fn(u64)->u32,W:Fn(u64,u32)> Weights for Backend<'_,'_,'_,R,W> {
    fn phase_mark(&self)->(u64,u64){(self.gpu.time_ns(),self.timing.iter().sum())}
    fn phase_end(&mut self,stage:usize,mark:(u64,u64)){
        let(end,tracked)=self.phase_mark();let elapsed=end-mark.0;let tracked=tracked-mark.1;
        if tracked>elapsed{self.profile_anomalies+=1;}
        let p=&mut self.graph_profile[stage];p.0+=elapsed;p.1+=elapsed.saturating_sub(tracked);p.2+=1;
    }
    fn batch(&mut self,requests:&[LinearRequest<'_>])->Result<Vec<Vec<f32>>> {
        if requests.is_empty()||requests.len()>super::frontier_batch::MAX{return Err(Error::Shape);}
        let profile_id=self.find(requests[0].name)?;
        let mut jobs=Vec::new();let mut leases=Vec::new();let mut prepared=Vec::new();let mut spans=Vec::new();
        let mut input_offset=0x10000usize;let mut output_offset=0usize;
        for request in requests {
            let id=self.find(request.name)?;let(_,t)=self.tensors[id];let x=request.input;
            if t.dims[0]as usize!=x.len()||x.len()>16384||!x.iter().all(|v|v.is_finite()){return Err(Error::Shape);}
            let m=if t.rank==1{1}else{t.dims[1]as usize};
            let input_end=input_offset.checked_add(x.len()*4).ok_or(Error::Shape)?;
            let output_end=output_offset.checked_add(m*4).ok_or(Error::Shape)?;
            if input_end>0x80000||output_end>0x200000{return Err(Error::Shape);}
            let key=(id,request.expert);
            let va=if let Some((_,va))=prepared.iter().find(|(k,_)|*k==key){*va}else{
                let va=if let Some(e)=request.expert{let(v,l)=self.expert(id,e)?;leases.push(l);v}
                    else{if t.rank>2{return Err(Error::Shape);}self.resident[id].ok_or(Error::Backend)?};
                prepared.push((key,va));va
            };
            unsafe{core::ptr::copy_nonoverlapping(x.as_ptr(),self.gpu.work.host.add(input_offset)as *mut f32,x.len());}
            jobs.push(super::frontier_batch::Dispatch{weight:va,input:WORK_VA+input_offset as u64,output:SCRATCH_VA+output_offset as u64,
                kind:t.kind,n:x.len()as u32,m:m as u32,row_bytes:t.row_bytes as u32});
            spans.push((output_offset/4,output_end/4));input_offset=input_end;output_offset=output_end;
        }
        fence(Ordering::SeqCst);
        let start=self.gpu.time_ns();
        let multi=super::frontier_dispatch::use_multi(jobs.len());
        let done=if multi {
            self.gpu.frontier_multi_launch(&jobs)
        }else{self.gpu.frontier_batch_launch(&jobs)};
        if !done{return Err(Error::Backend);}
        self.dispatch_groups[multi as usize]+=1;
        let elapsed=self.gpu.time_ns()-start;self.timing[2]+=elapsed;
        let p=&mut self.op_profile[profile_id];p.0+=elapsed;p.1+=1;p.2=p.2.max(jobs.len());
        // All private completions, or the combined completion, were checked.
        // No lease is released on timeout.
        for lease in leases{self.vram.release(lease).map_err(|_|Error::Backend)?;}
        self.linears+=jobs.len()as u64;
        let mut combined=vec![0f32;output_offset/4];
        let start=self.gpu.time_ns();
        for off in (0..output_offset).step_by(CHUNK){let n=(output_offset-off).min(CHUNK);
            if !self.gpu.frontier_copy_buffer(SCRATCH_VA+off as u64,WORK_VA+STAGING as u64,n){return Err(Error::Backend);}
            unsafe{core::ptr::copy_nonoverlapping(self.gpu.work.host.add(STAGING),(combined.as_mut_ptr()as *mut u8).add(off),n);}
        }
        self.timing[3]+=self.gpu.time_ns()-start;
        if !combined.iter().all(|v|v.is_finite()){return Err(Error::NonFinite);}
        Ok(spans.into_iter().map(|(lo,hi)|combined[lo..hi].to_vec()).collect())
    }
    fn recurrent_on_gpu(&self)->bool {true}
    fn recurrent(&mut self,layer:usize,position:usize,conv:&[f32],alpha:&[f32],beta:&[f32],a:&[f32],dt:&[f32],_state:&mut[f32])->Result<Vec<f32>> {
        let start=self.gpu.time_ns();
        if layer>=48||layer%4==3||conv.len()!=10240||[alpha,beta,a,dt].iter().any(|s|s.len()!=48){return Err(Error::Shape);}
        let index=(layer/4)*3+layer%4;let address=STATE_VA+(index*128*128*48*4)as u64;
        let mut offset=0;for input in [conv,alpha,beta,a,dt]{
            unsafe{core::ptr::copy_nonoverlapping(input.as_ptr(),(self.gpu.work.host.add(0x40000)as *mut f32).add(offset),input.len());}
            offset+=input.len();
        }
        fence(Ordering::SeqCst);
        if !self.gpu.frontier_launch(0xc0420000,34,GDN.len(),48*128/4,
            &[WORK_VA+0x40000,address,SCRATCH_VA],&[u32::from(position==0)]){return Err(Error::Backend);}
        if !self.gpu.frontier_copy_buffer(SCRATCH_VA,WORK_VA+STAGING as u64,6144*4){return Err(Error::Backend);}
        let mut out=vec![0f32;6144];
        unsafe{core::ptr::copy_nonoverlapping(self.gpu.work.host.add(STAGING)as *const f32,out.as_mut_ptr(),out.len());}
        if !out.iter().all(|v|v.is_finite()){return Err(Error::NonFinite);}
        self.timing[4]+=self.gpu.time_ns()-start;Ok(out)
    }
    fn linear(&mut self,name:&str,expert:Option<usize>,x:&[f32])->Result<Vec<f32>> {
        let id=self.find(name)?;let(_,t)=self.tensors[id];
        if t.dims[0]as usize!=x.len()||x.len()>16384||!x.iter().all(|v|v.is_finite()){
            return Err(Error::Shape);
        }
        let m=if t.rank==1{1}else{t.dims[1]as usize};if m*4>0x200000{return Err(Error::Shape);}
        let(va,lease)=if let Some(e)=expert {let(v,l)=self.expert(id,e)?;(v,Some(l))}
            else{if t.rank>2{return Err(Error::Shape);}(self.resident[id].ok_or(Error::Backend)?,None)};
        unsafe{core::ptr::copy_nonoverlapping(x.as_ptr(),self.gpu.work.host.add(0x40000)as *mut f32,x.len());}
        fence(Ordering::SeqCst);
        let start=self.gpu.time_ns();
        if !self.gpu.frontier_launch(0xc0410000,38,MATVEC.len(),(m+3)/4,
            &[va,WORK_VA+0x40000,SCRATCH_VA,WORK_VA,WORK_VA+1024],
            &[t.kind,x.len()as u32,m as u32,t.row_bytes as u32]) {
            // Keep cache lease held on timeout: GPU may still be using it.
            return Err(Error::Backend);
        }
        let elapsed=self.gpu.time_ns()-start;self.timing[2]+=elapsed;
        let p=&mut self.op_profile[id];p.0+=elapsed;p.1+=1;p.2=p.2.max(1);
        self.linears+=1;
        if let Some(l)=lease{self.vram.release(l).map_err(|_|Error::Backend)?;}
        let mut result=vec![0f32;m];
        let start=self.gpu.time_ns();
        for off in (0..m*4).step_by(CHUNK){let n=(m*4-off).min(CHUNK);
            if !self.gpu.frontier_copy_buffer(SCRATCH_VA+off as u64,WORK_VA+STAGING as u64,n){return Err(Error::Backend);}
            unsafe{core::ptr::copy_nonoverlapping(self.gpu.work.host.add(STAGING),
                (result.as_mut_ptr()as *mut u8).add(off),n);}
        }
        self.timing[3]+=self.gpu.time_ns()-start;
        if !result.iter().all(|v|v.is_finite()){return Err(Error::NonFinite);}
        if !self.quiet&&name=="output.weight"&&self.linears<3000{self.report();}
        Ok(result)
    }
    fn vector(&mut self,name:&str)->Result<Vec<f32>> {
        let id=self.find(name)?;if let Some(v)=self.vectors.get(&id){return Ok(v.clone());}
        let(_,t)=self.tensors[id];if t.bytes>1024*1024{return Err(Error::Shape);}
        let values=t.dims[..t.rank].iter().product::<u64>()as usize;
        let v=self.cpu_read(id,t.offset,t.bytes as usize,values)?;self.vectors.insert(id,v.clone());Ok(v)
    }
    fn row(&mut self,name:&str,row:usize)->Result<Vec<f32>> {
        let id=self.find(name)?;let(_,t)=self.tensors[id];let(offset,size)=t.rows(row as u64,1).map_err(|_|Error::Shape)?;
        if size>1024*1024{return Err(Error::Shape);}self.cpu_read(id,offset,size as usize,t.dims[0]as usize)
    }
    fn trace(&mut self,layer:usize,stage:&str,x:&[f32]) {
        if self.quiet{return;}
        let max=x.iter().copied().map(f32::abs).fold(0f32,f32::max);
        event(self.log,format!("  FRONTIER_LAYER layer={} stage={} max_abs={} linears={}",layer,stage,max,self.linears));
    }
}

pub(super) fn setup<R:Fn(u64)->u32,W:Fn(u64,u32)>(gpu:&mut Gpu<'_,R,W>,used:usize,
    fb:([u64;4],u32),log:&mut Vec<String>)->Result<()> {
    // Explicit nonoverlapping physical arenas, checked against RM's free FB.
    // Tables: always [0a000000,0b000000), code 0b000000, scratch0b200000,
    // cache [0e000000,0f000000). No physical write can approach WPR.
    let arenas=[(0x09800000u64,0x2000usize),(0x0a000000,0x1000000),
        (0x0b000000,0x200000),(0x0b200000,0x2000),(0x0b300000,0x100000),(0x0c000000,0x800000),
        (0x0c800000,0x200000),(0x0e000000,0x1000000),
        (0x10000000,used),(0x160000000,GPU_SLOTS*SLOT),(0x2a0000000,0x08000000)];
    if fb.0[2]!=0||fb.1!=0||0x10000000u64+used as u64>0x160000000 {
        return Err(Error::Backend);
    }
    for &(start,n) in &arenas {
        let end=start.checked_add(n as u64).ok_or(Error::Backend)?;
        if start<fb.0[0]||end>fb.0[1].saturating_add(1)||end>0x2f0920000{return Err(Error::Backend);}
    }
    for (i,&(a,n)) in arenas.iter().enumerate(){for &(b,m) in &arenas[i+1..]{
        if a<b+m as u64&&b<a+n as u64{return Err(Error::Backend);}
    }}
    let table_bytes=|n:usize|{let c=(n+0x1fffff)/0x200000;((c+255)/256+c)*4096};
    if table_bytes(used)>0x1000000||table_bytes(GPU_SLOTS*SLOT)>0x1000000{return Err(Error::Backend);}
    event(log,format!("  FRONTIER_ARENAS always_alloc={} cpu_cache_cap={} vram_cache={} lookup=SSD free=[0x{:x},0x{:x}]",
        used,CPU_BUDGET,GPU_SLOTS*SLOT,fb.0[0],fb.0[1]));
    gpu.map(WORK_VA,&gpu.work,0x09800000);
    gpu.map_memory(0xc0000000,0x0c000000,0x800000,0x0b000000,1);
    gpu.map_memory(SCRATCH_VA,0x0c800000,0x200000,0x0b200000,1);
    gpu.map_memory(ALWAYS_VA,0x10000000,used,0x0a000000,1);
    gpu.map_memory(CACHE_VA,0x160000000,GPU_SLOTS*SLOT,0x0e000000,1);
    gpu.map_memory(STATE_VA,0x2a0000000,0x08000000,0x0b300000,1);
    if MATVEC.len()!=16640||MATVEC_C2.len()!=80{return Err(Error::Backend);}
    for (i,b) in MATVEC.chunks_exact(4).enumerate(){gpu.write(0x0c410000+i as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
    if GDN.len()>65536||!include_bytes!("../kernels/frontier_gdn.c2").is_empty(){return Err(Error::Backend);}
    for (i,b) in GDN.chunks_exact(4).enumerate(){gpu.write(0x0c420000+i as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
    for i in 0..32{gpu.write(0x0c7f0000+i*4,0);}
    for (i,b) in MATVEC_C2.chunks_exact(4).enumerate(){gpu.write(0x0c7f0000+i as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
    let copy=include_bytes!("../kernels/copy_u32.sass");
    for(i,b)in copy.chunks_exact(4).enumerate(){gpu.write(gpu.data+0x6000+i as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
    // copy_buffer's template must not inherit frontier constant2/shared state.
    field(&mut gpu.q,592,607,128);field(&mut gpu.q,642,642,0);
    field(&mut gpu.q,544,561,0);field(&mut gpu.q,1075,1087,0x180/16);
    unsafe {
        for(i,v)in fc::iq3_grid().iter().enumerate(){core::ptr::write_volatile((gpu.work.host as *mut u32).add(i),*v);}
        for(i,v)in fc::iq4_values().iter().enumerate(){core::ptr::write_volatile((gpu.work.host.add(1024)as *mut i32).add(i),*v as i32);}
    }
    let pdb=gpu.data-0x5000;fence(Ordering::SeqCst);
    (gpu.wr)(gpu.bar+0xb830a0,(pdb>>8)as u32);(gpu.wr)(gpu.bar+0xb830a4,(pdb>>40)as u32);
    (gpu.wr)(gpu.bar+0xb830b0,0x80000007);
    let mut flushed=false;for _ in 0..20000{if(gpu.rd)(gpu.bar+0xb830b0)&0x80000000==0{flushed=true;break;}boot::stall(10);}
    if !flushed{return Err(Error::Backend);}
    gpu.write(gpu.data+0x900,128);gpu.write(gpu.data+0x904,1);gpu.write(gpu.data+0x908,1);
    gpu.frontier_batch_init();
    if !gpu.frontier_multi_init(){return Err(Error::Backend);}
    gpu.frontier_copy_init();
    event(log,String::from("  FRONTIER_COPY_TEMPLATE initialized=true private_QMD=true upload_full_readback=true"));
    Ok(())
}
