//! Physical-GPU operator gate for the new model's quantization formats.
use super::*;
use super::frontier_core as fc;

impl<R:Fn(u64)->u32,W:Fn(u64,u32)> Gpu<'_,R,W> {
    pub(super) fn frontier_launch(&mut self,code:u64,regs:u64,code_bytes:usize,grid:usize,args:&[u64],tail:&[u32])->bool {
        let mut q=self.q;self.stamp+=1;
        field(&mut q,384,415,grid as u64);field(&mut q,592,607,128);
        field(&mut q,648,656,regs);field(&mut q,832,863,self.stamp as u64);
        field(&mut q,1536,1567,code);field(&mut q,256,287,code>>8);
        field(&mut q,1641,1649,((code_bytes+255)/256) as u64);
        field(&mut q,544,561,0);field(&mut q,1075,1087,0x200/16);
        field(&mut q,642,642,1);
        // Multi-matvec has its own compiler-generated C2 layout.
        field(&mut q,1152,1183,if code==0xc0430000{0xc07f1000}else{0xc07f0000});
        field(&mut q,1184,1200,0);field(&mut q,1202,1202,1);field(&mut q,1203,1215,8);
        for(i,v)in q.iter().enumerate(){self.write(self.data+0x800+i as u64*4,*v);}
        for i in 0..128 {self.write(self.data+0x900+i*4,0);}
        self.write(self.data+0x900,128);self.write(self.data+0x904,1);self.write(self.data+0x908,1);
        for(i,v)in args.iter().enumerate(){self.write64(self.data+0xa60+i as u64*8,*v);}
        for(i,v)in tail.iter().enumerate(){self.write(self.data+0xa60+args.len()as u64*8+i as u64*4,*v);}
        let words=[0x20012000|(0x1698>>2),0x1011,0x20012000|(0x2b4>>2),0x40008,0x20012000|(0x2c0>>2),9];
        for(i,v)in words.iter().enumerate(){self.write(self.data+0x200+i as u64*4,*v);}
        self.write(self.data+(self.put%32)as u64*8,0x4000200);
        self.write(self.data+(self.put%32)as u64*8+4,(words.len()as u32)<<10);
        fence(Ordering::SeqCst);let _=(self.rd)(self.ap(self.data+(self.put%32)as u64*8+4));
        self.put=(self.put+1)%32;self.write(self.userd+0x8c,self.put);
        let _=(self.rd)(self.ap(self.userd+0x8c));(self.wr)(self.bar+0xbb0090,self.token);
        self.submits+=1;
        for _ in 0..100000 {
            if(self.rd)(self.ap(self.data+0xf10))==self.stamp {fence(Ordering::SeqCst);self.kernels+=1;return true;}
            boot::stall(10);
        }false
    }
}

fn fixture(kind:u32,n:usize)->Vec<u8> {
    let mut bytes=vec![0;fc::row_bytes(kind,n).unwrap()];let mut state=0x13243546u32;
    for v in &mut bytes {state^=state<<13;state^=state>>17;state^=state<<5;*v=state as u8;}
    if kind==0 || kind==30 {
        for i in 0..n {let bits=(((i%127)as f32-63.0)/64.0).to_bits();
            if kind==0 {bytes[i*4..i*4+4].copy_from_slice(&bits.to_le_bytes());}
            else{bytes[i*2..i*2+2].copy_from_slice(&((bits>>16)as u16).to_le_bytes());}}
    }else {
        let(block,size,off)=match kind {8=>(32,34,0),14=>(256,210,208),18=>(256,98,0),
            20=>(32,18,0),23=>(256,136,0),_=>unreachable!()};
        for b in 0..n/block {let half=[0x2000u16,0x2800,0x3000,0xb000][b%4];
            bytes[b*size+off..b*size+off+2].copy_from_slice(&half.to_le_bytes());}
    }bytes
}

pub fn probe<R:Fn(u64)->u32,W:Fn(u64,u32)>(rd:&R,wr:&W,bar:u64,data:u64,userd:u64,token:u32,
    dma:&mut crate::RbioDma,template:&[u32;64],fb:([u64;4],u32),log:&mut Vec<String>)->bool {
    let(cpu_ok,cases)=super::frontier_cpu_math::selftest();
    event(log,format!("  FRONTIER_CPU_NORM_GATE SSE2={} cases={} bit_exact={} softfloat_ABI_unchanged=true",cpu_ok,cases,cpu_ok));
    if !cpu_ok{return false;}
    let(dot_ok,cases)=super::frontier_cpu_math::dot_selftest();
    event(log,format!("  FRONTIER_CPU_DOT_GATE SSE2={} cases={} bit_exact={}",dot_ok,cases,dot_ok));
    if !dot_ok{return false;}
    let(exp_ok,cases)=super::frontier_cpu_expf::selftest();
    event(log,format!("  FRONTIER_CPU_EXP_GATE SSE2={} cases={} bit_exact={}",exp_ok,cases,exp_ok));
    if !exp_ok{return false;}
    let(act_ok,cases)=super::frontier_cpu_expf::activation_selftest();
    event(log,format!("  FRONTIER_CPU_ACTIVATION_GATE SSE2={} cases={} bit_exact={}",act_ok,cases,act_ok));
    if !act_ok{return false;}
    let(axpy_ok,cases)=super::frontier_cpu_axpy::selftest();
    event(log,format!("  FRONTIER_CPU_AXPY_GATE SSE2={} cases={} bit_exact={} softfloat_ABI_unchanged=true",axpy_ok,cases,axpy_ok));
    if !axpy_ok{return false;}
    if fb.0[2]!=0||fb.1!=0||fb.0[0]>0x0b000000||fb.0[1]<0x0c7fffff {return false;}
    let Some(work)=dma.alloc_pages(256)else{return false;};
    let saved=rd(bar+0x1700);
    let mut gpu=Gpu{rd,wr,bar,data,userd,token,window:frontier_window::Window::new(),work,q:*template,put:2,stamp:0x4e000000,kernels:0,model_data:0,
        vram_weights:Vec::new(),swizzled_weights:Vec::new(),jobs:Vec::new(),submits:0,graph:Vec::new(),profile:[0;4],prefill:Vec::new(),batch_build:false};
    let result=(||->bool {
        gpu.map(0x100000000,&gpu.work,0x09800000);
        gpu.map_memory(0xc0000000,0x0c000000,0x800000,0x0b000000,1);
        let kernels=[(include_bytes!("../kernels/frontier_dequant.sass").as_slice(),
            include_bytes!("../kernels/frontier_dequant.c2").as_slice(),18),
            (include_bytes!("../kernels/frontier_matvec.sass").as_slice(),
            include_bytes!("../kernels/frontier_matvec.c2").as_slice(),38)];
        for(i,(code,_,_))in kernels.iter().enumerate(){if code.len()>65536{return false;}
            for(j,v)in code.chunks_exact(4).enumerate(){gpu.write(0x0c400000+i as u64*65536+j as u64*4,u32::from_le_bytes(v.try_into().unwrap()));}}
        unsafe {for (i,v) in fc::iq3_grid().iter().enumerate() {core::ptr::write_volatile((work.host as *mut u32).add(i),*v);}
            for (i,v) in fc::iq4_values().iter().enumerate() {core::ptr::write_volatile((work.host.add(1024)as *mut i32).add(i),*v as i32);}}
        let pdb=data-0x5000;fence(Ordering::SeqCst);wr(bar+0xb830a0,(pdb>>8)as u32);wr(bar+0xb830a4,(pdb>>40)as u32);wr(bar+0xb830b0,0x80000007);
        let mut flushed=false;for _ in 0..20000 {if rd(bar+0xb830b0)&0x80000000==0{flushed=true;break;}boot::stall(10);}if !flushed{return false;}
        for (ki,(code,c2,regs)) in kernels.iter().enumerate() {
            if c2.len()>128||c2.len()%4!=0 {return false;}
            for i in 0..32 {gpu.write(0x0c7f0000+i*4,0);}
            for(i,v)in c2.chunks_exact(4).enumerate(){gpu.write(0x0c7f0000+i as u64*4,u32::from_le_bytes(v.try_into().unwrap()));}
            for kind in [0,30,8,14,18,20,23] {
                let (n,m)=if ki==0 {(8192usize,1usize)}else{(2560,17)};
                let bytes=fixture(kind,n*m);let mut expected=vec![0f32;n*m];
                if fc::dequantize(kind,&bytes,&mut expected).is_err(){return false;}
                let x:Vec<f32>=(0..n).map(|i|((i%29)as f32-14.0)/32.0).collect();
                unsafe {core::ptr::copy_nonoverlapping(bytes.as_ptr(),work.host.add(0x10000),bytes.len());
                    core::ptr::copy_nonoverlapping(x.as_ptr(),work.host.add(0x40000)as *mut f32,x.len());
                    core::ptr::write_bytes(work.host.add(0x80000),0xff,if ki==0{n*4}else{m*4});}
                let start=gpu.time_ns();
                let done=if ki==0 {gpu.frontier_launch(0xc0400000,*regs,code.len(),(n+127)/128,
                    &[0x100010000,0x100080000,0x100000000,0x100000400],&[kind,n as u32])}
                else {gpu.frontier_launch(0xc0410000,*regs,code.len(),(m+3)/4,
                    &[0x100010000,0x100040000,0x100080000,0x100000000,0x100000400],
                    &[kind,n as u32,m as u32,fc::row_bytes(kind,n).unwrap()as u32])};
                let elapsed=gpu.time_ns()-start;let mut maxerr=0f32;let mut pass=done;
                for i in 0..if ki==0{n}else{m} {
                    let actual=unsafe{read_volatile((work.host.add(0x80000)as *const f32).add(i))};
                    let reference=if ki==0{expected[i]}else{let mut sum=0f32;for j in 0..n{sum+=expected[i*n+j]*x[j];}sum};
                    let err=libm::fabsf(actual-reference);maxerr=maxerr.max(err);
                    pass&=actual.is_finite() && if ki==0{actual.to_bits()==reference.to_bits()}
                        else{err<=0.0002+libm::fabsf(reference)*0.00002};
                }
                event(log,format!("  FRONTIER_GPU_OP op={} kind={} n={} m={} completed={} elapsed_ns={} max_abs_error={} PASS={}",
                    if ki==0{"dequant"}else{"matvec"},kind,n,m,done,elapsed,maxerr,pass));
                if !pass{return false;}
            }
        }
        arena_probe(&mut gpu,fb,log)
    })();
    wr(bar+0x1700,saved);
    event(log,format!("  *** FRONTIER_GPU_QUANT_GATE PASS={} kernels={} inference=false ***",result,gpu.kernels));
    result
}

fn arena_probe<R:Fn(u64)->u32,W:Fn(u64,u32)>(gpu:&mut Gpu<'_,R,W>,fb:([u64;4],u32),log:&mut Vec<String>)->bool {
    use super::frontier_gguf::Shard;
    let result=(||->Option<()> {
        let shards=[
            Shard::parse(include_bytes!("../../reference/shard2-header.gguf"),49983253824).ok()?,
            Shard::parse(include_bytes!("../../reference/shard3-header.gguf"),39992153376).ok()?];
        let used:usize=shards.iter().flat_map(|s|s.tensors.iter())
            .filter(|t|t.name!="per_layer_token_embd.weight"&&!t.name.ends_with("_exps.weight"))
            .map(|t|(t.bytes as usize+4095)&!4095).sum();
        super::frontier_backend::setup(gpu,used,fb,log).ok()?;
        let mut addresses=Vec::new();
        for (base,len) in [(0x200000000u64,used),(0x400000000,5*1024*1024*1024),(0x600000000,0x08000000)] {
            for off in [0,0x1ff000,0x200000,0x1ffff000,0x20000000,0xfffff000,0x100000000,len-4096] {
                if off+4096<=len{addresses.push(base+off as u64);}
            }
        }
        addresses.push(0xe0000000);addresses.push(0xe01ff000);
        // Write every sentinel first, then read every sentinel: catches aliases.
        for (i,&va) in addresses.iter().enumerate(){
            unsafe{for j in 0..1024 {core::ptr::write_volatile((gpu.work.host.add(0x80000)as *mut u32).add(j),
                0x10203040u32.wrapping_mul(i as u32+1)^(j as u32));}}
            fence(Ordering::SeqCst);
            if !gpu.copy_buffer(0x100080000,va,4096){return None;}
        }
        for (i,&va) in addresses.iter().enumerate(){
            if !gpu.copy_buffer(va,0x100080000,4096){return None;}
            for j in 0..1024 {let actual=unsafe{read_volatile((gpu.work.host.add(0x80000)as *const u32).add(j))};
                if actual!=0x10203040u32.wrapping_mul(i as u32+1)^(j as u32){return None;}}
        }
        event(log,format!("  FRONTIER_ARENA_READBACK sentinels={} always_alloc={} cache_bytes=5368709120 PASS=true",addresses.len(),used));
        // Full vocabulary output span (>the old 512KiB output area), with a
        // narrower synthetic input. This checks storage, not model logits.
        let (n,m)=(256usize,248320usize);let kind=18;
        let bytes=fixture(kind,n*m);let x:Vec<f32>=(0..n).map(|i|((i%29)as f32-14.0)/32.0).collect();
        for off in (0..bytes.len()).step_by(0x80000){let amount=(bytes.len()-off).min(0x80000);
            unsafe{core::ptr::copy_nonoverlapping(bytes[off..].as_ptr(),gpu.work.host.add(0x80000),amount);}
            fence(Ordering::SeqCst);
            if !gpu.copy_buffer(0x100080000,0x200000000+off as u64,amount){return None;}
        }
        unsafe{core::ptr::copy_nonoverlapping(x.as_ptr(),gpu.work.host.add(0x40000)as *mut f32,n);}
        fence(Ordering::SeqCst);let start=gpu.time_ns();
        if !gpu.frontier_launch(0xc0410000,38,include_bytes!("../kernels/frontier_matvec.sass").len(),(m+3)/4,
            &[0x200000000,0x100040000,0xe0000000,0x100000000,0x100000400],&[kind,n as u32,m as u32,98]){return None;}
        let elapsed=gpu.time_ns()-start;let mut actual=vec![0f32;m];
        for off in (0..m*4).step_by(0x80000){let amount=(m*4-off).min(0x80000);
            if !gpu.copy_buffer(0xe0000000+off as u64,0x100080000,amount){return None;}
            unsafe{core::ptr::copy_nonoverlapping(gpu.work.host.add(0x80000),(actual.as_mut_ptr()as *mut u8).add(off),amount);}
        }
        if !actual.iter().all(|v|v.is_finite()){return None;}
        let mut maxerr=0f32;let mut checks=0;
        for row in (0..m).step_by(1021).chain([m-1]){
            let mut dequant=vec![0f32;n];fc::dequantize(kind,&bytes[row*98..(row+1)*98],&mut dequant).ok()?;
            let mut reference=0.;for j in 0..n{reference+=dequant[j]*x[j];}
            let err=(actual[row]-reference).abs();maxerr=maxerr.max(err);checks+=1;
            if err>0.0002+reference.abs()*0.00002{return None;}
        }
        event(log,format!("  FRONTIER_VOCAB_BUFFER synthetic=true n={} m={} checked_rows={} finite_rows={} elapsed_ns={} max_abs_error={} PASS=true inference=false",
            n,m,checks,m,elapsed,maxerr));
        // Actual weight bytes present in the pinned header-range fixtures:
        // first 17 vocabulary-projection rows and expert0 of layer15 up.
        // These do not constitute a complete model or a generated token.
        let sources=[(include_bytes!("../../reference/shard2-header.gguf").as_slice(),25408usize,14u32,17usize),
            (include_bytes!("../../reference/shard3-header.gguf").as_slice(),52512usize,18u32,640usize)];
        for (source,offset,kind,m) in sources {
            let n=2560usize;let rb=fc::row_bytes(kind,n).ok()?;let bytes=source.get(offset..offset+rb*m)?;
            for off in (0..bytes.len()).step_by(0x80000){let amount=(bytes.len()-off).min(0x80000);
                unsafe{core::ptr::copy_nonoverlapping(bytes[off..].as_ptr(),gpu.work.host.add(0x80000),amount);}
                fence(Ordering::SeqCst);if !gpu.copy_buffer(0x100080000,0x200000000+off as u64,amount){return None;}
            }
            let x:Vec<f32>=(0..n).map(|i|libm::sinf(i as f32*0.031)+0.1*libm::cosf(i as f32*0.013)).collect();
            unsafe{core::ptr::copy_nonoverlapping(x.as_ptr(),gpu.work.host.add(0x40000)as *mut f32,n);}
            fence(Ordering::SeqCst);let start=gpu.time_ns();
            if !gpu.frontier_launch(0xc0410000,38,include_bytes!("../kernels/frontier_matvec.sass").len(),(m+3)/4,
                &[0x200000000,0x100040000,0xe0000000,0x100000000,0x100000400],&[kind,n as u32,m as u32,rb as u32]){return None;}
            let elapsed=gpu.time_ns()-start;
            if !gpu.copy_buffer(0xe0000000,0x100080000,m*4){return None;}
            let mut dequant=vec![0f32;n];let mut maxerr=0f32;
            for row in 0..m{
                fc::dequantize(kind,&bytes[row*rb..(row+1)*rb],&mut dequant).ok()?;
                let mut reference=0.;for j in 0..n{reference+=dequant[j]*x[j];}
                let actual=unsafe{read_volatile((gpu.work.host.add(0x80000)as *const f32).add(row))};
                let err=(actual-reference).abs();maxerr=maxerr.max(err);
                if !actual.is_finite()||err>0.0002+reference.abs()*0.00002{return None;}
            }
            event(log,format!("  FRONTIER_ACTUAL_WEIGHT_OP kind={} n={} m={} elapsed_ns={} max_abs_error={} PASS=true full_model=false inference=false",
                kind,n,m,elapsed,maxerr));
        }
        expert_probe(gpu,log)?;
        cpu_gdn_probe(gpu,log)?;
        batch_probe(gpu,log)?;
        multi_format_probe(gpu,log)?;
        copy_template_probe(gpu,log)?;
        Some(())
    })();
    event(log,format!("  *** FRONTIER_ARENA_GATE PASS={} inference=false ***",result.is_some()));result.is_some()
}

fn copy_template_probe<R:Fn(u64)->u32,W:Fn(u64,u32)>(gpu:&mut Gpu<'_,R,W>,log:&mut Vec<String>)->Option<()> {
    let mut cases=0;
    for n in [4usize,12,508,512,516,4092,4096,65540,524288]{for mode in 0..3{
        let expected:Vec<u32>=(0..n/4).map(|i|(i as u32).wrapping_mul(0x10203041)^(n as u32)^mode).collect();
        unsafe{core::ptr::copy_nonoverlapping(expected.as_ptr(),gpu.work.host.add(0x80000)as *mut u32,n/4);}
        gpu.write(0x0c900000+n as u64,0xfedcba98);
        fence(Ordering::SeqCst);let start=gpu.time_ns();
        let ok=if mode==0{gpu.copy_buffer(0x100080000,0xe0100000,n)}
            else{gpu.frontier_copy_buffer(0x100080000,0xe0100000,n)};
        if !ok{return None;}
        unsafe{core::ptr::write_bytes(gpu.work.host.add(0x80000),0xa5,n);}
        fence(Ordering::SeqCst);
        let ok=if mode==1{gpu.copy_buffer(0xe0100000,0x100080000,n)}
            else{gpu.frontier_copy_buffer(0xe0100000,0x100080000,n)};
        if !ok{return None;}let elapsed=gpu.time_ns()-start;
        let seen=unsafe{core::slice::from_raw_parts(gpu.work.host.add(0x80000)as *const u32,n/4)};
        if seen!=expected||(gpu.rd)(gpu.ap(0x0c900000+n as u64))!=0xfedcba98{return None;}
        event(log,format!("  FRONTIER_COPY_TEMPLATE bytes={} mode={} roundtrip_ns={} bit_exact=true tail_intact=true",n,mode,elapsed));
        cases+=1;
    }}
    event(log,format!("  FRONTIER_COPY_TEMPLATE_GATE cases={} PASS=true full_readback=true",cases));Some(())
}

fn expert_probe<R:Fn(u64)->u32,W:Fn(u64,u32)>(gpu:&mut Gpu<'_,R,W>,log:&mut Vec<String>)->Option<()> {
    let weights=[include_bytes!("../../reference/blk0-expert0-gate.bin").as_slice(),
        include_bytes!("../../reference/blk0-expert0-up.bin").as_slice(),
        include_bytes!("../../reference/blk0-expert0-down.bin").as_slice()];
    let input:Vec<f32>=(0..2560).map(|i|libm::sinf(i as f32*0.031)+0.1*libm::cosf(i as f32*0.013)).collect();
    let mut device_results:Vec<Vec<f32>>=Vec::new();let mut reference_results:Vec<Vec<f32>>=Vec::new();
    for part in 0..3 {
        let(n,m,kind)=if part==2{(640,2560,20)}else{(2560,640,18)};
        let rb=fc::row_bytes(kind,n).ok()?;let bytes=weights[part];if bytes.len()!=m*rb{return None;}
        let silu=|v:f32|v/(1.0+libm::expf(-v));
        let xgpu=if part==2{device_results[0].iter().zip(&device_results[1]).map(|(&g,&u)|silu(g)*u).collect::<Vec<_>>()}else{input.clone()};
        let xref=if part==2{reference_results[0].iter().zip(&reference_results[1]).map(|(&g,&u)|silu(g)*u).collect::<Vec<_>>()}else{input.clone()};
        for off in (0..bytes.len()).step_by(0x80000){let amount=(bytes.len()-off).min(0x80000);
            unsafe{core::ptr::copy_nonoverlapping(bytes[off..].as_ptr(),gpu.work.host.add(0x80000),amount);}
            fence(Ordering::SeqCst);if !gpu.copy_buffer(0x100080000,0x200000000+off as u64,amount){return None;}
        }
        unsafe{core::ptr::copy_nonoverlapping(xgpu.as_ptr(),gpu.work.host.add(0x40000)as *mut f32,n);}
        fence(Ordering::SeqCst);let start=gpu.time_ns();
        if !gpu.frontier_launch(0xc0410000,38,include_bytes!("../kernels/frontier_matvec.sass").len(),(m+3)/4,
            &[0x200000000,0x100040000,0xe0000000,0x100000000,0x100000400],&[kind,n as u32,m as u32,rb as u32]){return None;}
        let elapsed=gpu.time_ns()-start;if !gpu.copy_buffer(0xe0000000,0x100080000,m*4){return None;}
        let mut device=vec![0f32;m];let mut reference=vec![0f32;m];let mut row=vec![0f32;n];let mut maxerr=0f32;
        for r in 0..m {
            device[r]=unsafe{read_volatile((gpu.work.host.add(0x80000)as *const f32).add(r))};
            fc::dequantize(kind,&bytes[r*rb..(r+1)*rb],&mut row).ok()?;
            for c in 0..n{reference[r]+=row[c]*xref[c];}
            let err=(device[r]-reference[r]).abs();maxerr=maxerr.max(err);
            if !device[r].is_finite()||err>0.0002+reference[r].abs()*0.0001{return None;}
        }
        event(log,format!("  FRONTIER_EXPERT_CHAIN layer=0 expert=0 part={} n={} m={} elapsed_ns={} max_abs_error={} PASS=true inference=false",
            ["gate","up","down"][part],n,m,elapsed,maxerr));
        device_results.push(device);reference_results.push(reference);
    }Some(())
}

// All seven loaded quant formats in one dispatch, including non-four-row tails.
fn multi_format_probe<R:Fn(u64)->u32,W:Fn(u64,u32)>(gpu:&mut Gpu<'_,R,W>,log:&mut Vec<String>)->Option<()> {
    use super::frontier_batch::Dispatch;
    let mut jobs=Vec::new();let mut out=0usize;
    for (i,(kind,m)) in [(0,1),(30,3),(8,17),(14,5),(18,640),(20,4),(23,127)].into_iter().enumerate(){
        let n=2560usize;let rb=fc::row_bytes(kind,n).ok()?;
        let row=fixture(kind,n);let mut bytes=Vec::with_capacity(rb*m);
        for _ in 0..m{bytes.extend_from_slice(&row);}
        let va=0x200000000+i as u64*0x200000;
        for off in (0..bytes.len()).step_by(0x80000){let amount=(bytes.len()-off).min(0x80000);
            unsafe{core::ptr::copy_nonoverlapping(bytes[off..].as_ptr(),gpu.work.host.add(0x80000),amount);}
            fence(Ordering::SeqCst);if !gpu.copy_buffer(0x100080000,va+off as u64,amount){return None;}
        }
        let x:Vec<f32>=(0..n).map(|j|libm::sinf(j as f32*0.031+i as f32*0.173)).collect();
        let input=0x10000+i*n*4;
        unsafe{core::ptr::copy_nonoverlapping(x.as_ptr(),gpu.work.host.add(input)as *mut f32,n);}
        jobs.push(Dispatch{weight:va,input:0x100000000+input as u64,output:0xe0000000+out as u64,
            kind,n:n as u32,m:m as u32,row_bytes:rb as u32});out+=m*4;
    }
    fence(Ordering::SeqCst);
    if !gpu.frontier_batch_launch(&jobs)||!gpu.copy_buffer(0xe0000000,0x100080000,out){return None;}
    let expected=unsafe{core::slice::from_raw_parts(gpu.work.host.add(0x80000)as *const f32,out/4)}.to_vec();
    let start=gpu.time_ns();if !gpu.frontier_multi_launch(&jobs){return None;}let elapsed=gpu.time_ns()-start;
    if !gpu.copy_buffer(0xe0000000,0x100080000,out){return None;}
    let actual=unsafe{core::slice::from_raw_parts(gpu.work.host.add(0x80000)as *const f32,out/4)};
    if !actual.iter().zip(&expected).all(|(a,b)|a.is_finite()&&a.to_bits()==b.to_bits()){return None;}
    event(log,format!("  FRONTIER_MULTI_FORMAT kinds=7 rows={} multi_including_table_ns={} bit_exact=true PASS=true inference=false",out/4,elapsed));
    Some(())
}

fn cpu_gdn_probe<R:Fn(u64)->u32,W:Fn(u64,u32)>(gpu:&mut Gpu<'_,R,W>,log:&mut Vec<String>)->Option<()> {
    let conv:Vec<f32>=(0..10240).map(|i|(i%127-63)as f32/256.).collect();
    let alpha:Vec<f32>=(0..48).map(|h|(h%11-5)as f32/8.).collect();
    let beta:Vec<f32>=(0..48).map(|h|(h%7-3)as f32/4.).collect();
    let a:Vec<f32>=(0..48).map(|h|-((h%5+1)as f32)/8.).collect();
    let dt:Vec<f32>=(0..48).map(|h|(h%3-1)as f32/8.).collect();
    let mut state:Vec<f32>=(0..128*128*48).map(|i|(i%37-18)as f32/1024.).collect();
    let start=gpu.time_ns();let out=super::frontier_forward::delta_step(&conv,&alpha,&beta,&a,&dt,&mut state).ok()?;
    let elapsed=gpu.time_ns()-start;let raw=include_bytes!("../../reference/gdn.ggml.bin");
    if raw.len()!=(out.len()+state.len())*4{return None;}
    let mut maxerr=0f32;
    for (actual,bytes) in out.iter().chain(&state).zip(raw.chunks_exact(4)){
        let expected=f32::from_le_bytes(bytes.try_into().ok()?);let err=(*actual-expected).abs();maxerr=maxerr.max(err);
        if !actual.is_finite()||err>1e-5+expected.abs()*1e-5{return None;}
    }
    event(log,format!("  FRONTIER_UEFI_CPU_GDN values={} elapsed_ns={} max_abs_error={} GGML_reference_match=true inference=false",raw.len()/4,elapsed,maxerr));
    gpu_gdn_probe(gpu,&conv,&alpha,&beta,&a,&dt,log)?;
    Some(())
}

fn gpu_gdn_probe<R:Fn(u64)->u32,W:Fn(u64,u32)>(gpu:&mut Gpu<'_,R,W>,conv:&[f32],alpha:&[f32],beta:&[f32],a:&[f32],dt:&[f32],log:&mut Vec<String>)->Option<()> {
    let state_address=0x600000000u64+35*128*128*48*4;
    let code=include_bytes!("../kernels/frontier_gdn.sass");
    if code.len()>65536||!include_bytes!("../kernels/frontier_gdn.c2").is_empty(){return None;}
    for (i,b) in code.chunks_exact(4).enumerate(){gpu.write(0x0c420000+i as u64*4,u32::from_le_bytes(b.try_into().ok()?));}
    let mut packed=conv.to_vec();for v in [alpha,beta,a,dt]{packed.extend_from_slice(v);}
    unsafe{core::ptr::copy_nonoverlapping(packed.as_ptr(),gpu.work.host.add(0x40000)as *mut f32,packed.len());}
    let mut state:Vec<f32>=(0..128*128*48).map(|i|(i%37-18)as f32/1024.).collect();
    let bytes=unsafe{core::slice::from_raw_parts(state.as_ptr()as *const u8,state.len()*4)};
    for off in (0..bytes.len()).step_by(0x80000){let amount=(bytes.len()-off).min(0x80000);
        unsafe{core::ptr::copy_nonoverlapping(bytes[off..].as_ptr(),gpu.work.host.add(0x80000),amount);}
        fence(Ordering::SeqCst);if !gpu.copy_buffer(0x100080000,state_address+off as u64,amount){return None;}
    }
    // Nonzero state, explicit reset despite dirty VRAM, then continuation.
    for (step,reset) in [0u32,1,0].into_iter().enumerate(){
        if reset!=0{state.fill(0.);}
        let expected_out=super::frontier_forward::delta_step(conv,alpha,beta,a,dt,&mut state).ok()?;
        fence(Ordering::SeqCst);let start=gpu.time_ns();
        if !gpu.frontier_launch(0xc0420000,34,code.len(),48*128/4,
            &[0x100040000,state_address,0xe0000000],&[reset]){return None;}
        let elapsed=gpu.time_ns()-start;
        let mut maxerr=0f32;let mut checked=0;
        for (va,reference) in [(0xe0000000u64,expected_out.as_slice()),(state_address,state.as_slice())]{
            for off in (0..reference.len()*4).step_by(0x80000){let amount=(reference.len()*4-off).min(0x80000);
                if !gpu.copy_buffer(va+off as u64,0x100080000,amount){return None;}
                for i in 0..amount/4{
                    let actual=unsafe{read_volatile((gpu.work.host.add(0x80000)as *const f32).add(i))};
                    let expected=reference[off/4+i];let err=(actual-expected).abs();maxerr=maxerr.max(err);checked+=1;
                    if !actual.is_finite()||err>1e-5+expected.abs()*1e-5{
                        event(log,format!("  FRONTIER_GPU_GDN_FAIL step={} i={} actual={} expected={}",step,off/4+i,actual,expected));return None;}
                }
            }
        }
        event(log,format!("  FRONTIER_GPU_GDN step={} reset={} values={} elapsed_ns={} max_abs_error={} PASS=true inference=false",step,reset,checked,elapsed,maxerr));
    }Some(())
}

fn batch_probe<R:Fn(u64)->u32,W:Fn(u64,u32)>(gpu:&mut Gpu<'_,R,W>,log:&mut Vec<String>)->Option<()> {
    use super::frontier_batch::Dispatch;
    if !gpu.frontier_multi_init(){return None;}
    let sources=[include_bytes!("../../reference/blk0-expert0-gate.bin").as_slice(),
        include_bytes!("../../reference/blk0-expert0-up.bin").as_slice(),
        include_bytes!("../../reference/blk0-expert0-down.bin").as_slice()];
    let mut decoded=Vec::new();
    for (i,bytes) in sources.iter().enumerate(){
        let mut d=vec![0f32;2560*640];fc::dequantize(if i==2{20}else{18},bytes,&mut d).ok()?;decoded.push(d);
        for off in (0..bytes.len()).step_by(0x80000){let amount=(bytes.len()-off).min(0x80000);
            unsafe{core::ptr::copy_nonoverlapping(bytes[off..].as_ptr(),gpu.work.host.add(0x80000),amount);}
            fence(Ordering::SeqCst);if !gpu.copy_buffer(0x100080000,0x200000000+i as u64*0x200000+off as u64,amount){return None;}
        }
    }
    for (round,count) in [2usize,3,4,11,23,32].into_iter().enumerate(){
        let mut jobs=Vec::new();let mut inputs=Vec::new();let mut output_off=0usize;let mut input_off=0x10000usize;
        for i in 0..count {
            let part=i%3;let(n,m,kind)=if part==2{(640,2560,20)}else{(2560,640,18)};
            let x:Vec<f32>=(0..n).map(|j|libm::sinf(j as f32*0.031+(i+round*7)as f32*0.17)).collect();
            if input_off+n*4>0x80000||output_off+m*4>0x200000{return None;}
            unsafe{core::ptr::copy_nonoverlapping(x.as_ptr(),gpu.work.host.add(input_off)as *mut f32,n);}
            jobs.push(Dispatch{weight:0x200000000+part as u64*0x200000,input:0x100000000+input_off as u64,
                output:0xe0000000+output_off as u64,kind,n:n as u32,m:m as u32,row_bytes:fc::row_bytes(kind,n).ok()?as u32});
            input_off+=n*4;output_off+=m*4;inputs.push(x);
        }
        fence(Ordering::SeqCst);let start=gpu.time_ns();
        for j in &jobs {if !gpu.frontier_launch(0xc0410000,38,include_bytes!("../kernels/frontier_matvec.sass").len(),(j.m as usize+3)/4,
            &[j.weight,j.input,j.output,0x100000000,0x100000400],&[j.kind,j.n,j.m,j.row_bytes]){return None;}}
        let serial_ns=gpu.time_ns()-start;
        if !gpu.copy_buffer(0xe0000000,0x100080000,output_off){return None;}
        let serial=unsafe{core::slice::from_raw_parts(gpu.work.host.add(0x80000)as *const f32,output_off/4)}.to_vec();
        let start=gpu.time_ns();if !gpu.frontier_batch_launch(&jobs){return None;}let batch_ns=gpu.time_ns()-start;
        if !gpu.copy_buffer(0xe0000000,0x100080000,output_off){return None;}
        let actual=unsafe{core::slice::from_raw_parts(gpu.work.host.add(0x80000)as *const f32,output_off/4)};
        if !actual.iter().zip(&serial).all(|(a,b)|a.is_finite()&&a.to_bits()==b.to_bits()){return None;}
        let mut off=0;let mut maxerr=0f32;
        for (i,job) in jobs.iter().enumerate(){let n=job.n as usize;
            for row in 0..job.m as usize {let mut reference=0f32;
                for j in 0..n{reference+=decoded[i%3][row*n+j]*inputs[i][j];}
                let err=(actual[off+row]-reference).abs();maxerr=maxerr.max(err);
                if err>0.0002+reference.abs()*0.0001{return None;}
            }off+=job.m as usize;
        }
        event(log,format!("  FRONTIER_BATCH count={} serial_ns={} batch_ns={} all_private_fences=true serial_bit_exact=true max_abs_error={} PASS=true inference=false",
            count,serial_ns,batch_ns,maxerr));
        let start=gpu.time_ns();if !gpu.frontier_multi_launch(&jobs){return None;}let multi_ns=gpu.time_ns()-start;
        if !gpu.copy_buffer(0xe0000000,0x100080000,output_off){return None;}
        let actual=unsafe{core::slice::from_raw_parts(gpu.work.host.add(0x80000)as *const f32,output_off/4)};
        if !actual.iter().zip(&serial).all(|(a,b)|a.is_finite()&&a.to_bits()==b.to_bits()){return None;}
        event(log,format!("  FRONTIER_MULTI count={} input=SYS batch_ns={} multi_including_table_ns={} bit_exact=true PASS=true inference=false",count,batch_ns,multi_ns));
        let start=gpu.time_ns();
        if !gpu.copy_buffer(0x100010000,0xe0100000,input_off-0x10000){return None;}
        let input_copy_ns=gpu.time_ns()-start;
        for job in &mut jobs{job.input=0xe0100000+(job.input-0x100010000);}
        let start=gpu.time_ns();if !gpu.frontier_batch_launch(&jobs){return None;}
        let vram_ns=gpu.time_ns()-start;
        if !gpu.copy_buffer(0xe0000000,0x100080000,output_off){return None;}
        let actual=unsafe{core::slice::from_raw_parts(gpu.work.host.add(0x80000)as *const f32,output_off/4)};
        if !actual.iter().zip(&serial).all(|(a,b)|a.is_finite()&&a.to_bits()==b.to_bits()){return None;}
        event(log,format!("  FRONTIER_VRAM_INPUT count={} input_copy_ns={} sysmem_batch_ns={} vram_batch_ns={} bit_exact=true PASS=true inference=false",
            count,input_copy_ns,batch_ns,vram_ns));
        let start=gpu.time_ns();if !gpu.frontier_multi_launch(&jobs){return None;}let multi_ns=gpu.time_ns()-start;
        if !gpu.copy_buffer(0xe0000000,0x100080000,output_off){return None;}
        let actual=unsafe{core::slice::from_raw_parts(gpu.work.host.add(0x80000)as *const f32,output_off/4)};
        if !actual.iter().zip(&serial).all(|(a,b)|a.is_finite()&&a.to_bits()==b.to_bits()){return None;}
        event(log,format!("  FRONTIER_MULTI count={} input=VRAM batch_ns={} multi_including_table_ns={} bit_exact=true PASS=true inference=false",count,vram_ns,multi_ns));
    }Some(())
}
