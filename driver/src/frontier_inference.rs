//! Full-media entry point. Never treats quant fixtures as a loaded model.
use super::*;
use super::frontier_forward::{Forward,VOCAB};
use super::frontier_gguf::{ModelIndex,Shard};
use super::frontier_uefi::{FullMedia,LOGICAL};

pub fn run<R:Fn(u64)->u32,W:Fn(u64,u32)>(rd:&R,wr:&W,bar:u64,data:u64,userd:u64,token:u32,
    dma:&mut crate::RbioDma,template:&[u32;64],fb:([u64;4],u32),mut media:FullMedia,log:&mut Vec<String>)->bool {
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
    let Some(headers)=media.headers()else{event(log,String::from("  FRONTIER_FULL header_read=FAIL inference=false"));return false;};
    let shards:Vec<_>=match headers.iter().zip(LOGICAL).map(|(h,n)|Shard::parse(h,n)).collect::<Result<_,_>>(){
        Ok(v)=>v,Err(e)=>{event(log,format!("  FRONTIER_FULL header_parse={:?} inference=false",e));return false;}};
    let model=match ModelIndex::new(shards){Ok(v)=>v,Err(_)=>return false};
    if model.inventory().ok()!=Some([5351626240,28800138240,55823564800]){return false;}
    let Some(work)=dma.alloc_pages(256)else{return false;};
    let saved=rd(bar+0x1700);
    let mut gpu=Gpu{rd,wr,bar,data,userd,token,window:frontier_window::Window::new(),work,q:*template,put:2,stamp:0x4e000000,kernels:0,model_data:0,
        vram_weights:Vec::new(),swizzled_weights:Vec::new(),jobs:Vec::new(),submits:0,graph:Vec::new(),profile:[0;4],prefill:Vec::new(),batch_build:false};
    let result=(||->Option<()> {
        let backend=match super::frontier_backend::Backend::new(&mut gpu,&model,media,fb,log){
            Ok(v)=>v,Err(e)=>{event(log,format!("  FRONTIER_FULL backend_init={:?} inference=false",e));return None;}};
        let mut forward=Forward::new(backend);
        let raw=include_bytes!("../../reference/prompt.ids");
        let prompt:Vec<u32>=raw.chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect();
        if raw.len()%4!=0||prompt.is_empty()||prompt.len()>128||prompt.iter().any(|&t|t as usize>=VOCAB){return None;}
        let time=||loop{let hi=rd(bar+0x9410);let lo=rd(bar+0x9400);if hi==rd(bar+0x9410){break ((hi as u64)<<32)|lo as u64;}};
        let start=time();let mut logits=Vec::new();
        for (position,&input) in prompt.iter().enumerate(){
            let before=time();
            logits=match forward.token(input){Ok(v)=>v,Err(e)=>{
                forward.backend.report();uefi::println!("  FRONTIER_FORWARD FAILED position={} error={:?} inference=false",position,e);return None;}};
            let next=greedy(&logits)?;
            // Diagnostic per-token output; not a throughput benchmark yet.
            uefi::println!("  FRONTIER_FORWARD position={} input={} next={} logit={} elapsed_ns={} all_layers=48",position,input,next,logits[next as usize],time()-before);
        }
        let prefill=time()-start;let mut output=Vec::new();let decode_start=time();
        for generated in 0..8 {
            let next=greedy(&logits)?;output.push(next);
            uefi::println!("  FRONTIER_GENERATED index={} token={} logit={}",generated,next,logits[next as usize]);
            if next==248046||next==248044||generated==7{break;}
            logits=match forward.token(next){Ok(v)=>v,Err(e)=>{
                forward.backend.report();uefi::println!("  FRONTIER_FORWARD FAILED decode={} error={:?}",generated,e);return None;}};
        }
        let decode=time()-decode_start;
        forward.backend.report();
        // IDs are authoritative; detokenize on the host with the pinned tokenizer.
        forward.backend.emit(format!("  FRONTIER_INFERENCE complete=true input_tokens={} output_ids={:?} prefill_ns={} decode_ns={} correctness=UNVERIFIED benchmark=false backend=GPU_LINEAR_GDN_CPU_NORMS_SSD_PLE",
            prompt.len(),output,prefill,decode));
        let mut backend=forward.backend;
        for (label,raw,cap) in [
            ("sentence",include_bytes!("../../reference/eval-sentence.ids").as_slice(),48usize),
            ("riddle",include_bytes!("../../reference/eval-riddle.ids").as_slice(),24usize)] {
            let input:Vec<u32>=raw.chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect();
            if raw.len()%4!=0||input.is_empty()||input.len()>128||input.iter().any(|&t|t as usize>=VOCAB){return None;}
            let mut expected=None;
            for round in 0..2 {
                backend.emit(format!("  FRONTIER_EVAL_BEGIN case={} round={} measured={} input_tokens={} generation_cap={} timer=PTIMER serial_during_timing=false",label,round,round==1,input.len(),cap));
                backend.begin_quiet();
                let mut run=Forward::new(backend);
                let mut pre_steps=Vec::with_capacity(input.len());let mut dec_steps=Vec::with_capacity(cap);
                let mut generated=Vec::with_capacity(cap);let mut scores=Vec::new();
                let begin=time();
                for (position,&token) in input.iter().enumerate(){let before=time();
                    scores=match run.advance(token,position+1==input.len()){
                        Ok(v)=>v,Err(e)=>{run.backend.emit(format!("  FRONTIER_EVAL_FAILED case={} round={} prefill_position={} error={:?}",label,round,position,e));run.backend.report();return None;}};
                    pre_steps.push(time()-before);
                }
                let prefill_ns=time()-begin;
                generated.push(greedy(&scores)?); // First output belongs to prefill, not a decode call.
                let begin=time();
                for _ in 1..cap {
                    let previous=*generated.last()?;if previous==248046||previous==248044{break;}
                    let before=time();
                    scores=match run.token(previous){Ok(v)=>v,Err(e)=>{
                        run.backend.emit(format!("  FRONTIER_EVAL_FAILED case={} round={} decode_calls={} error={:?}",label,round,dec_steps.len(),e));run.backend.report();return None;}};
                    generated.push(greedy(&scores)?);dec_steps.push(time()-before);
                }
                let decode_ns=time()-begin;let ended=matches!(generated.last(),Some(248046|248044));
                // All serial writes below occur after both timed phases finish.
                run.backend.emit(format!("  FRONTIER_EVAL case={} round={} measured={} prefill_tokens={} prefill_ns={} decode_calls={} decode_ns={} output_ids={:?} eos={} serial_during_timing=false prefill=sequential_final_head_only cache=retained",
                    label,round,round==1,input.len(),prefill_ns,dec_steps.len(),decode_ns,generated,ended));
                run.backend.emit(format!("  FRONTIER_EVAL_STEPS case={} round={} prefill_ns={:?} decode_ns={:?}",label,round,pre_steps,dec_steps));
                run.backend.report();
                if let Some(previous)=&expected {if previous!=&generated{
                    run.backend.emit(format!("  FRONTIER_EVAL_REPEAT_MISMATCH case={} FAIL=true",label));return None;}
                    run.backend.emit(format!("  FRONTIER_EVAL_REPEAT case={} generated_ids_equal=true",label));
                }else{expected=Some(generated);}
                backend=run.backend;
            }
        }
        drop(backend);
        event(log,String::from("  FRONTIER_EVAL_COMPLETE cases=2 measured_rounds=2 correctness=HOST_REVIEW_REQUIRED"));
        Some(())
    })();
    wr(bar+0x1700,saved);
    event(log,format!("  *** FRONTIER_FULL_FORWARD completed={} correctness=UNVERIFIED ***",result.is_some()));
    result.is_some()
}
fn greedy(logits:&[f32])->Option<u32>{
    if logits.len()!=VOCAB||!logits.iter().all(|v|v.is_finite()){return None;}
    let mut best=0;for i in 1..logits.len(){if logits[i]>logits[best]{best=i;}}Some(best as u32)
}
