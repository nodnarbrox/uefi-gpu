//! GPU transformer, embedding dequantization and stable greedy selection.
use super::*;
const BASE:u64=0x400000000;
const X:u64=BASE;const XN:u64=BASE+0x10000;const Q:u64=BASE+0x20000;
const K:u64=BASE+0x30000;const V:u64=BASE+0x40000;const ATT:u64=BASE+0x50000;
const TMP:u64=BASE+0x60000;const GATE:u64=BASE+0x80000;
const SCORES:u64=BASE+0x90000;const KEYS:u64=KV_KEYS;const VALUES:u64=KV_VALUES;
const TRIG:u64=BASE+0x1200000;const OUTNORM:u64=BASE+0x1300000;
const XQ:u64=BASE+0xa0000;
const LOGITS:u64=BASE+0x1400000;const PARTIAL:u64=BASE+0xb0000;
#[derive(Clone,Copy)]
pub(super) struct GraphJob {pub(super) pa:u64,pub(super) va:u64,pos_offset:usize,score_grid:bool}
impl<R:Fn(u64)->u32,W:Fn(u64,u32)> Gpu<'_,R,W> {
    pub(super) fn verify_gpu_greedy(&mut self,model:&Model,chosen:usize,log:&mut Vec<String>)->bool {
        // Outside benchmark timing: compare GPU greedy with a CPU scan of every
        // final logit, including the same finite-value and lowest-id tie rules.
        if !self.copy_buffer(LOGITS,0x100020000,model.tokens.len()*4){return false;}
        let mut best=f32::NEG_INFINITY;let mut next=0;
        for i in 0..model.tokens.len(){let v=unsafe{read_volatile((self.work.host.add(0x20000)as *const f32).add(i))};
            if !v.is_finite(){return false;}if v>best{best=v;next=i;}}
        let ok=next==chosen;
        event(log,format!("  M4 GPU_GREEDY_CPU_CHECK logits={} gpu_id={} cpu_id={} PASS={}",model.tokens.len(),chosen,next,ok));ok
    }
    pub(super) fn graph_add(&mut self,code:u64,regs:u64,code_bytes:usize,grid:usize,args:&[u64],tail:&[u32],pos_offset:usize,score_grid:bool) {
        let i=self.graph.len()as u64;
        let (base_pa,base_va)=if self.batch_build{(0x0d000000,0xe0000000)}else{(0x0c000000,0xc0000000)};
        let pa=base_pa+i*0x400;let va=base_va+i*0x400;
        let mut q=self.q;
        // Interior completions are never read by the CPU. Keep HOST WFI at
        // every dependency, but only release a CPU-visible semaphore at tails.
        field(&mut q,823,823,0);
        if code==0xc04b0000||code==0xc0520000 {field(&mut q,544,561,512);}
        if code==0xe0400000||code==0xe0450000 {field(&mut q,544,561,0);}
        if code==0xe0410000 {field(&mut q,544,561,512);}
        if code==0xe0440000 {
            field(&mut q,544,561,4608);field(&mut q,642,642,1);
            field(&mut q,1152,1183,0xe07f0000);field(&mut q,1184,1200,0);
            field(&mut q,1202,1202,1);field(&mut q,1203,1215,1);
        }
        field(&mut q,384,415,grid as u64);field(&mut q,648,656,regs);
        field(&mut q,1024,1055,va+0x100);field(&mut q,1075,1087,0x200/16);
        field(&mut q,1536,1567,code);field(&mut q,256,287,code>>8);
        field(&mut q,1641,1649,((code_bytes+255)/256)as u64);
        if code==0xc0470000 {
            field(&mut q,544,561,512); // fused_resnorm: 128 shared f32 values.
            field(&mut q,642,642,1);
            field(&mut q,1152,1183,0xc07f0000);field(&mut q,1184,1200,0);
            field(&mut q,1202,1202,1);field(&mut q,1203,1215,1);
        }
        if code==0xc04e0000||code==0xc04f0000 {field(&mut q,544,561,1536);}
        if code==0xc0500000 {
            field(&mut q,544,561,4608);field(&mut q,642,642,1);
            field(&mut q,1152,1183,0xc07f0100);field(&mut q,1184,1200,0);
            field(&mut q,1202,1202,1);field(&mut q,1203,1215,1);
        }
        field(&mut q,768,799,va+0x3f0);
        for(d,v)in q.iter().enumerate(){self.write(pa+d as u64*4,*v);}
        for d in 0..0x200/4 {self.write(pa+0x100+d*4,0);}
        // CUDA block dimensions in constant-bank zero, for compiler-generated code.
        self.write(pa+0x100,128);self.write(pa+0x104,1);self.write(pa+0x108,1);
        for(d,v)in args.iter().enumerate(){self.write64(pa+0x260+d as u64*8,*v);}
        for(d,v)in tail.iter().enumerate(){self.write(pa+0x260+args.len()as u64*8+d as u64*4,*v);}
        let pb=[0x2001001e,1,0x20012000|(0x1698>>2),0x1011,0x20012000|(0x2b4>>2),(va>>8)as u32,0x20012000|(0x2c0>>2),9];
        for(d,v)in pb.iter().enumerate(){
            self.write(pa+0x300+d as u64*4,*v);
            self.write(base_pa+0x600000+i*32+d as u64*4,*v);
        }
        self.write(pa+0x3f0,0);self.graph.push(GraphJob{pa,va,pos_offset,score_grid});
    }
    fn graph_quant(&mut self,input:u64,n:usize){
        self.graph_add(0xc04a0000,19,include_bytes!("../kernels/quant_act.sass").len(),(n+127)/128,
            &[input,XQ],&[n as u32],0,false);
    }
    fn graph_linear(&mut self,model:&Model,name:&str,input:u64,output:u64){
        let t=model.tensor(name);let weight=self.vram_weights.iter().find(|&&(off,_)|off==t.off).unwrap().1;
        self.graph_quant(input,t.n);
        let(code,regs,bytes)=if tensor_q4(name){(0xc0520000,30,include_bytes!("../kernels/linear_q4.sass").len())}else{(0xc04b0000,38,include_bytes!("../kernels/linear_dp4a.sass").len())};
        self.graph_add(code,regs,bytes,t.m,
            &[weight,XQ,output],&[t.n as u32,t.m as u32],0,false);
    }
    pub(super) fn init_graph(&mut self,model:&Model,fb:([u64;4],u32),log:&mut Vec<String>)->bool {
        if fb.0[2]!=0||fb.1!=0||fb.0[0]>0x0b000000||fb.0[1]<0xe3ffffff{return false;}
        self.map_memory(0xc0000000,0x0c000000,0x800000,0x0b000000,1);
        self.map_memory(BASE,0x80000000,0x2000000,0x0e000000,1);
        assert!(LAYERS*SEQ*KV*4<=0x2000000);
        self.map_memory(KV_KEYS,0xe0000000,0x4000000,0x0f300000,1);
        event(log,format!("  M4 KV_ARENA context={} PA=[0xe0000000..0xe4000000) keys_VA=0x{:x} values_VA=0x{:x} bytes_per_side={}",SEQ,KV_KEYS,KV_VALUES,LAYERS*SEQ*KV*4));
        let kernels:[(&[u8],u64);22]=[
            (include_bytes!("../kernels/rmsnorm.sass"),26),
            (include_bytes!("../kernels/rope_bias.sass"),20),
            (include_bytes!("../kernels/attn_scores.sass"),39),
            (include_bytes!("../kernels/attn_softmax.sass"),38),
            (include_bytes!("../kernels/attn_values.sass"),40),
            (include_bytes!("../kernels/residual.sass"),10),
            (include_bytes!("../kernels/silu_mul.sass"),18),
            (include_bytes!("../kernels/fused_resnorm.sass"),37),
            (include_bytes!("../kernels/fused_gate_up.sass"),38),
            (include_bytes!("../kernels/fused_qkv.sass"),39),
            (include_bytes!("../kernels/quant_act.sass"),19),
            (include_bytes!("../kernels/linear_dp4a.sass"),38),
            (include_bytes!("../kernels/gateup_dp4a.sass"),39),
            (include_bytes!("../kernels/qkv_dp4a.sass"),36),
            (include_bytes!("../kernels/greedy_tiles.sass"),33),
            (include_bytes!("../kernels/greedy_finish.sass"),31),
            (include_bytes!("../kernels/fused_attention.sass"),39),
            (include_bytes!("../kernels/embedding_q8.sass"),12),
            (include_bytes!("../kernels/linear_q4.sass"),30),
            (include_bytes!("../kernels/gateup_q4.sass"),38),
            (include_bytes!("../kernels/qkv_q4.sass"),37),
            (include_bytes!("../kernels/repack_q4.sass"),39)];
        for(i,(code,_))in kernels.iter().enumerate(){assert!(code.len()<0x10000);
            for(d,b)in code.chunks_exact(4).enumerate(){self.write(0x0c400000+i as u64*0x10000+d as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}}
        let c2=include_bytes!("../kernels/fused_resnorm.c2");assert_eq!(c2.len(),8);
        for d in 0..4 {self.write(0x0c7f0000+d*4,0);}
        for(d,b)in c2.chunks_exact(4).enumerate(){self.write(0x0c7f0000+d as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
        let c2=include_bytes!("../kernels/fused_attention.c2");assert_eq!(c2.len(),8);
        for d in 0..4 {self.write(0x0c7f0100+d*4,0);}
        for(d,b)in c2.chunks_exact(4).enumerate(){self.write(0x0c7f0100+d as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
        let pdb=self.data-0x5000;fence(Ordering::SeqCst);(self.wr)(self.bar+0xb830a0,(pdb>>8)as u32);
        (self.wr)(self.bar+0xb830a4,(pdb>>40)as u32);(self.wr)(self.bar+0xb830b0,0x80000007);
        let mut flushed=false;for _ in 0..20000{if(self.rd)(self.bar+0xb830b0)&0x80000000==0{flushed=true;break;}boot::stall(10);}if !flushed{return false;}
        if Q4_WEIGHTS&&!self.init_q4_weights(model,fb,log){event(log,String::from("  M4 Q4_INIT FAILED"));return false;}
        for layer in 0..LAYERS {
            let meta=BASE+0x1000000+layer as u64*0x10000;
            for(suffix,off)in [("attn_norm.weight",0),("ffn_norm.weight",0x2000),("attn_q.bias",0x4000),("attn_k.bias",0x6000),("attn_v.bias",0x7000)] {
                let t=model.tensor(&format!("blk.{}.{}",layer,suffix));assert_eq!(t.typ,0);
                if !self.copy_buffer(0x40000000+model.data as u64+t.off as u64,meta+off,t.n*4){return false;}
            }
        }
        let t=model.tensor("output_norm.weight");if !self.copy_buffer(0x40000000+model.data as u64+t.off as u64,OUTNORM,t.n*4){return false;}
        // Same libm coefficients as CPU RoPE, precomputed once outside timing.
        for pos in 0..SEQ {for i in 0..64 {let angle=pos as f32*libm::powf(1_000_000.0,-((2*i)as f32)/128.0);let(s,c)=libm::sincosf(angle);
            unsafe{core::ptr::write_volatile((self.work.host as *mut f32).add(pos*128+i),c);core::ptr::write_volatile((self.work.host as *mut f32).add(pos*128+64+i),s);}}}
        fence(Ordering::SeqCst);if !self.copy_buffer(0x100000000,TRIG,SEQ*128*4){return false;}
        let add=|gpu:&mut Self,id:usize,grid:usize,args:&[u64],tail:&[u32],pos:usize,sg:bool|{
            gpu.graph_add(0xc0400000+id as u64*0x10000,kernels[id].1,kernels[id].0.len(),grid,args,tail,pos,sg);};
        let emb=model.tensor("token_embd.weight");
        add(self,17,12,&[0x40000000+model.data as u64+emb.off as u64,X],&[0],0,false);
        for layer in 0..LAYERS {
            let name=|s:&str|format!("blk.{}.{}",layer,s);let meta=BASE+0x1000000+layer as u64*0x10000;
            let keys=KEYS+(layer*SEQ*KV*4)as u64;let values=VALUES+(layer*SEQ*KV*4)as u64;
            // The previous block's down projection remains in TMP until this
            // fused residual+norm. Token zero overwrites X, so no stale carry.
            add(self,7,1,&[X,TMP,meta,XN],&[if layer==0{0}else{1}],0,false);
            let weight=|s:&str|{let t=model.tensor(&name(s));self.vram_weights.iter().find(|&&(off,_)|off==t.off).unwrap().1};
            let(wq,wk,wv)=(weight("attn_q.weight"),weight("attn_k.weight"),weight("attn_v.weight"));
            let(wg,wu)=(weight("ffn_gate.weight"),weight("ffn_up.weight"));
            self.graph_quant(XN,DIM);
            add(self,13,512,&[wq,wk,wv,XQ,Q,K,V],&[],0,false);
            add(self,1,6,&[Q,K,V,meta+0x4000,meta+0x6000,meta+0x7000,TRIG,keys,values],&[0],0x2a8,false);
            add(self,16,12,&[Q,keys,values,ATT],&[0],0x280,false);
            self.graph_linear(model,&name("attn_output.weight"),ATT,TMP);
            add(self,7,1,&[X,TMP,meta+0x2000,XN],&[1],0,false);
            self.graph_quant(XN,DIM);
            add(self,if Q4_WEIGHTS{19}else{12},(HIDDEN+3)/4,&[wg,wu,XQ,GATE],&[],0,false);
            self.graph_linear(model,&name("ffn_down.weight"),GATE,TMP);
        }
        add(self,7,1,&[X,TMP,OUTNORM,XN],&[1],0,false);
        self.graph_linear(model,"output.weight",XN,LOGITS);
        let tiles=(model.tokens.len()+1023)/1024;
        add(self,14,tiles,&[LOGITS,PARTIAL],&[model.tokens.len()as u32],0,false);
        add(self,15,1,&[PARTIAL,0x100018000],&[tiles as u32],0,false);
        for idx in [self.graph.len()-6,self.graph.len()-1] {
            let pa=self.graph[idx].pa+0x64;let word=(self.rd)(self.ap(pa));
            self.write(pa,word|(1<<23)); // RELEASE0_ENABLE; original SYSMEMBAR retained.
        }
        assert!(self.graph.len()*32<0x100000);
        event(log,format!("  M4 GRAPH_READY linear=DP4A_Q8_ACT32 linear_vdr=4 qkv_gateup_vdr=2 linear_warps_per_row=4 dispatch=CONTIGUOUS_SINGLE_ENTRY fusion=QKV_GATEUP_SILU_RESNORM_ATTENTION embedding=GPU_SYSROW greedy=GPU release=TAIL_ONLY jobs={} KV_VRAM_bytes={} scratch_PA=[0x80000000..0x82000000) command_WFI_SCOPE_ALL=1",self.graph.len(),LAYERS*SEQ*KV*4*2));true
    }
    pub(super) fn forward_gpu(&mut self,model:&Model,token:usize,pos:usize,output:bool,log:&mut Vec<String>)->Option<usize>{
        let profile_start=self.time_ns();
        self.write(self.graph[0].pa+0x270,token as u32);
        let profile_graph=self.time_ns();self.profile[0]+=profile_graph-profile_start;
        let count=self.graph.len()-if output{0}else{5};
        for idx in 0..count {let j=self.graph[idx];
            if j.pos_offset!=0 {self.write(j.pa+j.pos_offset as u64,pos as u32);}
            if j.score_grid {self.write(j.pa+0x30,((12*(pos+1)+127)/128)as u32);}
        }
        // All dependencies retain HOST WFI. Only observe the final private
        // release; interior releases can keep old payloads without a CPU patch.
        let last=self.graph[count-1];self.stamp+=1;self.write(last.pa+0x68,self.stamp);
        self.write(self.data+self.put as u64*8,0xc0600000);
        self.write(self.data+self.put as u64*8+4,(count as u32*8)<<10);
        self.put=(self.put+1)%32;
        fence(Ordering::SeqCst);let _=(self.rd)(self.ap(last.pa+0x68));
        self.write(self.userd+0x8c,self.put);let _=(self.rd)(self.ap(self.userd+0x8c));(self.wr)(self.bar+0xbb0090,self.token);self.submits+=1;
        let mut done=false;for _ in 0..100000 {if(self.rd)(self.ap(last.pa+0x3f0))==self.stamp{done=true;break;}boot::stall(10);}
        if !done {event(log,format!("  M4 GRAPH TIMEOUT pos={} jobs=0..{} sem=0x{:x} want=0x{:x}",pos,count,(self.rd)(self.ap(last.pa+0x3f0)),self.stamp));return None;}
        self.kernels+=count as u64;
        let profile_greedy=self.time_ns();self.profile[2]+=profile_greedy-profile_graph;
        if !output{return Some(0);}fence(Ordering::SeqCst);
        let next=unsafe{read_volatile(self.work.host.add(0x18004)as *const u32)}as usize;
        let bad=unsafe{read_volatile(self.work.host.add(0x18008)as *const u32)};
        if bad!=0||next>=model.tokens.len(){event(log,format!("  M4 GPU_GREEDY invalid pos={} id={} bad={}",pos,next,bad));return None;}
        self.profile[3]+=self.time_ns()-profile_greedy;
        Some(next)
    }
}
