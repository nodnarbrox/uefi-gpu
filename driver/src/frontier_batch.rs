//! Reusable QMD/CB0 slots, one GPFIFO packet, private completion per job.
//! The validated HOST WFI packet is retained at every kernel boundary. This
//! reduces host submission work; it does not claim concurrent kernel execution.
use super::*;
const PA:u64=0x0c100000;const VA:u64=0xc0100000;
const PB_PA:u64=0x0c200000;const PB_VA:u64=0xc0200000;
pub use super::frontier_dispatch::{Dispatch,MAX};
impl<R:Fn(u64)->u32,W:Fn(u64,u32)> Gpu<'_,R,W> {
    pub(super) fn frontier_multi_init(&mut self)->bool {
        let code=include_bytes!("../kernels/frontier_multi_matvec.sass");
        let constants=include_bytes!("../kernels/frontier_multi_matvec.c2");
        // Specialized NVRTC metadata: 40 registers, no spills, independent C2.
        if code.len()!=15872||constants.len()!=80{return false;}
        for(i,b)in code.chunks_exact(4).enumerate(){self.write(0x0c430000+i as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
        for i in 0..32 {self.write(0x0c7f1000+i*4,0);}
        for(i,b)in constants.chunks_exact(4).enumerate(){self.write(0x0c7f1000+i as u64*4,u32::from_le_bytes(b.try_into().unwrap()));}
        true
    }
    pub(super) fn frontier_multi_launch(&mut self,jobs:&[Dispatch])->bool {
        for job in jobs {if super::frontier_core::row_bytes(job.kind,job.n as usize).ok()!=Some(job.row_bytes as usize){return false;}}
        let Some((bytes,grid))=super::frontier_dispatch::pack(jobs)else{return false;};
        // Table is disjoint from the lower-MiB output and the input experiment.
        // Copy and launch are both included by the caller's PTIMER interval.
        unsafe{core::ptr::copy_nonoverlapping(bytes.as_ptr(),self.work.host.add(0x80000),bytes.len());}
        fence(Ordering::SeqCst);
        if !self.copy_buffer(0x100080000,0xe01f8000,bytes.len()){return false;}
        self.frontier_launch(0xc0430000,40,include_bytes!("../kernels/frontier_multi_matvec.sass").len(),grid as usize,
            &[0xe01f8000,0x100000000,0x100000400],&[jobs.len()as u32])
    }
    pub(super) fn frontier_batch_init(&mut self) {
        for i in 0..MAX {
            let pa=PA+i as u64*0x400;let va=VA+i as u64*0x400;let mut q=self.q;
            field(&mut q,384,415,1);field(&mut q,592,607,128);field(&mut q,648,656,38);
            field(&mut q,823,823,1);field(&mut q,768,799,va+0x3f0);field(&mut q,832,863,0);
            field(&mut q,1024,1055,va+0x100);field(&mut q,1075,1087,0x200/16);
            field(&mut q,1536,1567,0xc0410000);field(&mut q,256,287,0xc0410000>>8);
            field(&mut q,1641,1649,((include_bytes!("../kernels/frontier_matvec.sass").len()+255)/256)as u64);
            field(&mut q,544,561,0);field(&mut q,642,642,1);
            field(&mut q,1152,1183,0xc07f0000);field(&mut q,1184,1200,0);
            field(&mut q,1202,1202,1);field(&mut q,1203,1215,8);
            for (j,v) in q.iter().enumerate(){self.write(pa+j as u64*4,*v);}
            for j in 0..128{self.write(pa+0x100+j*4,0);}
            self.write(pa+0x100,128);self.write(pa+0x104,1);self.write(pa+0x108,1);
            self.write64(pa+0x278,0x100000000);self.write64(pa+0x280,0x100000400);
            self.write(pa+0x3f0,0);
            let words=[0x2001001e,1,0x20012000|(0x1698>>2),0x1011,
                0x20012000|(0x2b4>>2),(va>>8)as u32,0x20012000|(0x2c0>>2),9];
            for (j,v) in words.iter().enumerate(){self.write(PB_PA+i as u64*32+j as u64*4,*v);}
        }
    }
    pub(super) fn frontier_batch_launch(&mut self,jobs:&[Dispatch])->bool {
        if jobs.is_empty()||jobs.len()>MAX{return false;}
        self.stamp+=1;
        for (i,job) in jobs.iter().enumerate(){
            if job.n==0||job.m==0||job.row_bytes==0{return false;}
            let pa=PA+i as u64*0x400;
            self.write(pa+0x30,(job.m+3)/4);self.write(pa+0x68,self.stamp);
            self.write64(pa+0x260,job.weight);self.write64(pa+0x268,job.input);self.write64(pa+0x270,job.output);
            self.write(pa+0x288,job.kind);self.write(pa+0x28c,job.n);self.write(pa+0x290,job.m);self.write(pa+0x294,job.row_bytes);
        }
        self.write(self.data+(self.put%32)as u64*8,PB_VA as u32);
        self.write(self.data+(self.put%32)as u64*8+4,(jobs.len()as u32*8)<<10);
        fence(Ordering::SeqCst);let _=(self.rd)(self.ap(PA+0x68));
        self.put=(self.put+1)%32;self.write(self.userd+0x8c,self.put);
        let _=(self.rd)(self.ap(self.userd+0x8c));(self.wr)(self.bar+0xbb0090,self.token);self.submits+=1;
        for _ in 0..100000 {
            let done=(0..jobs.len()).all(|i|(self.rd)(self.ap(PA+i as u64*0x400+0x3f0))==self.stamp);
            if done{fence(Ordering::SeqCst);self.kernels+=jobs.len()as u64;return true;}
            boot::stall(10);
        }
        false
    }
}
