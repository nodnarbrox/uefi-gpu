//! Lossy Q4_0 linear-weight experiment, independent of the preserved Q8 build.
use super::*;
fn half_rne(v:f32)->u16 {
    let bits=v.to_bits();let sign=((bits>>16)&0x8000)as u16;let mut e=((bits>>23)&255)as i32-127;
    if e>15{return sign|0x7c00;}if e < -25{return sign;}
    let mant=(bits&0x7fffff)|0x800000;
    let round=|shift:u32|{let a=mant>>shift;let rem=mant&((1<<shift)-1);let h=1<<(shift-1);a+u32::from(rem>h||(rem==h&&a&1!=0))};
    if e < -14{return sign|round((-e-1)as u32)as u16;}
    let mut m=round(13);if m==2048{e+=1;m=1024;}if e>15{return sign|0x7c00;}
    sign|(((e+15)as u16)<<10)|((m as u16)&1023)
}
pub(super) fn block_ref(src:&[u8])->[u8;18] {
    let dw=half(u16::from_le_bytes([src[0],src[1]]));let mut x=[0f32;32];let(mut a,mut max)=(0f32,0f32);
    for j in 0..32{x[j]=dw*(src[2+j]as i8 as f32);if a<libm::fabsf(x[j]){a=libm::fabsf(x[j]);max=x[j];}}
    let d=max/-8.0;let inv=if d!=0.0{1.0/d}else{0.0};let mut out=[0u8;18];out[..2].copy_from_slice(&half_rne(d).to_le_bytes());
    for j in 0..16{let lo=((x[j]*inv+8.5)as i32).min(15)as u8;let hi=((x[j+16]*inv+8.5)as i32).min(15)as u8;out[2+j]=lo|(hi<<4);}out
}
impl<R:Fn(u64)->u32,W:Fn(u64,u32)> Gpu<'_,R,W> {
    pub(super) fn init_q4_weights(&mut self,model:&Model,fb:([u64;4],u32),log:&mut Vec<String>)->bool {
        const PA:u64=0xa0000000;const VA:u64=0x800000000;const PT:u64=0x0f100000;
        let old=self.vram_weights.clone();let mut spans=Vec::new();let mut allocated=0usize;
        for &(off,src)in &old{let t=model.tensors.iter().find(|t|t.off==off&&t.typ==8).unwrap();if !tensor_q4(t.name){continue;}let blocks=t.n*t.m/32;
            spans.push((off,src,allocated,blocks));allocated+=(blocks*18+4095)&!4095;}
        let chunks=(allocated+0x1fffff)/0x200000;let tb=((chunks+255)/256+chunks)*4096;
        if fb.0[2]!=0||fb.1!=0||fb.0[0]>PT||PA+allocated as u64>fb.0[1]+1||PA+allocated as u64>0x2f0920000||PT+tb as u64>0x10000000{return false;}
        self.map_memory(VA,PA,allocated,PT,1);let pdb=self.data-0x5000;fence(Ordering::SeqCst);
        (self.wr)(self.bar+0xb830a0,(pdb>>8)as u32);(self.wr)(self.bar+0xb830a4,(pdb>>40)as u32);(self.wr)(self.bar+0xb830b0,0x80000007);
        let mut flushed=false;for _ in 0..20000{if(self.rd)(self.bar+0xb830b0)&0x80000000==0{flushed=true;break;}boot::stall(10);}if !flushed{return false;}
        event(log,format!("  M4 Q4_ARENA LOSSY=true format=Q4_0 tensors={} allocated_bytes={} PA=0x{:x} end=0x{:x}",spans.len(),allocated,PA,PA+allocated as u64));
        for &(_,src,dest,blocks)in &spans{
            let mut q=self.q;self.stamp+=1;field(&mut q,384,415,((blocks+127)/128)as u64);field(&mut q,648,656,39);field(&mut q,832,863,self.stamp as u64);
            field(&mut q,1536,1567,0xc0550000);field(&mut q,256,287,0xc0550000>>8);field(&mut q,1641,1649,((include_bytes!("../kernels/repack_q4.sass").len()+255)/256)as u64);
            for(i,v)in q.iter().enumerate(){self.write(self.data+0x800+i as u64*4,*v);}
            self.write64(self.data+0xa60,src);self.write64(self.data+0xa68,VA+dest as u64);self.write(self.data+0xa70,blocks as u32);
            let words=[0x20012000|(0x1698>>2),0x1011,0x20012000|(0x2b4>>2),0x40008,0x20012000|(0x2c0>>2),9];
            for(i,v)in words.iter().enumerate(){self.write(self.data+0x200+i as u64*4,*v);}
            self.write(self.data+self.put as u64*8,0x4000200);self.write(self.data+self.put as u64*8+4,6<<10);
            fence(Ordering::SeqCst);let _=(self.rd)(self.ap(self.data+self.put as u64*8+4));self.put=(self.put+1)%32;
            self.write(self.userd+0x8c,self.put);let _=(self.rd)(self.ap(self.userd+0x8c));(self.wr)(self.bar+0xbb0090,self.token);
            let mut done=false;for _ in 0..100000{if(self.rd)(self.ap(self.data+0xf10))==self.stamp{done=true;break;}boot::stall(10);}if !done{return false;}
        }
        let mut checked=0usize;let start=self.time_ns();
        for(ti,&(off,_,dest,blocks))in spans.iter().enumerate(){
            for first in (0..blocks).step_by(58254){let count=(blocks-first).min(58254);assert!(count%2==0);
                if !self.copy_buffer(VA+(dest+first*18)as u64,0x100000000,count*18){return false;}
                let seen=unsafe{core::slice::from_raw_parts(self.work.host,count*18)};
                for b in 0..count{let src=model.data+off+(first+b)*34;let expected=block_ref(&model.bytes[src..src+34]);
                    if seen[b*18..b*18+18]!=expected {event(log,format!("  M4 Q4_VERIFY MISMATCH tensor={} block={} GPU={:x?} CPU={:x?}",ti,first+b,&seen[b*18..b*18+18],expected));return false;}}
                checked+=count*18;
            }
            if ti%32==0{event(log,format!("  M4 Q4_VERIFY progress tensors={}/{} bytes={}",ti+1,spans.len(),checked));}
        }
        self.vram_weights=old;for &(off,_,dest,_)in &spans{self.vram_weights.iter_mut().find(|(o,_)|*o==off).unwrap().1=VA+dest as u64;}
        event(log,format!("  *** M4 Q4_VERIFY PASS=true checked_bytes={} elapsed_ns={} original_GGUF_unchanged=true ***",checked,self.time_ns()-start));true
    }
}
