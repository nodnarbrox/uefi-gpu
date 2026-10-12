//! Private reusable copy QMD. Same copy kernel, synchronous fence and full
//! caller readback validation; no asynchronous lifetime or DMA mapping changes.
use super::*;
const PA:u64=0x0c210000;
const VA:u64=0xc0210000;
impl<R:Fn(u64)->u32,W:Fn(u64,u32)> Gpu<'_,R,W>{
    pub(super) fn frontier_copy_init(&mut self){
        let mut q=self.q;
        field(&mut q,384,415,1);field(&mut q,592,607,128);
        field(&mut q,648,656,8);field(&mut q,832,863,0);
        field(&mut q,823,823,1);field(&mut q,768,799,VA+0x3f0);
        field(&mut q,800,807,0);
        field(&mut q,1024,1055,VA+0x100);field(&mut q,1056,1072,0);
        field(&mut q,1075,1087,0x200/16);
        field(&mut q,1536,1567,0x04002000);field(&mut q,256,287,0x04002000>>8);
        field(&mut q,1641,1649,2);field(&mut q,544,561,0);field(&mut q,642,642,0);
        for(i,v)in q.iter().enumerate(){self.write(PA+i as u64*4,*v);}
        for i in 0..128{self.write(PA+0x100+i*4,0);}
        self.write(PA+0x100,128);self.write(PA+0x104,1);self.write(PA+0x108,1);
        self.write(PA+0x3f0,0);
        let words=[0x20012000|(0x1698>>2),0x1011,0x20012000|(0x2b4>>2),(VA>>8)as u32,
            0x20012000|(0x2c0>>2),9];
        for(i,v)in words.iter().enumerate(){self.write(PA+0x400+i as u64*4,*v);}
    }
    pub(super) fn frontier_copy_buffer(&mut self,src:u64,dst:u64,bytes:usize)->bool{
        if bytes==0||bytes%4!=0||bytes/4>u32::MAX as usize
            ||src.checked_add(bytes as u64).is_none()||dst.checked_add(bytes as u64).is_none(){return false;}
        self.stamp+=1;
        self.write(PA+0x30,((bytes/4+127)/128)as u32);
        self.write(PA+0x68,self.stamp);
        self.write64(PA+0x260,src);self.write64(PA+0x268,dst);self.write(PA+0x270,(bytes/4)as u32);
        self.write(self.data+(self.put%32)as u64*8,(VA+0x400)as u32);
        self.write(self.data+(self.put%32)as u64*8+4,6<<10);
        fence(Ordering::SeqCst);let _=(self.rd)(self.ap(PA+0x68));
        self.put=(self.put+1)%32;self.write(self.userd+0x8c,self.put);
        let _=(self.rd)(self.ap(self.userd+0x8c));(self.wr)(self.bar+0xbb0090,self.token);
        for _ in 0..100000{
            if(self.rd)(self.ap(PA+0x3f0))==self.stamp{fence(Ordering::SeqCst);return true;}
            boot::stall(10);
        }false
    }
}
