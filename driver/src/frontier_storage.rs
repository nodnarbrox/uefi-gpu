//! Read-only bounded storage path. The UEFI adapter supplies a DMA-safe aligned
//! bounce buffer; no guest pointer is assumed to be a GPU bus address.
use super::frontier_core::block_range;

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Error { Alignment, Bounds, Buffer, Io }
pub trait ReadOnlyBlocks {
    fn block_size(&self)->usize;
    fn io_align(&self)->usize;
    fn device_bytes(&self)->u64;
    fn read_blocks(&mut self,lba:u64,output:&mut[u8])->Result<(),Error>;
}
impl<T:ReadOnlyBlocks> ReadOnlyBlocks for &mut T {
    fn block_size(&self)->usize {(**self).block_size()}
    fn io_align(&self)->usize {(**self).io_align()}
    fn device_bytes(&self)->u64 {(**self).device_bytes()}
    fn read_blocks(&mut self,lba:u64,output:&mut[u8])->Result<(),Error> {(**self).read_blocks(lba,output)}
}
#[derive(Default,Debug,Clone,Copy)]
pub struct Stats {pub requests:u64,pub calls:u64,pub requested:u64,pub transferred:u64,pub failures:u64}
pub struct Storage<'a,D:ReadOnlyBlocks> {
    pub device:D,bounce:&'a mut[u8],logical_bytes:u64,pub stats:Stats,
}
impl<'a,D:ReadOnlyBlocks> Storage<'a,D> {
    pub fn new(device:D,bounce:&'a mut[u8],logical_bytes:u64)->Result<Self,Error> {
        let block=device.block_size();let align=device.io_align().max(1);
        if block==0 || !block.is_power_of_two() || block>4096 || !align.is_power_of_two() ||
            align>4096 || bounce.as_ptr() as usize%align!=0 {return Err(Error::Alignment);}
        if bounce.len()<block || bounce.len()%block!=0 {return Err(Error::Buffer);}
        if logical_bytes==0 || logical_bytes>device.device_bytes() || device.device_bytes()%block as u64!=0 {
            return Err(Error::Bounds);
        }
        Ok(Self{device,bounce,logical_bytes,stats:Stats::default()})
    }
    pub fn read_exact(&mut self,offset:u64,output:&mut[u8])->Result<(),Error> {
        let end=offset.checked_add(output.len() as u64).ok_or(Error::Bounds)?;
        if end>self.logical_bytes {return Err(Error::Bounds);}
        if output.is_empty() {return Ok(());}
        self.stats.requests+=1;self.stats.requested+=output.len() as u64;
        let block=self.device.block_size();let mut done=0;
        while done<output.len() {
            let current=offset+done as u64;let skip=(current%block as u64) as usize;
            let amount=(output.len()-done).min(self.bounce.len()-skip);
            let (lba,read_bytes,skip)=block_range(current,amount,self.device.device_bytes(),block)
                .map_err(|_|Error::Bounds)?;
            if read_bytes>self.bounce.len() {return Err(Error::Buffer);}
            self.stats.calls+=1;
            if self.device.read_blocks(lba,&mut self.bounce[..read_bytes]).is_err() {
                self.stats.failures+=1;return Err(Error::Io);
            }
            self.stats.transferred+=read_bytes as u64;
            output[done..done+amount].copy_from_slice(&self.bounce[skip..skip+amount]);
            done+=amount;
        }Ok(())
    }
}

#[cfg(test)] mod tests {
    use super::*;
    struct Disk {fail_lba:Option<u64>,calls:usize}
    fn pattern(i:u64)->u8 {(i.wrapping_mul(131)^(i>>8)^(i>>16)) as u8}
    impl ReadOnlyBlocks for Disk {
        fn block_size(&self)->usize {512}
        fn io_align(&self)->usize {4096}
        fn device_bytes(&self)->u64 {4*1024*1024}
        fn read_blocks(&mut self,lba:u64,output:&mut[u8])->Result<(),Error> {
            assert_eq!(output.as_ptr() as usize%4096,0);assert_eq!(output.len()%512,0);
            assert!(output.len()<=65536);assert!(lba*512+output.len() as u64<=self.device_bytes());
            self.calls+=1;if self.fail_lba==Some(lba) {return Err(Error::Io);}
            for (i,v) in output.iter_mut().enumerate() {*v=pattern(lba*512+i as u64);}
            Ok(())
        }
    }
    #[repr(align(4096))] struct Bounce([u8;65536]);
    #[test] fn fixed_buffer_unaligned_and_large_reads() {
        let mut bounce=Bounce([0;65536]);
        let mut store=Storage::new(Disk{fail_lba:None,calls:0},&mut bounce.0,4*1024*1024-91).unwrap();
        for (offset,amount) in [(0,90),(511,90),(65530,190000),(4*1024*1024-181,90)] {
            let mut out=vec![0u8;amount];store.read_exact(offset,&mut out).unwrap();
            for (i,v) in out.iter().enumerate() {assert_eq!(*v,pattern(offset+i as u64));}
        }
        assert_eq!(store.stats.requests,4);assert_eq!(store.stats.failures,0);
        assert!(store.stats.transferred>=store.stats.requested);
        assert_eq!(store.stats.calls as usize,store.device.calls);
    }
    #[test] fn rejects_out_of_bounds_without_io() {
        let mut bounce=Bounce([0;65536]);
        let mut store=Storage::new(Disk{fail_lba:None,calls:0},&mut bounce.0,4097).unwrap();
        assert_eq!(store.read_exact(4090,&mut[0;8]),Err(Error::Bounds));
        assert_eq!(store.read_exact(u64::MAX,&mut[0;8]),Err(Error::Bounds));
        assert_eq!(store.device.calls,0);
        store.read_exact(4097,&mut[]).unwrap();assert_eq!(store.device.calls,0);
    }
    #[test] fn failure_propagates_not_a_partial_success() {
        let mut bounce=Bounce([0;65536]);
        let mut store=Storage::new(Disk{fail_lba:Some(128),calls:0},&mut bounce.0,4*1024*1024).unwrap();
        assert_eq!(store.read_exact(0,&mut vec![0;100000]),Err(Error::Io));
        assert_eq!(store.stats.failures,1);assert_eq!(store.stats.calls,2);
        assert_eq!(store.stats.transferred,65536);
    }
    #[test] fn larger_bounce_is_identical_and_reduces_read_calls(){
        struct LargeDisk;
        impl ReadOnlyBlocks for LargeDisk {
            fn block_size(&self)->usize{512}
            fn io_align(&self)->usize{4096}
            fn device_bytes(&self)->u64{4*1024*1024}
            fn read_blocks(&mut self,lba:u64,out:&mut[u8])->Result<(),Error>{
                assert_eq!(out.as_ptr()as usize%4096,0);assert!(out.len()<=524288);
                assert_eq!(out.len()%512,0);assert!(lba*512+out.len()as u64<=self.device_bytes());
                for(i,v)in out.iter_mut().enumerate(){*v=pattern(lba*512+i as u64);}Ok(())
            }
        }
        #[repr(align(4096))]struct Big([u8;524288]);
        let mut big=Box::<Big>::new_uninit();
        unsafe{core::ptr::write_bytes(big.as_mut_ptr().cast::<u8>(),0,core::mem::size_of::<Big>());}
        let mut big=unsafe{big.assume_init()};let mut small=Bounce([0;65536]);
        let mut a=Storage::new(LargeDisk,&mut small.0,4*1024*1024-91).unwrap();
        let mut b=Storage::new(LargeDisk,&mut big.0,4*1024*1024-91).unwrap();
        for(offset,n)in [(0,90),(511,90),(65530,1048576),(2097281,1500000),(4*1024*1024-181,90)]{
            let mut x=vec![0;n];let mut y=vec![0;n];a.read_exact(offset,&mut x).unwrap();b.read_exact(offset,&mut y).unwrap();
            assert_eq!(x,y);for(i,v)in x.iter().enumerate(){assert_eq!(*v,pattern(offset+i as u64));}
        }
        assert_eq!(a.stats.requested,b.stats.requested);assert_eq!(a.stats.transferred,b.stats.transferred);
        assert!(a.stats.calls>4*b.stats.calls);assert_eq!(b.stats.failures,0);
    }
}
