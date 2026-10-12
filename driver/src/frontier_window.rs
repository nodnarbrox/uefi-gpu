//! Cache the CPU-owned BAR0 framebuffer-window selector within one Gpu lifetime.
//! All bank changes during that lifetime must pass through this object. External
//! restoration happens only after the final use; a new Gpu starts invalid.
use core::cell::Cell;
pub struct Window {bank:Cell<Option<u32>>,calls:Cell<u64>,writes:Cell<u64>}
impl Window {
    pub fn new()->Self{Self{bank:Cell::new(None),calls:Cell::new(0),writes:Cell::new(0)}}
    pub fn address<W:Fn(u64,u32)>(&self,bar:u64,physical:u64,wr:&W)->u64{
        let bank=((physical&!0xfffff)>>16)as u32;
        self.calls.set(self.calls.get()+1);
        if self.bank.get()!=Some(bank){wr(bar+0x1700,bank);self.bank.set(Some(bank));self.writes.set(self.writes.get()+1);}
        bar+0x700000+(physical&0xfffff)
    }
    pub fn counts(&self)->(u64,u64){(self.calls.get(),self.writes.get())}
    pub fn reset_counts(&self){self.calls.set(0);self.writes.set(0);}
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn same_bank_crossings_and_new_owner(){
        let hardware=Cell::new(0u32);let n=Cell::new(0);
        let wr=|addr,value|{assert_eq!(addr,0x80001700);hardware.set(value);n.set(n.get()+1);};
        let window=Window::new();
        let addresses=[0x0c100000u64,0x0c100004,0x0c1ffffc,0x0c200000,
            0x2f3f00000,0x2f3f00004,0x0c100000];
        for p in addresses{
            let aperture=window.address(0x80000000,p,&wr);
            let reconstructed=((hardware.get()as u64)<<16)+(aperture-0x80700000);
            assert_eq!(reconstructed,p);
        }
        assert_eq!(window.counts(),(7,4));assert_eq!(n.get(),4);
        window.reset_counts();window.address(0x80000000,0x0c100000,&wr);assert_eq!(window.counts(),(1,0));
        hardware.set(0xdead);let next_owner=Window::new();
        next_owner.address(0x80000000,0x0c100000,&wr);assert_eq!(hardware.get(),0xc10);
        assert_eq!(next_owner.counts(),(1,1));
    }
    #[test]fn arbitrary_interleaved_command_and_readback_addresses(){
        let hardware=Cell::new(0);let wr=|_,v|hardware.set(v);let window=Window::new();
        let mut random=13u64;
        for i in 0..10000{
            random=random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let p=if i%5==0{(random%0x2f0000000)&!3}else{0x0c100000+(random%0x10000)&!3};
            let a=window.address(0x80000000,p,&wr);
            assert_eq!(((hardware.get()as u64)<<16)+(a-0x80700000),p);
        }
        assert_eq!(window.counts().0,10000);assert!(window.counts().1<6000);
    }
}
