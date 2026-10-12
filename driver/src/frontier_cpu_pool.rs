//! CPU expert buffers with an enforced sum-of-Vec-capacities budget.
//! Directory leases remain the caller's responsibility. Taking a buffer for an
//! upload does not free its budget charge until it is returned to the same slot.
extern crate alloc;
use alloc::{vec,vec::Vec};
#[derive(Debug,PartialEq,Eq)]pub enum Error{Slot,Size,Budget,Allocation,Loan}
pub struct Pool{data:Vec<Vec<u8>>,charge:Vec<usize>,loan:Vec<bool>,used:usize,limit:usize,max:usize}
pub struct Loan{slot:usize,bytes:Vec<u8>}
impl AsRef<[u8]> for Loan{fn as_ref(&self)->&[u8]{&self.bytes}}
impl Pool{
    pub fn new(slots:usize,max:usize,limit:usize)->Result<Self,Error>{
        if slots==0||max==0||limit<max{return Err(Error::Size);}
        Ok(Self{data:vec![Vec::new();slots],charge:vec![0;slots],loan:vec![false;slots],used:0,limit,max})
    }
    pub fn used(&self)->usize{self.used}
    pub fn prepare(&mut self,slot:usize,n:usize)->Result<&mut[u8],Error>{
        if slot>=self.data.len(){return Err(Error::Slot);}
        if self.loan[slot]{return Err(Error::Loan);}
        if n==0||n>self.max{return Err(Error::Size);}
        let other=self.used-self.charge[slot];
        if n>self.limit-other{return Err(Error::Budget);}
        // Reuse exact-sized allocations; otherwise release before reserving so
        // no old+new allocation transient can exceed the budget.
        if self.data[slot].capacity()!=n{
            self.used=other;self.charge[slot]=0;
            drop(core::mem::take(&mut self.data[slot]));
            let mut next=Vec::new();next.try_reserve_exact(n).map_err(|_|Error::Allocation)?;
            if next.capacity()>self.limit-other{return Err(Error::Budget);}
            self.charge[slot]=next.capacity();self.used=other+next.capacity();self.data[slot]=next;
        }
        self.data[slot].resize(n,0);Ok(&mut self.data[slot])
    }
    pub fn take(&mut self,slot:usize)->Result<Loan,Error>{
        if slot>=self.data.len(){return Err(Error::Slot);}
        if self.loan[slot]||self.data[slot].is_empty(){return Err(Error::Loan);}
        self.loan[slot]=true;Ok(Loan{slot,bytes:core::mem::take(&mut self.data[slot])})
    }
    pub fn put(&mut self,buffer:Loan)->Result<(),Error>{
        let Loan{slot,bytes:data}=buffer;
        if slot>=self.data.len(){return Err(Error::Slot);}
        if !self.loan[slot]||data.capacity()!=self.charge[slot]||data.is_empty(){return Err(Error::Loan);}
        self.data[slot]=data;self.loan[slot]=false;Ok(())
    }
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn replacement_and_loans_never_hide_capacity(){
        let mut p=Pool::new(3,100,180).unwrap();
        p.prepare(0,100).unwrap().fill(7);p.prepare(1,80).unwrap().fill(8);
        assert_eq!(p.used(),180);assert_eq!(p.prepare(2,1),Err(Error::Budget));
        let v=p.take(0).unwrap();assert_eq!(p.used(),180);
        assert_eq!(p.prepare(0,10),Err(Error::Loan));assert!(matches!(p.take(0),Err(Error::Loan)));
        assert!(v.as_ref().iter().all(|&x|x==7));p.put(v).unwrap();
        p.prepare(0,20).unwrap();assert_eq!(p.used(),100);
        p.prepare(2,80).unwrap();assert_eq!(p.used(),180);
        assert_eq!(p.prepare(3,1),Err(Error::Slot));assert_eq!(p.prepare(0,101),Err(Error::Size));
    }
    #[test]fn changing_expert_sizes_account_real_vec_capacity(){
        let mut p=Pool::new(16,200,2400).unwrap();
        for i in 0..10000{
            let slot=i%16;let n=1+(i*17)%200;
            match p.prepare(slot,n){Ok(b)=>b.fill(i as u8),Err(Error::Budget)=>{},Err(e)=>panic!("{e:?}")}
            assert_eq!(p.used(),p.data.iter().map(Vec::capacity).sum());assert!(p.used()<=2400);
        }
    }
}
