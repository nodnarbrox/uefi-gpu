//! Fixed-capacity LRU directory. Slots are never reusable while loading or held
//! by unfinished GPU work. CPU and VRAM pools use separate directories.
extern crate alloc;
use alloc::{vec,vec::Vec};
#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)] pub struct Key {pub tensor:usize,pub expert:usize}
#[derive(Clone,Copy,Debug,PartialEq,Eq)] pub struct Lease {pub slot:usize,epoch:u64}
#[derive(Clone,Copy,Debug,PartialEq,Eq)] pub enum Error { Capacity,Busy,Stale,Size,Clock }
#[derive(Clone,Copy)] struct Entry {key:Option<Key>,bytes:usize,epoch:u64,last:u64,held:bool,ready:bool}
pub struct Cache {entries:Vec<Entry>,slot_bytes:usize,clock:u64,pub hits:u64,pub misses:u64,pub evictions:u64}
impl Cache {
    pub fn resident_payload_bytes(&self)->usize {
        self.entries.iter().filter(|e|e.key.is_some()&&e.ready).map(|e|e.bytes).sum()
    }
    pub fn new(slots:usize,slot_bytes:usize)->Result<Self,Error> {
        if slots==0||slot_bytes==0||slots.checked_mul(slot_bytes).is_none(){return Err(Error::Capacity);}
        Ok(Self{entries:vec![Entry{key:None,bytes:0,epoch:0,last:0,held:false,ready:false};slots],
            slot_bytes,clock:0,hits:0,misses:0,evictions:0})
    }
    /// Returns (lease, hit). A miss reserves a held loading slot until commit.
    pub fn acquire(&mut self,key:Key,bytes:usize)->Result<(Lease,bool),Error> {
        if bytes==0||bytes>self.slot_bytes{return Err(Error::Size);}
        self.clock=self.clock.checked_add(1).ok_or(Error::Clock)?;
        if let Some((i,e))=self.entries.iter_mut().enumerate().find(|(_,e)|e.key==Some(key)) {
            if e.bytes!=bytes{return Err(Error::Size);}if e.held||!e.ready{return Err(Error::Busy);}
            e.held=true;e.last=self.clock;self.hits+=1;return Ok((Lease{slot:i,epoch:e.epoch},true));
        }
        let i=self.entries.iter().enumerate().filter(|(_,e)|!e.held)
            .min_by_key(|(_,e)|(e.key.is_some(),e.last)).map(|(i,_)|i).ok_or(Error::Busy)?;
        let e=&mut self.entries[i];if e.key.is_some(){self.evictions+=1;}
        e.epoch=e.epoch.checked_add(1).ok_or(Error::Clock)?;e.key=Some(key);e.bytes=bytes;
        e.last=self.clock;e.held=true;e.ready=false;self.misses+=1;
        Ok((Lease{slot:i,epoch:e.epoch},false))
    }
    fn entry(&mut self,l:Lease)->Result<&mut Entry,Error> {
        let e=self.entries.get_mut(l.slot).ok_or(Error::Stale)?;
        if e.key.is_none()||e.epoch!=l.epoch||!e.held{return Err(Error::Stale);}Ok(e)
    }
    pub fn commit(&mut self,l:Lease)->Result<(),Error> {self.entry(l)?.ready=true;Ok(())}
    /// Caller must have observed the consuming GPU fence before releasing.
    pub fn release(&mut self,l:Lease)->Result<(),Error> {
        let e=self.entry(l)?;if !e.ready{return Err(Error::Busy);}e.held=false;Ok(())
    }
    pub fn abort(&mut self,l:Lease)->Result<(),Error> {
        let e=self.entry(l)?;e.key=None;e.ready=false;e.held=false;Ok(())
    }
}
/// Variable-size VRAM reservations, with one global LRU over matrices. Unit
/// rounding is explicit; held/loading allocations cannot be reclaimed.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct PackedLease {entry:usize,epoch:u64,offset:usize}
impl PackedLease {pub fn offset(&self)->usize{self.offset}}
#[derive(Clone,Copy)]
struct PackedEntry {key:Option<Key>,bytes:usize,start:usize,units:usize,epoch:u64,last:u64,held:bool,ready:bool}
pub struct PackedCache {
    entries:Vec<PackedEntry>,owners:Vec<Option<usize>>,unit:usize,clock:u64,
    pub hits:u64,pub misses:u64,pub evictions:u64,
}
impl PackedCache {
    pub fn new(bytes:usize,unit:usize)->Result<Self,Error>{
        if unit==0||bytes==0||bytes%unit!=0{return Err(Error::Capacity);}
        let n=bytes/unit;
        Ok(Self{entries:vec![PackedEntry{key:None,bytes:0,start:0,units:0,epoch:0,last:0,held:false,ready:false};n],
            owners:vec![None;n],unit,clock:0,hits:0,misses:0,evictions:0})
    }
    pub fn resident_payload_bytes(&self)->usize{
        self.entries.iter().filter(|e|e.key.is_some()&&e.ready).map(|e|e.bytes).sum()
    }
    fn clear(&mut self,i:usize){
        let e=&mut self.entries[i];
        for owner in &mut self.owners[e.start..e.start+e.units]{assert_eq!(*owner,Some(i));*owner=None;}
        e.key=None;e.bytes=0;e.units=0;e.held=false;e.ready=false;
    }
    pub fn acquire(&mut self,key:Key,bytes:usize)->Result<(PackedLease,bool),Error>{
        if bytes==0{return Err(Error::Size);}
        let units=bytes.checked_add(self.unit-1).ok_or(Error::Size)?/self.unit;
        if units>self.owners.len(){return Err(Error::Size);}
        self.clock=self.clock.checked_add(1).ok_or(Error::Clock)?;
        if let Some((i,e))=self.entries.iter_mut().enumerate().find(|(_,e)|e.key==Some(key)){
            if e.bytes!=bytes{return Err(Error::Size);}if e.held||!e.ready{return Err(Error::Busy);}
            e.held=true;e.last=self.clock;self.hits+=1;
            return Ok((PackedLease{entry:i,epoch:e.epoch,offset:e.start*self.unit},true));
        }
        // Check feasibility before evicting: a held unit cannot be crossed.
        if !self.owners.windows(units).any(|w|w.iter().all(|o|o.is_none_or(|i|!self.entries[i].held))){return Err(Error::Busy);}
        let start=loop{
            if let Some(p)=self.owners.windows(units).position(|w|w.iter().all(Option::is_none)){break p;}
            let i=self.entries.iter().enumerate().filter(|(_,e)|e.key.is_some()&&!e.held)
                .min_by_key(|(_,e)|e.last).map(|(i,_)|i).ok_or(Error::Busy)?;
            self.clear(i);self.evictions+=1;
        };
        let i=self.entries.iter().position(|e|e.key.is_none()).ok_or(Error::Capacity)?;
        let e=&mut self.entries[i];e.epoch=e.epoch.checked_add(1).ok_or(Error::Clock)?;
        e.key=Some(key);e.bytes=bytes;e.start=start;e.units=units;e.last=self.clock;e.held=true;e.ready=false;
        for owner in &mut self.owners[start..start+units]{assert!(owner.is_none());*owner=Some(i);}
        self.misses+=1;Ok((PackedLease{entry:i,epoch:e.epoch,offset:start*self.unit},false))
    }
    fn entry(&mut self,l:PackedLease)->Result<&mut PackedEntry,Error>{
        let e=self.entries.get_mut(l.entry).ok_or(Error::Stale)?;
        if e.key.is_none()||e.epoch!=l.epoch||!e.held||e.start*self.unit!=l.offset{return Err(Error::Stale);}Ok(e)
    }
    pub fn commit(&mut self,l:PackedLease)->Result<(),Error>{self.entry(l)?.ready=true;Ok(())}
    pub fn release(&mut self,l:PackedLease)->Result<(),Error>{
        let e=self.entry(l)?;if !e.ready{return Err(Error::Busy);}e.held=false;Ok(())
    }
    pub fn abort(&mut self,l:PackedLease)->Result<(),Error>{self.entry(l)?;self.clear(l.entry);Ok(())}
}
#[cfg(test)] mod tests {
    use super::*;fn key(i:usize)->Key{Key{tensor:i,expert:1}}
    #[test] fn pinned_loading_and_stale_fence() {
        let mut c=Cache::new(1,100).unwrap();let (a,hit)=c.acquire(key(0),90).unwrap();assert!(!hit);
        assert_eq!(c.acquire(key(1),90),Err(Error::Busy));assert_eq!(c.release(a),Err(Error::Busy));
        c.commit(a).unwrap();assert_eq!(c.acquire(key(1),90),Err(Error::Busy));c.release(a).unwrap();
        let(b,hit)=c.acquire(key(1),90).unwrap();assert!(!hit);assert_eq!(c.release(a),Err(Error::Stale));
        c.abort(b).unwrap();assert_eq!(c.evictions,1);
    }
    #[test] fn least_recent_and_capacity() {
        let mut c=Cache::new(2,100).unwrap();
        for i in 0..2 {let(l,_)=c.acquire(key(i),80).unwrap();c.commit(l).unwrap();c.release(l).unwrap();}
        assert_eq!(c.resident_payload_bytes(),160);
        let(a,hit)=c.acquire(key(0),80).unwrap();assert!(hit);c.release(a).unwrap();
        let(b,hit)=c.acquire(key(2),80).unwrap();assert!(!hit);assert_eq!(b.slot,1);c.abort(b).unwrap();
        assert_eq!(c.resident_payload_bytes(),80);
        assert_eq!(c.acquire(key(3),101),Err(Error::Size));
        assert_eq!(c.acquire(key(0),81),Err(Error::Size));assert_eq!(c.hits,1);
    }
    #[test] fn packed_variable_sizes_pins_abort_and_stale_leases(){
        let mut c=PackedCache::new(400,100).unwrap();
        let(a,_)=c.acquire(key(0),190).unwrap();assert_eq!(a.offset(),0);c.commit(a).unwrap();
        let(b,_)=c.acquire(key(1),90).unwrap();assert_eq!(b.offset(),200);c.commit(b).unwrap();
        let(d,_)=c.acquire(key(2),90).unwrap();assert_eq!(d.offset(),300);c.commit(d).unwrap();
        assert_eq!(c.acquire(key(3),150),Err(Error::Busy));assert_eq!(c.evictions,0);
        c.release(a).unwrap();let(e,hit)=c.acquire(key(3),150).unwrap();assert!(!hit);assert_eq!(e.offset(),0);
        assert_eq!(c.release(a),Err(Error::Stale));assert_eq!(c.release(e),Err(Error::Busy));
        c.abort(e).unwrap();assert_eq!(c.resident_payload_bytes(),180);
        c.release(b).unwrap();c.release(d).unwrap();
        let(b2,hit)=c.acquire(key(1),90).unwrap();assert!(hit);assert_eq!(b2.offset(),200);c.release(b2).unwrap();
        assert_eq!(c.acquire(key(1),91),Err(Error::Size));assert_eq!(c.acquire(key(4),401),Err(Error::Size));
    }
    #[test] fn packed_trace_never_aliases_live_payloads(){
        let mut c=PackedCache::new(64*1024,1024).unwrap();let mut memory=vec![0u8;64*1024];
        let mut held:Vec<(PackedLease,u8,usize)>=Vec::new();let mut random=17u64;
        for step in 0..10000 {
            random=random.wrapping_mul(6364136223846793005).wrapping_add(1);
            if !held.is_empty()&&(random>>33)%4==0{
                let (l,value,bytes)=held.remove((random as usize)%held.len());
                assert!(memory[l.offset()..l.offset()+bytes].iter().all(|b|*b==value));c.release(l).unwrap();
            }
            let id=((random>>20)%100)as usize;let bytes=if id%7==0{1700}else{700};
            match c.acquire(key(id),bytes){
                Ok((l,hit))=>{
                    if hit{assert!(memory[l.offset()..l.offset()+bytes].iter().all(|b|*b==id as u8));}
                    else{memory[l.offset()..l.offset()+bytes].fill(id as u8);c.commit(l).unwrap();}
                    if held.len()<8 && step%3==0{held.push((l,id as u8,bytes));}else{c.release(l).unwrap();}
                },Err(Error::Busy)=>{},Err(e)=>panic!("{e:?}"),
            }
            for (i,e) in c.entries.iter().enumerate(){if e.key.is_some(){
                assert!(c.owners[e.start..e.start+e.units].iter().all(|o|*o==Some(i)));
            }}
            assert!(c.resident_payload_bytes()<=memory.len());
        }
        assert!(c.hits>0&&c.misses>0&&c.evictions>0);
    }
}
