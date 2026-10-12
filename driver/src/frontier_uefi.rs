//! First integration gate: actual UEFI Block I/O of pinned GGUF headers.
//! Header-only media are explicitly NOT complete model files or inference.
use alloc::{format,string::String,vec,vec::Vec};
use core::ffi::c_void;
use uefi::{boot,Identify};
use super::{frontier_gguf::{Shard,ModelIndex},frontier_storage::{ReadOnlyBlocks,Storage,Error}};
use alloc::boxed::Box;
use super::frontier_storage::Stats;
pub const LOGICAL:[u64;3]=[10946624,49983253824,39992153376];
pub const PREFIX:[usize;3]=[10946624,65536,1048576];
// Separate storage experiment: fewer synchronous reads, unchanged byte bounds.
// Allocate directly on the heap, not as a512KiB UEFI-stack temporary.
const BOUNCE_BYTES:usize=512*1024;
#[repr(align(4096))] struct Bounce([u8;BOUNCE_BYTES]);
pub struct FullMedia {blocks:Vec<UefiBlocks>,bounce:Box<Bounce>,pub stats:Stats}
impl FullMedia {
    /// Only complete, sector-padded media qualify. Header fixtures cannot enter
    /// inference. The host profile separately verifies published file hashes.
    pub fn discover()->Option<Self> {
        let handles=boot::locate_handle_buffer(boot::SearchType::ByProtocol(&crate::BlockIoProtocol::GUID)).ok()?;
        let mut found:[Option<UefiBlocks>;3]=core::array::from_fn(|_|None);
        for &handle in handles.iter() {
            let mut bio=match unsafe{boot::open_protocol::<crate::BlockIoProtocol>(boot::OpenProtocolParams{
                handle,agent:boot::image_handle(),controller:None},boot::OpenProtocolAttributes::GetProtocol)} {
                Ok(p)=>p,Err(_)=>continue};
            let Some(media)=(unsafe{bio.media.as_ref()})else{continue;};
            let block=media.block_size as usize;let mid=media.media_id;let align=media.io_align as usize;
            if media.media_present==0||block<16||!block.is_power_of_two()||block>4096||align>4096{continue;}
            let bytes=media.last_block.checked_add(1)?.checked_mul(block as u64)?;
            let mut first=crate::AlignedProbeBlock([0;4096]);
            if unsafe{(bio.read_blocks)(&mut*bio,mid,0,block,first.0.as_mut_ptr()as *mut c_void)}
                !=crate::RawStatus::SUCCESS||&first.0[..4]!=b"GGUF"{continue;}
            let i=match u64::from_le_bytes(first.0[8..16].try_into().ok()?){0=>0,402=>1,822=>2,_=>continue};
            let expected=(LOGICAL[i]+block as u64-1)&!(block as u64-1);
            if bytes!=expected {continue;}
            if found[i].is_some(){return None;}
            found[i]=Some(UefiBlocks{bio,mid,block,align,bytes});
        }
        let blocks=found.into_iter().collect::<Option<Vec<_>>>()?;
        let mut bounce=Box::<Bounce>::new_uninit();
        unsafe{core::ptr::write_bytes(bounce.as_mut_ptr().cast::<u8>(),0,core::mem::size_of::<Bounce>());}
        let bounce=unsafe{bounce.assume_init()};
        uefi::println!("FRONTIER_STORAGE bounce_bytes={} read_only=true",BOUNCE_BYTES);
        Some(Self{blocks,bounce,stats:Stats::default()})
    }
    pub fn read(&mut self,disk:usize,offset:u64,out:&mut[u8])->Result<(),Error> {
        let device=self.blocks.get_mut(disk).ok_or(Error::Bounds)?;
        let mut storage=Storage::new(device,&mut self.bounce.0,LOGICAL[disk])?;
        let result=storage.read_exact(offset,out);let s=storage.stats;
        self.stats.requests+=s.requests;self.stats.calls+=s.calls;self.stats.requested+=s.requested;
        self.stats.transferred+=s.transferred;self.stats.failures+=s.failures;result
    }
    pub fn headers(&mut self)->Option<[Vec<u8>;3]> {
        let mut result=core::array::from_fn(|i|vec![0;PREFIX[i]]);
        for i in 0..3{self.read(i,0,&mut result[i]).ok()?;}Some(result)
    }
}

struct UefiBlocks {
    bio:boot::ScopedProtocol<crate::BlockIoProtocol>,mid:u32,block:usize,align:usize,bytes:u64,
}
impl ReadOnlyBlocks for UefiBlocks {
    fn block_size(&self)->usize {self.block}
    fn io_align(&self)->usize {self.align}
    fn device_bytes(&self)->u64 {self.bytes}
    fn read_blocks(&mut self,lba:u64,output:&mut[u8])->Result<(),Error> {
        let media=unsafe{self.bio.media.as_ref()}.ok_or(Error::Io)?;
        if media.media_present==0 || media.media_id!=self.mid {return Err(Error::Io);}
        if unsafe{(self.bio.read_blocks)(&mut*self.bio,self.mid,lba,output.len(),output.as_mut_ptr() as *mut c_void)}
            !=crate::RawStatus::SUCCESS {return Err(Error::Io);}Ok(())
    }
}
fn event(log:&mut Vec<String>,s:String) {uefi::println!("{}",s);log.push(s);}
fn time<R:Fn(u64)->u32>(rd:&R,bar:u64)->u64 {
    loop {let hi=rd(bar+0x9410);let lo=rd(bar+0x9400);
        if hi==rd(bar+0x9410) {return ((hi as u64)<<32)|lo as u64;}}
}

pub fn header_probe<R:Fn(u64)->u32>(rd:&R,bar:u64,log:&mut Vec<String>)->bool {
    let result=(||->Option<()> {
        let handles=boot::locate_handle_buffer(boot::SearchType::ByProtocol(&crate::BlockIoProtocol::GUID)).ok()?;
        let mut prefixes:[Vec<u8>;3]=core::array::from_fn(|_|Vec::new());
        let lengths=[10946624,65536,1048576];
        let logical=[10946624,49983253824,39992153376];
        for &handle in handles.iter() {
            let mut bio=match unsafe{boot::open_protocol::<crate::BlockIoProtocol>(boot::OpenProtocolParams{
                handle,agent:boot::image_handle(),controller:None},boot::OpenProtocolAttributes::GetProtocol)} {
                Ok(p)=>p,Err(_)=>continue};
            let media=unsafe{bio.media.as_ref()}?;
            let block=media.block_size as usize;let mid=media.media_id;let align=media.io_align as usize;
            if media.media_present==0 || !block.is_power_of_two() || block>4096 ||
                align>4096 {continue;}
            let bytes=media.last_block.checked_add(1)?.checked_mul(block as u64)?;
            let mut first=crate::AlignedProbeBlock([0;4096]);
            if unsafe{(bio.read_blocks)(&mut*bio,mid,0,block,first.0.as_mut_ptr() as *mut c_void)}
                !=crate::RawStatus::SUCCESS || &first.0[..4]!=b"GGUF" {continue;}
            let count=u64::from_le_bytes(first.0[8..16].try_into().ok()?);
            let index=match count {0=>0,402=>1,822=>2,_=>continue};
            if !prefixes[index].is_empty() {return None;}
            let device=UefiBlocks{bio,mid,block,align,bytes};
            let mut bounce=crate::AlignedProbeBlock([0;4096]);
            let mut source=Storage::new(device,&mut bounce.0,bytes).ok()?;
            let mut prefix=vec![0;lengths[index]];
            let start=time(rd,bar);source.read_exact(0,&mut prefix).ok()?;
            let elapsed=time(rd,bar)-start;
            let parsed=Shard::parse(&prefix,logical[index]).ok()?;
            if parsed.split_no!=index as u64 {return None;}
            event(log,format!("  FRONTIER_HEADER shard={} media_bytes={} read_bytes={} calls={} elapsed_ns={} tensors={} data_offset={} header_only=true",
                index,bytes,source.stats.transferred,source.stats.calls,elapsed,parsed.tensors.len(),parsed.data_offset));
            prefixes[index]=prefix;
        }
        let shards:Vec<_>=prefixes.iter().zip(logical).map(|(p,size)|Shard::parse(p,size)).collect::<Result<_,_>>().ok()?;
        let model=ModelIndex::new(shards).ok()?;let inventory=model.inventory().ok()?;
        if inventory!=[5351626240,28800138240,55823564800] {return None;}
        let (_,lookup)=model.tensor("per_layer_token_embd.weight").ok()?;
        if lookup.kind!=20 || lookup.row_bytes!=90 {return None;}
        event(log,format!("  FRONTIER_DIRECTORY tensors=1224 always_bytes={} lookup_bytes={} expert_bytes={} lookup_row_bytes={} PASS=true inference=false",
            inventory[0],inventory[1],inventory[2],lookup.row_bytes));
        Some(())
    })();
    event(log,format!("  *** FRONTIER_HEADER_PROBE PASS={} full_model_loaded=false inference=false ***",result.is_some()));
    result.is_some()
}
