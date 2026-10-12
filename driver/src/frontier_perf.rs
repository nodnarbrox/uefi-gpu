//! Finite, firmware-managed performance request; no clock/voltage register writes.
//! Mirrors570.144 kperfBoostSet: public PERF_BOOST -> INTERNAL_PERF_BOOST_SET_2X.
use alloc::{format,string::String,vec::Vec};
use crate::{gsp,m3};
const SET:u32=0x20800a9a;
const GET:u32=0x20802068;
fn event(log:&mut Vec<String>,s:String){uefi::println!("{}",s);log.push(s);}
fn state<W:Fn(u64,u32)>(hs:&gsp::GspBootHandshake,wr:&W,falcon:u64,
    client:u32,subdevice:u32,next:&mut u32,label:&str,log:&mut Vec<String>){
    let mut params=u32::MAX.to_le_bytes();
    let seq=*next;*next+=1;
    let ok=m3::control(hs,wr,falcon,client,subdevice,GET,&mut params,seq,label,log);
    let raw=u32::from_le_bytes(params);
    let index=if ok&&raw.is_power_of_two()&&raw<=0x8000{Some(raw.trailing_zeros())}else{None};
    event(log,format!("FRONTIER_PSTATE phase={} query_ok={} raw=0x{:08x} index={:?}",label,ok,raw,index));
}
pub fn begin<W:Fn(u64,u32)>(hs:&gsp::GspBootHandshake,wr:&W,falcon:u64,
    client:u32,subdevice:u32,next:&mut u32,log:&mut Vec<String>)->bool {
    state(hs,wr,falcon,client,subdevice,next,"BEFORE_BOOST",log);
    // NvBool flags at0 (bitfield byte, NOT boolean1), padding1..4, NvU32 seconds4.
    // CUDA + high-priority + BOOST_TO_MAX. Finite2700s; normal thermal/power limits
    // still apply. Not overclocking, an infinite request, or a guaranteed P0 lock.
    let mut params=[0u8;8];params[0]=0x52;params[4..].copy_from_slice(&2700u32.to_le_bytes());
    let seq=*next;*next+=1;
    let ok=m3::control(hs,wr,falcon,client,subdevice,SET,&mut params,seq,"PERF_BOOST_2X",log);
    event(log,format!("FRONTIER_PERF_BOOST accepted={} cmd=0x{:08x} flags=0x52 duration_s=2700",ok,SET));
    if ok{uefi::boot::stall(200000);state(hs,wr,falcon,client,subdevice,next,"AFTER_BOOST",log);}
    ok
}
pub fn end<W:Fn(u64,u32)>(hs:&gsp::GspBootHandshake,wr:&W,falcon:u64,
    client:u32,subdevice:u32,next:&mut u32,log:&mut Vec<String>){
    state(hs,wr,falcon,client,subdevice,next,"BEFORE_CLEAR",log);
    let mut params=[0u8;8];params[0]=0x50; // Same CUDA request class, CLEAR, duration0.
    let seq=*next;*next+=1;
    let ok=m3::control(hs,wr,falcon,client,subdevice,SET,&mut params,seq,"PERF_BOOST_CLEAR",log);
    event(log,format!("FRONTIER_PERF_BOOST clear_ok={} cmd=0x{:08x} flags=0x50",ok,SET));
    if ok{uefi::boot::stall(100000);state(hs,wr,falcon,client,subdevice,next,"AFTER_CLEAR",log);}
}
