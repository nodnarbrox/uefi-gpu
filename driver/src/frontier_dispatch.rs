//! Explicit little-endian descriptor ABI; no native Rust struct layout dependency.
extern crate alloc;
use alloc::vec::Vec;
pub const MAX:usize=32;
pub const STRIDE:usize=48;
/// Only the large independent expert groups use descriptor-driven dispatch.
/// Small groups retain the existing per-job QMD path; this is a measured policy
/// candidate, not an assumption that combining every group is beneficial.
pub fn use_multi(count:usize)->bool { matches!(count,11|23) }
#[derive(Clone,Copy)]
pub struct Dispatch {pub weight:u64,pub input:u64,pub output:u64,pub kind:u32,pub n:u32,pub m:u32,pub row_bytes:u32}
pub fn pack(jobs:&[Dispatch])->Option<(Vec<u8>,u32)> {
    if jobs.is_empty()||jobs.len()>MAX{return None;}
    let mut bytes=Vec::with_capacity(jobs.len()*STRIDE);let mut rows=0u32;
    for job in jobs {
        if job.n==0||job.m==0||job.row_bytes==0||job.n>16384{return None;}
        let(row_end,input_bytes)=(job.m as u64*job.row_bytes as u64,job.n as u64*4);
        job.weight.checked_add(row_end)?;job.input.checked_add(input_bytes)?;
        let out_end=job.output.checked_add(job.m as u64*4)?;
        // The upper scratch MiB contains descriptors/input experiments, not output.
        if job.output<0xe0000000||out_end>0xe0100000{return None;}
        for previous in &jobs[..bytes.len()/STRIDE] {
            let end=previous.output.checked_add(previous.m as u64*4)?;
            if job.output<end&&previous.output<out_end{return None;}
        }
        for word in [job.weight,job.input,job.output]{bytes.extend_from_slice(&word.to_le_bytes());}
        for word in [job.kind,job.n,job.m,job.row_bytes,rows,0]{bytes.extend_from_slice(&word.to_le_bytes());}
        rows=rows.checked_add(job.m.checked_add(3)?&!3)?;
    }
    Some((bytes,rows/4))
}

#[cfg(test)]mod tests {
    use super::*;
    #[test]fn selective_policy(){
        for n in 0..=33 {assert_eq!(use_multi(n),n==11||n==23);}
    }
    fn job(m:u32,offset:u64)->Dispatch {Dispatch{weight:0x400000000,input:0x100010000,
        output:0xe0000000+offset,kind:18,n:2560,m,row_bytes:980}}
    #[test]fn abi_and_padded_row_mapping(){
        for count in [1,2,3,4,11,23,32] {
            let mut offset=0;let jobs:Vec<_>=(0..count).map(|i|{let m=[1,3,640,2560][i%4];
                let j=job(m,offset);offset+=m as u64*4;j}).collect();
            let(bytes,grid)=pack(&jobs).unwrap();assert_eq!(bytes.len(),count*48);
            let mut prefix=0;let mut seen=0;
            for (i,chunk) in bytes.chunks_exact(48).enumerate(){
                assert_eq!(u64::from_le_bytes(chunk[0..8].try_into().unwrap()),jobs[i].weight);
                assert_eq!(u64::from_le_bytes(chunk[8..16].try_into().unwrap()),jobs[i].input);
                assert_eq!(u64::from_le_bytes(chunk[16..24].try_into().unwrap()),jobs[i].output);
                assert_eq!(u32::from_le_bytes(chunk[40..44].try_into().unwrap()),prefix);
                assert_eq!(&chunk[44..48],&[0;4]);prefix+=(jobs[i].m+3)&!3;
            }
            for row in 0..grid*4 {
                let start=|i:usize|u32::from_le_bytes(bytes[i*48+40..i*48+44].try_into().unwrap());
                let(mut low,mut high)=(0,count);
                while low+1<high{let mid=(low+high)/2;if start(mid)<=row{low=mid;}else{high=mid;}}
                assert_eq!(row/4*4>=start(low),true);
                if row-start(low)<jobs[low].m {seen+=1;}
            }
            assert_eq!(seen,jobs.iter().map(|j|j.m).sum::<u32>());
        }
    }
    #[test]fn reject_aliases_overflows_and_descriptor_collision(){
        assert!(pack(&[]).is_none());assert!(pack(&[job(1,0);33]).is_none());
        assert!(pack(&[job(640,0),job(640,4)]).is_none());
        assert!(pack(&[job(0,0)]).is_none());assert!(pack(&[job(u32::MAX,0)]).is_none());
        assert!(pack(&[job(1,0x100000)]).is_none());
        let mut j=job(1,0);j.weight=u64::MAX;assert!(pack(&[j]).is_none());
        let mut j=job(1,0);j.input=u64::MAX;assert!(pack(&[j]).is_none());
    }
}
