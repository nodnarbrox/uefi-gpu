//! Bounded, platform-independent primitives for the pinned qwen4exp GGUF.
//! Quant layouts follow ggml-quants.c; this module is also tested natively.
#[path = "frontier_tables.rs"] mod tables;

pub const PLE_EOS: u32 = 248044;
pub const GENERATION_EOS: u32 = 248046;
pub const HEAD_OFFSETS: [u64;16] = [0,20000003,40000026,60000059,80000106,100000165,
    120000228,140000297,160000374,180000455,200000548,220000655,240000802,
    260000955,280001114,300001275];
pub const HEAD_VOCABS: [u64;16] = [20000003,20000023,20000033,20000047,20000059,
    20000063,20000069,20000077,20000081,20000093,20000107,20000147,20000153,
    20000159,20000161,20000171];
const MULT: [u64;3] = [23703573157769,20109073645365,8052911324071];

/// Predecessors are newest first. Missing/EOS predecessors cut older history;
/// the current token, even when EOS, does not reset its own predecessors.
pub fn ple_rows(token: u32, previous: [Option<u32>;2]) -> [u32;16] {
    let mut ctx = [token, PLE_EOS, PLE_EOS];
    let mut cut = false;
    for i in 0..2 {
        cut |= previous[i].is_none() || previous[i] == Some(PLE_EOS);
        if !cut {ctx[i+1] = previous[i].unwrap();}
    }
    let mut out = [0;16];
    let mut mixed = (ctx[0] as u64).wrapping_mul(MULT[0]);
    for n in 1..3 {
        mixed ^= (ctx[n] as u64).wrapping_mul(MULT[n]);
        for g in 0..8 {
            let h = (n-1)*8+g;
            out[h] = (mixed % HEAD_VOCABS[h] + HEAD_OFFSETS[h]) as u32;
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error { Bounds, Alignment, QuantType, Shape }

/// Convert an exact byte range to a bounded aligned block read. Never permits
/// reading beyond the device, integer wraparound, or a zero/invalid block size.
pub fn block_range(offset: u64, bytes: usize, device_bytes: u64, block: usize)
    -> Result<(u64, usize, usize), Error> {
    if block == 0 || !block.is_power_of_two() || block > 4096 {return Err(Error::Alignment);}
    let end = offset.checked_add(bytes as u64).ok_or(Error::Bounds)?;
    if bytes == 0 || end > device_bytes {return Err(Error::Bounds);}
    let first = offset & !(block as u64-1);
    let last = end.checked_add(block as u64-1).ok_or(Error::Bounds)? & !(block as u64-1);
    if last > device_bytes {return Err(Error::Bounds);}
    let amount = usize::try_from(last-first).map_err(|_|Error::Bounds)?;
    Ok((first/block as u64, amount, (offset-first) as usize))
}

pub fn row_bytes(kind: u32, n: usize) -> Result<usize,Error> {
    let (block,bytes) = match kind {0=>(1,4),8=>(32,34),14=>(256,210),
        18=>(256,98),20=>(32,18),23=>(256,136),30=>(1,2),_=>return Err(Error::QuantType)};
    if n == 0 || n % block != 0 {return Err(Error::Shape);}
    (n/block).checked_mul(bytes).ok_or(Error::Bounds)
}

pub fn half(h:u16)->f32 {
    let sign=((h as u32)&0x8000)<<16;let exp=(h>>10)&31;let frac=(h&1023) as u32;
    if exp==0 {let v=frac as f32*(1.0/16777216.0);if sign!=0 {-v}else{v}}
    else if exp==31 {f32::from_bits(sign|0x7f800000|(frac<<13))}
    else {f32::from_bits(sign|(((exp as u32)+112)<<23)|(frac<<13))}
}
fn u16le(b:&[u8],p:usize)->u16 {u16::from_le_bytes([b[p],b[p+1]])}
fn u32le(b:&[u8],p:usize)->u32 {u32::from_le_bytes(b[p..p+4].try_into().unwrap())}
const IQ4: [i8;16] = [-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113];
pub fn iq3_grid()->&'static[u32;256] {&tables::IQ3_GRID}
pub fn iq4_values()->&'static[i8;16] {&IQ4}

/// Exact decoded values, preserving ggml's scale multiplication order.
pub fn dequantize(kind:u32, input:&[u8], out:&mut[f32])->Result<(),Error> {
    if input.len()!=row_bytes(kind,out.len())? {return Err(Error::Bounds);}
    for (i,value) in out.iter_mut().enumerate() {
        *value = match kind {
            0=>f32::from_bits(u32le(input,4*i)),
            30=>f32::from_bits((u16le(input,2*i) as u32)<<16),
            8=>{let p=(i/32)*34;half(u16le(input,p))*(input[p+2+i%32] as i8 as f32)},
            20=>{let p=(i/32)*18;let j=i%32;
                let q=(input[p+2+j%16]>>(4*(j/16)))&15;
                half(u16le(input,p))*IQ4[q as usize] as f32},
            23=>{let p=(i/256)*136;let j=i%256;let sb=j/32;
                let sc=((input[p+4+sb/2]>>(4*(sb%2)))&15) as i32 |
                    (((u16le(input,p+2)>>(2*sb))&3) as i32)<<4;
                let q=(input[p+8+sb*16+j%16]>>(4*((j%32)/16)))&15;
                (half(u16le(input,p))*(sc-32) as f32)*IQ4[q as usize] as f32},
            14=>{let p=(i/256)*210;let j=i%256;let sub=j/128;let l=j%32;let group=(j%128)/32;
                let low=input[p+sub*64+l+(group%2)*32];
                let high=input[p+128+sub*32+l];
                let q=(((low>>(4*(group/2)))&15)|(((high>>(2*group))&3)<<4)) as i32-32;
                let sc=input[p+192+sub*8+l/16+group*2] as i8;
                (half(u16le(input,p+208))*sc as f32)*q as f32},
            18=>{let p=(i/256)*98;let j=i%256;let sb=j/32;let group=(j%32)/8;
                let aux=u32le(input,p+66+sb*4);
                let signs7=(aux>>(7*group))&127;
                let signs=signs7|((signs7.count_ones()&1)<<7);
                let index=input[p+2+sb*8+group*2+(j%8)/4] as usize;
                let q=(tables::IQ3_GRID[index]>>(8*(j%4)))&255;
                let scale=half(u16le(input,p))*(0.5+(aux>>28) as f32)*0.5;
                scale*q as f32*if signs&(1<<(j%8))!=0 {-1.0}else{1.0}},
            _=>return Err(Error::QuantType),
        };
    }
    Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn bounded_reads() {
        assert_eq!(block_range(511,90,4096,512),Ok((0,1024,511)));
        assert_eq!(block_range(4006,90,4096,512),Ok((7,512,422)));
        for (o,n,s,b) in [(0,0,4096,512),(u64::MAX,2,u64::MAX,512),(4090,7,4096,512),
            (0,1,4096,0),(0,1,4096,1000),(0,1,4096,8192)] {
            assert!(block_range(o,n,s,b).is_err());
        }
    }
    #[test] fn eos_history() {
        assert_eq!(ple_rows(42,[None,None]),ple_rows(42,[Some(PLE_EOS),Some(123)]));
        assert_ne!(ple_rows(PLE_EOS,[Some(12),Some(34)]),ple_rows(PLE_EOS,[None,None]));
        assert_ne!(ple_rows(42,[Some(GENERATION_EOS),Some(123)]),ple_rows(42,[None,None]));
        for t in [0,42,PLE_EOS,GENERATION_EOS,u32::MAX] {
            for(h,row)in ple_rows(t,[Some(u32::MAX),None]).iter().enumerate() {
                assert!((*row as u64)>=HEAD_OFFSETS[h]);
                assert!((*row as u64)<HEAD_OFFSETS[h]+HEAD_VOCABS[h]);
            }
        }
    }
    #[test] fn rejects_bad_quant() {
        assert_eq!(row_bytes(23,640),Err(Error::Shape));
        assert_eq!(row_bytes(20,160),Ok(90));
        assert_eq!(row_bytes(18,2560),Ok(980));
        assert_eq!(row_bytes(99,256),Err(Error::QuantType));
    }
}
