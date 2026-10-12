//! Exact-order expert/residual accumulation, separately hardware-gated.
//! Elementwise out[i] += scale * x[i], separate FP32 multiply then add.
//! No reduction reordering; pointer/integer-only Win64 assembly boundary.
unsafe extern "win64" {
    fn frontier_axpy_sse2(out: *mut f32, x: *const f32, n: usize, scale_bits: u32);
}
extern crate alloc;
use core::sync::atomic::{AtomicU64,Ordering};
static CALLS:AtomicU64=AtomicU64::new(0);
static FALLBACK:AtomicU64=AtomicU64::new(0);
pub fn counts()->(u64,u64){(CALLS.load(Ordering::Relaxed),FALLBACK.load(Ordering::Relaxed))}
pub fn reset(){CALLS.store(0,Ordering::Relaxed);FALLBACK.store(0,Ordering::Relaxed);}
pub fn axpy(out: &mut [f32], x: &[f32], scale: f32) -> Option<()> {
    if out.len() != x.len() { return None; }
    if out.is_empty() { return Some(()); }
    #[cfg(target_os="uefi")]
    if !super::frontier_cpu_math::enabled() { FALLBACK.fetch_add(1,Ordering::Relaxed);return None; }
    CALLS.fetch_add(1,Ordering::Relaxed);
    unsafe { frontier_axpy_sse2(out.as_mut_ptr(), x.as_ptr(), out.len(), scale.to_bits()); }
    Some(())
}
/// Scalar oracle remains soft-float on UEFI. Called before either physical path.
pub fn selftest()->(bool,usize){
    use alloc::{vec,vec::Vec};
    let mut cases=0;
    for n in [0,1,3,4,5,127,640,2560]{for scale in [-1.0,-0.0,0.0,0.125,1.0]{
        let x:Vec<f32>=(0..n).map(|i|(i%97)as f32/128.-0.25).collect();
        let mut out=vec![f32::from_bits(0x12345678);n+2];
        for v in &mut out[1..n+1]{*v=0.25;}
        let expected:Vec<f32>=x.iter().map(|v|0.25+scale*v).collect();
        if axpy(&mut out[1..n+1],&x,scale).is_none()||out[0].to_bits()!=0x12345678
            ||out[n+1].to_bits()!=0x12345678
            ||out[1..n+1].iter().zip(expected).any(|(a,b)|a.to_bits()!=b.to_bits()){return(false,cases);}
        cases+=1;
    }}
    let x=[f32::from_bits(1),-f32::from_bits(1),0.,-0.,1.,-1.,f32::MIN_POSITIVE];
    let mut out=[0.,-0.,-0.,-0.,-1.,1.,0.];let mut expected=out;
    for(i,v)in expected.iter_mut().enumerate(){*v+=0.5*x[i];}
    if axpy(&mut out,&x,0.5).is_none()||out.iter().zip(expected).any(|(a,b)|a.to_bits()!=b.to_bits()){return(false,cases);}
    (true,cases+1)
}
core::arch::global_asm!(r#"
.text
.p2align 4
.globl frontier_axpy_sse2
frontier_axpy_sse2:
    sub rsp, 8
    stmxcsr dword ptr [rsp]
    mov dword ptr [rsp + 4], 0x1f80
    ldmxcsr dword ptr [rsp + 4]
    movd xmm0, r9d
    shufps xmm0, xmm0, 0
    cmp r8, 4
    jb .Lfr_axpy_tail
.Lfr_axpy_four:
    movups xmm1, xmmword ptr [rdx]
    mulps xmm1, xmm0
    movups xmm2, xmmword ptr [rcx]
    addps xmm2, xmm1
    movups xmmword ptr [rcx], xmm2
    add rcx, 16
    add rdx, 16
    sub r8, 4
    cmp r8, 4
    jae .Lfr_axpy_four
.Lfr_axpy_tail:
    test r8, r8
    jz .Lfr_axpy_return
.Lfr_axpy_one:
    movss xmm1, dword ptr [rdx]
    mulss xmm1, xmm0
    movss xmm2, dword ptr [rcx]
    addss xmm2, xmm1
    movss dword ptr [rcx], xmm2
    add rcx, 4
    add rdx, 4
    dec r8
    jnz .Lfr_axpy_one
.Lfr_axpy_return:
    ldmxcsr dword ptr [rsp]
    add rsp, 8
    ret
"#);

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn physical_oracle_cases(){assert_eq!(selftest(),(true,41));}
    #[test] fn tails_and_guard_words() {
        for n in [0,1,2,3,4,5,7,8,9,127,128,640,2560,10240] {
            let x:Vec<f32>=(0..n).map(|i|(i%97) as f32/128.-0.25).collect();
            for scale in [-1.0,-0.0,0.0,0.125,1.0] {
                let mut guarded=vec![f32::from_bits(0x12345678);n+2];
                for v in &mut guarded[1..n+1] {*v=0.25;}
                let expected:Vec<_>=x.iter().map(|v|0.25+scale*v).collect();
                assert_eq!(axpy(&mut guarded[1..n+1],&x,scale),Some(()));
                assert_eq!(guarded[0].to_bits(),0x12345678);
                assert_eq!(guarded[n+1].to_bits(),0x12345678);
                assert!(guarded[1..n+1].iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()));
            }
        }
        let mut out=[9.0];
        assert_eq!(axpy(&mut out,&[],1.),None);
        assert_eq!(out,[9.0]);
    }
    #[test] fn sequential_expert_accumulation_bits() {
        let mut random=123u32;
        let mut expected=vec![0f32;2560];let mut actual=expected.clone();
        for _ in 0..11 {
            let x:Vec<f32>=(0..2560).map(|_|{
                random=random.wrapping_mul(1664525).wrapping_add(1013904223);
                f32::from_bits((random&0x807fffff)|(((random>>24)%80+80)<<23))
            }).collect();
            let scale=(random%61)as f32/32.-0.75;
            for (v,&x) in expected.iter_mut().zip(&x) {*v+=scale*x;}
            assert_eq!(axpy(&mut actual,&x,scale),Some(()));
            assert!(actual.iter().zip(&expected).all(|(a,b)|a.to_bits()==b.to_bits()));
        }
    }
    #[test] fn subnormals_signed_zero_and_mxcsr_restore() {
        let x=[f32::from_bits(1),-f32::from_bits(1),0.,-0.,1.,-1.,f32::MIN_POSITIVE];
        let mut output=[0.,-0.,-0.,-0.,-1.,1.,0.];let mut expected=output;
        for (o,x) in expected.iter_mut().zip(x) {*o+=0.5*x;}
        unsafe {
            let mut original=0u32;
            core::arch::asm!("stmxcsr [{}]",in(reg)&mut original,options(nostack,preserves_flags));
            // Non-default rounding and FTZ/DAZ must not leak into the helper or be lost.
            let altered=original|0xe040;
            core::arch::asm!("ldmxcsr [{}]",in(reg)&altered,options(nostack,preserves_flags));
            let result=axpy(&mut output,&x,0.5);
            let mut after=0u32;
            core::arch::asm!("stmxcsr [{}]",in(reg)&mut after,options(nostack,preserves_flags));
            core::arch::asm!("ldmxcsr [{}]",in(reg)&original,options(nostack,preserves_flags));
            assert_eq!(result,Some(()));assert_eq!(after,altered);
        }
        assert!(output.iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()));
    }
}
