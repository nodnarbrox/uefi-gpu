//! Exact common-range libm-0.2.16 expf polynomial and activation arithmetic.
//! Derived from FreeBSD e_expf.c (Ian Lance Taylor's float conversion).
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this software is freely
//! granted, provided that this notice is preserved.
use core::sync::atomic::{AtomicU64,Ordering};
static FAST:AtomicU64=AtomicU64::new(0);static FALLBACK:AtomicU64=AtomicU64::new(0);
static ACTIVATIONS:AtomicU64=AtomicU64::new(0);
unsafe extern "win64"{fn frontier_expf_sse2(bits:u32)->u32;}
unsafe extern "win64"{fn frontier_activation_sse2(bits:u32,silu:u32)->u32;}
pub fn activation(x:f32,silu:bool)->f32{
    if x.to_bits()&0x7fffffff<0x42a00000&&super::frontier_cpu_math::enabled(){
        ACTIVATIONS.fetch_add(1,Ordering::Relaxed);
        f32::from_bits(unsafe{frontier_activation_sse2(x.to_bits(),silu as u32)})
    }else{
        let value=1.0/(1.0+exp(-x));if silu{x*value}else{value}
    }
}
pub fn activation_count()->u64{ACTIVATIONS.load(Ordering::Relaxed)}
pub fn activation_selftest()->(bool,usize){
    if !super::frontier_cpu_math::enabled(){return(false,0);}
    let mut count=0;
    for i in 0..4096usize{
        let x=(((i*73)%170001)as i32-85000)as f32/1000.;
        for silu in [false,true]{
            let y=1.0/(1.0+libm::expf(-x));let expected=if silu{x*y}else{y};
            if activation(x,silu).to_bits()!=expected.to_bits(){return(false,count);}
            count+=1;
        }
    }(true,count)
}
pub fn exp(x:f32)->f32{
    if x.to_bits()&0x7fffffff<0x42a00000&&super::frontier_cpu_math::enabled(){
        FAST.fetch_add(1,Ordering::Relaxed);f32::from_bits(unsafe{frontier_expf_sse2(x.to_bits())})
    }else{FALLBACK.fetch_add(1,Ordering::Relaxed);libm::expf(x)}
}
pub fn counts()->(u64,u64){(FAST.load(Ordering::Relaxed),FALLBACK.load(Ordering::Relaxed))}
pub fn reset(){FAST.store(0,Ordering::Relaxed);FALLBACK.store(0,Ordering::Relaxed);ACTIVATIONS.store(0,Ordering::Relaxed);}
pub fn selftest()->(bool,usize){
    if !super::frontier_cpu_math::enabled(){return(false,0);}
    let specials=[-120.,-104.,-90.,-80.,80.,89.,100.,f32::INFINITY,f32::NEG_INFINITY,f32::NAN];
    for i in 0..4096usize{
        let x=if i%64==0{specials[(i/64)%specials.len()]}else{(((i*73)%160001)as i32-80000)as f32/1000.};
        if exp(x).to_bits()!=libm::expf(x).to_bits(){return(false,i);}
    }
    (true,4096)
}
core::arch::global_asm!(r#"
.text
.p2align 4
.globl frontier_activation_sse2
frontier_activation_sse2:
    // Win64: align stack and reserve32-byte outgoing shadow space. Only volatile
    // GPR/XMM registers; original x/mode and MXCSR survive the nested exp call.
    sub rsp, 56
    mov dword ptr [rsp + 32], ecx
    mov dword ptr [rsp + 36], edx
    stmxcsr dword ptr [rsp + 40]
    mov dword ptr [rsp + 44], 0x1f80
    ldmxcsr dword ptr [rsp + 44]
    xor ecx, 0x80000000
    call frontier_expf_sse2
    movd xmm0, eax
    mov eax, 0x3f800000
    movd xmm1, eax
    addss xmm0, xmm1
    divss xmm1, xmm0
    cmp dword ptr [rsp + 36], 0
    je .Lfr_activation_done
    movss xmm0, dword ptr [rsp + 32]
    mulss xmm1, xmm0
.Lfr_activation_done:
    movd eax, xmm1
    ldmxcsr dword ptr [rsp + 40]
    add rsp, 56
    ret
.p2align 4
.globl frontier_expf_sse2
frontier_expf_sse2:
    sub rsp, 8
    stmxcsr dword ptr [rsp]
    mov dword ptr [rsp + 4], 0x1f80
    ldmxcsr dword ptr [rsp + 4]
    movd xmm0, ecx
    mov r9d, ecx
    shr r9d, 31
    mov eax, ecx
    and eax, 0x7fffffff
    cmp eax, 0x39000000
    jbe .Lfr_exp_tiny
    cmp eax, 0x3eb17218
    jbe .Lfr_exp_no_reduce
    cmp eax, 0x3f851592
    ja .Lfr_exp_multiple
    mov r8d, 1
    sub r8d, r9d
    sub r8d, r9d
    jmp .Lfr_exp_reduce
.Lfr_exp_multiple:
    movaps xmm4, xmm0
    mov r10d, 0x3fb8aa3b
    movd xmm5, r10d
    mulss xmm4, xmm5
    shl r9d, 31
    or r9d, 0x3f000000
    movd xmm5, r9d
    addss xmm4, xmm5
    cvttss2si r8d, xmm4
.Lfr_exp_reduce:
    cvtsi2ss xmm3, r8d
    movaps xmm4, xmm3
    mov r10d, 0x3f317200
    movd xmm5, r10d
    mulss xmm4, xmm5
    movaps xmm1, xmm0
    subss xmm1, xmm4
    movaps xmm2, xmm3
    mov r10d, 0x35bfbe8e
    movd xmm5, r10d
    mulss xmm2, xmm5
    movaps xmm0, xmm1
    subss xmm0, xmm2
    jmp .Lfr_exp_primary
.Lfr_exp_no_reduce:
    xor r8d, r8d
    movaps xmm1, xmm0
    pxor xmm2, xmm2
.Lfr_exp_primary:
    movaps xmm3, xmm0
    mulss xmm3, xmm3
    movaps xmm4, xmm3
    mov r10d, 0xbb355215
    movd xmm5, r10d
    mulss xmm4, xmm5
    mov r10d, 0x3e2aaa8f
    movd xmm5, r10d
    addss xmm4, xmm5
    mulss xmm4, xmm3
    movaps xmm5, xmm0
    subss xmm5, xmm4
    mulss xmm0, xmm5
    mov r10d, 0x40000000
    movd xmm4, r10d
    subss xmm4, xmm5
    divss xmm0, xmm4
    subss xmm0, xmm2
    addss xmm0, xmm1
    mov r10d, 0x3f800000
    movd xmm4, r10d
    addss xmm0, xmm4
    test r8d, r8d
    jz .Lfr_exp_return
    lea eax, [r8 + 127]
    shl eax, 23
    movd xmm1, eax
    mulss xmm0, xmm1
    jmp .Lfr_exp_return
.Lfr_exp_tiny:
    mov eax, 0x3f800000
    movd xmm1, eax
    addss xmm0, xmm1
.Lfr_exp_return:
    movd eax, xmm0
    ldmxcsr dword ptr [rsp]
    add rsp, 8
    ret
"#);
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn exact_activation_arithmetic(){
        assert_eq!(activation_selftest(),(true,8192));
        for bits in [0u32,0x80000000,1,0x80000001,0x42a00000,0xc2a00000,0x7f800000,0xff800000,0x7fc00000]{
            let x=f32::from_bits(bits);
            for silu in [false,true]{let y=1.0/(1.0+libm::expf(-x));let expected=if silu{x*y}else{y};
                let actual=activation(x,silu);
                assert!(actual.to_bits()==expected.to_bits()||(actual.is_nan()&&expected.is_nan()));}
        }
    }
    #[test]fn same_polynomial_bits_dense_range_and_raw_float_inputs(){
        assert_eq!((1.6666625440e-1f32).to_bits(),0x3e2aaa8f);
        assert_eq!((-2.7667332906e-3f32).to_bits(),0xbb355215);
        let mut random=17u32;
        for i in 0..200000{
            random=random.wrapping_mul(1664525).wrapping_add(1013904223);
            let x=if i%2==0{((random%160000)as i32-80000)as f32/1000.}
                else{f32::from_bits((random&0x807fffff)|(((random>>24)%133)<<23))};
            assert_eq!(exp(x).to_bits(),libm::expf(x).to_bits(),"x={x} bits={:x}",x.to_bits());
        }
    }
    #[test]fn reduction_boundaries_and_fallback(){
        for base in [0u32,0x39000000,0x3eb17218,0x3f851592,0x42a00000,0x42b17218,0x42cff1b5,0x7f800000]{
            for d in 0..5{for sign in [0,0x80000000]{
                let x=f32::from_bits(base.saturating_sub(2).saturating_add(d)|sign);
                let a=exp(x);let b=libm::expf(x);
                assert!(a.to_bits()==b.to_bits()||(a.is_nan()&&b.is_nan()),"bits={:x}",x.to_bits());
            }}
        }
    }
}
