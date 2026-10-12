//! SSE2 leaf math across a pointer/integer-only Win64 boundary. Rust's UEFI
//! soft-float ABI remains unchanged. No CR0/CR4 writes or global feature flags.
use core::sync::atomic::{AtomicU8,AtomicU64,Ordering};
extern crate alloc;
static READY:AtomicU8=AtomicU8::new(0);
static CALLS:AtomicU64=AtomicU64::new(0);
static FALLBACK:AtomicU64=AtomicU64::new(0);
static DOT_CALLS:AtomicU64=AtomicU64::new(0);
static DOT_FALLBACK:AtomicU64=AtomicU64::new(0);
pub fn enabled()->bool{
    let mut state=READY.load(Ordering::Relaxed);
    if state==0{
        #[cfg(target_os="uefi")]
        let ok=unsafe{
            let cr0:u64;let cr4:u64;
            core::arch::asm!("mov {}, cr0","mov {}, cr4",out(reg)cr0,out(reg)cr4,options(nomem,nostack,preserves_flags));
            let features=core::arch::x86_64::__cpuid(1).edx;
            cr0&0x0c==0&&cr4&0x600==0x600&&features&(1<<26)!=0
        };
        #[cfg(not(target_os="uefi"))]
        let ok=true; // x86_64 native-test OS already owns/enables the FP context.
        state=if ok{2}else{1};READY.store(state,Ordering::Relaxed);
    }state==2
}
pub fn counts()->(bool,u64,u64){(enabled(),CALLS.load(Ordering::Relaxed),FALLBACK.load(Ordering::Relaxed))}
pub fn dot_counts()->(u64,u64){(DOT_CALLS.load(Ordering::Relaxed),DOT_FALLBACK.load(Ordering::Relaxed))}
pub fn reset_counts(){CALLS.store(0,Ordering::Relaxed);FALLBACK.store(0,Ordering::Relaxed);DOT_CALLS.store(0,Ordering::Relaxed);DOT_FALLBACK.store(0,Ordering::Relaxed);}
unsafe extern "win64"{fn frontier_dot_sse2(x:*const f32,y:*const f32,n:usize,stride:usize)->u32;}
pub fn dot(x:&[f32],y:&[f32],stride:usize)->Option<f32>{
    if x.is_empty(){return Some(0.);}
    if stride==0||stride>isize::MAX as usize/4||(x.len()-1).checked_mul(stride)?.checked_add(1)?>y.len(){return None;}
    if !enabled(){DOT_FALLBACK.fetch_add(1,Ordering::Relaxed);return None;}
    DOT_CALLS.fetch_add(1,Ordering::Relaxed);
    Some(f32::from_bits(unsafe{frontier_dot_sse2(x.as_ptr(),y.as_ptr(),x.len(),stride)}))
}
pub fn dot_selftest()->(bool,usize){
    use alloc::vec;
    if !enabled(){return(false,0);}
    let mut cases=0;
    for n in [1,3,40,128,256]{for stride in [1,3,512]{for seed in 0..2{
        let mut x=vec![0.;n];let mut y=vec![0.;(n-1)*stride+1];let mut expected=0f32;
        for i in 0..n{x[i]=((i*13+seed*7)%53)as f32/16.-1.;y[i*stride]=((i*7+seed)%37)as f32/32.-0.5;
            expected+=x[i]*y[i*stride];}
        let Some(actual)=dot(&x,&y,stride)else{return(false,cases);};
        if actual.to_bits()!=expected.to_bits(){return(false,cases);}cases+=1;
    }}}
    (true,cases)
}
/// Runs before the GPU operator gate; the scalar oracle is still soft-float on UEFI.
pub fn selftest()->(bool,usize){
    use alloc::vec;
    if !enabled(){return(false,0);}
    let mut cases=0;
    for group in [1,3,128,256,2560]{for seed in 0..4usize{
        let n=group*4;let mut x=vec![0f32;n];let mut gamma=vec![0f32;n];let mut out=vec![0f32;n];
        for i in 0..n{x[i]=((i*17+seed*11)%251)as f32/128.-1.;gamma[i]=(i%17)as f32/8.-1.;}
        let mut expected=vec![0f32;n];
        for base in (0..n).step_by(group){let mut sum=0f64;
            for &v in &x[base..base+group]{sum+=(v*v)as f64;}
            let scale=1./libm::sqrtf((sum/group as f64)as f32+1e-6);
            for i in base..base+group{expected[i]=(x[i]*scale)*gamma[i];}}
        if norm(&x,&gamma,&mut out,group)!=Some(true)||out.iter().zip(&expected).any(|(a,b)|a.to_bits()!=b.to_bits()){return(false,cases);}
        cases+=1;
    }}
    (true,cases)
}
unsafe extern "win64"{fn frontier_rmsnorm_sse2(x:*const f32,gamma:*const f32,out:*mut f32,n:usize,group:usize)->u32;}
pub fn norm(x:&[f32],gamma:&[f32],out:&mut[f32],group:usize)->Option<bool>{
    if group==0||group>i32::MAX as usize||x.is_empty()||x.len()%group!=0||gamma.len()!=x.len()||out.len()!=x.len(){return None;}
    if !enabled(){FALLBACK.fetch_add(1,Ordering::Relaxed);return None;}
    CALLS.fetch_add(1,Ordering::Relaxed);
    Some(unsafe{frontier_rmsnorm_sse2(x.as_ptr(),gamma.as_ptr(),out.as_mut_ptr(),x.len(),group)!=0})
}
core::arch::global_asm!(r#"
.text
.p2align 4
.globl frontier_rmsnorm_sse2
frontier_rmsnorm_sse2:
    mov r10, qword ptr [rsp + 40]
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    mov rdi, rcx
    mov rsi, rdx
    mov rbx, r8
    mov r12, r9
    mov r13, r10
    sub rsp, 8
    stmxcsr dword ptr [rsp]
    mov dword ptr [rsp + 4], 0x1f80
    ldmxcsr dword ptr [rsp + 4]
    cvtsi2sd xmm2, r13
    xor r14d, r14d
.Lfr_norm_group:
    pxor xmm0, xmm0
    xor r15d, r15d
.Lfr_norm_sum:
    lea rax, [r14 + r15]
    movss xmm1, dword ptr [rdi + rax*4]
    mulss xmm1, xmm1
    cvtss2sd xmm1, xmm1
    addsd xmm0, xmm1
    inc r15
    cmp r15, r13
    jb .Lfr_norm_sum
    divsd xmm0, xmm2
    cvtsd2ss xmm0, xmm0
    mov eax, 0x358637bd
    movd xmm1, eax
    addss xmm0, xmm1
    sqrtss xmm0, xmm0
    mov eax, 0x3f800000
    movd xmm1, eax
    divss xmm1, xmm0
    xor r15d, r15d
.Lfr_norm_out:
    lea rax, [r14 + r15]
    movss xmm0, dword ptr [rdi + rax*4]
    mulss xmm0, xmm1
    mulss xmm0, dword ptr [rsi + rax*4]
    movss dword ptr [rbx + rax*4], xmm0
    movd eax, xmm0
    and eax, 0x7f800000
    cmp eax, 0x7f800000
    je .Lfr_norm_bad
    inc r15
    cmp r15, r13
    jb .Lfr_norm_out
    add r14, r13
    cmp r14, r12
    jb .Lfr_norm_group
    mov eax, 1
    jmp .Lfr_norm_return
.Lfr_norm_bad:
    xor eax, eax
.Lfr_norm_return:
    ldmxcsr dword ptr [rsp]
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
.p2align 4
.globl frontier_dot_sse2
frontier_dot_sse2:
    sub rsp, 8
    stmxcsr dword ptr [rsp]
    mov dword ptr [rsp + 4], 0x1f80
    ldmxcsr dword ptr [rsp + 4]
    shl r9, 2
    xor r10d, r10d
    pxor xmm0, xmm0
.Lfr_dot_loop:
    movss xmm1, dword ptr [rcx + r10*4]
    mulss xmm1, dword ptr [rdx]
    addss xmm0, xmm1
    add rdx, r9
    inc r10
    cmp r10, r8
    jb .Lfr_dot_loop
    movd eax, xmm0
    ldmxcsr dword ptr [rsp]
    add rsp, 8
    ret
"#);
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn exact_strided_dot_and_bounds(){
        assert_eq!(dot_selftest(),(true,30));
        assert_eq!(dot(&[1.,2.],&[1.],1),None);
        assert_eq!(dot(&[1.,2.],&[1.,2.],usize::MAX),None);
        assert_eq!(dot(&[],&[],1),Some(0.));
        let x=[f32::from_bits(1),-f32::from_bits(1),-0.,0.];
        let mut reference=0f32;for v in x{reference+=v*1.;}
        assert_eq!(dot(&x,&[1.;4],1).unwrap().to_bits(),reference.to_bits());
    }
    #[test]fn exact_reference_order_across_groups_and_extremes(){
        assert_eq!(1e-6f32.to_bits(),0x358637bd);
        for group in [1,3,128,256,2560]{for seed in 0..12u32{
            let mut random=seed+1;
            let x:Vec<f32>=(0..group*4).map(|_|{random=random.wrapping_mul(1664525).wrapping_add(1013904223);
                f32::from_bits((random&0x807fffff)|(((random>>24)%240+1)<<23))}).collect();
            let gamma:Vec<f32>=(0..x.len()).map(|i|(i%17)as f32/8.-1.).collect();let mut out=vec![0.;x.len()];
            let mut expected=vec![0.;x.len()];
            for base in (0..x.len()).step_by(group){let mut sum=0f64;
                for &v in &x[base..base+group]{sum+=(v*v)as f64;}
                let scale=1./libm::sqrtf((sum/group as f64)as f32+1e-6);
                for i in base..base+group{expected[i]=(x[i]*scale)*gamma[i];}}
            assert_eq!(norm(&x,&gamma,&mut out,group),Some(expected.iter().all(|v|v.is_finite())));
            for(i,(a,b))in out.iter().zip(&expected).enumerate(){assert_eq!(a.to_bits(),b.to_bits(),"group={group} seed={seed} i={i}");}
        }}
    }
    #[test]fn signed_zeros_subnormal_and_nonfinite(){
        for x in [[0.,-0.,f32::from_bits(1),-f32::from_bits(1)],[f32::INFINITY,1.,2.,3.],[f32::NAN,1.,2.,3.]]{
            let mut out=[0.;4];let got=norm(&x,&[1.;4],&mut out,4);
            assert_eq!(got,Some(x.iter().all(|v|v.is_finite())));
        }
    }
}
