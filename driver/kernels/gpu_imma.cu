// Ampere INT8 tensor-core matvec. A 16x8x32 MMA repeats x in eight columns.
// Each Q8_0 block has its own scale, so integer MMA accumulation is reset per
// 32-value block before FP32 scaled accumulation. Four warps split K per CTA.
__device__ __forceinline__ float scale16(const unsigned char* p){
    unsigned short h=*(const unsigned short*)p;float f;
    asm("cvt.f32.f16 %0, %1;":"=f"(f):"h"(h));return f;
}
__device__ __forceinline__ int load4(const unsigned char* p){
    const unsigned short* h=(const unsigned short*)p;
    return (unsigned)h[0]|((unsigned)h[1]<<16);
}
struct Pair {float a,b;};
__device__ __forceinline__ Pair dot16(const unsigned char* w,const unsigned char* x,int blocks){
    int lane=threadIdx.x&31,g=lane>>2,t=lane&3,warp=threadIdx.x>>5;
    const unsigned char* r0=w+(unsigned long long)g*blocks*34;
    const unsigned char* r8=r0+(unsigned long long)8*blocks*34;
    Pair out={0,0};
    for(int b=warp;b<blocks;b+=4){
        const unsigned char* p0=r0+b*34;const unsigned char* p8=r8+b*34;
        const unsigned char* xp=x+b*36;
        int a0=load4(p0+2+t*4),a1=load4(p8+2+t*4);
        int a2=load4(p0+18+t*4),a3=load4(p8+18+t*4);
        int b0=*(const int*)(xp+4+t*4),b1=*(const int*)(xp+20+t*4);
        int c0,c1,c2,c3;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {0,0,0,0};"
            :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
            :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        if(t==0){float dx=*(const float*)xp;
            out.a=fmaf((float)c0,scale16(p0)*dx,out.a);
            out.b=fmaf((float)c2,scale16(p8)*dx,out.b);}
    }
    return out;
}
extern "C" __global__ void linear_imma(const unsigned char* w,const unsigned char* x,float* y,int cols,int rows){
    __shared__ float partial[64];int tid=threadIdx.x,row=blockIdx.x*16;
    if(row>=rows)return;int blocks=cols/32;
    Pair p=dot16(w+(unsigned long long)row*blocks*34,x,blocks);
    if((tid&3)==0){int i=(tid>>5)*16+((tid&31)>>2);partial[i]=p.a;partial[i+8]=p.b;}
    __syncthreads();
    if(tid<16)y[row+tid]=((partial[tid]+partial[16+tid])+partial[32+tid])+partial[48+tid];
}
extern "C" __global__ void qkv_imma(const unsigned char* wq,const unsigned char* wk,const unsigned char* wv,
    const unsigned char* x,float* q,float* k,float* v){
    __shared__ float partial[64];int tid=threadIdx.x,row=blockIdx.x*16;
    const unsigned char* w=wq;float* y=q;
    if(row>=1792){row-=1792;w=wv;y=v;}else if(row>=1536){row-=1536;w=wk;y=k;}
    Pair p=dot16(w+row*48*34,x,48);
    if((tid&3)==0){int i=(tid>>5)*16+((tid&31)>>2);partial[i]=p.a;partial[i+8]=p.b;}
    __syncthreads();
    if(tid<16)y[row+tid]=((partial[tid]+partial[16+tid])+partial[32+tid])+partial[48+tid];
}
extern "C" __global__ void gateup_imma(const unsigned char* wg,const unsigned char* wu,const unsigned char* x,float* gate){
    __shared__ float pg[64],pu[64];int tid=threadIdx.x,row=blockIdx.x*16;
    Pair g=dot16(wg+row*48*34,x,48);Pair u=dot16(wu+row*48*34,x,48);
    if((tid&3)==0){int i=(tid>>5)*16+((tid&31)>>2);pg[i]=g.a;pg[i+8]=g.b;pu[i]=u.a;pu[i+8]=u.b;}
    __syncthreads();
    if(tid<16){float a=((pg[tid]+pg[16+tid])+pg[32+tid])+pg[48+tid];
        float b=((pu[tid]+pu[16+tid])+pu[32+tid])+pu[48+tid];gate[row+tid]=(a/(1.0f+expf(-a)))*b;}
}
