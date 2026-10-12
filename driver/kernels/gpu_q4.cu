// Runtime Q8 -> standard Q4_0 linear weights, a separately labeled lossy variant.
// Embedding remains Q8; activations remain block-32 INT8; KV remains FP32.
__device__ __forceinline__ float d16(const unsigned char* p){unsigned short h=*(const unsigned short*)p;float d;
    asm("cvt.f32.f16 %0, %1;":"=f"(d):"h"(h));return d;}
__device__ __forceinline__ int unpack4(const unsigned char* p,int group){
    const unsigned short* h=(const unsigned short*)(p+2+(group&3)*4);
    unsigned v=(unsigned)h[0]|((unsigned)h[1]<<16);if(group>=4)v>>=4;v&=0x0f0f0f0f;
    return ((v|0x80808080u)-0x08080808u)^0x80808080u;
}
__device__ __forceinline__ int dot4(int a,int b,int c){int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;":"=r"(d):"r"(a),"r"(b),"r"(c));return d;}
__device__ __forceinline__ float sumwarp(float s){
    s+=__shfl_xor_sync(0xffffffff,s,16);s+=__shfl_xor_sync(0xffffffff,s,8);s+=__shfl_xor_sync(0xffffffff,s,4);
    s+=__shfl_xor_sync(0xffffffff,s,2);s+=__shfl_xor_sync(0xffffffff,s,1);return s;
}
extern "C" __global__ void repack_q4(const unsigned char* src,unsigned char* dst,int blocks){
    int b=blockIdx.x*128+threadIdx.x;if(b>=blocks)return;
    src+=(unsigned long long)b*34;dst+=(unsigned long long)b*18;float dw=d16(src),amax=0,maxval=0;
    for(int j=0;j<32;j++){float v=dw*(float)((const signed char*)src)[2+j];if(amax<fabsf(v)){amax=fabsf(v);maxval=v;}}
    float d=maxval/-8.0f,inv=d!=0?1.0f/d:0.0f;unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;":"=h"(h):"f"(d));*(unsigned short*)dst=h;
    for(int j=0;j<16;j++){int a=(int)((dw*(float)((const signed char*)src)[2+j])*inv+8.5f);
        int c=(int)((dw*(float)((const signed char*)src)[18+j])*inv+8.5f);dst[2+j]=min(15,a)|(min(15,c)<<4);}
}
__device__ __forceinline__ float rowdot(const unsigned char* w,const unsigned char* x,int blocks,int lane){
    float sum=0;int seg=lane&3;
    for(int b=lane>>2;b<blocks;b+=8){const unsigned char* p=w+b*18;const unsigned char* q=x+b*36;
        int s=dot4(unpack4(p,seg*2),*(const int*)(q+4+seg*8),0);
        s=dot4(unpack4(p,seg*2+1),*(const int*)(q+8+seg*8),s);sum=fmaf((float)s,d16(p)*(*(const float*)q),sum);}
    return sumwarp(sum);
}
extern "C" __global__ void linear_q4(const unsigned char* w,const unsigned char* x,float* y,int cols,int rows){
    __shared__ float parts[4];int tid=threadIdx.x,row=blockIdx.x;if(row>=rows)return;
    int blocks=cols/32;w+=(unsigned long long)row*blocks*18;float sum=0;int seg=tid&1;
    for(int b=tid>>1;b<blocks;b+=64){const unsigned char* p=w+b*18;const unsigned char* q=x+b*36;int s=0;
        #pragma unroll
        for(int k=0;k<4;k++)s=dot4(unpack4(p,seg*4+k),*(const int*)(q+4+seg*16+k*4),s);
        sum=fmaf((float)s,d16(p)*(*(const float*)q),sum);}
    sum=sumwarp(sum);if((tid&31)==0)parts[tid>>5]=sum;__syncthreads();
    if(tid==0)y[row]=((parts[0]+parts[1])+parts[2])+parts[3];
}
extern "C" __global__ void qkv_q4(const unsigned char* wq,const unsigned char* wk,const unsigned char* wv,
    const unsigned char* x,float* q,float* k,float* v){
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);if(row>=2048)return;
    const unsigned char* w=wq;float* y=q;if(row>=1792){row-=1792;w=wv;y=v;}else if(row>=1536){row-=1536;w=wk;y=k;}
    float sum=rowdot(w+row*48*18,x,48,lane);if(lane==0)y[row]=sum;
}
extern "C" __global__ void gateup_q4(const unsigned char* wg,const unsigned char* wu,const unsigned char* x,float* gate){
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);if(row>=8960)return;
    float g=rowdot(wg+row*48*18,x,48,lane),u=rowdot(wu+row*48*18,x,48,lane);
    if(lane==0)gate[row]=(g/(1.0f+expf(-g)))*u;
}
extern "C" __global__ void prefill_gemm_q4(const unsigned char* w,const unsigned char* x,float* y,int cols,int rows,int batch){
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,t=lane&3;
    int tiles=rows/16,row=(blockIdx.x%tiles)*16,token=(blockIdx.x/tiles)*32+warp*8,blocks=cols/32;
    if(token>=batch)return;
    const unsigned char* w0=w+(unsigned long long)(row+g)*blocks*18;const unsigned char* w8=w0+(unsigned long long)8*blocks*18;
    float f0=0,f1=0,f2=0,f3=0;
    for(int b=0;b<blocks;b++){const unsigned char* p0=w0+b*18;const unsigned char* p8=w8+b*18;
        const unsigned char* xb=x+((unsigned long long)(token+g)*blocks+b)*36;
        int a0=unpack4(p0,t),a1=unpack4(p8,t),a2=unpack4(p0,t+4),a3=unpack4(p8,t+4);
        int b0=*(const int*)(xb+4+t*4),b1=*(const int*)(xb+20+t*4),c0,c1,c2,c3;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {0,0,0,0};"
            :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        float d0=d16(p0),d8=d16(p8);float x0=*(const float*)(x+((unsigned long long)(token+2*t)*blocks+b)*36);
        float x1=*(const float*)(x+((unsigned long long)(token+2*t+1)*blocks+b)*36);
        f0=fmaf((float)c0,d0*x0,f0);f1=fmaf((float)c1,d0*x1,f1);f2=fmaf((float)c2,d8*x0,f2);f3=fmaf((float)c3,d8*x1,f3);}
    y[(unsigned long long)(token+2*t)*rows+row+g]=f0;
    y[(unsigned long long)(token+2*t+1)*rows+row+g]=f1;
    y[(unsigned long long)(token+2*t)*rows+row+g+8]=f2;
    y[(unsigned long long)(token+2*t+1)*rows+row+g+8]=f3;
}
