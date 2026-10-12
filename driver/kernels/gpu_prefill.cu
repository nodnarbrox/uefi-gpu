// Batched Q8 prefill: true 16-output x 8-token INT8 tensor-core tiles.
// Four warps cover distinct token tiles, sharing weight cache lines across N.
__device__ __forceinline__ float s16(const unsigned char* p){
    unsigned short h=*(const unsigned short*)p;float f;
    asm("cvt.f32.f16 %0, %1;":"=f"(f):"h"(h));return f;
}
__device__ __forceinline__ int q4bytes(const unsigned char* p){
    const unsigned short* h=(const unsigned short*)p;return (unsigned)h[0]|((unsigned)h[1]<<16);
}
extern "C" __global__ void prefill_gemm(const unsigned char* weights,const unsigned char* x,
    float* y,int cols,int rows,int batch){
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,t=lane&3;
    int tiles=rows/16,row=(blockIdx.x%tiles)*16,token=(blockIdx.x/tiles)*32+warp*8,blocks=cols/32;
    if(token>=batch)return;
    const int* packed=(const int*)weights+(unsigned long long)(row/16)*blocks*144;
    const int* xp=(const int*)x+(unsigned long long)(token/8)*blocks*72;
    float f0=0,f1=0,f2=0,f3=0;
    for(int b=0;b<blocks;b++){
        const int* p=packed+b*144;
        const int* xb=xp+b*72;
        int a0=p[lane],a1=p[32+lane],a2=p[64+lane],a3=p[96+lane];
        int b0=xb[lane],b1=xb[32+lane],c0,c1,c2,c3;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {0,0,0,0};"
            :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
            :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        float d0=__int_as_float(p[128+g]),d8=__int_as_float(p[136+g]);
        float x0=__int_as_float(xb[64+2*t]);
        float x1=__int_as_float(xb[65+2*t]);
        f0=fmaf((float)c0,d0*x0,f0);f1=fmaf((float)c1,d0*x1,f1);
        f2=fmaf((float)c2,d8*x0,f2);f3=fmaf((float)c3,d8*x1,f3);
    }
    y[(unsigned long long)(token+2*t)*rows+row+g]=f0;
    y[(unsigned long long)(token+2*t+1)*rows+row+g]=f1;
    y[(unsigned long long)(token+2*t)*rows+row+g+8]=f2;
    y[(unsigned long long)(token+2*t+1)*rows+row+g+8]=f3;
}
// Lossless Q8 -> native MMA-fragment planes, one-time setup outside timing.
// 16 rows x K32: 512 quant bytes + 16 exactly converted FP32 scales = 576 bytes.
extern "C" __global__ void prefill_repack(const unsigned char* src,int* dst,int blocks){
    int i=threadIdx.x,plane=i>>5,lane=i&31,g=lane>>2,t=lane&3;
    int tile=blockIdx.x/blocks,b=blockIdx.x%blocks;
    const unsigned char* p=src+((unsigned long long)(tile*16+g+(plane&1)*8)*blocks+b)*34;
    dst[(unsigned long long)blockIdx.x*144+i]=q4bytes(p+2+(plane>>1)*16+t*4);
    if(i<16){p=src+((unsigned long long)(tile*16+i)*blocks+b)*34;
        dst[(unsigned long long)blockIdx.x*144+128+i]=__float_as_int(s16(p));}
}
extern "C" __global__ void prefill_quant(const float* x,unsigned char* out,int cols,int batch){
    int lane=threadIdx.x&31,b=blockIdx.x*4+(threadIdx.x>>5),blocks=cols/32;
    if(b>=blocks*batch)return;
    float v=x[b*32+lane],a=fabsf(v);
    a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,16));a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,8));
    a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,4));a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,2));
    a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,1));
    int qi=__float2int_rn(v*(a>0?127.0f/a:0.0f));qi=max(-127,min(127,qi));
    int token=b/blocks,k=b%blocks,r=token&7;
    unsigned char* p=out+((unsigned long long)(token/8)*blocks+k)*288;
    p[(lane/16)*128+r*16+(lane%16)]=(unsigned char)(signed char)qi;
    if(lane==0)((float*)(p+256))[r]=a/127.0f;
}
extern "C" __global__ void prefill_embedding(const unsigned char* weights,const unsigned* tokens,float* x){
    int i=blockIdx.x*128+threadIdx.x,pos=i/1536,col=i%1536;
    const unsigned char* b=weights+(unsigned long long)tokens[pos]*48*34+(col/32)*34;
    x[i]=s16(b)*(float)((const signed char*)b)[2+(col&31)];
}
extern "C" __global__ void prefill_resnorm(float* x,const float* residual,const float* w,float* y,int add){
    __shared__ float sums[128];int tid=threadIdx.x,base=blockIdx.x*1536;float sum=0;
    for(int i=tid;i<1536;i+=128){float v=x[base+i]+(add?residual[base+i]:0.0f);sum+=v*v;}
    sums[tid]=sum;__syncthreads();
    for(int d=64;d;d>>=1){if(tid<d)sums[tid]+=sums[tid+d];__syncthreads();}
    float scale=1.0f/sqrtf(sums[0]/1536.0f+1.0e-6f);
    for(int i=tid;i<1536;i+=128){float v=x[base+i]+(add?residual[base+i]:0.0f);if(add)x[base+i]=v;y[base+i]=v*scale*w[i];}
}
extern "C" __global__ void prefill_rope(float* q,const float* k,const float* v,
    const float* qb,const float* kb,const float* vb,const float* trig,float* keys,float* values){
    int pos=blockIdx.x/6,i=(blockIdx.x%6)*128+threadIdx.x;
    if(i<768){int h=i/64,j=i%64,a=h*128+j,b=a+64;float cs=trig[pos*128+j],sn=trig[pos*128+64+j];
        float u=q[pos*1536+a]+qb[a],w=q[pos*1536+b]+qb[b];q[pos*1536+a]=u*cs-w*sn;q[pos*1536+b]=u*sn+w*cs;}
    if(i<128){int h=i/64,j=i%64,a=h*128+j,b=a+64;float cs=trig[pos*128+j],sn=trig[pos*128+64+j];
        float u=k[pos*256+a]+kb[a],w=k[pos*256+b]+kb[b];keys[pos*256+a]=u*cs-w*sn;keys[pos*256+b]=u*sn+w*cs;}
    if(i<256)values[pos*256+i]=v[pos*256+i]+vb[i];
}
extern "C" __global__ void prefill_attention(const float* q,const float* keys,const float* values,float* out){
    __shared__ float scores[1024];__shared__ float reduce[128];
    int tid=threadIdx.x,pos=blockIdx.x/12,h=blockIdx.x%12,kh=h/6,lane=tid&31,warp=tid>>5;
    for(int t=warp;t<=pos;t+=4){float sum=0;
        for(int j=lane;j<128;j+=32)sum+=q[pos*1536+h*128+j]*keys[t*256+kh*128+j];
        sum+=__shfl_xor_sync(0xffffffff,sum,16);sum+=__shfl_xor_sync(0xffffffff,sum,8);
        sum+=__shfl_xor_sync(0xffffffff,sum,4);sum+=__shfl_xor_sync(0xffffffff,sum,2);
        sum+=__shfl_xor_sync(0xffffffff,sum,1);if(lane==0)scores[t]=sum*0.08838834764831845f;}
    __syncthreads();float maximum=-3.402823466e38f;
    for(int t=tid;t<=pos;t+=128)maximum=fmaxf(maximum,scores[t]);
    reduce[tid]=maximum;__syncthreads();
    for(int d=64;d;d>>=1){if(tid<d)reduce[tid]=fmaxf(reduce[tid],reduce[tid+d]);__syncthreads();}
    maximum=reduce[0];float total=0;
    for(int t=tid;t<=pos;t+=128){float e=expf(scores[t]-maximum);scores[t]=e;total+=e;}
    __syncthreads();reduce[tid]=total;__syncthreads();
    for(int d=64;d;d>>=1){if(tid<d)reduce[tid]+=reduce[tid+d];__syncthreads();}
    total=reduce[0];for(int t=tid;t<=pos;t+=128)scores[t]/=total;
    __syncthreads();float sum=0;
    for(int t=0;t<=pos;t++)sum+=scores[t]*values[t*256+kh*128+tid];out[pos*1536+h*128+tid]=sum;
}
