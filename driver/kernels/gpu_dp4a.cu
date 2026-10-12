// Q8_0 weights remain byte-for-byte unchanged. Activations are quantized per
// 32 values to signed int8 with a float scale; DP4A is an integer SIMD op.
__device__ __forceinline__ float scale16(const unsigned char* p){
    unsigned short h=*(const unsigned short*)p;float f;
    asm("cvt.f32.f16 %0, %1;":"=f"(f):"h"(h));return f;
}
__device__ __forceinline__ int load4(const unsigned char* p){
    // GGUF's 34-byte blocks are only 2-byte aligned, never issue unaligned LD32.
    const unsigned short* h=(const unsigned short*)p;
    return (unsigned)h[0]|((unsigned)h[1]<<16);
}
__device__ __forceinline__ int dot4(int a,int b){
    int d;asm("dp4a.s32.s32 %0, %1, %2, 0;":"=r"(d):"r"(a),"r"(b));return d;
}
__device__ __forceinline__ int dot4acc(int a,int b,int c){
    int d;asm("dp4a.s32.s32 %0, %1, %2, %3;":"=r"(d):"r"(a),"r"(b),"r"(c));return d;
}
__device__ __forceinline__ float reduce(float x){
    x+=__shfl_xor_sync(0xffffffff,x,16);x+=__shfl_xor_sync(0xffffffff,x,8);
    x+=__shfl_xor_sync(0xffffffff,x,4);x+=__shfl_xor_sync(0xffffffff,x,2);
    x+=__shfl_xor_sync(0xffffffff,x,1);return x;
}
extern "C" __global__ void quant_act(const float* x,unsigned char* q,int cols){
    int lane=threadIdx.x&31,b=blockIdx.x*4+(threadIdx.x>>5);if(b>=cols/32)return;
    float v=x[b*32+lane],a=fabsf(v);
    a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,16));a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,8));
    a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,4));a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,2));
    a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,1));
    float inv=a>0?127.0f/a:0.0f;int qi=__float2int_rn(v*inv);qi=max(-127,min(127,qi));
    q+=b*36;if(lane==0)*(float*)q=a/127.0f;((signed char*)q)[4+lane]=(signed char)qi;
}
__device__ __forceinline__ float dotrow(const unsigned char* w,const unsigned char* q,int blocks,int lane,int step=8){
    float sum=0;int seg=lane&3;
    for(int b=lane>>2;b<blocks;b+=step){const unsigned char* wp=w+b*34;const unsigned char* qp=q+b*36;
        int wi=load4(wp+2+seg*8);int xi=*(const int*)(qp+4+seg*8);
        int acc=dot4(wi,xi);acc=dot4acc(load4(wp+6+seg*8),*(const int*)(qp+8+seg*8),acc);
        float d=scale16(wp)*(*(const float*)qp);sum=fmaf((float)acc,d,sum);}
    return reduce(sum);
}
extern "C" __global__ void linear_dp4a(const unsigned char* w,const unsigned char* q,float* y,int cols,int rows){
    __shared__ float partial[4];int tid=threadIdx.x,row=blockIdx.x;if(row>=rows)return;
    int blocks=cols/32;w+=(unsigned long long)row*blocks*34;float sum=0;int seg=tid&1;
    for(int b=tid>>1;b<blocks;b+=64){const unsigned char* wp=w+b*34;const unsigned char* qp=q+b*36;
        int acc=0;
        #pragma unroll
        for(int k=0;k<4;k++)acc=dot4acc(load4(wp+2+seg*16+k*4),*(const int*)(qp+4+seg*16+k*4),acc);
        sum=fmaf((float)acc,scale16(wp)*(*(const float*)qp),sum);
    }
    sum=reduce(sum);
    if((tid&31)==0)partial[tid>>5]=sum;__syncthreads();
    if(tid==0)y[row]=((partial[0]+partial[1])+partial[2])+partial[3];
}
extern "C" __global__ void qkv_dp4a(const unsigned char* wq,const unsigned char* wk,const unsigned char* wv,
    const unsigned char* x,float* q,float* k,float* v){
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);if(row>=2048)return;
    const unsigned char* w=wq;float* y=q;
    if(row>=1792){row-=1792;w=wv;y=v;}else if(row>=1536){row-=1536;w=wk;y=k;}
    float sum=dotrow(w+row*48*34,x,48,lane);if(lane==0)y[row]=sum;
}
extern "C" __global__ void gateup_dp4a(const unsigned char* wg,const unsigned char* wu,const unsigned char* x,float* gate){
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);if(row>=8960)return;
    float g=dotrow(wg+row*48*34,x,48,lane);float u=dotrow(wu+row*48*34,x,48,lane);
    if(lane==0)gate[row]=(g/(1.0f+expf(-g)))*u;
}
