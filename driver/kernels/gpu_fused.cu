__device__ __forceinline__ float q8scale(const unsigned char* p){
    unsigned short h=*(const unsigned short*)p;float f;
    asm("cvt.f32.f16 %0, %1;":"=f"(f):"h"(h));return f;
}
__device__ __forceinline__ float warp_sum(float x){
    x+=__shfl_xor_sync(0xffffffff,x,16);x+=__shfl_xor_sync(0xffffffff,x,8);
    x+=__shfl_xor_sync(0xffffffff,x,4);x+=__shfl_xor_sync(0xffffffff,x,2);
    x+=__shfl_xor_sync(0xffffffff,x,1);return x;
}
extern "C" __global__ void fused_qkv(const unsigned char* wq,const unsigned char* wk,const unsigned char* wv,
    const float* x,float* q,float* k,float* v){
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);if(row>=2048)return;
    const unsigned char* w=wq;float* y=q;
    if(row>=1792){row-=1792;w=wv;y=v;}else if(row>=1536){row-=1536;w=wk;y=k;}
    w+=row*48*34;float sum=0;
    for(int b=0;b<48;b++){const unsigned char* p=w+b*34;float d=q8scale(p);
        float z=d*(float)((const signed char*)p)[2+lane];sum=fmaf(z,x[b*32+lane],sum);}
    sum=warp_sum(sum);if(lane==0)y[row]=sum;
}
extern "C" __global__ void fused_gate_up(const unsigned char* wg,const unsigned char* wu,const float* x,float* gate){
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);if(row>=8960)return;
    wg+=row*48*34;wu+=row*48*34;float g=0,u=0;
    for(int b=0;b<48;b++){const unsigned char* pg=wg+b*34;const unsigned char* pu=wu+b*34;
        float v=x[b*32+lane];float dg=q8scale(pg)*(float)((const signed char*)pg)[2+lane];
        float du=q8scale(pu)*(float)((const signed char*)pu)[2+lane];g=fmaf(dg,v,g);u=fmaf(du,v,u);}
    g=warp_sum(g);u=warp_sum(u);if(lane==0)gate[row]=(g/(1.0f+expf(-g)))*u;
}
extern "C" __global__ void fused_resnorm(float* x,const float* residual,const float* w,float* y,int add){
    __shared__ float sums[128];int tid=threadIdx.x;float sum=0;
    for(int i=tid;i<1536;i+=128){float v=x[i]+(add?residual[i]:0.0f);sum+=v*v;}
    sums[tid]=sum;__syncthreads();
    for(int stride=64;stride;stride>>=1){if(tid<stride)sums[tid]+=sums[tid+stride];__syncthreads();}
    float scale=1.0f/sqrtf(sums[0]/1536.0f+1.0e-6f);
    for(int i=tid;i<1536;i+=128){float v=x[i]+(add?residual[i]:0.0f);if(add)x[i]=v;y[i]=v*scale*w[i];}
}
