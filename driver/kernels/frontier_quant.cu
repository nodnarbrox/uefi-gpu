// Correctness-first quant readers for the actual mixed-format frontier GGUF.
// Layout equations follow ggml-quants.c; grid data is passed explicitly so no
// relocation or implicit CUDA global-data loader is needed on bare metal.
typedef unsigned char u8;
typedef unsigned int u32;
__device__ __forceinline__ u32 rd16(const u8 *p) { return p[0]|((u32)p[1]<<8); }
__device__ __forceinline__ u32 rd32(const u8 *p) { return rd16(p)|(rd16(p+2)<<16); }
__device__ __forceinline__ float fp16(const u8 *p) {
    unsigned short h=(unsigned short)rd16(p);float x;
    asm("cvt.f32.f16 %0, %1;":"=f"(x):"h"(h));return x;
}
__device__ __forceinline__ float value(const u8 *in,const u32 *grid,const int *iq4,u32 kind,u32 i) {
    if(kind==0)return __uint_as_float(rd32(in+4*i));
    if(kind==30)return __uint_as_float(rd16(in+2*i)<<16);
    if(kind==8){const u8*p=in+(i/32)*34;return fp16(p)*(float)(signed char)p[2+i%32];}
    if(kind==20){const u8*p=in+(i/32)*18;u32 j=i%32,q=(p[2+j%16]>>(4*(j/16)))&15;
        return fp16(p)*(float)iq4[q];}
    if(kind==23){const u8*p=in+(i/256)*136;u32 j=i%256,sb=j/32;
        int sc=((p[4+sb/2]>>(4*(sb%2)))&15)|(((rd16(p+2)>>(2*sb))&3)<<4);
        u32 q=(p[8+sb*16+j%16]>>(4*((j%32)/16)))&15;
        return (fp16(p)*(float)(sc-32))*(float)iq4[q];}
    if(kind==14){const u8*p=in+(i/256)*210;u32 j=i%256,sub=j/128,l=j%32,g=(j%128)/32;
        u32 lo=p[sub*64+l+(g%2)*32],hi=p[128+sub*32+l];
        int q=(((lo>>(4*(g/2)))&15)|(((hi>>(2*g))&3)<<4))-32;
        int sc=(signed char)p[192+sub*8+l/16+g*2];
        return (fp16(p+208)*(float)sc)*(float)q;}
    if(kind==18){const u8*p=in+(i/256)*98;u32 j=i%256,sb=j/32,g=(j%32)/8;
        u32 aux=rd32(p+66+sb*4),s7=(aux>>(7*g))&127,s=s7|((__popc(s7)&1)<<7);
        u32 index=p[2+sb*8+g*2+(j%8)/4],q=(grid[index]>>(8*(j%4)))&255;
        float d=fp16(p)*(0.5f+(float)(aux>>28))*0.5f;
        return (d*(float)q)*(s&(1<<(j%8))?-1.0f:1.0f);}
    return __uint_as_float(0x7fc00000);
}
extern "C" __global__ void frontier_dequant(const u8 *in,float *out,const u32 *grid,const int *iq4,u32 kind,u32 n) {
    u32 i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=value(in,grid,iq4,kind,i);
}
// Choose a format once per warp/row; preserve the original per-lane accumulation
// and reduction. Constants let the compiler remove irrelevant quant-reader paths.
template<u32 Kind>
__device__ __forceinline__ float dot_kind(const u8 *w,const float *x,const u32 *grid,const int *iq4,u32 n,u32 lane) {
    float sum=0;
    for(u32 i=lane;i<n;i+=32)sum+=value(w,grid,iq4,Kind,i)*x[i];
    return sum;
}
__device__ __forceinline__ float dot_dispatch(const u8*w,const float*x,const u32*grid,const int*iq4,u32 kind,u32 n,u32 lane) {
    switch(kind){
        case 0:return dot_kind<0>(w,x,grid,iq4,n,lane);
        case 30:return dot_kind<30>(w,x,grid,iq4,n,lane);
        case 8:return dot_kind<8>(w,x,grid,iq4,n,lane);
        case 20:return dot_kind<20>(w,x,grid,iq4,n,lane);
        case 23:return dot_kind<23>(w,x,grid,iq4,n,lane);
        case 14:return dot_kind<14>(w,x,grid,iq4,n,lane);
        case 18:return dot_kind<18>(w,x,grid,iq4,n,lane);
        default:return __uint_as_float(0x7fc00000);
    }
}
extern "C" __global__ void frontier_matvec(const u8 *weights,const float*x,float*y,const u32 *grid,const int *iq4,
    u32 kind,u32 n,u32 m,u32 row_bytes) {
    u32 lane=threadIdx.x%32,row=blockIdx.x*4+threadIdx.x/32;
    if(row>=m)return;
    const u8*w=weights+(unsigned long long)row*row_bytes;
    float sum=dot_dispatch(w,x,grid,iq4,kind,n,lane);
    for(u32 delta=16;delta;delta/=2)sum+=__shfl_down_sync(0xffffffff,sum,delta);
    if(lane==0)y[row]=sum;
}

// Independent matrices share one launch. Each warp still computes exactly one
// row with the same quant reader, lane order and reduction as frontier_matvec.
// The host holds ALL weight/input/output leases until this launch completes.
struct FrontierJob {
    const u8 *weights;
    const float *input;
    float *output;
    u32 kind,n,m,row_bytes;
    u32 first_row,padding;
};
static_assert(sizeof(FrontierJob)==48,"Rust descriptor ABI");
extern "C" __global__ void frontier_multi_matvec(const FrontierJob *jobs,
    const u32 *grid,const int *iq4,u32 count) {
    u32 lane=threadIdx.x%32,global_row=blockIdx.x*4+threadIdx.x/32;
    // first_row is strictly increasing, rounded to four rows per job. Every
    // warp takes the same branch; no partial-warp shuffle mask is needed.
    u32 low=0,high=count;
    while(low+1<high){u32 mid=(low+high)/2;
        if(jobs[mid].first_row<=global_row)low=mid;else high=mid;}
    const FrontierJob &job=jobs[low];u32 row=global_row-job.first_row;
    if(row>=job.m)return;
    const u8 *w=job.weights+(unsigned long long)row*job.row_bytes;
    float sum=dot_dispatch(w,job.input,grid,iq4,job.kind,job.n,lane);
    for(u32 delta=16;delta;delta/=2)sum+=__shfl_down_sync(0xffffffff,sum,delta);
    if(lane==0)job.output[row]=sum;
}
