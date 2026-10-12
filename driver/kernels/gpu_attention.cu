// One CTA per query head. FP32 KV cache and exact expf softmax; no KV quantization.
extern "C" __global__ void fused_attention(const float* q,const float* keys,
    const float* values,float* out,int pos){
    __shared__ float scores[1024];__shared__ float reduce[128];
    int tid=threadIdx.x,h=blockIdx.x,kh=h/6;
    int lane=tid&31,warp=tid>>5;
    for(int t=warp;t<=pos;t+=4){float sum=0;
        for(int j=lane;j<128;j+=32)sum+=q[h*128+j]*keys[t*256+kh*128+j];
        sum+=__shfl_xor_sync(0xffffffff,sum,16);sum+=__shfl_xor_sync(0xffffffff,sum,8);
        sum+=__shfl_xor_sync(0xffffffff,sum,4);sum+=__shfl_xor_sync(0xffffffff,sum,2);
        sum+=__shfl_xor_sync(0xffffffff,sum,1);
        if(lane==0)scores[t]=sum*0.08838834764831845f;}
    __syncthreads();float maximum=-3.402823466e38f;
    for(int t=tid;t<=pos;t+=128)maximum=fmaxf(maximum,scores[t]);
    reduce[tid]=maximum;__syncthreads();
    for(int d=64;d;d>>=1){if(tid<d)reduce[tid]=fmaxf(reduce[tid],reduce[tid+d]);__syncthreads();}
    maximum=reduce[0];float total=0;
    for(int t=tid;t<=pos;t+=128){float e=expf(scores[t]-maximum);scores[t]=e;total+=e;}
    __syncthreads();reduce[tid]=total;__syncthreads();
    for(int d=64;d;d>>=1){if(tid<d)reduce[tid]+=reduce[tid+d];__syncthreads();}
    total=reduce[0];
    for(int t=tid;t<=pos;t+=128)scores[t]/=total;
    __syncthreads();float sum=0;
    for(int t=0;t<=pos;t++)sum+=scores[t]*values[t*256+kh*128+tid];
    out[h*128+tid]=sum;
}
