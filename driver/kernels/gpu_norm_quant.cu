// Same FP32 residual/RMSNorm reduction and block-32 activation quantization,
// fused to avoid a second dispatch and rereading the normalized vector.
extern "C" __global__ void fused_resnorm_quant(float* x,const float* residual,
    const float* w,float* y,unsigned char* q,int add){
    __shared__ float sums[128];int tid=threadIdx.x;float sum=0;
    for(int i=tid;i<1536;i+=128){float v=x[i]+(add?residual[i]:0.0f);sum+=v*v;}
    sums[tid]=sum;__syncthreads();
    for(int stride=64;stride;stride>>=1){if(tid<stride)sums[tid]+=sums[tid+stride];__syncthreads();}
    float scale=1.0f/sqrtf(sums[0]/1536.0f+1.0e-6f);
    for(int i=tid;i<1536;i+=128){float v=x[i]+(add?residual[i]:0.0f);if(add)x[i]=v;
        v=v*scale*w[i];y[i]=v;float a=fabsf(v);
        a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,16));a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,8));
        a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,4));a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,2));
        a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,1));
        int qi=__float2int_rn(v*(a>0?127.0f/a:0.0f));qi=max(-127,min(127,qi));
        unsigned char* b=q+(i/32)*36;if((i&31)==0)*(float*)b=a/127.0f;
        ((signed char*)b)[4+(i&31)]=(signed char)qi;
    }
}
