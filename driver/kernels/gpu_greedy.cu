// Stable greedy selection: lowest vocabulary id wins ties, and any nonfinite
// logit fails the entire result, matching the previous CPU loop.
struct Candidate {float value;int index;unsigned bad;unsigned pad;};
__device__ __forceinline__ bool better(float a,int ai,float b,int bi){return a>b||(a==b&&ai<bi);}
__device__ __forceinline__ void finish(float v,int id,unsigned bad,Candidate* out,
    float* vs,int* ids,unsigned* bs){
    int t=threadIdx.x;vs[t]=v;ids[t]=id;bs[t]=bad;__syncthreads();
    for(int d=64;d;d>>=1){if(t<d){
        if(better(vs[t+d],ids[t+d],vs[t],ids[t])){vs[t]=vs[t+d];ids[t]=ids[t+d];}
        bs[t]|=bs[t+d];}__syncthreads();}
    if(t==0){out->value=vs[0];out->index=ids[0];out->bad=bs[0];out->pad=0;}
}
extern "C" __global__ void greedy_tiles(const float* logits,Candidate* partial,int n){
    __shared__ float vs[128];__shared__ int ids[128];__shared__ unsigned bs[128];
    float best=-__int_as_float(0x7f800000);int id=0x7fffffff;unsigned bad=0;
    for(int j=0;j<8;j++){int i=blockIdx.x*1024+threadIdx.x+j*128;
        if(i<n){float v=logits[i];bad|=((__float_as_uint(v)&0x7f800000)==0x7f800000);
            if(better(v,i,best,id)){best=v;id=i;}}}
    finish(best,id,bad,partial+blockIdx.x,vs,ids,bs);
}
extern "C" __global__ void greedy_finish(const Candidate* partial,Candidate* result,int n){
    __shared__ float vs[128];__shared__ int ids[128];__shared__ unsigned bs[128];
    float best=-__int_as_float(0x7f800000);int id=0x7fffffff;unsigned bad=0;
    for(int j=threadIdx.x;j<n;j+=128){Candidate p=partial[j];bad|=p.bad;
        if(better(p.value,p.index,best,id)){best=p.value;id=p.index;}}
    finish(best,id,bad,result,vs,ids,bs);
}
