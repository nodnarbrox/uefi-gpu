// Qwen2.5-1.5B reference operators. No runtime CUDA calls: offline SASS only.
extern "C" __global__ void rmsnorm(const float* x,const float* w,float* y,int n) {
    int i=blockIdx.x*128+threadIdx.x;
    if(i>=n)return;
    // Deterministic sequential reduction initially; later fusion can improve it.
    float sum=0;for(int j=0;j<n;j++)sum+=x[j]*x[j];
    y[i]=x[i]*(1.0f/sqrtf(sum/(float)n+1.0e-6f))*w[i];
}
extern "C" __global__ void rope_bias(float* q,float* k,float* v,
    const float* qb,const float* kb,const float* vb,const float* trig,
    float* keys,float* values,int pos) {
    int i=blockIdx.x*128+threadIdx.x;
    if(i<768){int h=i/64,j=i%64,a=h*128+j,b=a+64;
        float cs=trig[pos*128+j],sn=trig[pos*128+64+j];
        float u=q[a]+qb[a],w=q[b]+qb[b];q[a]=u*cs-w*sn;q[b]=u*sn+w*cs;}
    if(i<128){int h=i/64,j=i%64,a=h*128+j,b=a+64;
        float cs=trig[pos*128+j],sn=trig[pos*128+64+j];
        float u=k[a]+kb[a],w=k[b]+kb[b];keys[pos*256+a]=u*cs-w*sn;keys[pos*256+b]=u*sn+w*cs;}
    if(i<256)values[pos*256+i]=v[i]+vb[i];
}
extern "C" __global__ void attn_scores(const float* q,const float* keys,float* scores,int pos) {
    int i=blockIdx.x*128+threadIdx.x;if(i>=12*(pos+1))return;
    int h=i/(pos+1),t=i%(pos+1),kh=h/6;float sum=0;
    for(int j=0;j<128;j++)sum+=q[h*128+j]*keys[t*256+kh*128+j];
    scores[h*256+t]=sum*0.08838834764831845f;
}
extern "C" __global__ void attn_softmax(float* scores,int pos) {
    int h=threadIdx.x;if(h>=12)return;float maximum=-3.402823466e38f;
    for(int t=0;t<=pos;t++)maximum=fmaxf(maximum,scores[h*256+t]);
    float sum=0;for(int t=0;t<=pos;t++){float e=expf(scores[h*256+t]-maximum);scores[h*256+t]=e;sum+=e;}
    for(int t=0;t<=pos;t++)scores[h*256+t]/=sum;
}
extern "C" __global__ void attn_values(const float* scores,const float* values,float* out,int pos) {
    int i=blockIdx.x*128+threadIdx.x;if(i>=1536)return;int h=i/128,j=i%128,kh=h/6;
    float sum=0;for(int t=0;t<=pos;t++)sum+=scores[h*256+t]*values[t*256+kh*128+j];out[i]=sum;
}
extern "C" __global__ void residual(float* x,const float* y,int n) {
    int i=blockIdx.x*128+threadIdx.x;if(i<n)x[i]+=y[i];
}
extern "C" __global__ void silu_mul(float* gate,const float* up,int n) {
    int i=blockIdx.x*128+threadIdx.x;if(i<n){float v=gate[i];gate[i]=(v/(1.0f+expf(-v)))*up[i];}
}
