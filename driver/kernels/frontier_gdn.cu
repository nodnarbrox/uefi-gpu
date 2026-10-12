// Full-dimension autoregressive GDN recurrence. One warp owns one value row;
// its 32 lanes cover K contiguously, retaining four state values per lane.
// inputs = conv[10240], alpha[48], beta[48], A[48], dt[48].
extern "C" __global__ void frontier_gdn(const float *inputs,float *state,float *out,unsigned reset) {
    unsigned row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=48*128)return;
    unsigned head=row/128,value=row%128,key_head=head%16;
    float alpha=inputs[10240+head]+inputs[10384+head];
    float sp=alpha>20.f?alpha:log1pf(expf(alpha));
    float decay=expf(inputs[10336+head]*sp);
    float beta=1.f/(1.f+expf(-inputs[10288+head]));
    float st[4],key[4];float sk=0.f;
    #pragma unroll
    for(unsigned j=0;j<4;++j){unsigned k=lane+32*j;
        st[j]=(reset?0.f:state[row*128+k])*decay;
        key[j]=inputs[2048+key_head*128+k];sk+=st[j]*key[j];}
    for(unsigned d=16;d;d/=2)sk+=__shfl_down_sync(0xffffffff,sk,d);
    sk=__shfl_sync(0xffffffff,sk,0);
    float delta=(inputs[4096+head*128+value]-sk)*beta;float result=0.f;
    #pragma unroll
    for(unsigned j=0;j<4;++j){unsigned k=lane+32*j;
        float updated=st[j]+key[j]*delta;state[row*128+k]=updated;
        result+=updated*(inputs[key_head*128+k]*0.08838834764831845f);}
    for(unsigned d=16;d;d/=2)result+=__shfl_down_sync(0xffffffff,result,d);
    if(lane==0)out[row]=result;
}
