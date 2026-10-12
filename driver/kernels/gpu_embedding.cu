// Direct dequantization of the selected GGUF embedding row; only ~1.6 KB of
// system-resident Q8 data is read. The transformer input stays in VRAM.
extern "C" __global__ void embedding_q8(const unsigned char* weights,float* x,int token){
    int i=blockIdx.x*128+threadIdx.x;if(i>=1536)return;
    const unsigned char* b=weights+(unsigned long long)token*48*34+(i/32)*34;
    unsigned short h=*(const unsigned short*)b;float d;
    asm("cvt.f32.f16 %0, %1;":"=f"(d):"h"(h));
    x[i]=d*(float)((const signed char*)b)[2+(i&31)];
}
