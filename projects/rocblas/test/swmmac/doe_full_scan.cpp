#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
typedef uint16_t u8 __attribute__((ext_vector_type(8)));
typedef uint16_t u16 __attribute__((ext_vector_type(16)));
typedef float    f8 __attribute__((ext_vector_type(8)));
__global__ void t(float*C,int*d,int base,int tw,int al){
    int lane=threadIdx.x%32;u8 a={0};u16 b={0};
    for(int i=0;i<16;i++)((uint16_t*)&b)[i]=0x3F80;  // B=1.0 all lanes
    if(lane==al)for(int i=0;i<8;i++)((uint16_t*)&a)[i]=0x4000; // A=2.0 this lane
    else for(int i=0;i<8;i++)((uint16_t*)&a)[i]=0x3F80;  // A=1.0 others
    int wb=0;if(lane==0)wb=atomicAdd(d,32)-base;wb=__builtin_amdgcn_readfirstlane(wb);
    if(wb>=tw)return;int cld=wb+lane;
    f8 c={0};c=__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a,b,c,0);
    *(f8*)(C+cld*16*8)=c;
}
int main(){
    int*d;float*dC;hipMalloc(&d,4);hipMalloc(&dC,32*128*4);float h[32*128];
    auto run=[&](int al){hipMemset(d,0,4);hipMemset(dC,0,32*128*4);
        t<<<2,32>>>(dC,d,0,1,al);hipDeviceSynchronize();hipMemcpy(h,dC,sizeof(h),hipMemcpyDeviceToHost);};
    run(-1);float base[256];for(int i=0;i<256;i++)base[i]=h[i];
    printf("Baseline=[%.0f]\n",h[0]);
    printf("=== A[lane=AL](all regs=2.0) → which outL affected? ===\n");
    for(int al=0;al<32;al+=4){
        run(al);
        printf("A[%2d]=2.0: ",al);
        for(int ol=0;ol<32;ol++){
            float d=h[ol*128]-base[ol*8];
            if(fabs(d)>0.1f)printf("L%d[0]=%.0f ",ol,d);
        }
        printf("\n");
    }
    // Lane→column mapping summary
    printf("\n=== Lane→Column mapping ===\n");
    for(int al=0;al<32;al++){
        run(al);
        int affected=0;
        for(int ol=0;ol<32;ol++)if(fabs(h[ol*128]-base[ol*8])>0.1f&&h[ol*128]!=base[ol*8]){affected=ol;break;}
        printf("A[%2d]→outL%02d\n",al,affected);
    }
    hipFree(d);hipFree(dC);return 0;
}
