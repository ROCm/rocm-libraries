// One-Hot DOE: set only SPECIFIC lane's SPECIFIC register
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
typedef uint16_t u8 __attribute__((ext_vector_type(8)));
typedef uint16_t u16 __attribute__((ext_vector_type(16)));
typedef float    f8 __attribute__((ext_vector_type(8)));

__global__ void hot(float*C,int*dcnt,int base,int tw,
    int a_lane,int a_pos, int b_lane,int b_pos)
{
    int lane=threadIdx.x%32;
    u8 a={0}; u16 b={0};
    if(a_lane<0){for(int i=0;i<8;i++)((uint16_t*)&a)[i]=0x3F80;}
    else if(lane==a_lane && a_pos>=0) ((uint16_t*)&a)[a_pos]=0x3F80;
    if(b_lane<0){for(int i=0;i<16;i++)((uint16_t*)&b)[i]=0x3F80;}
    else if(lane==b_lane && b_pos>=0) ((uint16_t*)&b)[b_pos]=0x3F80;

    int wb=0;if(lane==0)wb=atomicAdd(dcnt,32)-base;wb=__builtin_amdgcn_readfirstlane(wb);
    if(wb>=tw)return;int cld=wb+lane;
    f8 c={0};c=__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a,b,c,0);
    *(f8*)(C+cld*16*8)=c;
}

int main(){
    int*d;float*dC;hipMalloc(&d,4);hipMalloc(&dC,128*32*4);
    float h[128*32];

    auto run=[&](int al,int ap,int bl,int bp){
        hipMemset(d,0,4);hipMemset(dC,0,128*32*4);
        hot<<<2,32>>>(dC,d,0,1,al,ap,bl,bp);
        hipDeviceSynchronize();hipMemcpy(h,dC,sizeof(h),hipMemcpyDeviceToHost);
    };

    printf("=== One-Hot: Lane0 A[pos] + B all=1 (all lanes) ===\n");
    printf("A_pos| lane0 C[0..7]\n");
    // All lanes B=1, only lane0 A[pos]=1
    for(int ap=0;ap<8;ap++){
        run(0,ap, -1,-1); // B not set (all zero!)
        printf("A[%d]  [",ap);
        for(int j=0;j<8;j++)printf("%7.1f",h[j]);printf("]\n");
    }

    // B=1 for ALL lanes, A[lane0][pos]=1
    printf("\n=== B=ALL lanes=1, A[lane0][pos]=1 ===\n");
    printf("A_pos| lane0 C[0..7]\n");
    for(int ap=0;ap<8;ap++){
        run(0,ap, -1,-1); // All B=1, only lane0 A[pos]=1
        printf("A[%d]  [",ap);
        for(int j=0;j<8;j++)printf("%7.3f",h[j]);printf("]\n");
    }

    // Alternative: fill all B registers for all lanes via a separate mechanism
    // For now: what does A[lane0][pos]=1 with B=0 produce? (above already tested)

    printf("\n=== A=ALL lanes=1, B[lane0][pos]=1 ===\n");
    printf("B_pos| lane0 C[0..7]\n");
    for(int bp=0;bp<16;bp++){
        run(-1,-1, 0,bp); // All A=1, only lane0 B[pos]=1
        printf("B[%2d] [",bp);
        for(int j=0;j<8;j++)printf("%7.1f",h[j]);printf("]\n");
    }

    // Both A and B set for specific lane:ress
    printf("\n=== A[lane0][pos]=1 AND B[lane0][pos]=1 ===\n");
    for(int ap=0;ap<8;ap++){
        run(0,ap, 0,ap); // A[lane0][ap]=1, B[lane0][ap]=1
        printf("A[%d]B[%d][",ap,ap);
        for(int j=0;j<8;j++)printf("%7.1f",h[j]);printf("]\n");
    }

    hipFree(d);hipFree(dC);
    return 0;
}
