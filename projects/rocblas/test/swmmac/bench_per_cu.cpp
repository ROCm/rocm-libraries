// bench_per_cu.cpp — Per-CU atomic counters for tighter wave staggering
//
// Single global counter: 1024 waves serialized at ~30 cyc L2 latency
//   → 30,720 cycles total → sparse staggering → IPC 0.55-0.62
//
// 32 per-CU counters (separate cache lines): 32 waves each
//   → 32 × ~4 cyc = 128 cycles total → tight staggering → target IPC 0.65-0.70
//
// N14 calibration: GPU cycle = 0.3597 ns (9,374,984 Hz reference)
//
// Build:
//   /opt/llvm-amd/bin/clang++ -x hip --offload-arch=gfx1200 \
//     -I/opt/rocm/include -DROCWMMA_WAVE32_MODE=1 -O3 \
//     -L/opt/rocm/lib -lamdhip64 -o bench_per_cu bench_per_cu.cpp

#include <rocwmma/rocwmma_16chain.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
static constexpr double O=32768.0;
static constexpr int LO=160, TT=1024, NCU=32;

// ======================================================================
// Per-CU counter array: 32 counters, each in own 128-byte cache line
// Avoids false sharing at L2 cache line granularity (64-128 bytes)
// ======================================================================
struct CUCounter {
    int cnt;
    char _pad[124];  // pad to 128 bytes = 2 cache lines
};
static_assert(sizeof(CUCounter)==128, "CUCounter must be 128 bytes");

// ======================================================================
// K_BASE: global single-counter atomic (baseline)
// ======================================================================
__global__ __launch_bounds__(32,2)
void k_base(int32_t*C,const int32_t*A,const int32_t*B,int L,int* cnt){
    int wb=0;if(threadIdx.x==0)wb=atomicAdd(cnt,32);wb=__builtin_amdgcn_readfirstlane(wb);if(wb>=TT)return;int w=wb+threadIdx.x;
    int32_t bt[1][4];for(int t=0;t<1;++t)for(int j=0;j<4;++j)bt[t][j]=B[(w+t)*4+j];
    alignas(32)int32_t ac[16][8]={};
    const rocwmma::SwmmacARegsT& ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
    for(int t=0;t<1;++t){
        const rocwmma::SwmmacBRegsT& rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt[t]);
        for(int i=0;i<L;++i){
            #pragma unroll
            for(int cc=0;cc<16;++cc){
                rocwmma::SwmmacAccumT& rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);
                rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);
            }
        }
    }
    for(int t=0;t<1;++t)for(int cc=0;cc<16;++cc)for(int j=0;j<8;++j)C[((w+t)*16+cc)*8+j]=ac[cc][j];
}

// ======================================================================
// K_PERCU: per-CU counter staggering (no global atomic for work claim)
//   blockIdx.x determines work assignment. Per-CU counter for stagger only.
// ======================================================================
__global__ __launch_bounds__(32,2)
void k_percu(int32_t*C,const int32_t*A,const int32_t*B,int L,
             CUCounter* cu_cnt, int* cu_global_cnt){
    int w=blockIdx.x;if(w>=TT)return;

    // Determine CU: use blockIdx.x to compute a pseudo-CU assignment
    // (real CU ID not available from HIP without arch-specific intrinsics)
    // Block scheduling is typically round-robin-ish, so w%NCU is a proxy
    int cu = w % NCU;

    // Per-CU sequence number → creates INTRA-CU staggering
    // Serializes only ~32 waves per counter instead of 1024
    int seq = atomicAdd(&cu_cnt[cu].cnt, 1);

    // Tighter stagger: each consecutive wave on same CU delayed by PHASE cycles
    // 32 waves × PHASE = max delay within a CU
    constexpr int PHASE = 2;  // 2 GPU cycles between consecutive waves
    uint32_t delay = (uint32_t)seq * PHASE;
    #pragma unroll 1
    for(uint32_t p=0;p<delay;++p){__asm__ __volatile__("s_nop 0");}

    int32_t bt[1][4];for(int t=0;t<1;++t)for(int j=0;j<4;++j)bt[t][j]=B[(w+t)*4+j];
    alignas(32)int32_t ac[16][8]={};
    const rocwmma::SwmmacARegsT& ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
    for(int t=0;t<1;++t){
        const rocwmma::SwmmacBRegsT& rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt[t]);
        for(int i=0;i<L;++i){
            #pragma unroll
            for(int cc=0;cc<16;++cc){
                rocwmma::SwmmacAccumT& rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);
                rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);
            }
        }
    }
    for(int t=0;t<1;++t)for(int cc=0;cc<16;++cc)for(int j=0;j<8;++j)C[((w+t)*16+cc)*8+j]=ac[cc][j];
}

// ======================================================================
// K_HYBRID: global atomic for claim + per-CU counter for stagger
//   Global serialization + local tight phasing
// ======================================================================
__global__ __launch_bounds__(32,2)
void k_hybrid(int32_t*C,const int32_t*A,const int32_t*B,int L,
              int* global_cnt, CUCounter* cu_cnt){
    int wb=0;
    if(threadIdx.x==0)wb=atomicAdd(global_cnt,32);
    wb=__builtin_amdgcn_readfirstlane(wb);
    if(wb>=TT)return;
    int w=wb+threadIdx.x;
    int cu = w % NCU;
    int seq = atomicAdd(&cu_cnt[cu].cnt, 1);
    constexpr int PHASE = 1;
    uint32_t delay = (uint32_t)seq * PHASE;
    #pragma unroll 1
    for(uint32_t p=0;p<delay;++p){__asm__ __volatile__("s_nop 0");}

    int32_t bt[1][4];for(int t=0;t<1;++t)for(int j=0;j<4;++j)bt[t][j]=B[(w+t)*4+j];
    alignas(32)int32_t ac[16][8]={};
    const rocwmma::SwmmacARegsT& ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
    for(int t=0;t<1;++t){
        const rocwmma::SwmmacBRegsT& rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt[t]);
        for(int i=0;i<L;++i){
            #pragma unroll
            for(int cc=0;cc<16;++cc){
                rocwmma::SwmmacAccumT& rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);
                rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);
            }
        }
    }
    for(int t=0;t<1;++t)for(int cc=0;cc<16;++cc)for(int j=0;j<8;++j)C[((w+t)*16+cc)*8+j]=ac[cc][j];
}

// ======================================================================
struct Stats { double min,max,mean,std; int n; };
Stats compute(std::vector<double>& v){
    Stats st;st.n=(int)v.size();std::sort(v.begin(),v.end());
    st.min=v.front();st.max=v.back();
    double sum=0;for(double x:v)sum+=x;st.mean=sum/v.size();
    double s2=0;for(double x:v)s2+=(x-st.mean)*(x-st.mean);st.std=sqrt(s2/v.size());
    return st;
}

Stats bench_base(int32_t*dC,int32_t*dA,int32_t*dB,int32_t*cnt,int it,int ns){
    std::vector<double> v;
    for(int s=0;s<ns;++s){
        hipMemset(cnt,0,4);k_base<<<1024,32>>>(dC,dA,dB,LO,cnt);hipDeviceSynchronize();
        hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
        hipEventRecord(e1,0);
        for(int i=0;i<it;++i){hipMemset(cnt,0,4);k_base<<<1024,32>>>(dC,dA,dB,LO,cnt);}
        hipDeviceSynchronize();hipEventRecord(e2,0);hipEventSynchronize(e2);
        float ms;hipEventElapsedTime(&ms,e1,e2);
        hipEventDestroy(e1);hipEventDestroy(e2);
        v.push_back(O*1024*16*LO/(ms/it*1e-3)/1e12);
    }
    return compute(v);
}

Stats bench_percu(int32_t*dC,int32_t*dA,int32_t*dB,CUCounter*cu_cnt,int it,int ns){
    std::vector<double> v;
    for(int s=0;s<ns;++s){
        hipMemset(cu_cnt,0,sizeof(CUCounter)*NCU);
        k_percu<<<1024,32>>>(dC,dA,dB,LO,cu_cnt,nullptr);hipDeviceSynchronize();
        hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
        hipEventRecord(e1,0);
        for(int i=0;i<it;++i){hipMemset(cu_cnt,0,sizeof(CUCounter)*NCU);k_percu<<<1024,32>>>(dC,dA,dB,LO,cu_cnt,nullptr);}
        hipDeviceSynchronize();hipEventRecord(e2,0);hipEventSynchronize(e2);
        float ms;hipEventElapsedTime(&ms,e1,e2);
        hipEventDestroy(e1);hipEventDestroy(e2);
        v.push_back(O*1024*16*LO/(ms/it*1e-3)/1e12);
    }
    return compute(v);
}

Stats bench_hybrid(int32_t*dC,int32_t*dA,int32_t*dB,int32_t*cnt,CUCounter*cu_cnt,int it,int ns){
    std::vector<double> v;
    for(int s=0;s<ns;++s){
        hipMemset(cnt,0,4);hipMemset(cu_cnt,0,sizeof(CUCounter)*NCU);
        k_hybrid<<<1024,32>>>(dC,dA,dB,LO,cnt,cu_cnt);hipDeviceSynchronize();
        hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
        hipEventRecord(e1,0);
        for(int i=0;i<it;++i){hipMemset(cnt,0,4);hipMemset(cu_cnt,0,sizeof(CUCounter)*NCU);k_hybrid<<<1024,32>>>(dC,dA,dB,LO,cnt,cu_cnt);}
        hipDeviceSynchronize();hipEventRecord(e2,0);hipEventSynchronize(e2);
        float ms;hipEventElapsedTime(&ms,e1,e2);
        hipEventDestroy(e1);hipEventDestroy(e2);
        v.push_back(O*1024*16*LO/(ms/it*1e-3)/1e12);
    }
    return compute(v);
}

int main(){
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);
    printf("GPU: %s (%d CUs) @ %d MHz\n",p.name,p.multiProcessorCount*2,p.clockRate/1000);
    printf("N14: 9,374,984 Hz → GPU cycle = %.4f ns (±1.706 PPM)\n",1.0/p.clockRate*1e6);
    printf("═══ Per-CU Counter Stagger ═══\n\n");

    int32_t *dC,*dA,*dB,*cnt;
    CUCounter *cu_cnt;
    hipMalloc(&dC,1024*16*8*4);hipMalloc(&dA,1024*2*4);
    hipMalloc(&dB,4096*4);hipMalloc(&cnt,4);
    hipMalloc(&cu_cnt,sizeof(CUCounter)*NCU);
    std::vector<int32_t>hA(2048,0x32103210),hB(4096,0x76547654);
    hipMemcpy(dA,hA.data(),8192,hipMemcpyHostToDevice);
    hipMemcpy(dB,hB.data(),16384,hipMemcpyHostToDevice);

    printf("Warmup (30s)...\n");
    for(int i=0;i<300;++i){hipMemset(cnt,0,4);k_base<<<1024,32>>>(dC,dA,dB,LO,cnt);}
    hipDeviceSynchronize();

    int it=10, ns=30;
    printf("\n%-28s %8s %8s %8s %8s %8s\n","Kernel","Min","Max","Mean","Std","IPC");
    printf("%-28s %8s %8s %8s %8s %8s\n","------","---","---","----","---","---");

    Stats b=bench_base(dC,dA,dB,cnt,it,ns);
    printf("%-28s %8.0f %8.0f %8.0f %8.0f %6.3f\n","base (global atomic)",b.min,b.max,b.mean,b.std,b.mean/5830);

    Stats pcu=bench_percu(dC,dA,dB,cu_cnt,it,ns);
    printf("%-28s %8.0f %8.0f %8.0f %8.0f %6.3f\n","per-CU only",pcu.min,pcu.max,pcu.mean,pcu.std,pcu.mean/5830);

    Stats hy=bench_hybrid(dC,dA,dB,cnt,cu_cnt,it,ns);
    printf("%-28s %8.0f %8.0f %8.0f %8.0f %6.3f\n","hybrid (global+per-CU)",hy.min,hy.max,hy.mean,hy.std,hy.mean/5830);

    printf("\n═══ Analysis ═══\n");
    printf("  base:         %.0f TOPs mean (IPC=%.3f)\n",b.mean,b.mean/5830);
    printf("  per-CU only:  %.0f TOPs mean (IPC=%.3f)\n",pcu.mean,pcu.mean/5830);
    printf("  hybrid:       %.0f TOPs mean (IPC=%.3f)\n",hy.mean,hy.mean/5830);
    double best = std::max({b.mean,pcu.mean,hy.mean});
    printf("  best vs peak: %.0f%% (target: 4080 TOPs = 70.0%% theory)\n",best/4080*100);

    hipFree(dC);hipFree(dA);hipFree(dB);hipFree(cnt);hipFree(cu_cnt);
    return 0;
}
