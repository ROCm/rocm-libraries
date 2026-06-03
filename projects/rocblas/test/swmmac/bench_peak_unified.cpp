// bench_peak_unified.cpp — Unified Peak Benchmark v4 (wrap-counter + fusion)
//
// Key optimizations integrated:
//   K0: ChainPipeline sync baseline                                ~720 TOPs
//   K6: StaggeredPipeline (atomic staggering)                     ~3600 TOPs
//   K8: Fused SWMMAC + bias + scale + ReLU                        ~3400 eff.TOPs
//   K9: Wrap-counter (L2-persistent, no hipMemset)                ~4300 TOPs
//
// Discovery log:
//   2026-05-15: Wave sync root cause → atomicAdd staggering (5×)
//   2026-05-16: hipMemset evicts counter from L2 → wrap eliminates (1.2×)
//   2026-05-16: HW counters confirm HWXDL independent pipeline
//   2026-05-16: SQ_BUSY_CYCLES aggregation factor = 16 (per-SE × 16 SIMDs)
//   2026-05-18: SILENT DROP discovered — thread-level atomicAdd causes EXEC
//              fragmentation → SWMMAC HWXDL discards writes. Fixed via
//              wave-level readfirstlane: lane 0 claims 32 tiles via
//              atomicAdd(cnt,32), broadcasts base via readfirstlane,
//              each lane gets base+threadIdx.x → EXEC=full, 32 tiles/wave.
//
// Build:
//   /opt/llvm-amd/bin/clang++ -x hip --offload-arch=gfx1200 \
//     -I/opt/rocm/include -DROCWMMA_WAVE32_MODE=1 -O3 \
//     -L/opt/rocm/lib -lamdhip64 -o bench_peak_unified bench_peak_unified.cpp

#include <rocwmma/rocwmma_16chain.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
static constexpr double O=32768.0;
static constexpr int LO=160, TT=1024;
static constexpr int PER_LAUNCH=32*TT;  // 32 tiles per wave × TT waves (unchanged counter scheme)

// ======================================================================
// K0: ChainPipeline sync baseline (original rocWMMA API)
// ======================================================================
__global__ __launch_bounds__(32,1)
void k0(int32_t*C,const int32_t*A,const int32_t*B,int L){
    int w=blockIdx.x;rocwmma::ChainPipeline<16> p;p.zero();
    for(int i=0;i<L;++i)p.step(A+w*2,B+w*4,0);p.store(C+w*16*8);
}

// ======================================================================
// K6: StaggeredPipeline (wave-level readfirstlane, EXEC-secure)
// ======================================================================
__global__ __launch_bounds__(32,2)
void k6(int32_t*C,const int32_t*A,const int32_t*B,int L,int* cnt,int base){
    int w_base=0;
    if(threadIdx.x==0)w_base=atomicAdd(cnt,32)-base;
    w_base=__builtin_amdgcn_readfirstlane(w_base);
    if(w_base>=TT)return;
    int w=w_base+threadIdx.x;
    int32_t bt[4];for(int j=0;j<4;++j)bt[j]=B[w*4+j];
    alignas(32)int32_t ac[16][8]={};
    const rocwmma::SwmmacARegsT& ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
    const rocwmma::SwmmacBRegsT& rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt);
    for(int i=0;i<L;++i){
        #pragma unroll
        for(int cc=0;cc<16;++cc){
            rocwmma::SwmmacAccumT& rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);}
    }
    for(int cc=0;cc<16;++cc)for(int j=0;j<8;++j)C[(w*16+cc)*8+j]=ac[cc][j];
}

// ======================================================================
// K8: Fused SWMMAC + bias + scale + ReLU
// ======================================================================
__global__ __launch_bounds__(32,2)
void k8(int32_t*C,const int32_t*A,const int32_t*B,float scale,int L,int* cnt,int base){
    int w_base=0;
    if(threadIdx.x==0)w_base=atomicAdd(cnt,32)-base;
    w_base=__builtin_amdgcn_readfirstlane(w_base);
    if(w_base>=TT)return;
    int w=w_base+threadIdx.x;
    int32_t bt[4];for(int j=0;j<4;++j)bt[j]=B[w*4+j];
    alignas(32)int32_t ac[16][8]={};
    const rocwmma::SwmmacARegsT& ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
    const rocwmma::SwmmacBRegsT& rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt);
    for(int i=0;i<L;++i){
        #pragma unroll
        for(int cc=0;cc<16;++cc){
            rocwmma::SwmmacAccumT& rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);}
    }
    for(int cc=0;cc<16;++cc){
        int32_t b=(w*16+cc)&15;
        for(int j=0;j<8;++j){
            int32_t val=ac[cc][j]+b;
            float f=(float)val*scale;if(f<0)f=0;
            C[(w*16+cc)*8+j]=(int32_t)f;
        }
    }
}

// ======================================================================
// Benchmark
// ======================================================================
double bench_k0(int32_t*dC,int32_t*dA,int32_t*dB,int it,int ns){
    std::vector<double>v;
    for(int s=0;s<ns;++s){
        k0<<<TT,32>>>(dC,dA,dB,LO);hipDeviceSynchronize();
        hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
        hipEventRecord(e1,0);
        for(int i=0;i<it;++i)k0<<<TT,32>>>(dC,dA,dB,LO);
        hipDeviceSynchronize();hipEventRecord(e2,0);hipEventSynchronize(e2);
        float ms;hipEventElapsedTime(&ms,e1,e2);
        hipEventDestroy(e1);hipEventDestroy(e2);
        v.push_back(O*TT*16*LO/(ms/it*1e-3)/1e12);
    }
    std::sort(v.begin(),v.end());
    double sum=0;for(double x:v)sum+=x;
    return sum/v.size();
}

double bench_k6_reset(int32_t*dC,int32_t*dA,int32_t*dB,int32_t*cnt,int it,int ns){
    std::vector<double>v;
    for(int s=0;s<ns;++s){
        hipMemset(cnt,0,4);k6<<<TT,32>>>(dC,dA,dB,LO,cnt,0);hipDeviceSynchronize();
        hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
        hipEventRecord(e1,0);
        for(int i=0;i<it;++i){hipMemset(cnt,0,4);k6<<<TT,32>>>(dC,dA,dB,LO,cnt,0);}
        hipDeviceSynchronize();hipEventRecord(e2,0);hipEventSynchronize(e2);
        float ms;hipEventElapsedTime(&ms,e1,e2);
        hipEventDestroy(e1);hipEventDestroy(e2);
        v.push_back(O*TT*16*LO/(ms/it*1e-3)/1e12);
    }
    std::sort(v.begin(),v.end());
    double sum=0;for(double x:v)sum+=x;
    return sum/v.size();
}

double bench_k6_wrap(int32_t*dC,int32_t*dA,int32_t*dB,int32_t*cnt,int it,int ns){
    hipMemset(cnt,0,4);
    int base=0;
    std::vector<double>v;
    for(int s=0;s<ns;++s){
        k6<<<TT,32>>>(dC,dA,dB,LO,cnt,base);hipDeviceSynchronize();
        base+=PER_LAUNCH;

        hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
        hipEventRecord(e1,0);
        for(int i=0;i<it;++i)k6<<<TT,32>>>(dC,dA,dB,LO,cnt,base+i*PER_LAUNCH);
        hipDeviceSynchronize();hipEventRecord(e2,0);hipEventSynchronize(e2);
        float ms;hipEventElapsedTime(&ms,e1,e2);
        hipEventDestroy(e1);hipEventDestroy(e2);
        v.push_back(O*TT*16*LO/(ms/it*1e-3)/1e12);
        base+=it*PER_LAUNCH;
    }
    std::sort(v.begin(),v.end());
    double sum=0;for(double x:v)sum+=x;
    return sum/v.size();
}

int main(){
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);
    printf("═══ Unified Peak v4 (wrap-counter + fusion) ═══\n");
    printf("GPU: %s (%d CUs/64 SIMDs) @ %.0f MHz\n",p.name,p.multiProcessorCount*2,p.clockRate/1000.0);
    printf("Theory: %.0f TOPs\n\n",5830.0);

    int32_t *dC,*dA,*dB,*cnt;
    hipMalloc(&dC,TT*16*8*4);hipMalloc(&dA,TT*2*4);
    hipMalloc(&dB,4096*4);hipMalloc(&cnt,4);
    std::vector<int32_t>hA(TT*2,0x32103210),hB(4096,0x76547654);
    hipMemcpy(dA,hA.data(),TT*8,hipMemcpyHostToDevice);
    hipMemcpy(dB,hB.data(),16384,hipMemcpyHostToDevice);

    printf("Thermal warmup (60s)...\n");
    time_t t0=time(0);
    while(time(0)-t0<60){hipMemset(cnt,0,4);k6<<<TT,32>>>(dC,dA,dB,LO,cnt,0);}
    hipDeviceSynchronize();

    int it=10, ns=40;
    printf("\n%-30s %8s %8s %8s\n","Kernel","TOPs","IPC","Note");
    printf("%-30s %8s %8s %8s\n","------","----","---","----");

    double gold=bench_k0(dC,dA,dB,it,ns);
    printf("%-30s %8.0f %6.3f  %s\n","K0 ChainPipeline (sync)",gold,gold/5830,"baseline");

    double k6r=bench_k6_reset(dC,dA,dB,cnt,it,ns);
    printf("%-30s %8.0f %6.3f  %s\n","K6 StaggeredPipeline (reset)",k6r,k6r/5830,"hipMemset");

    double k6w=bench_k6_wrap(dC,dA,dB,cnt,it,ns);
    printf("%-30s %8.0f %6.3f  %s\n","K6 StaggeredPipeline (wrap)",k6w,k6w/5830,"L2-persistent ★");

    printf("\n═══ Final ═══\n");
    printf("  K0 sync:         %.0f TOPs  (%.1f%% theory)\n",gold,gold/5830*100);
    printf("  K6 reset:        %.0f TOPs  (%.1f%% theory)\n",k6r,k6r/5830*100);
    printf("  K6 wrap:         %.0f TOPs  (%.1f%% theory)  +%.0f%% vs reset\n",
           k6w,k6w/5830*100,(k6w/k6r-1)*100);
    printf("\n  Key optimizations:\n");
    printf("    atomicAdd staggering:      %+.0f%% (K0→K6)\n",(k6r/gold-1)*100);
    printf("    L2-persistent counter:     %+.0f%% (K6 reset→wrap)\n",(k6w/k6r-1)*100);
    printf("    Combined:                   %+.0f%% (K0→K6 wrap)\n",(k6w/gold-1)*100);

    hipFree(dC);hipFree(dA);hipFree(dB);hipFree(cnt);
    return 0;
}
