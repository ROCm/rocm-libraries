// bench_persistent.cpp — Persistent SWMMAC kernel (never-exit loop)
//
// Launches ONCE, processes multiple work batches. Atomic counter stays in L2.
// Host signals work via a device-mapped volatile flag.
//
// Eliminates: kernel launch overhead, hipMemset counter eviction, cold-start L2 miss.
// Target: stable 23.7 μs per batch, no cold/hot bimodal distribution.
//
// Build: /opt/llvm-amd/bin/clang++ -x hip --offload-arch=gfx1200 \
//   -I/opt/rocm/include -DROCWMMA_WAVE32_MODE=1 -O3 \
//   -L/opt/rocm/lib -lamdhip64 -o bench_persistent bench_persistent.cpp

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
static constexpr double TH=5306.0;

// Persistent kernel: 1024 waves loop, counter accumulates (never memset)
// Each batch: waves claim via atomicAdd with host-managed base offset.
// Counter starts at 0, host advances base by TT each batch.
// Shutdown: host sets cnt to negative value.
__global__ __launch_bounds__(32,2)
void k_persistent(int32_t*C,const int32_t*A,const int32_t*B,
                  int L,volatile int* cnt, volatile int* flag, volatile int* done){
    int my_work;
    int base=0;
    while(true){
        // Wave-level claim: lane 0 claims, broadcast to all 32 lanes
        int wb=0;
        if(threadIdx.x==0)wb=atomicAdd((int*)cnt,32)-base;
        wb=__builtin_amdgcn_readfirstlane(wb);
        if(wb<0){
            if(threadIdx.x==0)atomicAdd((int*)cnt,32);
            break;
        }
        if(wb>=TT){
            if(threadIdx.x==0)atomicAdd((int*)cnt,-32);
            __threadfence_system();
            while(*(volatile int*)cnt >= base+TT && *(volatile int*)cnt >= 0);
            base+=TT;
            continue;
        }

        int w=wb+threadIdx.x;
        // SWMMAC work (same as k6)
        int32_t bt[4];for(int j=0;j<4;++j)bt[j]=B[w*4+j];
        alignas(32)int32_t ac[16][8]={};
        const rocwmma::SwmmacARegsT& ra=*reinterpret_cast<const rocwmma::SwmmacARegsT*>(A+w*2);
        const rocwmma::SwmmacBRegsT& rb=*reinterpret_cast<const rocwmma::SwmmacBRegsT*>(bt);
        for(int i=0;i<L;++i){
#pragma unroll
            for(int cc=0;cc<16;++cc){rocwmma::SwmmacAccumT& rc=*reinterpret_cast<rocwmma::SwmmacAccumT*>(ac[cc]);rc=rocwmma::SwmmacI4::exec(ra,rb,rc,0);}
        }
        for(int cc=0;cc<16;++cc)for(int j=0;j<8;++j)C[(w*16+cc)*8+j]=ac[cc][j];

        // Wave-level done signal: lane 0 signals, broadcast result
        int my_done=0;
        if(threadIdx.x==0)my_done=atomicAdd((int*)done,1);
        my_done=__builtin_amdgcn_readfirstlane(my_done);
        if(my_done==TT-1){
            if(threadIdx.x==0)atomicAdd((int*)done,-TT);
            base+=TT;
        }
    }
}

int main(){
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);
    printf("═══ Persistent Kernel: Counter L2-Permanent ═══\n");
    printf("GPU: %s @ %.0f MHz  Theory: %.0f TOPs\n\n",p.name,p.clockRate/1000.0,TH);

    int32_t *dC,*dA,*dB,*cnt,*d_flag,*d_done;
    hipMalloc(&dC,1024*16*8*4);hipMalloc(&dA,1024*2*4);
    hipMalloc(&dB,4096*4);hipMalloc(&cnt,4);
    hipMalloc(&d_flag,4);hipMalloc(&d_done,4);
    hipMemset(cnt,0,4);hipMemset(d_flag,0,4);hipMemset(d_done,0,4);

    std::vector<int32_t>hA(2048,0x32103210),hB(4096,0x76547654);
    hipMemcpy(dA,hA.data(),8192,hipMemcpyHostToDevice);
    hipMemcpy(dB,hB.data(),16384,hipMemcpyHostToDevice);

    // Launch persistent kernel ONCE (2 waves = 64 threads for flag polling)
    k_persistent<<<1,64>>>(dC,dA,dB,LO,cnt,d_flag,d_done);

    // Thermal warmup: run 30 batches
    printf("Warmup (30 batches)...\n");
    for(int b=0;b<30;++b){
        hipMemset(d_flag,1,4);  // signal: work available
        while(true){  // wait for batch completion
            int done;hipMemcpy(&done,d_done,4,hipMemcpyDeviceToHost);
            if(done>b)break;
        }
        hipMemset(d_flag,0,4);  // signal: idle
    }
    printf("Done.\n\n");

    // Timed measurement: 100 batches
    int n_batches=100;
    printf("Phase 2: Timed %d batches...\n",n_batches);

    // Synchronize at start
    hipMemset(d_flag,0,4);
    hipDeviceSynchronize();

    hipEvent_t e1,e2;hipEventCreate(&e1);hipEventCreate(&e2);
    int base_done;hipMemcpy(&base_done,d_done,4,hipMemcpyDeviceToHost);

    hipEventRecord(e1,0);
    for(int b=0;b<n_batches;++b){
        hipMemset(d_flag,1,4);
        while(true){
            int done;hipMemcpy(&done,d_done,4,hipMemcpyDeviceToHost);
            if(done>base_done+b)break;
        }
        hipMemset(d_flag,0,4);
    }
    hipEventRecord(e2,0);hipEventSynchronize(e2);
    float ms;hipEventElapsedTime(&ms,e1,e2);
    hipEventDestroy(e1);hipEventDestroy(e2);

    double batch_ms=ms/n_batches;
    double tops=O*1024*16*LO/(batch_ms*1e-3)/1e12;
    double ipc=tops/TH;

    printf("\n═══ Persistent Kernel Results ═══\n");
    printf("  Total time:   %.2f ms for %d batches\n",ms,n_batches);
    printf("  Per batch:    %.3f ms\n",batch_ms);
    printf("  TOPs:         %.0f\n",tops);
    printf("  IPC:          %.3f\n",ipc);

    // Also time individual batches for distribution analysis
    printf("\n=== Per-batch timing (20 samples) ===\n");
    std::vector<double> samples;
    for(int b=0;b<20;++b){
        int prev;hipMemcpy(&prev,d_done,4,hipMemcpyDeviceToHost);
        hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);
        hipEventRecord(s,0);
        hipMemset(d_flag,1,4);
        while(true){
            int cur;hipMemcpy(&cur,d_done,4,hipMemcpyDeviceToHost);
            if(cur>prev)break;
        }
        hipMemset(d_flag,0,4);
        hipEventRecord(e,0);hipEventSynchronize(e);
        float tm;hipEventElapsedTime(&tm,s,e);
        hipEventDestroy(s);hipEventDestroy(e);
        double tp=O*1024*16*LO/(tm*1e-3)/1e12;
        samples.push_back(tp);
        printf("  batch %2d: %.3f ms → %.0f TOPs\n",b,tm,tp);
    }

    std::sort(samples.begin(),samples.end());
    double mean=0;for(double x:samples)mean+=x;mean/=samples.size();
    double s2=0;for(double x:samples)s2+=(x-mean)*(x-mean);double stddev=sqrt(s2/samples.size());

    printf("\n  Mean: %.0f ± %.0f TOPs  IPC=%.3f  [%.0f–%.0f]\n",
           mean,stddev,mean/TH,samples.front(),samples.back());
    printf("  Stability: ±%.1f%%\n",stddev/mean*100);

    // Shutdown persistent kernel
    hipMemset(d_flag,2,4);
    hipDeviceSynchronize();

    hipFree(dC);hipFree(dA);hipFree(dB);hipFree(cnt);hipFree(d_flag);hipFree(d_done);
    return 0;
}
