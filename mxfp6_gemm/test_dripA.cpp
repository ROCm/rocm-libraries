// drip-A experiment: correctness (fresh-alloc, incl partial-grid / k_tiles==1) + bench
// @8192^3 comparing baseline LDS / hybrid no-drip / hybrid drip-A (HARD_WAIT on & off).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_lds.hpp"
#include "mxfp6_lds_hybrid.hpp"
using namespace mxfp6;
static constexpr int MT=256,NT=256,KT=192,WM=2,WN=2,M_PW=4,N_PW=4;

template<bool HARD> static int correct(int M,int N,int K,int reps){
  int Kp=((K+KT-1)/KT)*KT; std::mt19937 rng(M+N+K);std::uniform_real_distribution<float> d(-2,2);
  std::vector<float> Af((size_t)M*Kp,0.f),Bf((size_t)Kp*N,0.f);
  for(int i=0;i<M;i++)for(int k=0;k<K;k++)Af[(size_t)i*Kp+k]=d(rng);
  for(int k=0;k<K;k++)for(int j=0;j<N;j++)Bf[(size_t)k*N+j]=d(rng);
  auto Aq=quantize_to_mxfp6(Af.data(),M,Kp);auto Bq=preprocess_B(Bf.data(),Kp,N);auto Bsh=preshuffle_B(Bq);
  auto saC=tile_scale(preprocess_scale(Aq.scales.data(),M,Kp),M_PW,KT/64);
  auto sbC=tile_scale(preprocess_scale(Bq.scales.data(),N,Kp),N_PW,KT/64);
  std::vector<float> Dref((size_t)M*N);mxfp6_gemm_ref(Aq,Bq,Dref.data(),M,Kp,N);
  void*dA,*dBsh;uint8_t*dsA,*dsB;
  hipMalloc(&dA,Aq.packed_data.size());hipMalloc(&dBsh,Bsh.data.size());
  hipMalloc(&dsA,saC.data.size());hipMalloc(&dsB,sbC.data.size());
  hipMemcpy(dA,Aq.packed_data.data(),Aq.packed_data.size(),hipMemcpyHostToDevice);
  hipMemcpy(dBsh,Bsh.data.data(),Bsh.data.size(),hipMemcpyHostToDevice);
  hipMemcpy(dsA,saC.data.data(),saC.data.size(),hipMemcpyHostToDevice);
  hipMemcpy(dsB,sbC.data.data(),sbC.data.size(),hipMemcpyHostToDevice);
  dim3 g(M/MT,N/NT),blk(256);int lds=2*(MT*(KT*6/8));int fails=0;
  for(int r=0;r<reps;r++){float*dD;hipMalloc(&dD,(size_t)M*N*4);hipMemset(dD,0x5A,(size_t)M*N*4);
    lds_gemm_hybrid_dripA<MT,NT,KT,WM,WN,1,0,true,float,6,HARD><<<g,blk,lds>>>(dA,dBsh,dsA,dsB,dD,N,Kp/64,Aq.packed_row_bytes,Bq.packed_row_bytes);
    if(hipDeviceSynchronize()!=hipSuccess){fails++;hipFree(dD);break;}
    std::vector<float> Dg((size_t)M*N);hipMemcpy(Dg.data(),dD,(size_t)M*N*4,hipMemcpyDeviceToHost);hipFree(dD);
    float er=0,mx=0;for(size_t i=0;i<(size_t)M*N;i++){er=fmaxf(er,fabsf(Dg[i]-Dref[i]));mx=fmaxf(mx,fabsf(Dref[i]));}
    if(!(er<2e-2f*fmaxf(1.f,mx)))fails++;}
  printf("  HARD=%d %dx%dx%d : %d/%d %s\n",(int)HARD,M,N,K,reps-fails,reps,fails?"FAIL<<<":"");
  hipFree(dA);hipFree(dBsh);hipFree(dsA);hipFree(dsB);return fails;}

template<class F> static double bench(F run){for(int i=0;i<10;i++)run();hipDeviceSynchronize();double best=1e30;
  for(int r=0;r<4;r++){hipEvent_t a,b;hipEventCreate(&a);hipEventCreate(&b);hipEventRecord(a);
   for(int i=0;i<20;i++)run();hipEventRecord(b);hipDeviceSynchronize();float ms=0;hipEventElapsedTime(&ms,a,b);
   hipEventDestroy(a);hipEventDestroy(b);best=fmin(best,ms/20.0);}return best;}
static double tf(int M,int N,int K,double ms){return 2.0*M*N*K/(ms*1e-3)/1e12;}

int main(){
  printf("=== drip-A correctness (fresh-alloc poison) ===\n");
  int f=0;
  // HARD_WAIT=true
  f+=correct<true>(256,256,64,100);     // k_tiles==1 degenerate
  f+=correct<true>(1024,1024,64,100);
  f+=correct<true>(3072,3072,384,8);    // partial grid
  f+=correct<true>(768,1792,960,8);     // non-square odd
  f+=correct<true>(2048,2048,2048,8);
  // HARD_WAIT=false (rely on margin) — race-prone path, hammer it
  f+=correct<false>(256,256,64,200);
  f+=correct<false>(1024,1024,64,200);
  f+=correct<false>(3072,3072,384,8);
  f+=correct<false>(2048,2048,2048,8);
  if(f){printf("CORRECTNESS FAILED (f=%d)\n",f);}

  printf("\n=== bench @8192^3 FP16 ===\n");
  const int M=8192,N=8192,K=8192,Kp=((K+KT-1)/KT)*KT;
  std::mt19937 rng(42);std::uniform_real_distribution<float> d(-1,1);
  std::vector<float> Af((size_t)M*Kp,0.f),Bf((size_t)Kp*N,0.f);
  for(int i=0;i<M;i++)for(int k=0;k<K;k++)Af[(size_t)i*Kp+k]=d(rng);
  for(int k=0;k<K;k++)for(int j=0;j<N;j++)Bf[(size_t)k*N+j]=d(rng);
  auto Aq=quantize_to_mxfp6(Af.data(),M,Kp);auto Bq=preprocess_B(Bf.data(),Kp,N);auto Bsh=preshuffle_B(Bq);
  auto saC=tile_scale(preprocess_scale(Aq.scales.data(),M,Kp),M_PW,KT/64);
  auto sbC=tile_scale(preprocess_scale(Bq.scales.data(),N,Kp),N_PW,KT/64);
  void*dA,*dB,*dBsh;uint8_t*dsA,*dsB;__half*dD;
  hipMalloc(&dA,Aq.packed_data.size());hipMalloc(&dB,Bq.packed_data.size());hipMalloc(&dBsh,Bsh.data.size());
  hipMalloc(&dsA,saC.data.size());hipMalloc(&dsB,sbC.data.size());hipMalloc(&dD,(size_t)M*N*2);
  hipMemcpy(dA,Aq.packed_data.data(),Aq.packed_data.size(),hipMemcpyHostToDevice);
  hipMemcpy(dB,Bq.packed_data.data(),Bq.packed_data.size(),hipMemcpyHostToDevice);
  hipMemcpy(dBsh,Bsh.data.data(),Bsh.data.size(),hipMemcpyHostToDevice);
  hipMemcpy(dsA,saC.data.data(),saC.data.size(),hipMemcpyHostToDevice);
  hipMemcpy(dsB,sbC.data.data(),sbC.data.size(),hipMemcpyHostToDevice);
  int Ar=Aq.packed_row_bytes,Br=Bq.packed_row_bytes;dim3 g(M/MT,N/NT),blk(256);
  int ldsB=2*(MT*(KT*6/8)+NT*(KT*6/8)), ldsH=2*(MT*(KT*6/8));
  double b0=bench([&]{lds_gemm_db<MT,NT,KT,WM,WN,1,0,true,__half><<<g,blk,ldsB>>>(dA,dB,dsA,dsB,dD,N,Kp/64,Ar,Br);});
  double h6=bench([&]{lds_gemm_hybrid<MT,NT,KT,WM,WN,1,0,true,__half,6,true><<<g,blk,ldsH>>>(dA,dBsh,dsA,dsB,dD,N,Kp/64,Ar,Br);});
  double dT=bench([&]{lds_gemm_hybrid_dripA<MT,NT,KT,WM,WN,1,0,true,__half,6,true><<<g,blk,ldsH>>>(dA,dBsh,dsA,dsB,dD,N,Kp/64,Ar,Br);});
  double dF=bench([&]{lds_gemm_hybrid_dripA<MT,NT,KT,WM,WN,1,0,true,__half,6,false><<<g,blk,ldsH>>>(dA,dBsh,dsA,dsB,dD,N,Kp/64,Ar,Br);});
  printf("  baseline LDS              : %.0f TFLOPs\n",tf(M,N,K,b0));
  printf("  hybrid no-drip PFD6       : %.0f TFLOPs (%+.1f%% vs base)\n",tf(M,N,K,h6),100*(tf(M,N,K,h6)/tf(M,N,K,b0)-1));
  printf("  hybrid drip-A HARD_WAIT=1 : %.0f TFLOPs (%+.1f%% vs base, %+.1f%% vs no-drip)\n",tf(M,N,K,dT),100*(tf(M,N,K,dT)/tf(M,N,K,b0)-1),100*(tf(M,N,K,dT)/tf(M,N,K,h6)-1));
  printf("  hybrid drip-A HARD_WAIT=0 : %.0f TFLOPs (%+.1f%% vs base, %+.1f%% vs no-drip)\n",tf(M,N,K,dF),100*(tf(M,N,K,dF)/tf(M,N,K,b0)-1),100*(tf(M,N,K,dF)/tf(M,N,K,h6)-1));
  hipFree(dA);hipFree(dB);hipFree(dBsh);hipFree(dsA);hipFree(dsB);hipFree(dD);
  return 0;
}
