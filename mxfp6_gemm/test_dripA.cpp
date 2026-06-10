// drip-A tunable-schedule experiment: correctness (fresh-alloc, incl partial-grid /
// k_tiles==1) + bench @8192^3 sweeping the (START,STRIDE,PER,STOP) A-drip schedule.
// Variants (from the design doc): baseline / V-batch2 / V-offset1 / V-offset-batch / V-batch3.
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

template<bool HARD,int ST,int STR,int PER,int STOP>
static int correct(int M,int N,int K,int reps){
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
    lds_gemm_hybrid_dripA<MT,NT,KT,WM,WN,1,0,true,float,6,HARD,ST,STR,PER,STOP><<<g,blk,lds>>>(dA,dBsh,dsA,dsB,dD,N,Kp/64,Aq.packed_row_bytes,Bq.packed_row_bytes);
    if(hipDeviceSynchronize()!=hipSuccess){fails++;hipFree(dD);break;}
    std::vector<float> Dg((size_t)M*N);hipMemcpy(Dg.data(),dD,(size_t)M*N*4,hipMemcpyDeviceToHost);hipFree(dD);
    float er=0,mx=0;for(size_t i=0;i<(size_t)M*N;i++){er=fmaxf(er,fabsf(Dg[i]-Dref[i]));mx=fmaxf(mx,fabsf(Dref[i]));}
    if(!(er<2e-2f*fmaxf(1.f,mx)))fails++;}
  printf("  [%d,%d,%d,%d] HARD=%d %dx%dx%d : %d/%d %s\n",ST,STR,PER,STOP,(int)HARD,M,N,K,reps-fails,reps,fails?"FAIL<<<":"");
  hipFree(dA);hipFree(dBsh);hipFree(dsA);hipFree(dsB);return fails;}

template<class F> static double bench(F run){for(int i=0;i<10;i++)run();hipDeviceSynchronize();double best=1e30;
  for(int r=0;r<4;r++){hipEvent_t a,b;hipEventCreate(&a);hipEventCreate(&b);hipEventRecord(a);
   for(int i=0;i<20;i++)run();hipEventRecord(b);hipDeviceSynchronize();float ms=0;hipEventElapsedTime(&ms,a,b);
   hipEventDestroy(a);hipEventDestroy(b);best=fmin(best,ms/20.0);}return best;}
static double tf(int M,int N,int K,double ms){return 2.0*M*N*K/(ms*1e-3)/1e12;}

int main(){
  printf("=== drip-A schedule correctness (fresh-alloc poison) ===\n");
  int f=0;
  // default (baseline schedule) must still be bit-correct
  f+=correct<true,0,1,1,0>(256,256,64,100); f+=correct<true,0,1,1,0>(3072,3072,384,8);
  // V-batch2 (PER=2): finishes A early -> safe for HARD_WAIT=0 too (structural margin)
  f+=correct<true ,0,1,2,0>(256,256,64,100); f+=correct<true ,0,1,2,0>(3072,3072,384,8); f+=correct<true ,0,1,2,0>(768,1792,960,8);
  f+=correct<false,0,1,2,0>(256,256,64,200); f+=correct<false,0,1,2,0>(3072,3072,384,8);
  // V-offset-batch (START=1,PER=2)
  f+=correct<true ,1,1,2,0>(256,256,64,100); f+=correct<true ,1,1,2,0>(3072,3072,384,8); f+=correct<true ,1,1,2,0>(2048,2048,2048,8);
  f+=correct<false,1,1,2,0>(256,256,64,200); f+=correct<false,1,1,2,0>(3072,3072,384,8);
  // V-offset1 (START=1): last chunk at q9 -> borderline, verify HARD_WAIT=0 explicitly
  f+=correct<true ,1,1,1,0>(3072,3072,384,8);
  f+=correct<false,1,1,1,0>(256,256,64,200); f+=correct<false,1,1,1,0>(3072,3072,384,8);
  if(f){printf("CORRECTNESS FAILED (f=%d)\n",f);}

  printf("\n=== bench @8192^3 FP16 (HARD_WAIT=1, swz0) ===\n");
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
  printf("  baseline LDS                         : %.0f\n",tf(M,N,K,b0));
  // schedule sweep at the tuned PFD=5 (B-ring sweet spot); V-offset1 [1,1,1,0] = best.
#define V(NAME,ST,STR,PER,STOP) {auto run=[&]{lds_gemm_hybrid_dripA<MT,NT,KT,WM,WN,1,0,true,__half,5,true,ST,STR,PER,STOP><<<g,blk,ldsH>>>(dA,dBsh,dsA,dsB,dD,N,Kp/64,Ar,Br);}; \
   double ms=bench(run); printf("  %-22s [%d,%d,%d,%d] : %.0f (%+.1f%% vs base)\n",NAME,ST,STR,PER,STOP,tf(M,N,K,ms),100*(tf(M,N,K,ms)/tf(M,N,K,b0)-1));}
  V("front-loaded(orig)",0,1,1,0)
  V("V-batch2",          0,1,2,0)
  V("V-offset1(default)",1,1,1,0)
  V("V-offset-batch",    1,1,2,0)
  V("V-batch3",          0,1,3,0)
  hipFree(dA);hipFree(dB);hipFree(dBsh);hipFree(dsA);hipFree(dsB);hipFree(dD);
  return 0;}
