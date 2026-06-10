// End-to-end test of libmxfp6gemm through its PUBLIC API (mxfp6/gemm.hpp): choose_tile
// routing + host tiled-scale (mxfp6/preprocess.hpp) + mxfp6::gemm() launch. Correctness
// (fresh-alloc + CPU ref) on both tile paths, then an indicative perf sweep over 12 shapes.
// (For precise numbers warm to steady state — see HANDOFF; this is a functional gate.)
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "mxfp6/gemm.hpp"
#include "mxfp6/preprocess.hpp"
#include "mxfp6/reference.hpp"
#include "mxfp6/types.hpp"
using namespace mxfp6;
static constexpr int KT=192;

template<class F> static double bench(F run){for(int i=0;i<8;i++)run();hipDeviceSynchronize();double best=1e30;
  for(int r=0;r<4;r++){hipEvent_t a,b;hipEventCreate(&a);hipEventCreate(&b);hipEventRecord(a);
   for(int i=0;i<20;i++)run();hipEventRecord(b);hipDeviceSynchronize();float ms=0;hipEventElapsedTime(&ms,a,b);
   hipEventDestroy(a);hipEventDestroy(b);best=fmin(best,ms/20.0);}return best;}
static double tf(int M,int N,int K,double ms){return 2.0*M*N*K/(ms*1e-3)/1e12;}

// build device inputs for (M,N,K) using the chosen tile's MPW/NPW for scale tiling
struct Dev { void*dA,*dBsh; uint8_t*dsA,*dsB; int Ar,Br,Kp; QuantizedMatrix Aq,Bq; };
static Dev setup(int M,int N,int K,TileChoice tc){
  int Kp=((K+KT-1)/KT)*KT; std::mt19937 rng(M*3+N*7+K);std::uniform_real_distribution<float> d(-1,1);
  std::vector<float> Af((size_t)M*Kp,0.f),Bf((size_t)Kp*N,0.f);
  for(int i=0;i<M;i++)for(int k=0;k<K;k++)Af[(size_t)i*Kp+k]=d(rng);
  for(int k=0;k<K;k++)for(int j=0;j<N;j++)Bf[(size_t)k*N+j]=d(rng);
  Dev x; x.Aq=quantize_to_mxfp6(Af.data(),M,Kp); x.Bq=preprocess_B(Bf.data(),Kp,N);
  auto Bsh=preshuffle_B(x.Bq);
  auto saC=tile_scale(preprocess_scale(x.Aq.scales.data(),M,Kp),tc.MPW,KT/64);
  auto sbC=tile_scale(preprocess_scale(x.Bq.scales.data(),N,Kp),tc.NPW,KT/64);
  hipMalloc(&x.dA,x.Aq.packed_data.size());hipMalloc(&x.dBsh,Bsh.data.size());
  hipMalloc(&x.dsA,saC.data.size());hipMalloc(&x.dsB,sbC.data.size());
  hipMemcpy(x.dA,x.Aq.packed_data.data(),x.Aq.packed_data.size(),hipMemcpyHostToDevice);
  hipMemcpy(x.dBsh,Bsh.data.data(),Bsh.data.size(),hipMemcpyHostToDevice);
  hipMemcpy(x.dsA,saC.data.data(),saC.data.size(),hipMemcpyHostToDevice);
  hipMemcpy(x.dsB,sbC.data.data(),sbC.data.size(),hipMemcpyHostToDevice);
  x.Ar=x.Aq.packed_row_bytes; x.Br=x.Bq.packed_row_bytes; x.Kp=Kp; return x;
}
static void teardown(Dev&x){hipFree(x.dA);hipFree(x.dBsh);hipFree(x.dsA);hipFree(x.dsB);}

static bool verify(int M,int N,int K){
  TileChoice tc=choose_tile(M,N); Dev x=setup(M,N,K,tc);
  std::vector<float> Dref((size_t)M*N);mxfp6_gemm_ref(x.Aq,x.Bq,Dref.data(),M,x.Kp,N);
  float*dD;hipMalloc(&dD,(size_t)M*N*4);hipMemset(dD,0x5A,(size_t)M*N*4);
  gemm(OutType::F32,M,N,x.Kp,x.dA,x.dBsh,x.dsA,x.dsB,dD,x.Ar,x.Br);hipDeviceSynchronize();
  std::vector<float> Dg((size_t)M*N);hipMemcpy(Dg.data(),dD,(size_t)M*N*4,hipMemcpyDeviceToHost);hipFree(dD);
  float er=0,mx=0;for(size_t i=0;i<(size_t)M*N;i++){er=fmaxf(er,fabsf(Dg[i]-Dref[i]));mx=fmaxf(mx,fabsf(Dref[i]));}
  bool ok=er<2e-2f*fmaxf(1.f,mx);
  printf("  %4dx%4dx%4d -> %dx%d : %s\n",M,N,K,tc.MT,tc.NT,ok?"OK":"FAIL<<<");
  teardown(x);return ok;
}
static void perf(int M,int N,int K){
  TileChoice tc=choose_tile(M,N); Dev x=setup(M,N,K,tc);
  __half*dD;hipMalloc(&dD,(size_t)M*N*2);
  auto run=[&]{gemm(OutType::F16,M,N,x.Kp,x.dA,x.dBsh,x.dsA,x.dsB,dD,x.Ar,x.Br);};
  double ms=bench(run);int wg=(M/256)*(N/256);
  printf("  %5dx%5dx%5d wg256=%4d -> %3dx%3d : %.0f TFLOPs\n",M,N,K,wg,tc.MT,tc.NT,tf(M,N,K,ms));
  hipFree(dD);teardown(x);
}
int main(){
  printf("=== libmxfp6gemm correctness (end-to-end, CPU ref) ===\n");
  int f=0;
  f+=!verify(512,512,768);      // -> 128x256 path (wg256<CU)
  f+=!verify(768,1280,960);     // -> 128x256, non-square
  f+=!verify(4096,4096,768);    // -> 256x256 path (wg256=256), small K for fast ref
  if(f){printf("  CORRECTNESS FAILED\n");return 1;}
  printf("  all OK\n");
  printf("\n=== indicative perf (12 shapes), FP16 ===\n");
  int sh[][3]={{8192,8192,8192},{8192,4096,8192},{4096,8192,8192},{8192,9216,8192},
    {8192,7680,8192},{8192,5120,8192},{4096,5120,8192},{4096,4096,8192},{2048,8192,8192},
    {2048,4096,8192},{2048,2048,8192},{1024,4096,4096}};
  for(auto&s:sh)perf(s[0],s[1],s[2]);
  return 0;}
