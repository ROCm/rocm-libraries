// HYBRID driver: A-LDS / B-direct-register experiment.
//  (1) correctness gate (fresh-alloc, F32) vs CPU reference
//  (2) bench @8192^3: baseline lds_gemm_db  vs  lds_gemm_hybrid across PFD ring depths
// K is padded to a multiple of KT (no K-tail in the hybrid kernel).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_lds.hpp"
#include "mxfp6_lds_hybrid.hpp"

using namespace mxfp6;

static inline float of(float x) { return x; }
static inline float of(__half x) { return __half2float(x); }

static constexpr int MT = 256, NT = 256, KT = 192, WM = 2, WN = 2;
static constexpr int M_PW = (MT / 32) / WM, N_PW = (NT / 32) / WN;

// ---- correctness: hybrid kernel vs reference, fresh-alloc, REPS times ----
template <int PFD, int SWZ = 0, bool SHUF = false>
static bool correct(int M, int N, int K, int reps = 40) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-2, 2);
    int Kp = ((K + KT - 1) / KT) * KT;
    std::vector<float> Af((size_t)M * Kp, 0.f), Bf((size_t)Kp * N, 0.f);
    for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) Af[(size_t)i * Kp + k] = d(rng);
    for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = d(rng);

    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, Kp);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), Kp, N);
    PreshuffledB Bsh = preshuffle_B(Bq);
    TiledScale saC = tile_scale(preprocess_scale(Aq.scales.data(), M, Kp), M_PW, KT / 64);
    TiledScale sbC = tile_scale(preprocess_scale(Bq.scales.data(), N, Kp), N_PW, KT / 64);
    std::vector<float> Dref((size_t)M * N);
    // reference on padded K (extra cols are zero -> identical result)
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, Kp, N);

    void *dA, *dB, *dBsh; uint8_t *dsA, *dsB;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dBsh, Bsh.data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dBsh, Bsh.data.data(), Bsh.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);

    dim3 g(M / MT, N / NT), blk(256);
    int lds = 2 * (MT * (KT * 6 / 8));   // hybrid: only A in LDS, double-buffered
    int fails = 0;
    for (int r = 0; r < reps; r++) {
        float* dD; hipMalloc(&dD, (size_t)M * N * sizeof(float));
        hipMemset(dD, 0x5A, (size_t)M * N * sizeof(float));  // poison
        lds_gemm_hybrid<MT, NT, KT, WM, WN, 1, SWZ, true, float, PFD, SHUF>
            <<<g, blk, lds>>>(dA, SHUF ? dBsh : dB, dsA, dsB, dD, N, Kp / 64, Aq.packed_row_bytes, Bq.packed_row_bytes);
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) { printf("  PFD=%d launch err %s\n", PFD, hipGetErrorString(e)); hipFree(dD); hipFree(dA);hipFree(dB);hipFree(dsA);hipFree(dsB); return false; }
        std::vector<float> Dg((size_t)M * N);
        hipMemcpy(Dg.data(), dD, (size_t)M * N * sizeof(float), hipMemcpyDeviceToHost);
        hipFree(dD);
        float er = 0, mx = 0;
        for (size_t i = 0; i < (size_t)M * N; i++) { er = fmaxf(er, fabsf(Dg[i] - Dref[i])); mx = fmaxf(mx, fabsf(Dref[i])); }
        if (!(er < 2e-2f * fmaxf(1.f, mx))) { fails++; if (fails <= 2) printf("  PFD=%d rep%d FAIL er=%.3e\n", PFD, r, er); }
    }
    hipFree(dA); hipFree(dB); hipFree(dBsh); hipFree(dsA); hipFree(dsB);
    printf("  hybrid PFD=%d SHUF=%d  %dx%dx%d: %d/%d pass\n", PFD, (int)SHUF, M, N, K, reps - fails, reps);
    return fails == 0;
}

// ---- bench helpers ----
template <typename Launch>
static double bench_ms(Launch run) {
    for (int i = 0; i < 10; i++) run();
    hipDeviceSynchronize();
    double best = 1e30;
    for (int r = 0; r < 4; r++) {
        hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
        hipEventRecord(a);
        for (int i = 0; i < 20; i++) run();
        hipEventRecord(b); hipDeviceSynchronize();
        float ms = 0; hipEventElapsedTime(&ms, a, b);
        hipEventDestroy(a); hipEventDestroy(b);
        best = fmin(best, ms / 20.0);
    }
    return best;
}

struct Dev { void *dA,*dB,*dBsh; uint8_t *dsA,*dsB; __half* dD; int A_rs,B_rs,Kp; };

static Dev setup(int M, int N, int K) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
    int Kp = ((K + KT - 1) / KT) * KT;
    std::vector<float> Af((size_t)M * Kp, 0.f), Bf((size_t)Kp * N, 0.f);
    for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) Af[(size_t)i * Kp + k] = d(rng);
    for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, Kp);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), Kp, N);
    PreshuffledB Bsh = preshuffle_B(Bq);
    TiledScale saC = tile_scale(preprocess_scale(Aq.scales.data(), M, Kp), M_PW, KT / 64);
    TiledScale sbC = tile_scale(preprocess_scale(Bq.scales.data(), N, Kp), N_PW, KT / 64);
    Dev dv; dv.Kp = Kp; dv.A_rs = Aq.packed_row_bytes; dv.B_rs = Bq.packed_row_bytes;
    hipMalloc(&dv.dA, Aq.packed_data.size());
    hipMalloc(&dv.dB, Bq.packed_data.size());
    hipMalloc(&dv.dBsh, Bsh.data.size());
    hipMalloc(&dv.dsA, saC.data.size());
    hipMalloc(&dv.dsB, sbC.data.size());
    hipMalloc(&dv.dD, (size_t)M * N * sizeof(__half));
    hipMemcpy(dv.dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dv.dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dv.dBsh, Bsh.data.data(), Bsh.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dv.dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dv.dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    return dv;
}
static double tflops(int M,int N,int K,double ms){ return 2.0*M*N*K/(ms*1e-3)/1e12; }

int main() {
    const int M = 8192, N = 8192, K = 8192;

    printf("=== correctness (hybrid, fresh-alloc F32) ===\n");
    int ok = 1;
    ok &= correct<2>(512, 512, 1024);
    ok &= correct<4>(512, 512, 1024);
    ok &= correct<4, 16>(512, 8192, 1024);             // +swz, raw B
    ok &= correct<6>(256, 512, 768);
    ok &= correct<4, 0, true>(512, 512, 1024);         // SHUF coalesced B
    ok &= correct<4, 16, true>(512, 8192, 1024);       // SHUF + swz
    if (!ok) { printf("CORRECTNESS FAILED — stop.\n"); return 1; }

    printf("\n=== bench @%d^3 (FP16) ===\n", M);
    Dev dv = setup(M, N, K);
    dim3 g(M / MT, N / NT), blk(256);

    // baseline: lds_gemm_db (A+B in LDS), 144KB
    {
        int lds = 2 * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8));
        auto run = [&] {
            lds_gemm_db<MT, NT, KT, WM, WN, 1, 0, true, __half>
                <<<g, blk, lds>>>(dv.dA, dv.dB, dv.dsA, dv.dsB, dv.dD, N, dv.Kp / 64, dv.A_rs, dv.B_rs);
        };
        double ms = bench_ms(run);
        printf("  BASELINE lds_gemm_db          : %.3f ms  %.0f TFLOPs\n", ms, tflops(M,N,K,ms));
    }
    // hybrid: A-LDS / B-direct, 72KB LDS, sweep PFD
    int ldsH = 2 * (MT * (KT * 6 / 8));
#define HB(P) do{ auto run=[&]{ lds_gemm_hybrid<MT,NT,KT,WM,WN,1,0,true,__half,P,false> \
        <<<g,blk,ldsH>>>(dv.dA,dv.dB,dv.dsA,dv.dsB,dv.dD,N,dv.Kp/64,dv.A_rs,dv.B_rs); }; \
        double ms=bench_ms(run); printf("  HYBRID raw-B   PFD=%-2d         : %.3f ms  %.0f TFLOPs\n",P,ms,tflops(M,N,K,ms)); }while(0)
#define HS(P) do{ auto run=[&]{ lds_gemm_hybrid<MT,NT,KT,WM,WN,1,0,true,__half,P,true> \
        <<<g,blk,ldsH>>>(dv.dA,dv.dBsh,dv.dsA,dv.dsB,dv.dD,N,dv.Kp/64,dv.A_rs,dv.B_rs); }; \
        double ms=bench_ms(run); printf("  HYBRID shuf-B  PFD=%-2d         : %.3f ms  %.0f TFLOPs\n",P,ms,tflops(M,N,K,ms)); }while(0)
    HB(3); HB(4);
    HS(2); HS(3); HS(4); HS(6); HS(8);

    hipFree(dv.dA); hipFree(dv.dB); hipFree(dv.dsA); hipFree(dv.dsB); hipFree(dv.dD);
    return 0;
}
