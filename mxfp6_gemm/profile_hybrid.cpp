// Single-dispatch profiling driver for the HYBRID kernel (A-LDS / B-direct coalesced)
// at 8192^3, FP16 output. Config: lds_gemm_hybrid<256,256,192,2,2,occ1,swz0,DB,__half,PFD=6,SHUF=true>.
// B uses the preshuffle_B coalesced layout. K padded to a multiple of 192.
//
// Usage: ./profile_hybrid [warmup] [repeat]
//   warmup=0 repeat=1  -> exactly ONE lds_gemm_hybrid dispatch (PMC/ATT/RCV capture)
//   warmup=3 repeat=10 -> timed benchmark (reports TFLOPs)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_lds.hpp"
#include "mxfp6_lds_hybrid.hpp"

using namespace mxfp6;

static constexpr int MT = 256, NT = 256, KT = 192, WM = 2, WN = 2, SWZ = 0, PFD = 6;

int main(int argc, char** argv) {
    int warmup = argc > 1 ? atoi(argv[1]) : 0;
    int repeat = argc > 2 ? atoi(argv[2]) : 1;
    const int M = 8192, N = 8192, K = 8192;
    constexpr int M_PW = (MT / 32) / WM, N_PW = (NT / 32) / WN;  // 4,4
    int Kp = ((K + KT - 1) / KT) * KT;                          // pad K -> multiple of 192

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> Af((size_t)M * Kp, 0.f), Bf((size_t)Kp * N, 0.f);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) Af[(size_t)i * Kp + k] = d(rng);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = d(rng);

    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, Kp);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), Kp, N);
    PreshuffledB    Bsh = preshuffle_B(Bq);
    TiledScale saC = tile_scale(preprocess_scale(Aq.scales.data(), M, Kp), M_PW, KT / 64);
    TiledScale sbC = tile_scale(preprocess_scale(Bq.scales.data(), N, Kp), N_PW, KT / 64);

    void *dA, *dBsh;
    uint8_t *dsA, *dsB;
    __half* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dBsh, Bsh.data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(__half));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dBsh, Bsh.data.data(), Bsh.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    int A_rs = Aq.packed_row_bytes, B_rs = Bq.packed_row_bytes;

    dim3 g(M / MT, N / NT), blk(256);
    int lds = 2 * (MT * (KT * 6 / 8));   // hybrid: only A in LDS, double-buffered
    auto launch = [&] {
        lds_gemm_hybrid<MT, NT, KT, WM, WN, 1, SWZ, true, __half, PFD, true>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, Kp / 64, A_rs, B_rs);
    };

    for (int i = 0; i < warmup; i++) launch();
    hipDeviceSynchronize();

    if (repeat <= 1) {
        launch();
        hipDeviceSynchronize();
        hipError_t e = hipGetLastError();
        if (e != hipSuccess) { printf("err %s\n", hipGetErrorString(e)); return 1; }
        printf("single dispatch done (M=%d N=%d K=%d, Kp=%d, HYBRID 256x256 KT192 swz%d PFD%d shufB)\n",
               M, N, K, Kp, SWZ, PFD);
    } else {
        double best = 1e30;
        for (int r = 0; r < 4; r++) {
            hipEvent_t a, b;
            hipEventCreate(&a); hipEventCreate(&b);
            hipEventRecord(a);
            for (int i = 0; i < repeat; i++) launch();
            hipEventRecord(b);
            hipDeviceSynchronize();
            float ms = 0; hipEventElapsedTime(&ms, a, b);
            hipEventDestroy(a); hipEventDestroy(b);
            best = fmin(best, ms / repeat);
        }
        printf("M=%d N=%d K=%d: %.3f ms  %.0f TFLOPs\n", M, N, K, best,
               2.0 * M * N * K / (best * 1e-3) / 1e12);
    }
    hipFree(dA); hipFree(dBsh); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return 0;
}
