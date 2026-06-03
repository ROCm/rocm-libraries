// ============================================================================
// LDS deep-K staged MXFP6 GEMM — paradigm experiment (vs V17 register-direct).
//
// Idea (Step 22 follow-up): the occ1 ceiling is "MFMA window (~512cyc) < load
// latency (~880cyc)". Staging a DEEP K-tile in LDS enlarges the MFMA compute
// window per global load WITHOUT register spill (registers can't hold deep K).
// KEEP 32x32x64 MFMA (2x the FLOPs/instr and HALF the operand bandwidth/FLOP of
// CK's 16x16x128 — see density analysis; CK's mxfp6 is slow partly because of it).
//
// Data path: global_load_lds (async, zero VGPR; lane*16 LDS layout, see probe)
//   -> __syncthreads -> ds_read MFMA operands -> 32x32x64 MFMA.
//
// This file: correctness-first minimal version (WG 128x128, K_TILE=128, single
// buffer, naive scales). Scale-up / double-buffer / tuning come after err=0.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_lds.hpp"

using namespace mxfp6;


// ---- harness ----
template <int MT, int NT, int KT, int WM, int WN, int OCC = 1, int SWZ = 0, bool DB = true>
static bool correct(int M, int N, int K) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-2, 2);
    int Kp = ((K + KT - 1) / KT) * KT;  // pad K to multiple of K_TILE (zeros)
    std::vector<float> Af((size_t)M * Kp, 0.f), Bf((size_t)Kp * N, 0.f);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) Af[(size_t)m * Kp + k] = d(rng);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) Bf[(size_t)k * N + n] = d(rng);
    constexpr int M_PW = (MT / 32) / WM, N_PW = (NT / 32) / WN;
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, Kp);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), Kp, N);  // -> B^T[N][Kp] packed
    TiledScale saC = tile_scale(preprocess_scale(Aq.scales.data(), M, Kp), M_PW, KT / 64);
    TiledScale sbC = tile_scale(preprocess_scale(Bq.scales.data(), N, Kp), N_PW, KT / 64);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, Kp, N);  // padded zeros contribute 0

    void *dA, *dB;
    uint8_t *dsA, *dsB;
    float* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    hipMemset(dD, 0, (size_t)M * N * 4);

    dim3 g(M / MT, N / NT), blk(256);
    int lds = (DB ? 2 : 1) * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8));
    lds_gemm_db<MT, NT, KT, WM, WN, OCC, SWZ, DB><<<g, blk, lds>>>(dA, dB, dsA, dsB, dD, N, Kp / 64,
                                                     Aq.packed_row_bytes, Bq.packed_row_bytes);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("  launch err %s\n", hipGetErrorString(e)); return false; }
    std::vector<float> Dg((size_t)M * N);
    hipMemcpy(Dg.data(), dD, (size_t)M * N * 4, hipMemcpyDeviceToHost);
    float er = 0, mx = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        er = fmaxf(er, fabsf(Dg[i] - Dref[i]));
        mx = fmaxf(mx, fabsf(Dref[i]));
    }
    bool ok = er < 1e-2f * fmaxf(1.f, mx);
    printf("  LDS M=%d N=%d K=%d: err=%.3e %s\n", M, N, K, er, ok ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return ok;
}

template <int MT, int NT, int KT, int WM, int WN, int OCC = 1, int SWZ = 0, bool DB = true>
static double bench(int M, int N, int K) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
    // Pad K up to a multiple of K_TILE with zeros so SUBS | k_iters for ANY K
    // (e.g. KT192 needs k_iters % 3 == 0). Padded region is zero -> no effect on
    // result; adds <1% useless compute. TFLOPS still uses the real K.
    int Kp = ((K + KT - 1) / KT) * KT;
    std::vector<float> Af((size_t)M * Kp, 0.f), Bf((size_t)Kp * N, 0.f);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) Af[(size_t)m * Kp + k] = d(rng);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) Bf[(size_t)k * N + n] = d(rng);
    constexpr int M_PW = (MT / 32) / WM, N_PW = (NT / 32) / WN;
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, Kp);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), Kp, N);
    TiledScale saC = tile_scale(preprocess_scale(Aq.scales.data(), M, Kp), M_PW, KT / 64);
    TiledScale sbC = tile_scale(preprocess_scale(Bq.scales.data(), N, Kp), N_PW, KT / 64);
    void *dA, *dB;
    uint8_t *dsA, *dsB;
    float* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    dim3 g(M / MT, N / NT), blk(256);
    int lds = (DB ? 2 : 1) * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8));
    auto run = [&] {
        lds_gemm_db<MT, NT, KT, WM, WN, OCC, SWZ, DB><<<g, blk, lds>>>(dA, dB, dsA, dsB, dD, N, Kp / 64,
                                                         Aq.packed_row_bytes, Bq.packed_row_bytes);
    };
    for (int i = 0; i < 10; i++) run();
    hipDeviceSynchronize();
    double best = 1e30;
    for (int r = 0; r < 4; r++) {
        hipEvent_t a, b;
        hipEventCreate(&a); hipEventCreate(&b);
        hipEventRecord(a);
        for (int i = 0; i < 20; i++) run();
        hipEventRecord(b);
        hipDeviceSynchronize();
        float ms = 0; hipEventElapsedTime(&ms, a, b);
        hipEventDestroy(a); hipEventDestroy(b);
        best = fmin(best, ms / 20.0);
    }
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return 2.0 * M * N * K / (best * 1e-3) / 1e12;
}

int main() {
    printf("=== LDS deep-K correctness (only SUBS|k_iters configs) ===\n");
    int ok = 0, tot = 0;
    // K=1024 -> k_iters=16: KT128(SUBS2)=8 tiles, KT256(SUBS4)=4 tiles, both exact.
    ok += correct<256, 256, 128, 2, 2, 1, 0, true>(512, 512, 1024); tot++;   // KT128 double
    ok += correct<256, 256, 256, 2, 2, 1, 0, false>(512, 512, 1024); tot++;  // KT256 single
    ok += correct<256, 256, 256, 2, 2, 1, 16, false>(512, 8192, 1024); tot++; // +swz16
    ok += correct<128, 256, 256, 2, 2, 1, 0, false>(256, 512, 1024); tot++;  // 8acc KT256 single
    ok += correct<256, 256, 192, 2, 2, 1, 0, true>(512, 512, 1024); tot++;   // KT192 DB (K padded 1024->1152)
    ok += correct<256, 256, 192, 2, 2, 1, 16, true>(512, 8192, 1024); tot++; // KT192 DB swz16
    printf("%d/%d\n", ok, tot);
#ifndef NOSCALE
    if (ok != tot) return 1;
#endif

    // Deeper-K sweep @8192^3 (baseline = 256x256 KT192 DB swz16 = 1671). Goal: beat it.
    printf("\n=== deeper-K sweep @8192^3 (KT192 DB = 1671 baseline) ===\n");
    printf("  256x256 KT192 DB swz16 (16acc,144KB)  : %.0f\n", bench<256, 256, 192, 2, 2, 1, 16, true>(8192, 8192, 8192));
    printf("  256x256 KT256 single swz16 (16acc,96KB): %.0f\n", bench<256, 256, 256, 2, 2, 1, 16, false>(8192, 8192, 8192));
    printf("  256x128 KT256 DB swz16 (8acc,144KB)    : %.0f\n", bench<256, 128, 256, 2, 2, 1, 16, true>(8192, 8192, 8192));
    printf("  128x256 KT256 DB swz16 (8acc,144KB)    : %.0f\n", bench<128, 256, 256, 2, 2, 1, 16, true>(8192, 8192, 8192));
    printf("  256x256 KT128 DB swz16 (16acc,96KB)    : %.0f\n", bench<256, 256, 128, 2, 2, 1, 16, true>(8192, 8192, 8192));
    return 0;
}
