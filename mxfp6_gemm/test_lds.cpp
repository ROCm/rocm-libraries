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
// Standalone LDS dev/correctness driver. The kernel lives in mxfp6_lds.hpp (shared
// with the v18 production dispatcher). Winning config: 256x256, KT192 double-buffer,
// tiled scales, swz (N>=M) — beats V17 +3% @8192^3. Also exercises the K-tail
// (arbitrary K, no padding). Hybrid (A-LDS/B-direct) was disproven; see git history.
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

// Host-side OutT -> float (for comparing F16/BF16 kernel output to the F32 reference).
static inline float out_to_float(float x) { return x; }
static inline float out_to_float(__half x) { return __half2float(x); }
static inline float out_to_float(__hip_bfloat16 x) { return __bfloat162float(x); }

// ---- harness ----
template <int MT, int NT, int KT, int WM, int WN, int OCC = 1, int SWZ = 0, bool DB = true,
          typename OutT = float>
static bool correct(int M, int N, int K) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-2, 2);
    // No K padding: K-tail in the kernel handles k_iters % SUBS != 0 (plain scales).
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    constexpr int M_PW = (MT / 32) / WM, N_PW = (NT / 32) / WN;
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);  // plain (tail)
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    TiledScale saC = tile_scale(saP, M_PW, KT / 64);                   // tiled (main loop)
    TiledScale sbC = tile_scale(sbP, N_PW, KT / 64);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);

    void *dA, *dB;
    uint8_t *dsA, *dsB, *dsAp, *dsBp;
    OutT* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dsAp, saP.data.size());
    hipMalloc(&dsBp, sbP.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(OutT));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsAp, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsBp, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
    hipMemset(dD, 0, (size_t)M * N * sizeof(OutT));

    dim3 g(M / MT, N / NT), blk(256);
    int lds = (DB ? 2 : 1) * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8));
    lds_gemm_db<MT, NT, KT, WM, WN, OCC, SWZ, DB, OutT><<<g, blk, lds>>>(dA, dB, dsA, dsB, dD, N, K / 64,
                                                     Aq.packed_row_bytes, Bq.packed_row_bytes, dsAp, dsBp);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("  launch err %s\n", hipGetErrorString(e)); return false; }
    std::vector<OutT> Dg((size_t)M * N);
    hipMemcpy(Dg.data(), dD, (size_t)M * N * sizeof(OutT), hipMemcpyDeviceToHost);
    float er = 0, mx = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        er = fmaxf(er, fabsf(out_to_float(Dg[i]) - Dref[i]));
        mx = fmaxf(mx, fabsf(Dref[i]));
    }
    bool ok = er < 2e-2f * fmaxf(1.f, mx);
    const char* ty = sizeof(OutT) == 4 ? "f32" : "16b";
    printf("  LDS[%s] M=%d N=%d K=%d: err=%.3e %s\n", ty, M, N, K, er, ok ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dsAp); hipFree(dsBp); hipFree(dD);
    return ok;
}

template <int MT, int NT, int KT, int WM, int WN, int OCC = 1, int SWZ = 0, bool DB = true,
          typename OutT = float>
static double bench(int M, int N, int K) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
    // No K padding: the kernel's K-tail handles k_iters % SUBS != 0 (plain scales).
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    constexpr int M_PW = (MT / 32) / WM, N_PW = (NT / 32) / WN;
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    TiledScale saC = tile_scale(saP, M_PW, KT / 64);
    TiledScale sbC = tile_scale(sbP, N_PW, KT / 64);
    void *dA, *dB;
    uint8_t *dsA, *dsB, *dsAp, *dsBp;
    OutT* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dsAp, saP.data.size());
    hipMalloc(&dsBp, sbP.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(OutT));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsAp, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsBp, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
    dim3 g(M / MT, N / NT), blk(256);
    int lds = (DB ? 2 : 1) * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8));
    auto run = [&] {
        lds_gemm_db<MT, NT, KT, WM, WN, OCC, SWZ, DB, OutT><<<g, blk, lds>>>(dA, dB, dsA, dsB, dD, N, K / 64,
                                                 Aq.packed_row_bytes, Bq.packed_row_bytes, dsAp, dsBp);
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
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dsAp); hipFree(dsBp); hipFree(dD);
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
    ok += correct<256, 256, 192, 2, 2, 1, 0, true>(512, 512, 1024); tot++;   // KT192 DB (K-tail: rem=1 k64)
    ok += correct<256, 256, 192, 2, 2, 1, 16, true>(512, 8192, 1024); tot++; // KT192 DB swz16
    printf("%d/%d\n", ok, tot);
#ifndef NOSCALE
    if (ok != tot) return 1;
#endif

    // (3) K-tail (no padding) KT192 perf @8192^3 (padded baseline was 1671).
    // k_iters=128, SUBS=3 -> 42 full KT192 tiles + 2-k64 KT64 tail.
    printf("\n=== (3) K-tail KT192 (no pad) @8192^3 (padded was ~1671) ===\n");
    printf("  256x256 KT192 DB swz16 (K-tail)   : %.0f\n", bench<256, 256, 192, 2, 2, 1, 16, true>(8192, 8192, 8192));

    // (4) Epilogue output types (HANDOFF Next Step #3): F16 / BF16 output.
    // Output type only affects the epilogue (one store after all K) -> perf ~unchanged
    // vs F32; this confirms correctness + no regression.
    printf("\n=== (4) Output-type epilogue: F32 / F16 / BF16 (256x256 KT192 DB swz16) ===\n");
    int ok2 = 0, t2 = 0;
    ok2 += correct<256, 256, 192, 2, 2, 1, 16, true, float>(512, 8192, 1024); t2++;
    ok2 += correct<256, 256, 192, 2, 2, 1, 16, true, __half>(512, 8192, 1024); t2++;
    ok2 += correct<256, 256, 192, 2, 2, 1, 16, true, __hip_bfloat16>(512, 8192, 1024); t2++;
    printf("  %d/%d\n", ok2, t2);
    // NOTE: in THIS K-tail (no-pad) dev path F16/BF16 run ~6% slower than F32 (half output
    // needs AccVGPR->VGPR->convert->store, raising the VGPR floor 440->500; the KT64
    // single-buffer tail schedules worse under that pressure). The PRODUCTION K-padded path
    // (v18 dispatcher) does NOT show this — there F16/BF16 are neutral-to-faster vs F32.
    // See test_pipeline_v18.cpp end-to-end section.
    printf("  F32  @8192^3 : %.0f\n", bench<256, 256, 192, 2, 2, 1, 16, true, float>(8192, 8192, 8192));
    printf("  F16  @8192^3 : %.0f\n", bench<256, 256, 192, 2, 2, 1, 16, true, __half>(8192, 8192, 8192));
    printf("  BF16 @8192^3 : %.0f\n", bench<256, 256, 192, 2, 2, 1, 16, true, __hip_bfloat16>(8192, 8192, 8192));
    return 0;
}
