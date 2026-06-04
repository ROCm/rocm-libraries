// ############################################################################
// STATUS: NO-GO — ARCHIVED REFERENCE (8-wave wave-level ping-pong; Idea 2, ported
//         from HipKittens arXiv:2511.08083 / HazyResearch/HipKittens 8_wave.cu).
//         Kept for archive; NOT in the production path. Production = deep-K
//         lds_gemm_db (mxfp6_lds.hpp), 1741 F16 @8192^3.
//
// RESULT: MS1 best F16 @8192^3 = 1432  (-17.8% vs deep-K 1741). occ-2 held with
//         ZERO spill (F16 128/128, F32 61/128 VGPR/AGPR); correctness 6/6 bit-exact.
//
// ROOT CAUSE (why ping-pong loses to deep-K on FP6):
//   Cooperative glds->LDS staging forces a per-cluster vmcnt(0) drain + s_barrier
//   at the cross-wave hand-off (MFMA operands gather across multiple waves' loaded
//   chunks -> the drain+visibility is CORRECTNESS-REQUIRED: TOPSYNC=0 races,
//   err=2e2). At 8-wave shallow K_STEP=64 that drain fires every ~8 MFMAs — 12-27x
//   more often than deep-K's ~96-MFMA window. Deep-K's large MFMA window exists
//   precisely to amortize this unavoidable drain; occ-2's 2x wave parallelism is
//   eaten by the synchronization overhead instead. A deeper prefetch ring does NOT
//   help (the mandatory drain empties it every iter -> perf flat across nbuf 2/3/4/6).
//
// SEE: memory mxfp6_8wave_pingpong_nogo + team knowledge.md for the full analysis.
// ############################################################################
// ============================================================================
// MS0 driver: 8-wave scaffold correctness + perf-ref for lds_gemm_pp8.
//
// Idea 2 step 0: verify the clean-room 8-wave kernel (256x256, 32x32x64 MFMA,
// 4->8 waves, occ-2 target, NO ping-pong) accumulates 256x256 correctly and
// (via .s / rocprofv3, checked separately) reaches occ-2 with no spill.
// Correctness in F32 (QA judges); the @8192^3 number is a reference only — MS0
// is NOT expected to beat the 1741 production baseline.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_pp8.hpp"   // pulls in mxfp6_lds.hpp primitives

using namespace mxfp6;

static inline float out_to_float(float x) { return x; }
static inline float out_to_float(__half x) { return __half2float(x); }
static inline float out_to_float(__hip_bfloat16 x) { return __bfloat162float(x); }

// MS0 fixed config: 256x256 block, K_STEP=64, 8 waves = 2x4.
static constexpr int MT = 256, NT = 256, KSTEP = 64, WM = 2, WN = 4;
static constexpr int M_PW = (MT / 32) / WM;   // 4
static constexpr int N_PW = (NT / 32) / WN;   // 2

template <typename OutT = float>
static bool correct(int M, int N, int K) {
    // Requirements: M,N multiples of 256; K multiple of 64 (no K-tail in MS0).
    if (M % MT || N % NT || K % 64) { printf("  BADSHAPE M=%d N=%d K=%d\n", M, N, K); return false; }
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-2, 2);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);
    TiledScale saC = tile_scale(saP, M_PW, KSTEP / 64);
    TiledScale sbC = tile_scale(sbP, N_PW, KSTEP / 64);

    void *dA, *dB;
    uint8_t *dsA, *dsB;
    OutT* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(OutT));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    hipMemset(dD, 0, (size_t)M * N * sizeof(OutT));

    dim3 g(M / MT, N / NT), blk(512);
    int lds = MT * (KSTEP * 6 / 8) + NT * (KSTEP * 6 / 8);   // single buffer
    lds_gemm_pp8<MT, NT, KSTEP, WM, WN, OutT><<<g, blk, lds>>>(
        dA, dB, dsA, dsB, dD, N, K / 64, Aq.packed_row_bytes, Bq.packed_row_bytes);
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
    printf("  PP8[%s] M=%d N=%d K=%d: err=%.3e %s\n", ty, M, N, K, er, ok ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return ok;
}

template <typename OutT = __half>
static double bench(int M, int N, int K) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    TiledScale saC = tile_scale(saP, M_PW, KSTEP / 64);
    TiledScale sbC = tile_scale(sbP, N_PW, KSTEP / 64);
    void *dA, *dB;
    uint8_t *dsA, *dsB;
    OutT* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(OutT));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    dim3 g(M / MT, N / NT), blk(512);
    int lds = MT * (KSTEP * 6 / 8) + NT * (KSTEP * 6 / 8);
    auto run = [&] {
        lds_gemm_pp8<MT, NT, KSTEP, WM, WN, OutT><<<g, blk, lds>>>(
            dA, dB, dsA, dsB, dD, N, K / 64, Aq.packed_row_bytes, Bq.packed_row_bytes);
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

// ---- MS1 ping-pong harness ----
template <typename OutT = float, int PF_FRONT = 0, int LGK_MANUAL = 1>
static bool correct_pp(int M, int N, int K) {
    if (M % MT || N % NT || K % 64) { printf("  BADSHAPE M=%d N=%d K=%d\n", M, N, K); return false; }
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-2, 2);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);
    TiledScale saC = tile_scale(saP, M_PW, KSTEP / 64);
    TiledScale sbC = tile_scale(sbP, N_PW, KSTEP / 64);
    void *dA, *dB;
    uint8_t *dsA, *dsB;
    OutT* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(OutT));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    hipMemset(dD, 0, (size_t)M * N * sizeof(OutT));
    dim3 g(M / MT, N / NT), blk(512);
    int lds = 2 * (MT * (KSTEP * 6 / 8) + NT * (KSTEP * 6 / 8));   // tic/toc double buffer
    lds_gemm_pp8_pingpong<MT, NT, KSTEP, WM, WN, OutT, PF_FRONT, LGK_MANUAL><<<g, blk, lds>>>(
        dA, dB, dsA, dsB, dD, N, K / 64, Aq.packed_row_bytes, Bq.packed_row_bytes);
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
    printf("  PP8pp[%s] M=%d N=%d K=%d: err=%.3e %s\n", ty, M, N, K, er, ok ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return ok;
}

template <typename OutT = __half, int PF_FRONT = 0, int LGK_MANUAL = 1>
static double bench_pp(int M, int N, int K) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    TiledScale saC = tile_scale(saP, M_PW, KSTEP / 64);
    TiledScale sbC = tile_scale(sbP, N_PW, KSTEP / 64);
    void *dA, *dB;
    uint8_t *dsA, *dsB;
    OutT* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(OutT));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    dim3 g(M / MT, N / NT), blk(512);
    int lds = 2 * (MT * (KSTEP * 6 / 8) + NT * (KSTEP * 6 / 8));
    auto run = [&] {
        lds_gemm_pp8_pingpong<MT, NT, KSTEP, WM, WN, OutT, PF_FRONT, LGK_MANUAL><<<g, blk, lds>>>(
            dA, dB, dsA, dsB, dD, N, K / 64, Aq.packed_row_bytes, Bq.packed_row_bytes);
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

// ---- MS1 attempt-3 deep-prefetch ring harness ----
template <typename OutT, int NBUF, int TOPSYNC = 0>
static bool correct_ring(int M, int N, int K) {
    if (M % MT || N % NT || K % 64) { printf("  BADSHAPE\n"); return false; }
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-2, 2);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);
    TiledScale saC = tile_scale(saP, M_PW, KSTEP / 64);
    TiledScale sbC = tile_scale(sbP, N_PW, KSTEP / 64);
    void *dA, *dB; uint8_t *dsA, *dsB; OutT* dD;
    hipMalloc(&dA, Aq.packed_data.size()); hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size()); hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(OutT));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    hipMemset(dD, 0, (size_t)M * N * sizeof(OutT));
    dim3 g(M / MT, N / NT), blk(512);
    int lds = NBUF * (MT * (KSTEP * 6 / 8) + NT * (KSTEP * 6 / 8));
    lds_gemm_pp8_ring<MT, NT, KSTEP, WM, WN, OutT, NBUF, TOPSYNC><<<g, blk, lds>>>(
        dA, dB, dsA, dsB, dD, N, K / 64, Aq.packed_row_bytes, Bq.packed_row_bytes);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("  launch err %s\n", hipGetErrorString(e)); return false; }
    std::vector<OutT> Dg((size_t)M * N);
    hipMemcpy(Dg.data(), dD, (size_t)M * N * sizeof(OutT), hipMemcpyDeviceToHost);
    float er = 0, mx = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        er = fmaxf(er, fabsf(out_to_float(Dg[i]) - Dref[i])); mx = fmaxf(mx, fabsf(Dref[i]));
    }
    bool ok = er < 2e-2f * fmaxf(1.f, mx);
    printf("  RING[nbuf=%d topsync=%d] M=%d N=%d K=%d: err=%.3e %s\n", NBUF, TOPSYNC, M, N, K, er, ok ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return ok;
}

template <typename OutT, int NBUF, int TOPSYNC = 0>
static double bench_ring(int M, int N, int K) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    TiledScale saC = tile_scale(preprocess_scale(Aq.scales.data(), M, K), M_PW, KSTEP / 64);
    TiledScale sbC = tile_scale(preprocess_scale(Bq.scales.data(), N, K), N_PW, KSTEP / 64);
    void *dA, *dB; uint8_t *dsA, *dsB; OutT* dD;
    hipMalloc(&dA, Aq.packed_data.size()); hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size()); hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(OutT));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    dim3 g(M / MT, N / NT), blk(512);
    int lds = NBUF * (MT * (KSTEP * 6 / 8) + NT * (KSTEP * 6 / 8));
    auto run = [&] {
        lds_gemm_pp8_ring<MT, NT, KSTEP, WM, WN, OutT, NBUF, TOPSYNC><<<g, blk, lds>>>(
            dA, dB, dsA, dsB, dD, N, K / 64, Aq.packed_row_bytes, Bq.packed_row_bytes);
    };
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
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return 2.0 * M * N * K / (best * 1e-3) / 1e12;
}

// ============================================================================
// CURRENT STATE = MS1 attempt-2 BEST config (the 1429 producer): ping-pong with
// front-load prefetch (PF_FRONT=1) + compiler-managed lgkmcnt (LGK_MANUAL=0).
// MS1 verdict = NO-GO (best 1429 vs deep-K 1741, -18%); attempt-3 deep-ring is
// structurally defeated by the mandatory cross-wave vmcnt(0) drain (kept below as
// a flat-across-depth reference for the profiler's drain-mechanism archive).
// The full attempt-1/2 sweeps + the TOPSYNC=0 race diagnostic live in the
// implementer logs; main() here is the clean reproduction driver.
// ============================================================================
int main() {
    printf("=== MS1 BEST (attempt-2: ping-pong, front-load prefetch + compiler-lgkmcnt) ===\n");
    printf("--- correctness (F32; PF=1 LGK=0) ---\n");
    int ok = 0, tot = 0;
    ok += (correct_pp<float, 1, 0>(256, 256, 1024));   tot++;   // single WG
    ok += (correct_pp<float, 1, 0>(512, 512, 1024));   tot++;
    ok += (correct_pp<float, 1, 0>(512, 768, 1024));   tot++;
    ok += (correct_pp<float, 1, 0>(256, 1024, 1024));  tot++;
    ok += (correct_pp<float, 1, 0>(768, 512, 1024));   tot++;
    ok += (correct_pp<float, 1, 0>(1024, 1024, 512));  tot++;
    printf("  %d/%d\n", ok, tot);
    printf("--- perf @8192^3 (unified-ish; profiler nails 6-run; gate>=~1700, deep-K=1741) ---\n");
    printf("  F16 : %.0f\n", (bench_pp<__half, 1, 0>(8192, 8192, 8192)));
    printf("  F32 : %.0f\n", (bench_pp<float, 1, 0>(8192, 8192, 8192)));

    printf("\n=== ARCHIVE: MS1 attempt-3 deep-ring (DEFEATED; flat across depth) ===\n");
    printf("  RING correctness (TOPSYNC=1, nbuf=4): ");
    correct_ring<float, 4, 1>(512, 768, 1024);
    printf("  RING F16 nbuf=4 @8192^3 : %.0f  (flat vs nbuf=2/3/6 -> deep prefetch negated by drain)\n",
           (bench_ring<__half, 4, 1>(8192, 8192, 8192)));
    return 0;
}
