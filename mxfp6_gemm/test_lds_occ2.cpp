// ============================================================================
// occ2 MXFP6 LDS GEMM — Phase 1 compile-gate + bench driver (NEW FILE).
//
// Goal: break the occ1 wall (buffer_load issue back-pressure ~9.3M stall, ~30%,
// single wave can't issue) by co-residing 2 WGs per CU (occupancy 2). The 2nd
// WG (from a DIFFERENT block, no barrier sync) overlaps MFMA while the 1st waits
// on loads/barrier -> natural inter-wave overlap, no ping-pong required.
//
// Phase 1 does NOT write a new kernel. The production lds_gemm_db template is
// already fully parameterized (M/N/K_TILE, WAVES, MIN_OCC, SWZ, DB, OutT); the
// occ2 configs are direct instantiations with MIN_OCC=2 and an 8-acc/wave tile
// (128x256, M_PW=2 N_PW=4 = 8 acc = 128 AGPR) so that 2 WGs fit the 256-combined
// VGPR+AGPR/wave budget AND the 80KB-LDS/WG (per-CU 160KB shared by 2 WGs) limit.
//
// Candidates (from Phase-0 researcher five-axis analysis):
//   PRIMARY  occ2-DEEP : <128,256,256, 2,2, OCC=2, SWZ=0, DB=false>  KT256 single
//                        LDS 72KB/WG (2WG=144KB OK), per-wave window 1024cyc (1.16x load lat)
//   SECONDARY occ2-DB  : <128,256,128, 2,2, OCC=2, SWZ=0, DB=true>   KT128 double
//                        LDS 72KB/WG OK, window 512cyc (relies on inter-wave overlap)
//
// GATE (the real, paper-undecidable test): compile & inspect ISA metadata
//   .vgpr_spill_count == 0  (any hot-loop spill -> NO-GO)
//   occupancy / .agpr+.vgpr  must allow 2 blocks/CU (arch VGPR must drop 244->=128).
// Use scripts/dump ISA separately (--save-temps); this driver validates
// correctness (F32, race-sensitive partial-fill squares) and benches gate-passers.
//
// DOES NOT touch the production mxfp6_lds.hpp / test_lds.cpp.
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

static inline float out_to_float(float x) { return x; }
static inline float out_to_float(__half x) { return __half2float(x); }
static inline float out_to_float(__hip_bfloat16 x) { return __bfloat162float(x); }

// ---- harness (identical structure to test_lds.cpp) ----
template <int MT, int NT, int KT, int WM, int WN, int OCC = 1, int SWZ = 0, bool DB = true,
          typename OutT = float>
static bool correct(int M, int N, int K) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-2, 2);
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
    printf("  occ2[%s] %3dx%3d KT%d DB=%d  M=%d N=%d K=%d (WG=%d): err=%.3e %s\n",
           ty, MT, NT, KT, (int)DB, M, N, K, (M / MT) * (N / NT), er, ok ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dsAp); hipFree(dsBp); hipFree(dD);
    return ok;
}

template <int MT, int NT, int KT, int WM, int WN, int OCC = 1, int SWZ = 0, bool DB = true,
          typename OutT = __half>
static double bench(int M, int N, int K) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
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
    for (int r = 0; r < 5; r++) {  // 5x median-of-best per perf memory
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
    // ---- (1) Correctness: F32, race-sensitive partial-fill squares (N<=M, WG 48-480, K>=3072)
    //         per war_barrier memory. K=3072 is exact for BOTH KT256 (12 tiles) and KT128 (24).
    printf("=== (1) occ2 correctness (F32, race-sensitive partial-fill: N<=M K>=3072) ===\n");
    int ok = 0, tot = 0;
    // PRIMARY occ2-DEEP <128,256,256,2,2,OCC=2,SWZ0,DB=false>
    ok += correct<128, 256, 256, 2, 2, 2, 0, false>(1536, 1536, 3072); tot++; // WG 72
    ok += correct<128, 256, 256, 2, 2, 2, 0, false>(3072, 3072, 3072); tot++; // WG 288
    ok += correct<128, 256, 256, 2, 2, 2, 0, false>(3072, 1536, 3072); tot++; // WG 144 N<M
    ok += correct<128, 256, 256, 2, 2, 2, 16, false>(3072, 1536, 3072); tot++; // +swz16
    // SECONDARY occ2-DB <128,256,128,2,2,OCC=2,SWZ0,DB=true>
    ok += correct<128, 256, 128, 2, 2, 2, 0, true>(1536, 1536, 3072); tot++;
    ok += correct<128, 256, 128, 2, 2, 2, 0, true>(3072, 3072, 3072); tot++;
    ok += correct<128, 256, 128, 2, 2, 2, 0, true>(3072, 1536, 3072); tot++;
    ok += correct<128, 256, 128, 2, 2, 2, 16, true>(3072, 1536, 3072); tot++;
    printf("  F32: %d/%d\n", ok, tot);

    // ---- (2) FP16 correctness sanity (production output type) ----
    printf("\n=== (2) occ2 correctness (FP16 epilogue) ===\n");
    int ok2 = 0, t2 = 0;
    ok2 += correct<128, 256, 256, 2, 2, 2, 0, false, __half>(1536, 1536, 3072); t2++;
    ok2 += correct<128, 256, 128, 2, 2, 2, 0, true,  __half>(1536, 1536, 3072); t2++;
    printf("  FP16: %d/%d\n", ok2, t2);

    if (ok != tot || ok2 != t2) { printf("CORRECTNESS FAIL -> stop before bench\n"); return 1; }

    // ---- (3) Bench gate-passers @8192^3 FP16 vs production SWZ0 baseline 1869 ----
    // (Compile gate .vgpr_spill_count/occupancy is checked separately via ISA dump.)
    // 8192: KT256 -> 32 tiles exact; KT128 -> 64 tiles exact. No K-tail.
    printf("\n=== (3) occ2 bench @8192^3 FP16 (production SWZ0 baseline = 1869) ===\n");
    printf("            SWZ=0   SWZ=16\n");
    {
        double d0  = bench<128,256,256,2,2,2, 0,false,__half>(8192,8192,8192);
        double d16 = bench<128,256,256,2,2,2,16,false,__half>(8192,8192,8192);
        printf("  occ2-DEEP 128x256 KT256 SB: %6.0f  %6.0f\n", d0, d16);
    }
    {
        double b0  = bench<128,256,128,2,2,2, 0,true,__half>(8192,8192,8192);
        double b16 = bench<128,256,128,2,2,2,16,true,__half>(8192,8192,8192);
        printf("  occ2-DB   128x256 KT128 DB: %6.0f  %6.0f\n", b0, b16);
    }
    return 0;
}
