// ============================================================================
// occ2 axis-3 drip kernel (lds_gemm_occ2) — Phase 2 GATE driver.
// F32-first: correctness on race-sensitive partial-fill squares (N<=M, WG48-480,
// K>=3072) + FP16 sanity. Big bench deferred to QA/profiler (lead directive).
// Does NOT touch production files.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_lds_occ2.hpp"   // brings in mxfp6_lds.hpp too

using namespace mxfp6;

static inline float out_to_float(float x) { return x; }
static inline float out_to_float(__half x) { return __half2float(x); }
static inline float out_to_float(__hip_bfloat16 x) { return __bfloat162float(x); }

template <int MT, int NT, int KT, int WM, int WN, int OCC = 2, int SWZ = 0, bool DB = true,
          typename OutT = float>
static bool correct_drip(int M, int N, int K) {
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
    lds_gemm_occ2<MT, NT, KT, WM, WN, OCC, SWZ, DB, OutT><<<g, blk, lds>>>(
        dA, dB, dsA, dsB, dD, N, K / 64, Aq.packed_row_bytes, Bq.packed_row_bytes, dsAp, dsBp);
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
    printf("  drip[%s] %3dx%3d KT%d  M=%d N=%d K=%d (WG=%d): err=%.3e %s\n",
           ty, MT, NT, KT, M, N, K, (M / MT) * (N / NT), er, ok ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dsAp); hipFree(dsBp); hipFree(dD);
    return ok;
}

int main() {
    // F32-first gate: drip structure + correctness + occ=2 (occ checked by occ_probe).
    printf("=== Phase 2 drip GATE: F32 correctness (partial-fill N<=M K>=3072) ===\n");
    int ok = 0, tot = 0;
    ok += correct_drip<128, 256, 128, 2, 2, 2, 0, true, float>(1536, 1536, 3072); tot++; // WG72
    ok += correct_drip<128, 256, 128, 2, 2, 2, 0, true, float>(3072, 3072, 3072); tot++; // WG288
    ok += correct_drip<128, 256, 128, 2, 2, 2, 0, true, float>(3072, 1536, 3072); tot++; // WG144 N<M
    ok += correct_drip<128, 256, 128, 2, 2, 2, 16, true, float>(3072, 1536, 3072); tot++; // +swz16
    ok += correct_drip<128, 256, 128, 2, 2, 2, 0, true, float>(4608, 3072, 3072); tot++;  // WG=432 (near 480 cap)
    printf("  F32: %d/%d\n", ok, tot);
    // NOTE: full-8192^3 correctness omitted here — naive CPU ref is O(5.5e11), impractical.
    // Partial-fill squares above are the meaningful inter-wave-skew race test; QA benches @8192.

    printf("\n=== FP16 sanity ===\n");
    int ok2 = 0, t2 = 0;
    ok2 += correct_drip<128, 256, 128, 2, 2, 2, 0, true, __half>(1536, 1536, 3072); t2++;
    ok2 += correct_drip<128, 256, 128, 2, 2, 2, 16, true, __half>(3072, 1536, 3072); t2++;
    printf("  FP16: %d/%d\n", ok2, t2);

    printf("\n%s\n", (ok == tot && ok2 == t2) ? "GATE correctness: ALL PASS" : "GATE correctness: FAIL");
    return (ok == tot && ok2 == t2) ? 0 : 1;
}
