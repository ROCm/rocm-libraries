// Partial-grid / non-square / odd-K correctness gate for the hybrid (A-LDS / B-direct
// coalesced) kernel. Memory flags 3072^2 + partial-grid inter-wave skew as the risk.
// All shapes tile-aligned (M,N multiples of 256); K arbitrary (driver pads to *192).
// Fresh-alloc 0x5A poison every rep; compares to CPU reference on padded K.
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
static constexpr int MT = 256, NT = 256, KT = 192, WM = 2, WN = 2;
static constexpr int M_PW = (MT / 32) / WM, N_PW = (NT / 32) / WN;  // 4,4

static int total = 0, passed = 0;

template <int PFD, int SWZ>
static void run(int M, int N, int K, int reps = 8) {
    total++;
    int Kp = ((K + KT - 1) / KT) * KT;
    std::mt19937 rng(M * 131 + N * 17 + K + SWZ);
    std::uniform_real_distribution<float> d(-2, 2);
    std::vector<float> Af((size_t)M * Kp, 0.f), Bf((size_t)Kp * N, 0.f);
    for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) Af[(size_t)i * Kp + k] = d(rng);
    for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, Kp);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), Kp, N);
    PreshuffledB Bsh = preshuffle_B(Bq);
    TiledScale saC = tile_scale(preprocess_scale(Aq.scales.data(), M, Kp), M_PW, KT / 64);
    TiledScale sbC = tile_scale(preprocess_scale(Bq.scales.data(), N, Kp), N_PW, KT / 64);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, Kp, N);

    void *dA, *dBsh; uint8_t *dsA, *dsB;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dBsh, Bsh.data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dBsh, Bsh.data.data(), Bsh.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    dim3 g(M / MT, N / NT), blk(256);
    int lds = 2 * (MT * (KT * 6 / 8));
    int fails = 0; float worst = 0;
    for (int r = 0; r < reps; r++) {
        float* dD; hipMalloc(&dD, (size_t)M * N * sizeof(float));
        hipMemset(dD, 0x5A, (size_t)M * N * sizeof(float));
        lds_gemm_hybrid<MT, NT, KT, WM, WN, 1, SWZ, true, float, PFD, true>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, Kp / 64, Aq.packed_row_bytes, Bq.packed_row_bytes);
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) { printf("  [%dx%dx%d swz%d PFD%d] LAUNCH ERR %s\n", M,N,K,SWZ,PFD,hipGetErrorString(e)); fails++; hipFree(dD); break; }
        std::vector<float> Dg((size_t)M * N);
        hipMemcpy(Dg.data(), dD, (size_t)M * N * sizeof(float), hipMemcpyDeviceToHost);
        hipFree(dD);
        float er = 0, mx = 0;
        for (size_t i = 0; i < (size_t)M * N; i++) { er = fmaxf(er, fabsf(Dg[i] - Dref[i])); mx = fmaxf(mx, fabsf(Dref[i])); }
        worst = fmaxf(worst, er / fmaxf(1.f, mx));
        if (!(er < 2e-2f * fmaxf(1.f, mx))) fails++;
    }
    bool ok = (fails == 0);
    if (ok) passed++;
    printf("  %-6s %5dx%5dx%5d (Kp%d g=%dx%d) swz%-2d PFD%d : %d/%d %s (relerr %.1e)\n",
           ok ? "PASS" : "FAIL", M, N, K, Kp, M/MT, N/NT, SWZ, PFD, reps - fails, reps, ok ? "" : "<<<<", worst);
    hipFree(dA); hipFree(dBsh); hipFree(dsA); hipFree(dsB);
}

int main() {
    printf("=== hybrid shuf-B : partial-grid / non-square / odd-K correctness ===\n");

    // single-WG & tiny grids (most partial-fill)
    run<6,0>(256, 256, 1024);
    run<6,0>(256, 512, 1024);
    run<6,0>(512, 256, 1024);

    // 3072^2 — explicitly flagged in memory (inter-wave skew on partial CU fill)
    run<6,0>(3072, 3072, 384);
    run<4,0>(3072, 3072, 384);

    // odd number of 256-tiles per dim (grid 3,5,7,...) + odd k_tiles
    run<6,0>(768, 768, 1152);     // g=3x3, Kp=1152 (6 tiles)
    run<6,0>(1280, 1280, 640);    // g=5x5, K=640 -> Kp=768 (4 tiles)
    run<6,0>(1792, 768, 960);     // g=7x3, Kp=960 (5 tiles, odd)
    run<6,0>(768, 1792, 960);

    // non-square, varied K incl non-192 multiples (force odd/tail-pad k_tiles)
    run<6,0>(256, 4096, 1000);    // wide N, K=1000 -> Kp=1152
    run<6,0>(4096, 256, 1000);    // tall M
    run<6,0>(512, 3072, 2048);
    run<6,0>(3072, 512, 2048);
    run<6,0>(1024, 1024, 100);    // tiny K -> Kp=192 (1 tile, all odd-tail)
    run<6,0>(1024, 1024, 64);     // K=64 -> Kp=192

    // SWZ remap on grids where nb not a multiple of SWZ (boundary of the band logic)
    run<6,16>(512, 8192, 1024);
    run<6,16>(768, 1280, 1024);   // nb=5, SWZ=16 -> gs clamps
    run<6,32>(512, 8192, 1024);
    run<6,32>(1024, 1536, 768);   // nb=6, SWZ=32
    run<4,16>(2048, 2048, 1536);

    // larger square at scale (deep K)
    run<6,0>(2048, 2048, 2048);
    run<6,0>(2048, 2048, 4096);

    printf("\n==== %d/%d shape-configs PASS ====\n", passed, total);
    return passed == total ? 0 : 1;
}
