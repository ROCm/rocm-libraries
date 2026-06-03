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

// ===========================================================================
// (2) Asymmetric hybrid: A staged DEEP in LDS (KT256 double-buffer, 96KB), B read
// directly from preshuffled global into registers. LDS holds only A -> KT reaches
// 256 (symmetric all-LDS caps at KT192 @144KB). Tests whether deep-A + direct-B
// beats symmetric KT192 (1671). Scales tile-grouped for both (asm). B double-buffered.
// ===========================================================================
__device__ __forceinline__ v6i load_B_op(const char* p, int lane) {
    int4 lo; int2 hi;  // preshuffled B: dwordx4 @ sec0 (lane*16) + dwordx2 @ sec1 (1024+lane*8)
    asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(lo) : "v"(p + lane * 16) : "memory");
    asm volatile("global_load_dwordx2 %0, %1, off" : "=v"(hi) : "v"(p + 1024 + lane * 8) : "memory");
    return v6i{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y};
}

template <int M_TILE, int N_TILE, int K_TILE, int WAVES_M, int WAVES_N, int SWZ = 0>
__global__ void __launch_bounds__(256, 1)
    lds_gemm_hyb(const void* __restrict__ A, const void* __restrict__ Bsh,
                 const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                 float* __restrict__ D, int N, int k_iters, int A_rs) {
    constexpr int KT_BYTES = K_TILE * 6 / 8, A_CPR = KT_BYTES / 16, SUBS = K_TILE / 64;
    constexpr int MB = M_TILE / 32, NB = N_TILE / 32, M_PW = MB / WAVES_M, N_PW = NB / WAVES_N;
    constexpr int A_BYTES = M_TILE * KT_BYTES, BUF = A_BYTES;  // LDS = A only
    constexpr int SA_PAD = ((M_PW + 3) / 4) * 4, SB_PAD = ((N_PW + 3) / 4) * 4;
    static_assert(SA_PAD == 4 && SB_PAD == 4, "tiled-scale path: <=4 blocks/wave");
    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64, wm = wave / WAVES_N, wn = wave % WAVES_N;
    int wg_m, wg_n;
    if constexpr (SWZ > 0) {
        int mb = gridDim.x, nb = gridDim.y, pid = blockIdx.y * mb + blockIdx.x;
        const int G = SWZ, span = G * mb;
        int grp = pid / span, fn = grp * G, gs = (nb - fn) < G ? (nb - fn) : G, r = pid % span;
        wg_m = r / gs; wg_n = fn + r % gs;
    } else { wg_m = blockIdx.x; wg_n = blockIdx.y; }
    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_rs;
    int sa_grp = wg_m * WAVES_M + wm, sb_grp = wg_n * WAVES_N + wn;
    int k_tiles = k_iters / SUBS;

    AccTileA acc[M_PW][N_PW];
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) clear_acc(acc[mi][ni]);

    auto prefetch = [&](int kt, uint32_t base, v6i (*b)[N_PW], int (*sa)[1], int (*sb)[1]) {
        issue_tile<M_TILE, KT_BYTES, A_CPR>(smem, base, Ag, A_rs, kt * KT_BYTES, wave, lane);
#pragma unroll
        for (int sub = 0; sub < SUBS; sub++)
#pragma unroll
            for (int ni = 0; ni < N_PW; ni++) {
                int nbg = wg_n * NB + wn * N_PW + ni, k64 = kt * SUBS + sub;
                const char* p = reinterpret_cast<const char*>(Bsh) + (size_t)(nbg * k_iters + k64) * 1536;
                b[sub][ni] = load_B_op(p, lane);
            }
        const char* pa = reinterpret_cast<const char*>(sA) + (size_t)((sa_grp * k_tiles + kt) * 64 + lane) * SUBS * SA_PAD;
        const char* pbs = reinterpret_cast<const char*>(sB) + (size_t)((sb_grp * k_tiles + kt) * 64 + lane) * SUBS * SB_PAD;
        int ta[SUBS], tbs[SUBS];
        asm_load_dwordxN_nowait(ta, pa, SUBS);
        asm_load_dwordxN_nowait(tbs, pbs, SUBS);
#pragma unroll
        for (int sub = 0; sub < SUBS; sub++) { sa[sub][0] = ta[sub]; sb[sub][0] = tbs[sub]; }
    };
    auto compute = [&](uint32_t cur, v6i (*b)[N_PW], int (*sa)[1], int (*sb)[1]) {
#pragma unroll
        for (int sub = 0; sub < SUBS; sub++) {
            v6i a[M_PW]; int sav[M_PW], sbv[N_PW];
#pragma unroll
            for (int mi = 0; mi < M_PW; mi++) {
                a[mi] = read_op<KT_BYTES>(smem, cur, wm * M_PW + mi, sub, lane);
                sav[mi] = (sa[sub][0] >> (8 * mi)) & 0xff;
            }
#pragma unroll
            for (int ni = 0; ni < N_PW; ni++) sbv[ni] = (sb[sub][0] >> (8 * ni)) & 0xff;
#pragma unroll
            for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
                for (int ni = 0; ni < N_PW; ni++)
                    mfma_scale_f32_32x32x64_fp6<0>(acc[mi][ni], a[mi], b[sub][ni], sav[mi], sbv[ni]);
        }
    };

    v6i b0[SUBS][N_PW], b1[SUBS][N_PW];
    int sa0[SUBS][1], sa1[SUBS][1], sb0[SUBS][1], sb1[SUBS][1];
    prefetch(0, 0, b0, sa0, sb0);
    int kt = 0;
    for (; kt + 1 < k_tiles; kt += 2) {
        prefetch(kt + 1, BUF, b1, sa1, sb1);
        wait_vmcnt(0); __syncthreads();
        compute(0, b0, sa0, sb0);
        __syncthreads();
        if (kt + 2 < k_tiles) prefetch(kt + 2, 0, b0, sa0, sb0);
        wait_vmcnt(0); __syncthreads();
        compute(BUF, b1, sa1, sb1);
        __syncthreads();
    }
    if (kt < k_tiles) { wait_vmcnt(0); __syncthreads(); compute(0, b0, sa0, sb0); }

#pragma unroll
    for (int mi = 0; mi < M_PW; mi++) {
        int m = wg_m * M_TILE + (wm * M_PW + mi) * 32;
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) {
            int n = wg_n * N_TILE + (wn * N_PW + ni) * 32;
            store_acc_f32(D, N, acc[mi][ni], m, n);
        }
    }
}

// ---- harness ----
template <int MT, int NT, int KT, int WM, int WN, int OCC = 1, int SWZ = 0, bool DB = true>
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
    float* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dsAp, saP.data.size());
    hipMalloc(&dsBp, sbP.data.size());
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsAp, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsBp, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
    hipMemset(dD, 0, (size_t)M * N * 4);

    dim3 g(M / MT, N / NT), blk(256);
    int lds = (DB ? 2 : 1) * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8));
    lds_gemm_db<MT, NT, KT, WM, WN, OCC, SWZ, DB><<<g, blk, lds>>>(dA, dB, dsA, dsB, dD, N, K / 64,
                                                     Aq.packed_row_bytes, Bq.packed_row_bytes, dsAp, dsBp);
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
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dsAp); hipFree(dsBp); hipFree(dD);
    return ok;
}

template <int MT, int NT, int KT, int WM, int WN, int OCC = 1, int SWZ = 0, bool DB = true>
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
    float* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dsAp, saP.data.size());
    hipMalloc(&dsBp, sbP.data.size());
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsAp, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsBp, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
    dim3 g(M / MT, N / NT), blk(256);
    int lds = (DB ? 2 : 1) * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8));
    auto run = [&] {
        lds_gemm_db<MT, NT, KT, WM, WN, OCC, SWZ, DB><<<g, blk, lds>>>(dA, dB, dsA, dsB, dD, N, K / 64,
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

// ---- hybrid (A-LDS-deep + B-direct) harness: 256x256 KT256 ----
template <int KT, int SWZ>
static void hyb_prep(int M, int N, int K, std::vector<float>& Af, std::vector<float>& Bf,
                     void** dA, void** dB, uint8_t** dsA, uint8_t** dsB, float** dD,
                     int* A_rs, int* Kp) {
    *Kp = ((K + KT - 1) / KT) * KT;
    int Kpv = *Kp;
    std::vector<float> Ap((size_t)M * Kpv, 0.f), Bp((size_t)Kpv * N, 0.f);
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) Ap[(size_t)m * Kpv + k] = Af[(size_t)m * K + k];
    for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) Bp[(size_t)k * N + n] = Bf[(size_t)k * N + n];
    QuantizedMatrix Aq = quantize_to_mxfp6(Ap.data(), M, Kpv);
    QuantizedMatrix Bq = preprocess_B(Bp.data(), Kpv, N);
    PreshuffledB pbB = preshuffle_B(Bq);
    TiledScale saC = tile_scale(preprocess_scale(Aq.scales.data(), M, Kpv), 4, KT / 64);
    TiledScale sbC = tile_scale(preprocess_scale(Bq.scales.data(), N, Kpv), 4, KT / 64);
    hipMalloc(dA, Aq.packed_data.size()); hipMalloc(dB, pbB.data.size());
    hipMalloc(dsA, saC.data.size()); hipMalloc(dsB, sbC.data.size());
    hipMalloc(dD, (size_t)M * N * 4);
    hipMemcpy(*dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(*dB, pbB.data.data(), pbB.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(*dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(*dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    *A_rs = Aq.packed_row_bytes;
}
template <int KT, int SWZ>
static bool hyb_correct(int M, int N, int K) {
    std::mt19937 rng(5); std::uniform_real_distribution<float> d(-2, 2);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng); for (auto& x : Bf) x = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    std::vector<float> Dref((size_t)M * N); mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);
    void *dA, *dB; uint8_t *dsA, *dsB; float* dD; int A_rs, Kp;
    hyb_prep<KT, SWZ>(M, N, K, Af, Bf, &dA, &dB, &dsA, &dsB, &dD, &A_rs, &Kp);
    hipMemset(dD, 0, (size_t)M * N * 4);
    dim3 g(M / 256, N / 256), blk(256); int lds = 2 * 256 * (KT * 6 / 8);
    lds_gemm_hyb<256, 256, KT, 2, 2, SWZ><<<g, blk, lds>>>(dA, dB, dsA, dsB, dD, N, Kp / 64, A_rs);
    if (hipDeviceSynchronize() != hipSuccess) { printf("  hyb err %s\n", hipGetErrorString(hipGetLastError())); return false; }
    std::vector<float> Dg((size_t)M * N); hipMemcpy(Dg.data(), dD, (size_t)M * N * 4, hipMemcpyDeviceToHost);
    float er = 0, mx = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) { er = fmaxf(er, fabsf(Dg[i] - Dref[i])); mx = fmaxf(mx, fabsf(Dref[i])); }
    bool ok = er < 1e-2f * fmaxf(1.f, mx);
    printf("  HYB KT%d%s M=%d N=%d K=%d: err=%.3e %s\n", KT, SWZ ? "[swz]" : "", M, N, K, er, ok ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return ok;
}
template <int KT, int SWZ>
static double hyb_bench(int M, int N, int K) {
    std::mt19937 rng(42); std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng); for (auto& x : Bf) x = d(rng);
    void *dA, *dB; uint8_t *dsA, *dsB; float* dD; int A_rs, Kp;
    hyb_prep<KT, SWZ>(M, N, K, Af, Bf, &dA, &dB, &dsA, &dsB, &dD, &A_rs, &Kp);
    dim3 g(M / 256, N / 256), blk(256); int lds = 2 * 256 * (KT * 6 / 8);
    auto run = [&] { lds_gemm_hyb<256, 256, KT, 2, 2, SWZ><<<g, blk, lds>>>(dA, dB, dsA, dsB, dD, N, Kp / 64, A_rs); };
    for (int i = 0; i < 10; i++) run(); hipDeviceSynchronize();
    double best = 1e30;
    for (int r = 0; r < 4; r++) {
        hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); hipEventRecord(a);
        for (int i = 0; i < 20; i++) run();
        hipEventRecord(b); hipDeviceSynchronize();
        float ms = 0; hipEventElapsedTime(&ms, a, b); hipEventDestroy(a); hipEventDestroy(b);
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
    ok += hyb_correct<256, 0>(512, 512, 1024); tot++;     // hybrid KT256
    ok += hyb_correct<256, 16>(512, 8192, 1024); tot++;   // hybrid KT256 swz16
    printf("%d/%d\n", ok, tot);
#ifndef NOSCALE
    if (ok != tot) return 1;
#endif

    // (3) K-tail (no padding) KT192 perf @8192^3 (padded baseline was 1671).
    // k_iters=128, SUBS=3 -> 42 full KT192 tiles + 2-k64 KT64 tail.
    printf("\n=== (3) K-tail KT192 (no pad) @8192^3 (padded was ~1671) ===\n");
    printf("  256x256 KT192 DB swz16 (K-tail)   : %.0f\n", bench<256, 256, 192, 2, 2, 1, 16, true>(8192, 8192, 8192));
    return 0;
}
