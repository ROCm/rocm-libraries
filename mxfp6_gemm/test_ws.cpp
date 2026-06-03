// ============================================================================
// Warp-Specialized MXFP6 GEMM — producer/consumer experiment (vs LDS deep-K v18).
//
// Bet (HANDOFF Next Step #2): the occ1 18% ceiling = 9% L2-miss latency TAIL that
// a depth-2 double buffer can't hide (hides the mean, not the tail). A DEEPER LDS
// ring (producer runs N buffers ahead) hides the tail. Deep ring needs load/compute
// DECOUPLED = warp specialization.
//
// gfx950 has NO split/named barrier (only whole-WG s_barrier) -> producer/consumer
// handshake via LDS flags (volatile read + threadfence), polled.
//
// HIP allocates registers uniformly across all waves of a block -> a producer wave
// still reserves the consumer's AGPR. 5 waves x 256 AGPR > 1024/CU won't fit. So the
// validation config uses 128x256 (8 pure-AGPR acc/consumer, 5x128=640<=1024). The
// 256x256 16-acc target needs V2 mixed acc (12 AGPR + 4 Arch) to fit 5 waves.
//
// Layout: 5 waves = 4 consumers (split M) + 1 producer. Consumer w owns M-blocks
// [w*M_PW .. ] x all N-blocks. Producer (solo wave) fills the ring; consumers MFMA.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_lds.hpp"  // load_tile_lds / read_op / tile primitives

using namespace mxfp6;

// Single-wave (producer) cooperative load of a ROWS x K_TILE FP6 tile into LDS.
// Same row-major LDS layout as load_tile_lds, but distributed over ONE wave's 64
// lanes (chunk = i*64 + lane) instead of 4 waves. M0 = lds_base + i*64*16.
template <int ROWS, int KT_BYTES, int CPR>
__device__ __forceinline__ void load_tile_solo(char* smem, uint32_t lds_base,
                                               const char* gbase, int row_stride,
                                               int kt_byte, int lane) {
    constexpr int TOTAL = ROWS * CPR;
    static_assert(TOTAL % 64 == 0, "tile chunks must be a multiple of 64 (one wave)");
    constexpr int ISSUES = TOTAL / 64;
#pragma unroll
    for (int i = 0; i < ISSUES; i++) {
        int chunk = i * 64 + lane;
        int m  = chunk / CPR;
        int ck = chunk % CPR;
        const void* g = gbase + (size_t)m * row_stride + kt_byte + ck * 16;
        set_m0(__builtin_amdgcn_readfirstlane(lds_base + (uint32_t)(i * 64 * 16)));
        async_load_lds_b128(smem, g);
    }
}

// Warp-specialized deep-ring GEMM. NC consumers + 1 producer (blockDim = (NC+1)*64).
// Plain (lane-ordered) scales [block][k64][64]; consumers load their own (no glds on
// the consumer -> the scale typed-load's vmcnt only drains scales, no glds conflict).
//
// NPW_A = N-blocks/consumer kept in AGPR (AccTileA); the rest (NB-NPW_A) spill to Arch
// VGPR (AccTileV, V2 mixed-acc trick) so 5 waves fit: HIP allocates AGPR uniformly, so
// to seat 5 waves on 4 SIMDs we need per-wave AGPR <= 1024/5 = 204 (<=12 acc tiles).
// 256x256 needs 16 acc/consumer -> NPW_A=6 (12 AGPR + 4 Arch). 128x256 -> NPW_A=8 (all AGPR).
template <int M_TILE, int N_TILE, int K_TILE, int DEPTH, int NC = 4, int NPW_A = N_TILE / 32,
          int SWZ = 0>
__global__ void __launch_bounds__((NC + 1) * 64, 1)
    lds_gemm_ws(const void* __restrict__ A, const void* __restrict__ B,
                const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                float* __restrict__ D, int N, int k_iters, int A_rs, int B_rs) {
    constexpr int KT_BYTES = K_TILE * 6 / 8;
    constexpr int CPR      = KT_BYTES / 16;
    constexpr int SUBS     = K_TILE / 64;
    constexpr int MB = M_TILE / 32, NB = N_TILE / 32;
    constexpr int NV = NB - NPW_A;                 // N-blocks/consumer in Arch VGPR
    constexpr int M_PW = MB / NC;                 // M-blocks per consumer
    constexpr int A_BYTES = M_TILE * KT_BYTES, B_BYTES = N_TILE * KT_BYTES;
    constexpr int BUF = A_BYTES + B_BYTES;        // one ring slot's bytes
    static_assert(MB % NC == 0, "M-blocks must split evenly across consumers");

    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    bool producer = (wave == NC);

    // Flags live past the ring. fill[s] = highest buffer-seq present in slot s (-1
    // init, monotonic, written by producer lane0). drain[s] = count of consumer
    // completions on slot s (monotonic, atomicAdd by each consumer lane0).
    constexpr uint32_t FLAG_OFF = (uint32_t)DEPTH * BUF;
    volatile int* fill  = reinterpret_cast<volatile int*>(smem + FLAG_OFF);
    volatile int* drain = reinterpret_cast<volatile int*>(smem + FLAG_OFF + DEPTH * 4);

    // L2-aware WG remap (same as lds_gemm_db).
    int wg_m, wg_n;
    if constexpr (SWZ > 0) {
        int mb = gridDim.x, nb = gridDim.y, pid = blockIdx.y * mb + blockIdx.x;
        const int G = SWZ, span = G * mb;
        int grp = pid / span, fn = grp * G, gs = (nb - fn) < G ? (nb - fn) : G, r = pid % span;
        wg_m = r / gs;
        wg_n = fn + r % gs;
    } else {
        wg_m = blockIdx.x;
        wg_n = blockIdx.y;
    }
    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_rs;
    const char* Bg = reinterpret_cast<const char*>(B) + (size_t)(wg_n * N_TILE) * B_rs;

    // One-time flag init (whole-WG barrier allowed once at start).
    for (int s = tid; s < DEPTH; s += blockDim.x) { fill[s] = -1; drain[s] = 0; }
    __syncthreads();

    int k_tiles = k_iters / SUBS;

    if (producer) {
        for (int p = 0; p < k_tiles; p++) {
            int s = p % DEPTH;
            uint32_t base = (uint32_t)s * BUF;
            if (p >= DEPTH) {                       // wait slot's prior occupant drained
                int need = NC * (p / DEPTH);
                while (drain[s] < need) { __builtin_amdgcn_s_sleep(1); }
            }
            int kb = p * KT_BYTES;
            load_tile_solo<M_TILE, KT_BYTES, CPR>(smem, base + 0, Ag, A_rs, kb, lane);
            load_tile_solo<N_TILE, KT_BYTES, CPR>(smem, base + A_BYTES, Bg, B_rs, kb, lane);
            wait_vmcnt(0);                           // data landed in LDS (all lanes)
            __threadfence_block();                   // order data before flag
            if (lane == 0) fill[s] = p;
        }
        return;
    }

    // ---- consumer ----
    int w = wave;                                    // 0..NC-1
    AccTileA acc_a[M_PW][NPW_A];                      // N-blocks 0..NPW_A-1 in AGPR
    AccTileV acc_v[M_PW][NV > 0 ? NV : 1];            // N-blocks NPW_A..NB-1 in Arch VGPR
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++) {
#pragma unroll
        for (int ni = 0; ni < NPW_A; ni++) clear_acc(acc_a[mi][ni]);
#pragma unroll
        for (int ni = 0; ni < NV; ni++) clear_acc(acc_v[mi][ni]);
    }

    for (int c = 0; c < k_tiles; c++) {
        int s = c % DEPTH;
        uint32_t base = (uint32_t)s * BUF;
        while (fill[s] < c) { __builtin_amdgcn_s_sleep(1); }   // wait buffer ready
        __threadfence_block();                                  // order flag before data
#pragma unroll
        for (int sub = 0; sub < SUBS; sub++) {
            int kc = c * SUBS + sub;
            v6i a[M_PW], b[NB];
            int sav[M_PW], sbv[NB];
#pragma unroll
            for (int mi = 0; mi < M_PW; mi++) {
                int blk = w * M_PW + mi;
                a[mi] = read_op<KT_BYTES>(smem, base, blk, sub, lane);
#ifdef NOSCALE
                sav[mi] = 127;
#else
                sav[mi] = sA[(size_t)((wg_m * MB + blk) * k_iters + kc) * 64 + lane];
#endif
            }
#pragma unroll
            for (int ni = 0; ni < NB; ni++) {
                b[ni] = read_op<KT_BYTES>(smem, base + A_BYTES, ni, sub, lane);
#ifdef NOSCALE
                sbv[ni] = 127;
#else
                sbv[ni] = sB[(size_t)((wg_n * NB + ni) * k_iters + kc) * 64 + lane];
#endif
            }
#pragma unroll
            for (int mi = 0; mi < M_PW; mi++) {
#pragma unroll
                for (int ni = 0; ni < NPW_A; ni++)
                    mfma_scale_f32_32x32x64_fp6<0>(acc_a[mi][ni], a[mi], b[ni], sav[mi], sbv[ni]);
#pragma unroll
                for (int ni = 0; ni < NV; ni++)
                    mfma_scale_f32_32x32x64_fp6<0>(acc_v[mi][ni], a[mi], b[NPW_A + ni],
                                                   sav[mi], sbv[NPW_A + ni]);
            }
        }
        __threadfence_block();                                  // MFMA done before freeing
        if (lane == 0) atomicAdd(const_cast<int*>(&drain[s]), 1);
    }

#pragma unroll
    for (int mi = 0; mi < M_PW; mi++) {
        int m = wg_m * M_TILE + (w * M_PW + mi) * 32;
#pragma unroll
        for (int ni = 0; ni < NPW_A; ni++)
            store_acc_f32(D, N, acc_a[mi][ni], m, wg_n * N_TILE + ni * 32);
#pragma unroll
        for (int ni = 0; ni < NV; ni++)
            store_acc_f32(D, N, acc_v[mi][ni], m, wg_n * N_TILE + (NPW_A + ni) * 32);
    }
}

// ---- harness ----
template <int MT, int NT, int KT, int DEPTH, int NC = 4, int NPW_A = NT / 32, int SWZ = 0>
static bool correct(int M, int N, int K) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-2, 2);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);  // plain [blk][k64][64]
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);

    void *dA, *dB;
    uint8_t *dsA, *dsB;
    float* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saP.data.size());
    hipMalloc(&dsB, sbP.data.size());
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
    hipMemset(dD, 0, (size_t)M * N * 4);

    dim3 g(M / MT, N / NT), blk((NC + 1) * 64);
    int lds = DEPTH * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8)) + DEPTH * 8 + 64;
    lds_gemm_ws<MT, NT, KT, DEPTH, NC, NPW_A, SWZ><<<g, blk, lds>>>(
        dA, dB, dsA, dsB, dD, N, K / 64, Aq.packed_row_bytes, Bq.packed_row_bytes);
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
    printf("  WS M=%d N=%d K=%d MT=%d NT=%d KT=%d D=%d: err=%.3e %s\n", M, N, K, MT, NT, KT, DEPTH,
           er, ok ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return ok;
}

template <int MT, int NT, int KT, int DEPTH, int NC = 4, int NPW_A = NT / 32, int SWZ = 0>
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
    void *dA, *dB;
    uint8_t *dsA, *dsB;
    float* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saP.data.size());
    hipMalloc(&dsB, sbP.data.size());
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
    dim3 g(M / MT, N / NT), blk((NC + 1) * 64);
    int lds = DEPTH * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8)) + DEPTH * 8 + 64;
    auto run = [&] {
        lds_gemm_ws<MT, NT, KT, DEPTH, NC, NPW_A, SWZ><<<g, blk, lds>>>(
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

// Baseline lds_gemm_db bench (symmetric 4-wave deep-K) at an arbitrary tile, for the
// equal-intensity comparison. K padded to KT multiple (matches v18 production path).
template <int MT, int NT, int KT, int WM, int WN, int OCC = 1, int SWZ = 0, bool DB = true>
static double lds_db_bench(int M, int N, int K) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
    int Kp = ((K + KT - 1) / KT) * KT;
    std::vector<float> Af((size_t)M * Kp, 0.f), Bf((size_t)Kp * N, 0.f);
    for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) Af[(size_t)i * Kp + k] = d(rng);
    for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = d(rng);
    constexpr int M_PW = (MT / 32) / WM, N_PW = (NT / 32) / WN;
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, Kp);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), Kp, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, Kp);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, Kp);
    TiledScale saC = tile_scale(saP, M_PW, KT / 64), sbC = tile_scale(sbP, N_PW, KT / 64);
    void *dA, *dB; uint8_t *dsA, *dsB; float* dD;
    hipMalloc(&dA, Aq.packed_data.size()); hipMalloc(&dB, Bq.packed_data.size());
    hipMalloc(&dsA, saC.data.size()); hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    dim3 g(M / MT, N / NT), blk(256);
    int lds = (DB ? 2 : 1) * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8));
    auto run = [&] {
        lds_gemm_db<MT, NT, KT, WM, WN, OCC, SWZ, DB><<<g, blk, lds>>>(
            dA, dB, dsA, dsB, dD, N, Kp / 64, Aq.packed_row_bytes, Bq.packed_row_bytes);
    };
    for (int i = 0; i < 10; i++) run();
    hipDeviceSynchronize();
    double best = 1e30;
    for (int r = 0; r < 4; r++) {
        hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
        hipEventRecord(a); for (int i = 0; i < 20; i++) run(); hipEventRecord(b);
        hipDeviceSynchronize();
        float ms = 0; hipEventElapsedTime(&ms, a, b); hipEventDestroy(a); hipEventDestroy(b);
        best = fmin(best, ms / 20.0);
    }
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return 2.0 * M * N * K / (best * 1e-3) / 1e12;
}

int main() {
    printf("=== Warp-spec correctness (8-acc fitting configs) ===\n");
    int ok = 0, tot = 0;
    ok += correct<128, 256, 64, 4>(256, 512, 1024); tot++;
    ok += correct<128, 256, 128, 3>(512, 512, 2048); tot++;
    ok += correct<128, 256, 256, 2>(512, 512, 2048); tot++;
    printf("%d/%d\n", ok, tot);
#ifndef NOSCALE
    if (ok != tot) return 1;
#endif

    // Map the warp-spec ceiling at 8-acc (the max that fits a producer-shared SIMD).
    // Hypothesis check: bigger KT (amortize flag sync) and depth (hide L2-miss tail).
    printf("\n=== Warp-spec 8-acc ceiling sweep @8192^3 (LDS 16-acc baseline 1639) ===\n");
    printf("  WS  128x256 KT192 d2 : %.0f\n", bench<128, 256, 192, 2>(8192, 8192, 8192));
    printf("  WS  128x256 KT256 d2 : %.0f\n", bench<128, 256, 256, 2>(8192, 8192, 8192));

    // DECISIVE equal-intensity comparison: baseline lds_gemm_db ALSO at 128x256 8-acc.
    // Isolates the warp-spec decoupling benefit from the (lower) 8-acc tile intensity.
    printf("\n=== Equal-intensity: WS vs baseline lds_gemm_db, both @128x256 8-acc KT128 ===\n");
    printf("  WS  128x256 KT128 d3 : %.0f\n", bench<128, 256, 128, 3>(8192, 8192, 8192));
    printf("  BL  128x256 KT128 d2 : %.0f\n", lds_db_bench<128, 256, 128, 2, 2>(8192, 8192, 8192));
    printf("  BL  128x256 KT256 d2 : %.0f\n", lds_db_bench<128, 256, 256, 2, 2>(8192, 8192, 8192));
    printf("  BL  256x256 KT192 d2 (16-acc ref): %.0f\n", lds_db_bench<256, 256, 192, 2, 2>(8192, 8192, 8192));
    return 0;
}
