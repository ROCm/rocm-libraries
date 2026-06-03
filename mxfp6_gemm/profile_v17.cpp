// Single-dispatch profiling driver for the v17 production kernel.
// Hard-wired to the 8192^3 dispatcher pick: T512, 2x8 waves, depth-1 prefetch,
// L2-aware swizzle SWZ=16 (choose_tile(8192,8192)=T512, choose_swz=16).
//
// Usage: ./profile_v17 [warmup] [repeat]
//   warmup=0 repeat=1  -> exactly ONE mxfp6_gemm_pipeline dispatch (PMC/ATT)
//   warmup=3 repeat=10 -> timed benchmark (reports TFLOPs)
//
// Kernel template + prep copied verbatim from test_pipeline_v17.cpp so the
// compiled ISA is bit-identical to the production path.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_types.hpp"

using namespace mxfp6;

template <int M_TILE, int NPW_A, int NPW_V, int N_WAVES, int MIN_OCC, int WAVES_M = 2,
          int WAVES_N = 2, int SWZ = 0>
__global__ void __launch_bounds__(256, MIN_OCC)
    mxfp6_gemm_pipeline(const void* __restrict__ A_packed, const void* __restrict__ B_shuffled,
                        const uint8_t* __restrict__ scale_A, const uint8_t* __restrict__ scale_B,
                        float* __restrict__ D, int D_stride, int k_iters, int A_row_stride) {
    constexpr int M_PER_WAVE = (M_TILE / 32) / WAVES_M;
    constexpr int NPW = NPW_A + NPW_V;
    constexpr int N_TILE = WAVES_N * NPW * 32;
    constexpr int NPW_V_ALLOC = NPW_V > 0 ? NPW_V : 1;
    int tid = threadIdx.x, wave_id = tid / 64, lane = tid % 64;
    int wave_m = wave_id / WAVES_N, wave_n = wave_id % WAVES_N;
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
    int m_tile_base = wg_m * (M_TILE / 32), n_tile_base = wg_n * (N_TILE / 32);

    AccTileA acc_a[M_PER_WAVE][NPW_A];
    AccTileV acc_v[M_PER_WAVE][NPW_V_ALLOC];
#pragma unroll
    for (int mi = 0; mi < M_PER_WAVE; mi++) {
#pragma unroll
        for (int ni = 0; ni < NPW_A; ni++) clear_acc(acc_a[mi][ni]);
#pragma unroll
        for (int ni = 0; ni < NPW_V; ni++) clear_acc(acc_v[mi][ni]);
    }
    int lane_m = lane & 31, lane_kh = lane >> 5;
    auto load_B = [&](int t) -> v6i {
        const char* p = reinterpret_cast<const char*>(B_shuffled) + t * 1536;
        float4 lo = *reinterpret_cast<const float4*>(p + lane * 16);
        double hr = *reinterpret_cast<const double*>(p + 1024 + lane * 8);
        int2 hi = *reinterpret_cast<const int2*>(&hr);
        return v6i{__float_as_int(lo.x),
                   __float_as_int(lo.y),
                   __float_as_int(lo.z),
                   __float_as_int(lo.w),
                   hi.x,
                   hi.y};
    };
    auto load_A = [&](int m_tile, int ki) -> v6i {
        using v6i_a = int __attribute__((__vector_size__(24), __aligned__(4)));
        const char* a = reinterpret_cast<const char*>(A_packed) +
                        (size_t)(m_tile * 32 + lane_m) * A_row_stride + ki * 48 + lane_kh * 24;
        v6i_a x = *reinterpret_cast<const v6i_a*>(a);
        return v6i{x[0], x[1], x[2], x[3], x[4], x[5]};
    };
    constexpr int SA_PAD = ((M_PER_WAVE + 3) / 4) * 4, SB_PAD = ((NPW + 3) / 4) * 4;
    using sav = int __attribute__((__vector_size__(SA_PAD), __aligned__(4)));
    using sbv = int __attribute__((__vector_size__(SB_PAD), __aligned__(4)));
    int sa_grp = wg_m * WAVES_M + wave_m, sb_grp = wg_n * WAVES_N + wave_n;
    auto ld_a = [&](v6i* a, int ki) {
#pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++) a[mi] = load_A(m_tile_base + wave_m * M_PER_WAVE + mi, ki);
    };
    auto ld_b = [&](v6i* b, int ki) {
#pragma unroll
        for (int ni = 0; ni < NPW; ni++) b[ni] = load_B((n_tile_base + wave_n * NPW + ni) * k_iters + ki);
    };
    auto ld_sa = [&](int* sa, int ki) {
        sav w = *reinterpret_cast<const sav*>(reinterpret_cast<const char*>(scale_A) +
                                              (size_t)((sa_grp * k_iters + ki) * 64 + lane) * SA_PAD);
#pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++) sa[mi] = (w[mi / 4] >> (8 * (mi % 4))) & 0xff;
    };
    auto ld_sb = [&](int* sb, int ki) {
        sbv w = *reinterpret_cast<const sbv*>(reinterpret_cast<const char*>(scale_B) +
                                              (size_t)((sb_grp * k_iters + ki) * 64 + lane) * SB_PAD);
#pragma unroll
        for (int ni = 0; ni < NPW; ni++) sb[ni] = (w[ni / 4] >> (8 * (ni % 4))) & 0xff;
    };
    auto do_mfma = [&](v6i* a, v6i* b, int* sa, int* sb) {
#pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++) {
#pragma unroll
            for (int ni = 0; ni < NPW_A; ni++)
                mfma_scale_f32_32x32x64_fp6<0>(acc_a[mi][ni], a[mi], b[ni], sa[mi], sb[ni]);
#pragma unroll
            for (int ni = 0; ni < NPW_V; ni++)
                mfma_scale_f32_32x32x64_fp6<0>(acc_v[mi][ni], a[mi], b[NPW_A + ni], sa[mi],
                                               sb[NPW_A + ni]);
        }
    };
    if constexpr (MIN_OCC == 1) {
        // EXPERIMENT: compile-time 2x-unrolled PING-PONG (static buffers, no dynamic index).
        // do_mfma(buf0) consumes tiles loaded a full half-iteration earlier, so the compiler
        // ties its vmcnt to buf0's (mostly-complete) loads while buf1's loads overlap the
        // buf0 MFMA cluster — real load/MFMA overlap, with loads kept compiler-managed.
        v6i a0[M_PER_WAVE], b0[NPW], a1[M_PER_WAVE], b1[NPW];
        int sa0[M_PER_WAVE], sb0[NPW], sa1[M_PER_WAVE], sb1[NPW];
        ld_a(a0, 0); ld_b(b0, 0); ld_sa(sa0, 0); ld_sb(sb0, 0);  // prologue: buf0 = ki0
        int ki = 0;
        for (; ki + 1 < k_iters; ki += 2) {
            ld_a(a1, ki + 1); ld_b(b1, ki + 1); ld_sa(sa1, ki + 1); ld_sb(sb1, ki + 1);  // issue buf1
            do_mfma(a0, b0, sa0, sb0);                                                    // MFMA buf0
            if (ki + 2 < k_iters) {
                ld_a(a0, ki + 2); ld_b(b0, ki + 2); ld_sa(sa0, ki + 2); ld_sb(sb0, ki + 2);  // issue buf0
            }
            do_mfma(a1, b1, sa1, sb1);                                                    // MFMA buf1
        }
        if (ki < k_iters) do_mfma(a0, b0, sa0, sb0);  // odd-k tail (buf0 already loaded)
    } else {
        for (int ki = 0; ki < k_iters; ki++) {
            v6i ac[M_PER_WAVE], bc[NPW]; int sac[M_PER_WAVE], sbc[NPW];
            ld_a(ac, ki); ld_b(bc, ki); ld_sa(sac, ki); ld_sb(sbc, ki);
            do_mfma(ac, bc, sac, sbc);
        }
    }
#pragma unroll
    for (int mi = 0; mi < M_PER_WAVE; mi++) {
        int m = wg_m * M_TILE + (wave_m * M_PER_WAVE + mi) * 32;
#pragma unroll
        for (int ni = 0; ni < NPW_A; ni++) {
            int n = wg_n * N_TILE + (wave_n * NPW + ni) * 32;
            store_acc_f32(D, D_stride, acc_a[mi][ni], m, n);
        }
#pragma unroll
        for (int ni = 0; ni < NPW_V; ni++) {
            int n = wg_n * N_TILE + (wave_n * NPW + NPW_A + ni) * 32;
            store_acc_f32(D, D_stride, acc_v[mi][ni], m, n);
        }
    }
}

// T512 + 2x8 + SWZ=16 launch (the 8192^3 dispatcher pick).
static void launch(int M, int N, int K, const void* dA, const void* dB, const uint8_t* dsA,
                   const uint8_t* dsB, float* dD, int prb) {
    dim3 block(256);
    int kit = K / 64;
    dim3 g(M / 128, N / 512);
    mxfp6_gemm_pipeline<128, 8, 0, 4, 1, 2, 2, 16><<<g, block>>>(dA, dB, dsA, dsB, dD, N, kit, prb);
}

int main(int argc, char** argv) {
    int warmup = argc > 1 ? atoi(argv[1]) : 0;
    int repeat = argc > 2 ? atoi(argv[2]) : 1;
    const int M = 8192, N = 8192, K = 8192;
    const int mpw = 2, npw = 8;  // shape_of(T512, square=false)

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);

    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    CoalescedScale saC = preshuffle_scale(saP, mpw);
    CoalescedScale sbC = preshuffle_scale(sbP, npw);
    PreshuffledB pbB = preshuffle_B(Bq);

    void *dA, *dB;
    uint8_t *dsA, *dsB;
    float* dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, pbB.data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, pbB.data.data(), pbB.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    int prb = Aq.packed_row_bytes;

    for (int i = 0; i < warmup; i++) launch(M, N, K, dA, dB, dsA, dsB, dD, prb);
    hipDeviceSynchronize();

    if (repeat <= 1) {
        // Single-dispatch mode for PMC/ATT: exactly one kernel launch.
        launch(M, N, K, dA, dB, dsA, dsB, dD, prb);
        hipDeviceSynchronize();
        hipError_t e = hipGetLastError();
        if (e != hipSuccess) { printf("err %s\n", hipGetErrorString(e)); return 1; }
        printf("single dispatch done (M=%d N=%d K=%d)\n", M, N, K);
    } else {
        double best = 1e30;
        for (int r = 0; r < 4; r++) {
            hipEvent_t a, b;
            hipEventCreate(&a);
            hipEventCreate(&b);
            hipEventRecord(a);
            for (int i = 0; i < repeat; i++) launch(M, N, K, dA, dB, dsA, dsB, dD, prb);
            hipEventRecord(b);
            hipDeviceSynchronize();
            float ms = 0;
            hipEventElapsedTime(&ms, a, b);
            hipEventDestroy(a);
            hipEventDestroy(b);
            best = fmin(best, ms / repeat);
        }
        double tflops = 2.0 * M * N * K / (best * 1e-3) / 1e12;
        printf("M=%d N=%d K=%d: %.3f ms  %.0f TFLOPs\n", M, N, K, best, tflops);
    }
    hipFree(dA);
    hipFree(dB);
    hipFree(dsA);
    hipFree(dsB);
    hipFree(dD);
    return 0;
}
