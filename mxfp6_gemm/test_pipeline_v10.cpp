#include "mxfp6_asm_utils.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace mxfp6;

static constexpr int LDS_ROW_BYTES = 52;

// v10: v5's direct raw-A load (valid for runtime activation) + BIG register tile.
// A's uncoalesced load is unavoidable for runtime row-major A, so amortize it:
// each wave does M_PER_WAVE×N_PER_WAVE MFMAs/step (default 2×4=8), reusing each
// loaded A tile across N_PER_WAVE B tiles → A traffic per FLOP halves vs v5.
// 8 acc tiles = 128 AGPR → occupancy ~2 (fine; bigger tile amortizes load cost).
template <int M_TILE = 128, int N_TILE = 256, int N_WAVES = 4>
__global__ void __launch_bounds__(256, 2) mxfp6_gemm_pipeline(
    const void* __restrict__ A_packed,
    const void* __restrict__ B_shuffled,
    const uint8_t* __restrict__ scale_A,
    const uint8_t* __restrict__ scale_B,
    float* __restrict__ D, int D_stride,
    int k_iters, int A_row_stride)
{
    constexpr int WAVES_M = 2, WAVES_N = 2;
    constexpr int M_PER_WAVE = (M_TILE / 32) / WAVES_M;
    constexpr int N_PER_WAVE = (N_TILE / 32) / WAVES_N;

    int tid = threadIdx.x;
    int wave_id = tid / 64;
    int lane = tid % 64;
    int wave_m = wave_id / WAVES_N;
    int wave_n = wave_id % WAVES_N;

    int wg_m = blockIdx.x;
    int wg_n = blockIdx.y;
    int m_tile_base = wg_m * (M_TILE / 32);
    int n_tile_base = wg_n * (N_TILE / 32);

    AccTileA acc[M_PER_WAVE][N_PER_WAVE];
    #pragma unroll
    for (int mi = 0; mi < M_PER_WAVE; mi++)
        #pragma unroll
        for (int ni = 0; ni < N_PER_WAVE; ni++) clear_acc(acc[mi][ni]);

    int lane_m = lane & 31;     // M-row within tile
    int lane_kh = lane >> 5;    // K-half (0/1)

    // B: pre-shuffled tile (1536B). section0 = lane*16, section1 = 1024+lane*8.
    auto load_B = [&](int tile_idx) -> v6i {
        const char* t = reinterpret_cast<const char*>(B_shuffled) + tile_idx * 1536;
        float4 lo = *reinterpret_cast<const float4*>(t + lane * 16);
        double hi_raw = *reinterpret_cast<const double*>(t + 1024 + lane * 8);
        int2 hi = *reinterpret_cast<const int2*>(&hi_raw);
        return v6i{__float_as_int(lo.x), __float_as_int(lo.y),
                   __float_as_int(lo.z), __float_as_int(lo.w), hi.x, hi.y};
    };
    // A: raw row-major [M][K]. lane reads 24B at row (m_tile*32+lane_m), K-half.
    auto load_A = [&](int m_tile, int ki) -> v6i {
        using v6i_a = int __attribute__((__vector_size__(24), __aligned__(4)));
        const char* a = reinterpret_cast<const char*>(A_packed)
                      + (size_t)(m_tile * 32 + lane_m) * A_row_stride
                      + ki * 48 + lane_kh * 24;
        v6i_a x = *reinterpret_cast<const v6i_a*>(a);
        return v6i{x[0], x[1], x[2], x[3], x[4], x[5]};
    };

    for (int ki = 0; ki < k_iters; ki++) {
        v6i a_reg[M_PER_WAVE], b_reg[N_PER_WAVE];
        #pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++)
            a_reg[mi] = load_A(m_tile_base + wave_m * M_PER_WAVE + mi, ki);
        #pragma unroll
        for (int ni = 0; ni < N_PER_WAVE; ni++)
            b_reg[ni] = load_B((n_tile_base + wave_n * N_PER_WAVE + ni) * k_iters + ki);

        int sa[M_PER_WAVE], sb[N_PER_WAVE];
        #pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++)
            sa[mi] = static_cast<int>(scale_A[((m_tile_base + wave_m * M_PER_WAVE + mi) * k_iters + ki) * 64 + lane]);
        #pragma unroll
        for (int ni = 0; ni < N_PER_WAVE; ni++)
            sb[ni] = static_cast<int>(scale_B[((n_tile_base + wave_n * N_PER_WAVE + ni) * k_iters + ki) * 64 + lane]);

        #pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++)
            #pragma unroll
            for (int ni = 0; ni < N_PER_WAVE; ni++)
                mfma_scale_f32_32x32x64_fp6<0>(acc[mi][ni], a_reg[mi], b_reg[ni], sa[mi], sb[ni]);
    }

    // ---- Store ----
    #pragma unroll
    for (int mi = 0; mi < M_PER_WAVE; mi++) {
        int m = wg_m * M_TILE + (wave_m * M_PER_WAVE + mi) * 32;
        #pragma unroll
        for (int ni = 0; ni < N_PER_WAVE; ni++) {
            int n = wg_n * N_TILE + (wave_n * N_PER_WAVE + ni) * 32;
            store_acc_f32(D, D_stride, acc[mi][ni], m, n);
        }
    }
}

// ---- Correctness test ----

static bool run_test(int M, int N, int K, const float* Af, const float* Bf) {
    printf("  M=%d N=%d K=%d: ", M, N, K);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af, M, K);
    QuantizedMatrix Bq = preprocess_B(Bf, K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    PreshuffledB pbB = preshuffle_B(Bq);
    std::vector<float> Dref(M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);

    void *dA, *dB; uint8_t *dsA, *dsB; float *dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, pbB.data.size());
    hipMalloc(&dsA, saP.data.size());
    hipMalloc(&dsB, sbP.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(float));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, pbB.data.data(), pbB.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
    hipMemset(dD, 0, (size_t)M * N * sizeof(float));

    constexpr int M_TILE = 128, N_TILE = 256, N_WAVES = 4;
    dim3 grid(M / M_TILE, N / N_TILE);
    dim3 block(N_WAVES * 64);
    mxfp6_gemm_pipeline<M_TILE, N_TILE, N_WAVES><<<grid, block>>>(
        dA, dB, dsA, dsB, dD, N, K/64, Aq.packed_row_bytes);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("HIP error: %s\n", hipGetErrorString(err));
        hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
        return false;
    }

    std::vector<float> Dgpu(M * N);
    hipMemcpy(Dgpu.data(), dD, (size_t)M * N * sizeof(float), hipMemcpyDeviceToHost);
    float maxerr = 0;
    for (int i = 0; i < M * N; i++)
        maxerr = fmaxf(maxerr, fabsf(Dgpu[i] - Dref[i]));
    bool pass = (maxerr < 0.01f);
    printf("max_err=%.4e → %s\n", maxerr, pass ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return pass;
}

// ---- Benchmark ----

static void run_bench(int M, int N, int K) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> Af(M * K), Bf(K * N);
    for (auto& v : Af) v = dist(rng);
    for (auto& v : Bf) v = dist(rng);

    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    PreshuffledB pbB = preshuffle_B(Bq);

    void *dA, *dB; uint8_t *dsA, *dsB; float *dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, pbB.data.size());
    hipMalloc(&dsA, saP.data.size());
    hipMalloc(&dsB, sbP.data.size());
    hipMalloc(&dD, (size_t)M * N * sizeof(float));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, pbB.data.data(), pbB.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);

    constexpr int M_TILE = 128, N_TILE = 256, N_WAVES = 4;
    dim3 grid(M / M_TILE, N / N_TILE);
    dim3 block(N_WAVES * 64);
    int k_iters = K / 64;

    for (int i = 0; i < 5; i++)
        mxfp6_gemm_pipeline<M_TILE, N_TILE, N_WAVES><<<grid, block>>>(
            dA, dB, dsA, dsB, dD, N, k_iters, Aq.packed_row_bytes);
    hipDeviceSynchronize();

    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0);
    for (int i = 0; i < 20; i++)
        mxfp6_gemm_pipeline<M_TILE, N_TILE, N_WAVES><<<grid, block>>>(
            dA, dB, dsA, dsB, dD, N, k_iters, Aq.packed_row_bytes);
    hipEventRecord(t1);
    hipDeviceSynchronize();

    float ms = 0;
    hipEventElapsedTime(&ms, t0, t1);
    float avg = ms / 20;
    double tflops = 2.0 * M * N * K / (avg * 1e-3) / 1e12;
    printf("M=%5d N=%5d K=%5d | %.3f ms | %.1f TFLOPS\n", M, N, K, avg, tflops);

    hipEventDestroy(t0); hipEventDestroy(t1);
    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
}

int main() {
    // ---- Correctness ----
    printf("=== Correctness ===\n");
    int total = 0, passed = 0;
    struct TC { int M, N, K; };
    TC tests[] = {{128,256,64},{128,256,256},{256,256,256},{512,512,512}};
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto [M, N, K] : tests) {
        std::vector<float> Af(M*K), Bf(K*N);
        for (auto& v : Af) v = dist(rng);
        for (auto& v : Bf) v = dist(rng);
        total++;
        if (run_test(M, N, K, Af.data(), Bf.data())) passed++;
    }
    printf("%d/%d passed\n", passed, total);
    if (passed != total) return 1;

    // ---- Benchmark ----
    printf("\n=== Benchmark (pipeline) ===\n");
    TC benches[] = {
        {2048,4096,4096},{4096,4096,4096},{8192,4096,4096},
        {4096,8192,4096},{4096,4096,8192},{8192,8192,8192},
    };
    for (auto [M, N, K] : benches)
        run_bench(M, N, K);

    return 0;
}
