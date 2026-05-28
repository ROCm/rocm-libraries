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

template <int M_TILE = 128, int N_TILE = 128, int N_WAVES = 4>
__global__ void mxfp6_gemm_multitile(
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

    __shared__ uint32_t lds[M_TILE * LDS_ROW_BYTES / 4];
    asm volatile("" : : "r"(lds) : "memory");

    int tid = threadIdx.x;
    int wave_id = tid / 64;
    int lane = tid % 64;
    int wave_m = wave_id / WAVES_N;
    int wave_n = wave_id % WAVES_N;
    int dim = lane % 32;
    int khalf = lane / 32;

    int wg_m = blockIdx.x;
    int wg_n = blockIdx.y;
    int m_tile_base = wg_m * (M_TILE / 32);
    int n_tile_base = wg_n * (N_TILE / 32);

    AccTileA acc00, acc01, acc10, acc11;
    clear_acc(acc00); clear_acc(acc01);
    clear_acc(acc10); clear_acc(acc11);

    const char* A_base = reinterpret_cast<const char*>(A_packed)
                       + wg_m * M_TILE * A_row_stride;

    for (int ki = 0; ki < k_iters; ki++) {
        // ---- A: all 256 threads cooperate to load M_TILE rows to LDS ----
        // Phase 1: rows 0..127, bytes 0-15 (tid<128) and 16-31 (tid>=128)
        {
            int row = (tid < 128) ? tid : (tid - 128);
            int seg = (tid < 128) ? 0 : 16;
            v4i tmp = *reinterpret_cast<const v4i*>(
                A_base + row * A_row_stride + ki * 48 + seg);
            ds_write_b128(static_cast<uint32_t>(row * LDS_ROW_BYTES + seg), tmp);
        }
        // Phase 2: rows 0..127, bytes 32-47
        if (tid < 128) {
            v4i tmp = *reinterpret_cast<const v4i*>(
                A_base + tid * A_row_stride + ki * 48 + 32);
            ds_write_b128(static_cast<uint32_t>(tid * LDS_ROW_BYTES + 32), tmp);
        }
        wait_lgkmcnt(0);
        __syncthreads();

        // ---- B: per-wave VMEM load (N_PER_WAVE tiles) ----
        v8i b_reg[N_PER_WAVE];
        #pragma unroll
        for (int ni = 0; ni < N_PER_WAVE; ni++) {
            int nt = n_tile_base + wave_n * N_PER_WAVE + ni;
            const char* b_tile = reinterpret_cast<const char*>(B_shuffled)
                               + (nt * k_iters + ki) * 1536;
            float4 b0 = *reinterpret_cast<const float4*>(b_tile + lane * 16);
            double b1_raw = *reinterpret_cast<const double*>(b_tile + 1024 + lane * 8);
            int2 b45 = *reinterpret_cast<const int2*>(&b1_raw);
            b_reg[ni] = v8i{__float_as_int(b0.x), __float_as_int(b0.y),
                            __float_as_int(b0.z), __float_as_int(b0.w),
                            b45.x, b45.y, 0, 0};
        }

        // ---- Scales ----
        int sa[M_PER_WAVE], sb[N_PER_WAVE];
        #pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++) {
            int mt = m_tile_base + wave_m * M_PER_WAVE + mi;
            sa[mi] = static_cast<int>(scale_A[(mt * k_iters + ki) * 64 + lane]);
            asm volatile("v_and_b32 %0, 0xFF, %0" : "+v"(sa[mi]));
        }
        #pragma unroll
        for (int ni = 0; ni < N_PER_WAVE; ni++) {
            int nt = n_tile_base + wave_n * N_PER_WAVE + ni;
            sb[ni] = static_cast<int>(scale_B[(nt * k_iters + ki) * 64 + lane]);
            asm volatile("v_and_b32 %0, 0xFF, %0" : "+v"(sb[ni]));
        }

        // ---- MFMA: iterate A tiles, reuse B tiles ----
        {
            int lds_row0 = (wave_m * M_PER_WAVE + 0) * 32 + dim;
            v8i a0 = ds_read_fp6x32(
                static_cast<uint32_t>(lds_row0 * LDS_ROW_BYTES + khalf * 24));
            mfma_scale_f32_32x32x64_fp6<0>(acc00, a0, b_reg[0], sa[0], sb[0]);
            mfma_scale_f32_32x32x64_fp6<0>(acc01, a0, b_reg[1], sa[0], sb[1]);

            int lds_row1 = (wave_m * M_PER_WAVE + 1) * 32 + dim;
            v8i a1 = ds_read_fp6x32(
                static_cast<uint32_t>(lds_row1 * LDS_ROW_BYTES + khalf * 24));
            mfma_scale_f32_32x32x64_fp6<0>(acc10, a1, b_reg[0], sa[1], sb[0]);
            mfma_scale_f32_32x32x64_fp6<0>(acc11, a1, b_reg[1], sa[1], sb[1]);
        }

        __syncthreads();
    }

    // ---- Store ----
    {
        int m0 = wg_m * M_TILE + (wave_m * M_PER_WAVE + 0) * 32;
        int m1 = wg_m * M_TILE + (wave_m * M_PER_WAVE + 1) * 32;
        int n0 = wg_n * N_TILE + (wave_n * N_PER_WAVE + 0) * 32;
        int n1 = wg_n * N_TILE + (wave_n * N_PER_WAVE + 1) * 32;
        store_acc_f32(D, D_stride, acc00, m0, n0);
        store_acc_f32(D, D_stride, acc01, m0, n1);
        store_acc_f32(D, D_stride, acc10, m1, n0);
        store_acc_f32(D, D_stride, acc11, m1, n1);
    }
}

static bool run_test(int M, int N, int K, const float* Af, const float* Bf) {
    printf("  M=%d N=%d K=%d: ", M, N, K);

    QuantizedMatrix Aq = quantize_to_mxfp6(Af, M, K);
    QuantizedMatrix Bq = preprocess_B(Bf, K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    PreshuffledB pbB = preshuffle_B(Bq);

    std::vector<float> Dref(M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);

    int k_iters = K / 64;

    void *dA, *dB; uint8_t *dsA, *dsB; float *dD;
    hipMalloc(&dA, Aq.packed_data.size());
    hipMalloc(&dB, pbB.data.size());
    hipMalloc(&dsA, saP.data.size());
    hipMalloc(&dsB, sbP.data.size());
    hipMalloc(&dD, M * N * sizeof(float));
    hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, pbB.data.data(), pbB.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
    hipMemset(dD, 0, M * N * sizeof(float));

    constexpr int M_TILE = 128, N_TILE = 128, N_WAVES = 4;
    dim3 grid(M / M_TILE, N / N_TILE);
    dim3 block(N_WAVES * 64);

    mxfp6_gemm_multitile<M_TILE, N_TILE, N_WAVES><<<grid, block>>>(
        dA, dB, dsA, dsB, dD, N, k_iters, Aq.packed_row_bytes);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("HIP error: %s\n", hipGetErrorString(err));
        hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
        return false;
    }

    std::vector<float> Dgpu(M * N);
    hipMemcpy(Dgpu.data(), dD, M * N * sizeof(float), hipMemcpyDeviceToHost);

    float maxerr = 0;
    int mis = 0;
    for (int i = 0; i < M * N; i++) {
        float e = fabsf(Dgpu[i] - Dref[i]);
        if (e > maxerr) maxerr = e;
        if (e > 0.5f) mis++;
    }
    bool pass = (maxerr < 0.01f);
    printf("max_err=%.4e mis=%d → %s\n", maxerr, mis, pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("    D[0][0] ref=%.4f gpu=%.4f\n", Dref[0], Dgpu[0]);
        int mid = M/2 * N + N/2;
        printf("    D[%d][%d] ref=%.4f gpu=%.4f\n", M/2, N/2, Dref[mid], Dgpu[mid]);
    }

    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return pass;
}

int main() {
    struct TC { int M, N, K; };
    TC tests[] = {
        {128, 128,  64},
        {128, 128, 256},
        {256, 256, 256},
        {512, 512, 512},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);
    int total = 0, passed = 0;

    printf("=== Constant tests ===\n");
    for (int t = 0; t < ntests; t++) {
        auto [M, N, K] = tests[t];
        std::vector<float> Af(M * K, 1.0f), Bf(K * N, 1.0f);
        total++;
        if (run_test(M, N, K, Af.data(), Bf.data())) passed++;
    }

    printf("\n=== Random tests ===\n");
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int t = 0; t < ntests; t++) {
        auto [M, N, K] = tests[t];
        std::vector<float> Af(M * K), Bf(K * N);
        for (auto& v : Af) v = dist(rng);
        for (auto& v : Bf) v = dist(rng);
        total++;
        if (run_test(M, N, K, Af.data(), Bf.data())) passed++;
    }

    printf("\n=== Multi-tile: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
