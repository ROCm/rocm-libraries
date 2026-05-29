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
__global__ void mxfp6_gemm_pipeline(
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
    constexpr int LDS_BUF = M_TILE * LDS_ROW_BYTES;

    __shared__ uint32_t lds[2 * LDS_BUF / 4];
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

    // ---- Helpers (register prefetch of B + scales for iteration `ki`) ----
    auto load_B = [&](int ki, v6i (&b_reg)[N_PER_WAVE]) {
        #pragma unroll
        for (int ni = 0; ni < N_PER_WAVE; ni++) {
            int nt = n_tile_base + wave_n * N_PER_WAVE + ni;
            const char* b_tile = reinterpret_cast<const char*>(B_shuffled)
                               + (nt * k_iters + ki) * 1536;
            float4 b0 = *reinterpret_cast<const float4*>(b_tile + lane * 16);
            double b1_raw = *reinterpret_cast<const double*>(b_tile + 1024 + lane * 8);
            int2 b45 = *reinterpret_cast<const int2*>(&b1_raw);
            b_reg[ni] = v6i{__float_as_int(b0.x), __float_as_int(b0.y),
                            __float_as_int(b0.z), __float_as_int(b0.w),
                            b45.x, b45.y};
        }
    };
    auto load_scale = [&](int ki, int (&sa)[M_PER_WAVE], int (&sb)[N_PER_WAVE]) {
        #pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++) {
            int mt = m_tile_base + wave_m * M_PER_WAVE + mi;
            sa[mi] = static_cast<int>(scale_A[(mt * k_iters + ki) * 64 + lane]);
        }
        #pragma unroll
        for (int ni = 0; ni < N_PER_WAVE; ni++) {
            int nt = n_tile_base + wave_n * N_PER_WAVE + ni;
            sb[ni] = static_cast<int>(scale_B[(nt * k_iters + ki) * 64 + lane]);
        }
    };
    // Cooperative global→LDS write of A tile for iteration `ki` into buffer `buf`.
    auto write_A = [&](int ki, int buf) {
        int row = (tid < 128) ? tid : (tid - 128);
        int seg = (tid < 128) ? 0 : 16;
        v4i tmp = *reinterpret_cast<const v4i*>(
            A_base + row * A_row_stride + ki * 48 + seg);
        ds_write_b128_plain(lds, static_cast<uint32_t>(buf + row * LDS_ROW_BYTES + seg), tmp);
        if (tid < 128) {
            tmp = *reinterpret_cast<const v4i*>(
                A_base + tid * A_row_stride + ki * 48 + 32);
            ds_write_b128_plain(lds, static_cast<uint32_t>(buf + tid * LDS_ROW_BYTES + 32), tmp);
        }
    };

    // ---- Prologue: stage A[0] to LDS, prefetch B[0]+scale[0] to registers ----
    write_A(0, 0);
    v6i b_cur[N_PER_WAVE];
    int sa_cur[M_PER_WAVE], sb_cur[N_PER_WAVE];
    load_B(0, b_cur);
    load_scale(0, sa_cur, sb_cur);
    __syncthreads();   // A[0] visible; compiler fences lgkmcnt for the LDS writes

    for (int ki = 0; ki < k_iters; ki++) {
        int cur_buf = (ki & 1) * LDS_BUF;

        // ---- Read this iter's A from LDS (needs the barrier above) ----
        uint32_t off0 = cur_buf
            + ((wave_m * M_PER_WAVE + 0) * 32 + dim) * LDS_ROW_BYTES + khalf * 24;
        uint32_t off1 = cur_buf
            + ((wave_m * M_PER_WAVE + 1) * 32 + dim) * LDS_ROW_BYTES + khalf * 24;
        v6i a0 = ds_read_fp6x32_plain(lds, off0);
        v6i a1 = ds_read_fp6x32_plain(lds, off1);

        // ---- Prefetch NEXT iter: B+scale to regs, A to other LDS buffer ----
        // B/scale are global → independent of the barrier, so issuing them here
        // lets their latency overlap this iter's MFMA + the barrier.
        v6i b_next[N_PER_WAVE];
        int sa_next[M_PER_WAVE], sb_next[N_PER_WAVE];
        if (ki < k_iters - 1) {
            load_B(ki + 1, b_next);
            load_scale(ki + 1, sa_next, sb_next);
            write_A(ki + 1, (1 - (ki & 1)) * LDS_BUF);
        }

        // ---- MFMA on current (B already loaded last iter) ----
        mfma_scale_f32_32x32x64_fp6<0>(acc00, a0, b_cur[0], sa_cur[0], sb_cur[0]);
        mfma_scale_f32_32x32x64_fp6<0>(acc01, a0, b_cur[1], sa_cur[0], sb_cur[1]);
        mfma_scale_f32_32x32x64_fp6<0>(acc10, a1, b_cur[0], sa_cur[1], sb_cur[0]);
        mfma_scale_f32_32x32x64_fp6<0>(acc11, a1, b_cur[1], sa_cur[1], sb_cur[1]);

        // ---- Rotate register buffers ----
        #pragma unroll
        for (int ni = 0; ni < N_PER_WAVE; ni++) { b_cur[ni] = b_next[ni]; sb_cur[ni] = sb_next[ni]; }
        #pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++) { sa_cur[mi] = sa_next[mi]; }

        // ---- Barrier: make next A buffer visible (cross-wave dep) ----
        if (ki < k_iters - 1) {
            __syncthreads();
        }
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

    constexpr int M_TILE = 128, N_TILE = 128, N_WAVES = 4;
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

    constexpr int M_TILE = 128, N_TILE = 128, N_WAVES = 4;
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
    TC tests[] = {{128,128,64},{128,128,256},{256,256,256},{512,512,512}};
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
