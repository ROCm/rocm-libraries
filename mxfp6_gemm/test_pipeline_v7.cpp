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

// v7: A is a RUNTIME activation (row-major, no preprocessing). To get COALESCED
// global loads despite A's row-major layout, cooperatively load a wide K=256 slab
// into LDS: each row's 256-K run = 192 bytes = exactly 3 cache lines, loaded by a
// contiguous thread group (100% coalesced). LDS then transposes it for the MFMA
// ds_read. K=64's 48-byte run can't tile a 64B cache line — only a 4-step slab can.
// B stays pre-shuffled (weights). Single LDS buffer → occupancy stays 4.
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
    constexpr int KSLAB = 4;                  // K64 steps per LDS slab (K=256)
    constexpr int LROW = 49 * 4;              // LDS row bytes: 49 dw, gcd(49,64)=1
    constexpr int LBUF = M_TILE * LROW;       // one slab buffer

    __shared__ uint32_t lds[LBUF / 4];
    asm volatile("" : : "r"(lds) : "memory");

    int tid = threadIdx.x;
    int wave_id = tid / 64;
    int lane = tid % 64;
    int wave_m = wave_id / WAVES_N;
    int wave_n = wave_id % WAVES_N;
    int dim = lane & 31;
    int khalf = lane >> 5;

    int wg_m = blockIdx.x;
    int wg_n = blockIdx.y;
    int m_tile_base = wg_m * (M_TILE / 32);
    int n_tile_base = wg_n * (N_TILE / 32);

    AccTileA acc00, acc01, acc10, acc11;
    clear_acc(acc00); clear_acc(acc01);
    clear_acc(acc10); clear_acc(acc11);

    const char* A_base = reinterpret_cast<const char*>(A_packed)
                       + (size_t)wg_m * M_TILE * A_row_stride;

    // B: pre-shuffled tile (1536B). section0 = lane*16, section1 = 1024+lane*8.
    auto load_B = [&](int tile_idx) -> v6i {
        const char* t = reinterpret_cast<const char*>(B_shuffled) + tile_idx * 1536;
        float4 lo = *reinterpret_cast<const float4*>(t + lane * 16);
        double hi_raw = *reinterpret_cast<const double*>(t + 1024 + lane * 8);
        int2 hi = *reinterpret_cast<const int2*>(&hi_raw);
        return v6i{__float_as_int(lo.x), __float_as_int(lo.y),
                   __float_as_int(lo.z), __float_as_int(lo.w), hi.x, hi.y};
    };

    // Coalesced cooperative load of a K-slab (cur_steps×48 B/row) into LDS.
    // 16B chunks; consecutive threads read consecutive 16B within a row's run.
    auto load_slab = [&](int slab, int cur_steps) {
        using v4i_a = int __attribute__((__vector_size__(16), __aligned__(4)));
        int cpr = cur_steps * 3;            // 16B chunks per row
        int total = M_TILE * cpr;
        int koff = slab * (KSLAB * 48);     // byte offset of slab within a row
        for (int c = tid; c < total; c += 256) {
            int row = c / cpr;
            int cir = c % cpr;
            const char* g = A_base + (size_t)row * A_row_stride + koff + cir * 16;
            char* l = reinterpret_cast<char*>(lds) + row * LROW + cir * 16;
            *reinterpret_cast<v4i_a*>(l) = *reinterpret_cast<const v4i_a*>(g);
        }
    };

    int n_slabs = (k_iters + KSLAB - 1) / KSLAB;
    for (int slab = 0; slab < n_slabs; slab++) {
        int cur_steps = k_iters - slab * KSLAB;
        if (cur_steps > KSLAB) cur_steps = KSLAB;

        if (slab > 0) __syncthreads();      // WAR: prev slab's ds_read done
        load_slab(slab, cur_steps);
        __syncthreads();                    // RAW: slab visible before ds_read

        for (int s = 0; s < cur_steps; s++) {
            int ki = slab * KSLAB + s;
            uint32_t off0 = ((wave_m * M_PER_WAVE + 0) * 32 + dim) * LROW + s * 48 + khalf * 24;
            uint32_t off1 = ((wave_m * M_PER_WAVE + 1) * 32 + dim) * LROW + s * 48 + khalf * 24;
            v6i a0 = ds_read_fp6x32_plain(lds, off0);
            v6i a1 = ds_read_fp6x32_plain(lds, off1);

            v6i b_reg[N_PER_WAVE];
            #pragma unroll
            for (int ni = 0; ni < N_PER_WAVE; ni++) {
                int nt = n_tile_base + wave_n * N_PER_WAVE + ni;
                b_reg[ni] = load_B(nt * k_iters + ki);
            }
            int sa[M_PER_WAVE], sb[N_PER_WAVE];
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

            mfma_scale_f32_32x32x64_fp6<0>(acc00, a0, b_reg[0], sa[0], sb[0]);
            mfma_scale_f32_32x32x64_fp6<0>(acc01, a0, b_reg[1], sa[0], sb[1]);
            mfma_scale_f32_32x32x64_fp6<0>(acc10, a1, b_reg[0], sa[1], sb[0]);
            mfma_scale_f32_32x32x64_fp6<0>(acc11, a1, b_reg[1], sa[1], sb[1]);
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
