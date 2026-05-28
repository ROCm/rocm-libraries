#include "mxfp6_asm_utils.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace mxfp6;

// LDS padding: 52 bytes per row (48 data + 4 pad), stride=13 DWORDs, gcd(13,64)=1
static constexpr int LDS_ROW_BYTES = 52;
static constexpr int LDS_TOTAL_BYTES = 32 * LDS_ROW_BYTES; // 1664

// K-loop GEMM: M=32, N=32, K=any (multiple of 64)
// A is loaded directly from row-major packed data (no host-side retiling).
__global__ void gemm_k_loop(
    const void* __restrict__ A_packed,      // [32][packed_row_bytes], row-major FP6
    const void* __restrict__ B_shuffled,    // [K/64 * 1536] pre-shuffled B tiles
    const uint8_t* __restrict__ scale_A,    // [K/64 * 64] lane-ordered
    const uint8_t* __restrict__ scale_B,    // [K/64 * 64] lane-ordered
    float* __restrict__ D, int D_stride,
    int k_iters, int A_row_stride)          // k_iters=K/64, A_row_stride=fp6_packed_bytes(K)
{
    __shared__ uint32_t lds[LDS_TOTAL_BYTES / 4];
    asm volatile("" : : "r"(lds) : "memory");
    int tid = threadIdx.x;

    AccTile acc;
    clear_acc(acc);

    const char* A_base = reinterpret_cast<const char*>(A_packed);

    for (int ki = 0; ki < k_iters; ki++) {
        // ---- A: strided HBM → VGPR → LDS (with padding) ----
        // Phase 1: 64 threads load rows 0-31, bytes 0-15 and 16-31
        int row = tid % 32;
        int seg = tid / 32;  // 0 or 1
        int hbm_off = row * A_row_stride + ki * 48 + seg * 16;
        v4i tmp = *reinterpret_cast<const v4i*>(A_base + hbm_off);
        uint32_t lds_off = row * LDS_ROW_BYTES + seg * 16;
        ds_write_b128(lds_off, tmp);

        // Phase 2: 32 threads load rows 0-31, bytes 32-47
        if (tid < 32) {
            hbm_off = tid * A_row_stride + ki * 48 + 32;
            tmp = *reinterpret_cast<const v4i*>(A_base + hbm_off);
            lds_off = tid * LDS_ROW_BYTES + 32;
            ds_write_b128(lds_off, tmp);
        }
        wait_lgkmcnt(0);
        __syncthreads();

        // ds_read: 24 bytes per half-K per dimension
        int dim   = tid % 32;
        int khalf = tid / 32;
        v8i a_reg = ds_read_fp6x32(static_cast<uint32_t>(dim * LDS_ROW_BYTES + khalf * 24));

        // ---- B: pre-shuffled VMEM load ----
        const char* b_tile = reinterpret_cast<const char*>(B_shuffled) + ki * 1536;
        const float4* b_s0 = reinterpret_cast<const float4*>(b_tile + tid * 16);
        const double* b_s1 = reinterpret_cast<const double*>(b_tile + 1024 + tid * 8);

        float4 b_load0 = *b_s0;
        double b_load1_raw = *b_s1;

        v8i b_reg;
        b_reg[0] = __float_as_int(b_load0.x);
        b_reg[1] = __float_as_int(b_load0.y);
        b_reg[2] = __float_as_int(b_load0.z);
        b_reg[3] = __float_as_int(b_load0.w);
        int2 b_dw45 = *reinterpret_cast<const int2*>(&b_load1_raw);
        b_reg[4] = b_dw45.x;
        b_reg[5] = b_dw45.y;
        b_reg[6] = 0;
        b_reg[7] = 0;

        // ---- Scales ----
        int sa = static_cast<int>(scale_A[ki * 64 + tid]);
        int sb = static_cast<int>(scale_B[ki * 64 + tid]);
        asm volatile("v_and_b32 %0, 0xFF, %0" : "+v"(sa));
        asm volatile("v_and_b32 %0, 0xFF, %0" : "+v"(sb));

        // ---- MFMA accumulate ----
        mfma_scale_f32_32x32x64_fp6<0>(acc, a_reg, b_reg, sa, sb);
    }

    // ---- Store ----
    store_acc_f32(D, D_stride, acc, 0, 0);
}

static bool run_test(const char* label, int M, int N, int K,
                     const float* Af, const float* Bf) {
    printf("  K=%d: ", K);

    QuantizedMatrix Aq = quantize_to_mxfp6(Af, M, K);
    QuantizedMatrix Bq = preprocess_B(Bf, K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    PreshuffledB pbB = preshuffle_B(Bq);

    std::vector<float> Dref(M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);

    int k_iters = K / 64;
    int A_row_stride = Aq.packed_row_bytes;

    void *dA, *dB;
    uint8_t *dsA, *dsB;
    float *dD;
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

    gemm_k_loop<<<1, 64>>>(dA, dB, dsA, dsB, dD, N, k_iters, A_row_stride);
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
    printf("max_err=%.4e mismatches=%d → %s\n", maxerr, mis, pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("    D[0][0] ref=%.4f gpu=%.4f\n", Dref[0], Dgpu[0]);
        printf("    D[15][17] ref=%.4f gpu=%.4f\n", Dref[15*N+17], Dgpu[15*N+17]);
        printf("    D[31][31] ref=%.4f gpu=%.4f\n", Dref[31*N+31], Dgpu[31*N+31]);
    }

    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    return pass;
}

int main() {
    const int M = 32, N = 32;
    int test_Ks[] = {64, 128, 256, 512};
    int num_tests = sizeof(test_Ks) / sizeof(test_Ks[0]);
    int total = 0, passed = 0;

    // ---- Constant tests ----
    printf("=== Constant tests (all 1.0) ===\n");
    for (int t = 0; t < num_tests; t++) {
        int K = test_Ks[t];
        std::vector<float> Af(M * K, 1.0f), Bf(K * N, 1.0f);
        total++;
        if (run_test("const", M, N, K, Af.data(), Bf.data())) passed++;
    }

    // ---- Random tests ----
    printf("\n=== Random tests ===\n");
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int t = 0; t < num_tests; t++) {
        int K = test_Ks[t];
        std::vector<float> Af(M * K), Bf(K * N);
        for (auto& v : Af) v = dist(rng);
        for (auto& v : Bf) v = dist(rng);
        total++;
        if (run_test("rand", M, N, K, Af.data(), Bf.data())) passed++;
    }

    printf("\n=== Summary: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
