#include "mxfp6_asm_utils.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

using namespace mxfp6;

// Single MFMA correctness test: M=N=32, K=64.
// A loaded via async HBM→LDS→VGPR path.
// B loaded the same way (⚠️ B layout not yet correct — see HANDOFF.md).
__global__ void single_mfma_kernel(
    const void* __restrict__ A_packed,
    const void* __restrict__ B_packed,
    const uint8_t* __restrict__ scale_A,
    const uint8_t* __restrict__ scale_B,
    float* __restrict__ D, int N_stride)
{
    __shared__ uint32_t lds[1024]; // 4096 bytes
    void* smem = (void*)lds;
    asm volatile("" : : "r"(smem) : "memory");

    int tid = threadIdx.x;
    int lane = tid & 0xF;
    int group = (tid >> 4) & 3;
    int row = lane + ((group & 1) << 4);
    int k_half = (group >> 1) & 1;

    // ---- Load A[32][48 bytes] into LDS[0..1535] ----
    const char* a_src = reinterpret_cast<const char*>(A_packed);
    set_m0(0);
    async_load_lds_b128(smem, a_src + tid * 16);
    wait_vmcnt(0);
    __syncthreads();
    if (tid < 32) {
        set_m0(1024);
        async_load_lds_b128(smem, a_src + 1024 + tid * 16);
    }
    wait_vmcnt(0);
    __syncthreads();

    v8i a_reg = ds_read_fp6x32(static_cast<uint32_t>(row * 48 + k_half * 24));

    // ---- Load B[32][48 bytes] into LDS[2048..3583] ----
    const char* b_src = reinterpret_cast<const char*>(B_packed);
    __syncthreads();
    set_m0(2048);
    async_load_lds_b128(smem, b_src + tid * 16);
    wait_vmcnt(0);
    __syncthreads();
    if (tid < 32) {
        set_m0(2048 + 1024);
        async_load_lds_b128(smem, b_src + 1024 + tid * 16);
    }
    wait_vmcnt(0);
    __syncthreads();

    v8i b_reg = ds_read_fp6x32(2048 + static_cast<uint32_t>(row * 48 + k_half * 24));

    // ---- Scales ----
    int sa = static_cast<int>(scale_A[tid]);
    int sb = static_cast<int>(scale_B[tid]);
    asm volatile("v_and_b32 %0, 0xFF, %0" : "+v"(sa));
    asm volatile("v_and_b32 %0, 0xFF, %0" : "+v"(sb));

    // ---- MFMA ----
    AccTile acc;
    clear_acc(acc);
    mfma_scale_f32_32x32x64_fp6<0>(acc, a_reg, b_reg, sa, sb);

    // ---- Store ----
    store_acc_f32(D, N_stride, acc, 0, 0);
}

int main() {
    const int M = 32, K = 64, N = 32;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> A_float(M * K), B_float(K * N);
    for (auto& v : A_float) v = dist(rng);
    for (auto& v : B_float) v = dist(rng);

    QuantizedMatrix A_q = quantize_to_mxfp6(A_float.data(), M, K);
    QuantizedMatrix B_q = preprocess_B(B_float.data(), K, N);

    PreprocessedScale scaleA_p = preprocess_scale(A_q.scales.data(), M, K);
    PreprocessedScale scaleB_p = preprocess_scale(B_q.scales.data(), N, K);

    std::vector<float> D_ref(M * N);
    mxfp6_gemm_ref(A_q, B_q, D_ref.data(), M, K, N);

    void *d_A, *d_B;
    uint8_t *d_scA, *d_scB;
    float *d_D;

    hipMalloc(&d_A, A_q.packed_data.size());
    hipMalloc(&d_B, B_q.packed_data.size());
    hipMalloc(&d_scA, scaleA_p.data.size());
    hipMalloc(&d_scB, scaleB_p.data.size());
    hipMalloc(&d_D, M * N * sizeof(float));

    hipMemcpy(d_A, A_q.packed_data.data(), A_q.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B_q.packed_data.data(), B_q.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_scA, scaleA_p.data.data(), scaleA_p.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_scB, scaleB_p.data.data(), scaleB_p.data.size(), hipMemcpyHostToDevice);
    hipMemset(d_D, 0, M * N * sizeof(float));

    single_mfma_kernel<<<1, 64>>>(d_A, d_B, d_scA, d_scB, d_D, N);
    hipDeviceSynchronize();

    std::vector<float> D_gpu(M * N);
    hipMemcpy(D_gpu.data(), d_D, M * N * sizeof(float), hipMemcpyDeviceToHost);

    int mismatches = 0;
    float max_abs_err = 0.0f;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float ref = D_ref[m * N + n];
            float gpu = D_gpu[m * N + n];
            float abs_err = std::abs(ref - gpu);
            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (abs_err > 1e-4f) {
                if (mismatches < 10)
                    printf("MISMATCH [%d][%d]: ref=%f, gpu=%f, diff=%e\n",
                           m, n, ref, gpu, ref - gpu);
                mismatches++;
            }
        }
    }

    printf("\n========================================\n");
    printf("Single MFMA Test (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("  max abs error: %e\n", max_abs_err);
    printf("  mismatches (abs>1e-4): %d / %d\n", mismatches, M * N);
    printf("========================================\n");

    hipFree(d_A); hipFree(d_B); hipFree(d_scA); hipFree(d_scB); hipFree(d_D);
    return mismatches > 0 ? 1 : 0;
}
