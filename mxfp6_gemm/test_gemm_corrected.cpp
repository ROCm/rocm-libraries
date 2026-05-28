#include "mxfp6_asm_utils.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

using namespace mxfp6;

// Single-tile GEMM: M=32, N=32, K=64
// A via LDS (per-lane, each thread holds one M-row's K-half)
// B via per-lane global load (each thread holds one N-column's K-half)
// Both have per-lane scales
__global__ void gemm_32x32x64(
    const void* __restrict__ A_packed,     // [32 * 48 bytes]
    const void* __restrict__ B_packed,     // [32 * 48 bytes] = B^T[N=32][K=64]
    const uint8_t* __restrict__ scale_A,   // [64] preprocessed A scales
    const uint8_t* __restrict__ scale_B,   // [64] preprocessed B scales
    float* __restrict__ D, int D_stride)
{
    __shared__ uint32_t lds[512]; // 2048 bytes
    void* smem = (void*)lds;
    asm volatile("" : : "r"(smem) : "memory");

    int tid = threadIdx.x;

    // Thread-to-data mapping (ISA Page 59):
    //   A[m][k]: m = tid%32, k_half = tid/32
    //   B[k][n]: n = tid%32, k_half = tid/32
    int dim   = tid % 32;  // M-index for A, N-index for B
    int khalf = tid / 32;  // 0 or 1

    // ---- Load A into LDS, then read per-lane ----
    // A is 32 rows × 48 bytes = 1536 bytes
    const char* a_src = reinterpret_cast<const char*>(A_packed);

    // Phase 1: 64 threads × 16 bytes = 1024 bytes
    set_m0(0);
    async_load_lds_b128(smem, a_src + tid * 16);
    wait_vmcnt(0);
    __syncthreads();

    // Phase 2: 32 threads × 16 bytes = 512 bytes (remaining)
    if (tid < 32) {
        set_m0(1024);
        async_load_lds_b128(smem, a_src + 1024 + tid * 16);
    }
    wait_vmcnt(0);
    __syncthreads();

    // Read A[m=dim][k_half] from LDS: 24 bytes at row*48 + khalf*24
    v8i a_reg = ds_read_fp6x32(static_cast<uint32_t>(dim * 48 + khalf * 24));

    // ---- Load B per-lane from global memory ----
    // B^T is stored as [N=32][K=64], each row = 48 bytes packed FP6
    // Thread tid reads B^T[n=dim][k_half]: 24 bytes at dim*48 + khalf*24
    const uint32_t* b_ptr = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const char*>(B_packed) + dim * 48 + khalf * 24);
    v8i b_reg;
    b_reg[0] = b_ptr[0]; b_reg[1] = b_ptr[1]; b_reg[2] = b_ptr[2];
    b_reg[3] = b_ptr[3]; b_reg[4] = b_ptr[4]; b_reg[5] = b_ptr[5];
    b_reg[6] = 0; b_reg[7] = 0;

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
    store_acc_f32(D, D_stride, acc, 0, 0);
}

int main() {
    // ========== Test 1: Constant ==========
    printf("=== Test 1: Constant A=B=1.0 ===\n");
    {
        const int M = 32, K = 64, N = 32;
        std::vector<float> Af(M*K, 1.0f), Bf(K*N, 1.0f);
        QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
        QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
        PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
        PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);

        std::vector<float> Dref(M*N);
        mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);

        void *dA, *dB; uint8_t *dsA, *dsB; float *dD;
        hipMalloc(&dA, Aq.packed_data.size());
        hipMalloc(&dB, Bq.packed_data.size());
        hipMalloc(&dsA, saP.data.size());
        hipMalloc(&dsB, sbP.data.size());
        hipMalloc(&dD, M*N*sizeof(float));

        hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dsA, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dsB, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
        hipMemset(dD, 0, M*N*sizeof(float));

        gemm_32x32x64<<<1, 64>>>(dA, dB, dsA, dsB, dD, N);
        hipDeviceSynchronize();

        std::vector<float> Dgpu(M*N);
        hipMemcpy(Dgpu.data(), dD, M*N*sizeof(float), hipMemcpyDeviceToHost);

        float maxerr = 0;
        for (int i = 0; i < M*N; i++)
            maxerr = fmaxf(maxerr, fabsf(Dgpu[i] - Dref[i]));
        printf("Ref D[0][0]=%.2f, GPU D[0][0]=%.2f, max_err=%.4e → %s\n\n",
               Dref[0], Dgpu[0], maxerr, maxerr < 0.01f ? "PASS" : "FAIL");

        hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    }

    // ========== Test 2: Random ==========
    printf("=== Test 2: Random M=32, N=32, K=64 ===\n");
    {
        const int M = 32, K = 64, N = 32;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        std::vector<float> Af(M*K), Bf(K*N);
        for (auto& v : Af) v = dist(rng);
        for (auto& v : Bf) v = dist(rng);

        QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
        QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
        PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
        PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);

        std::vector<float> Dref(M*N);
        mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);

        void *dA, *dB; uint8_t *dsA, *dsB; float *dD;
        hipMalloc(&dA, Aq.packed_data.size());
        hipMalloc(&dB, Bq.packed_data.size());
        hipMalloc(&dsA, saP.data.size());
        hipMalloc(&dsB, sbP.data.size());
        hipMalloc(&dD, M*N*sizeof(float));

        hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dB, Bq.packed_data.data(), Bq.packed_data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dsA, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dsB, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
        hipMemset(dD, 0, M*N*sizeof(float));

        gemm_32x32x64<<<1, 64>>>(dA, dB, dsA, dsB, dD, N);
        hipDeviceSynchronize();

        std::vector<float> Dgpu(M*N);
        hipMemcpy(Dgpu.data(), dD, M*N*sizeof(float), hipMemcpyDeviceToHost);

        int mismatches = 0;
        float maxerr = 0;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float err = fabsf(Dgpu[m*N+n] - Dref[m*N+n]);
                if (err > maxerr) maxerr = err;
                if (err > 0.5f) {
                    if (mismatches < 5)
                        printf("  MISMATCH [%d][%d]: ref=%.4f gpu=%.4f\n",
                               m, n, Dref[m*N+n], Dgpu[m*N+n]);
                    mismatches++;
                }
            }
        }

        printf("max_err=%.4e, mismatches(>0.5)=%d/%d\n", maxerr, mismatches, M*N);
        printf("Sample: D[0][0] ref=%.4f gpu=%.4f\n", Dref[0], Dgpu[0]);
        printf("        D[1][1] ref=%.4f gpu=%.4f\n", Dref[1*N+1], Dgpu[1*N+1]);
        printf("        D[31][31] ref=%.4f gpu=%.4f\n", Dref[31*N+31], Dgpu[31*N+31]);
        printf("\n%s\n", mismatches == 0 ? "PASS" : "FAIL");

        hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    }

    return 0;
}
