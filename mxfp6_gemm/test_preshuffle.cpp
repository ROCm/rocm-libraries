#include "mxfp6_asm_utils.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace mxfp6;

// Single-tile GEMM with pre-shuffled B: M=32, N=32, K=64
__global__ void gemm_preshuffle(
    const void* __restrict__ A_packed,      // [1536 bytes] A[32][K=64] packed FP6
    const void* __restrict__ B_shuffled,    // [1536 bytes] pre-shuffled B tile
    const uint8_t* __restrict__ scale_A,    // [64] A scales in lane order
    const uint8_t* __restrict__ scale_B,    // [64] B scales in lane order
    float* __restrict__ D, int D_stride)
{
    __shared__ uint32_t lds[512];
    void* smem = (void*)lds;
    asm volatile("" : : "r"(smem) : "memory");

    int tid = threadIdx.x;

    // ---- A: async HBM → LDS → VGPR ----
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

    int dim   = tid % 32;
    int khalf = tid / 32;
    v8i a_reg = ds_read_fp6x32(static_cast<uint32_t>(dim * 48 + khalf * 24));

    // ---- B: coalesced VMEM load from pre-shuffled data ----
    // Section 0: tid*16 bytes → dwordx4 (DWORDs 0-3)
    // Section 1: 1024 + tid*8 bytes → dwordx2 (DWORDs 4-5)
    const char* b_base = reinterpret_cast<const char*>(B_shuffled);
    const float4* b_s0 = reinterpret_cast<const float4*>(b_base + tid * 16);
    const double* b_s1 = reinterpret_cast<const double*>(b_base + 1024 + tid * 8);

    float4 b_load0 = *b_s0;
    double b_load1_raw = *b_s1;

    v8i b_reg;
    b_reg[0] = __float_as_int(b_load0.x);
    b_reg[1] = __float_as_int(b_load0.y);
    b_reg[2] = __float_as_int(b_load0.z);
    b_reg[3] = __float_as_int(b_load0.w);
    // dwordx2 from section 1
    int2 b_dw45 = *reinterpret_cast<const int2*>(&b_load1_raw);
    b_reg[4] = b_dw45.x;
    b_reg[5] = b_dw45.y;
    b_reg[6] = 0;
    b_reg[7] = 0;

    // ---- Scales ----
    int sa = static_cast<int>(scale_A[tid]);
    int sb = static_cast<int>(scale_B[tid]);
    asm volatile("v_and_b32 %0, 0xFF, %0" : "+v"(sa));
    asm volatile("v_and_b32 %0, 0xFF, %0" : "+v"(sb));

    // ---- MFMA (TransposeC: src0=B, src1=A) ----
    AccTile acc;
    clear_acc(acc);
    mfma_scale_f32_32x32x64_fp6<0>(acc, a_reg, b_reg, sa, sb);

    // ---- Store ----
    store_acc_f32(D, D_stride, acc, 0, 0);
}

int main() {
    const int M = 32, K = 64, N = 32;

    // ---- Constant test ----
    printf("=== Constant test ===\n");
    {
        std::vector<float> Af(M*K, 1.0f), Bf(K*N, 1.0f);
        QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
        QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
        PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
        PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
        PreshuffledB pbB = preshuffle_B(Bq);

        std::vector<float> Dref(M*N);
        mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);

        void *dA; uint8_t *dsA, *dsB; float *dD; void *dB;
        hipMalloc(&dA, Aq.packed_data.size());
        hipMalloc(&dB, 1536);
        hipMalloc(&dsA, 64);
        hipMalloc(&dsB, 64);
        hipMalloc(&dD, M*N*sizeof(float));
        hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dB, pbB.data.data(), 1536, hipMemcpyHostToDevice);
        hipMemcpy(dsA, saP.data.data(), 64, hipMemcpyHostToDevice);
        hipMemcpy(dsB, sbP.data.data(), 64, hipMemcpyHostToDevice);
        hipMemset(dD, 0, M*N*sizeof(float));

        gemm_preshuffle<<<1, 64>>>(dA, dB, dsA, dsB, dD, N);
        hipDeviceSynchronize();

        std::vector<float> Dgpu(M*N);
        hipMemcpy(Dgpu.data(), dD, M*N*sizeof(float), hipMemcpyDeviceToHost);
        float maxerr = 0;
        for (int i = 0; i < M*N; i++)
            maxerr = fmaxf(maxerr, fabsf(Dgpu[i] - Dref[i]));
        printf("max_err=%.4e → %s\n", maxerr, maxerr < 0.01f ? "PASS" : "FAIL");

        hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    }

    // ---- Random test ----
    printf("\n=== Random test ===\n");
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        std::vector<float> Af(M*K), Bf(K*N);
        for (auto& v : Af) v = dist(rng);
        for (auto& v : Bf) v = dist(rng);

        QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
        QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
        PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
        PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
        PreshuffledB pbB = preshuffle_B(Bq);

        std::vector<float> Dref(M*N);
        mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);

        void *dA; uint8_t *dsA, *dsB; float *dD; void *dB;
        hipMalloc(&dA, Aq.packed_data.size());
        hipMalloc(&dB, 1536);
        hipMalloc(&dsA, 64);
        hipMalloc(&dsB, 64);
        hipMalloc(&dD, M*N*sizeof(float));
        hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dB, pbB.data.data(), 1536, hipMemcpyHostToDevice);
        hipMemcpy(dsA, saP.data.data(), 64, hipMemcpyHostToDevice);
        hipMemcpy(dsB, sbP.data.data(), 64, hipMemcpyHostToDevice);
        hipMemset(dD, 0, M*N*sizeof(float));

        gemm_preshuffle<<<1, 64>>>(dA, dB, dsA, dsB, dD, N);
        hipDeviceSynchronize();

        std::vector<float> Dgpu(M*N);
        hipMemcpy(Dgpu.data(), dD, M*N*sizeof(float), hipMemcpyDeviceToHost);
        int mis = 0;
        float maxerr = 0;
        for (int i = 0; i < M*N; i++) {
            float e = fabsf(Dgpu[i] - Dref[i]);
            if (e > maxerr) maxerr = e;
            if (e > 0.5f) mis++;
        }
        printf("max_err=%.4e, mismatches=%d/%d\n", maxerr, mis, M*N);
        printf("D[0][0] ref=%.4f gpu=%.4f\n", Dref[0], Dgpu[0]);
        printf("D[15][17] ref=%.4f gpu=%.4f\n", Dref[15*N+17], Dgpu[15*N+17]);
        printf("D[31][31] ref=%.4f gpu=%.4f\n", Dref[31*N+31], Dgpu[31*N+31]);
        printf("%s\n", mis == 0 ? "PASS" : "FAIL");

        hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    }

    return 0;
}
