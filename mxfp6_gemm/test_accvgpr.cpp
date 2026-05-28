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
static constexpr int LDS_TOTAL_BYTES = 32 * LDS_ROW_BYTES;

// Same K-loop kernel but using AccTileA (accumulator in AccVGPR)
__global__ void gemm_k_loop_accvgpr(
    const void* __restrict__ A_packed,
    const void* __restrict__ B_shuffled,
    const uint8_t* __restrict__ scale_A,
    const uint8_t* __restrict__ scale_B,
    float* __restrict__ D, int D_stride,
    int k_iters, int A_row_stride)
{
    __shared__ uint32_t lds[LDS_TOTAL_BYTES / 4];
    asm volatile("" : : "r"(lds) : "memory");
    int tid = threadIdx.x;

    AccTileA acc;    // <-- AccVGPR
    clear_acc(acc);

    const char* A_base = reinterpret_cast<const char*>(A_packed);

    for (int ki = 0; ki < k_iters; ki++) {
        int row = tid % 32;
        int seg = tid / 32;
        int hbm_off = row * A_row_stride + ki * 48 + seg * 16;
        v4i tmp = *reinterpret_cast<const v4i*>(A_base + hbm_off);
        uint32_t lds_off = row * LDS_ROW_BYTES + seg * 16;
        ds_write_b128(lds_off, tmp);

        if (tid < 32) {
            hbm_off = tid * A_row_stride + ki * 48 + 32;
            tmp = *reinterpret_cast<const v4i*>(A_base + hbm_off);
            lds_off = tid * LDS_ROW_BYTES + 32;
            ds_write_b128(lds_off, tmp);
        }
        wait_lgkmcnt(0);
        __syncthreads();

        int dim   = tid % 32;
        int khalf = tid / 32;
        v8i a_reg = ds_read_fp6x32(static_cast<uint32_t>(dim * LDS_ROW_BYTES + khalf * 24));

        const char* b_tile = reinterpret_cast<const char*>(B_shuffled) + ki * 1536;
        float4 b_load0 = *reinterpret_cast<const float4*>(b_tile + tid * 16);
        double b_load1_raw = *reinterpret_cast<const double*>(b_tile + 1024 + tid * 8);

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

        int sa = static_cast<int>(scale_A[ki * 64 + tid]);
        int sb = static_cast<int>(scale_B[ki * 64 + tid]);
        asm volatile("v_and_b32 %0, 0xFF, %0" : "+v"(sa));
        asm volatile("v_and_b32 %0, 0xFF, %0" : "+v"(sb));

        mfma_scale_f32_32x32x64_fp6<0>(acc, a_reg, b_reg, sa, sb);
    }

    store_acc_f32(D, D_stride, acc, 0, 0);
}

int main() {
    const int M = 32, N = 32;
    int test_Ks[] = {64, 128, 256, 512};
    int num_tests = 4;
    int total = 0, passed = 0;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (int t = 0; t < num_tests; t++) {
        int K = test_Ks[t];
        std::vector<float> Af(M * K), Bf(K * N);
        for (auto& v : Af) v = dist(rng);
        for (auto& v : Bf) v = dist(rng);

        QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
        QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
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
        hipMalloc(&dD, M * N * sizeof(float));
        hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dB, pbB.data.data(), pbB.data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dsA, saP.data.data(), saP.data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dsB, sbP.data.data(), sbP.data.size(), hipMemcpyHostToDevice);
        hipMemset(dD, 0, M * N * sizeof(float));

        gemm_k_loop_accvgpr<<<1, 64>>>(dA, dB, dsA, dsB, dD, N, K/64, Aq.packed_row_bytes);
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            printf("K=%d: HIP error: %s\n", K, hipGetErrorString(err));
            hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
            total++; continue;
        }

        std::vector<float> Dgpu(M * N);
        hipMemcpy(Dgpu.data(), dD, M * N * sizeof(float), hipMemcpyDeviceToHost);
        float maxerr = 0;
        for (int i = 0; i < M * N; i++)
            maxerr = fmaxf(maxerr, fabsf(Dgpu[i] - Dref[i]));

        bool pass = (maxerr < 0.01f);
        printf("K=%d: max_err=%.4e → %s\n", K, maxerr, pass ? "PASS" : "FAIL");
        total++;
        if (pass) passed++;

        hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dD);
    }

    printf("\n=== AccVGPR test: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
