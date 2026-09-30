/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Wave-split-K skinny GEMM (M=4, FP16). Source: rocBLAS-internal skinnyGemm
// wvSpltK_hf_m4_<BETA_EQ_ZERO, TRANSA> in
// library/src/blas_ex/rocblas_gemm_ex_kernels.cpp.
//
// Specialized to the TRANSA=true instantiation, which is the one rocBLAS uses
// for the NN small-M path and the only layout hipBLASLt NN produces: A is
// column-major M x K and D is column-major M x N.
//
// Differences from the rocBLAS original:
//   * Column tile retuned from YTILE=7 / UNRL=1 to YTILE=3 / UNRL=2. rocBLAS
//     picked 7 for an 80-CU part; at 304 CUs a 7-wide tile needs N > 34000 just
//     to give every wave one tile, so small-N decode shapes left most of the
//     machine idle. Measured on MI300X over 20 decode shapes, this moves the
//     kernel from 0.89x to 1.11x versus hipBLASLt, and from 7/20 to 17/20 shapes
//     faster.
//   * CuCount is a kernarg, not a compile-time 80, so the persistent stride
//     matches the launch grid on MI300 / MI350.
//   * C and D are separate pointers; hipBLASLt also dispatches out-of-place.
//   * beta is a runtime value rather than a template parameter.
//   * M is a kernarg. Tile rows past it read the last real row of A and are
//     never stored, so this kernel serves every M <= 4, including M=3.
//   * Leading dimensions are kernargs. With a runtime M, lda == ldc == ldd == M
//     is no longer a constant a stride predicate can pin.
//   * A is read only from LDS. The original's global-memory fallback indexes A
//     as row-major, which is wrong for this layout; rocBLAS never reached it
//     because its host path requires M*K <= 32768. That bound is a predicate in
//     custom.config (K <= 8192 for every M <= 4), so the fallback is unreachable
//     here too and is dropped rather than left as a trap.
//
// Regenerate assembly (code object v4 is the version hipBLASLt links at):
//   hipcc -S --cuda-device-only --offload-arch=gfx942 -O3 -mcode-object-version=4 \
//     -mllvm -amdgpu-kernarg-preload-count=14 \
//     -o wvSpltK_hf_m4.s wvSpltK_hf_m4.cpp
// Keep .amdgcn_target / .amdhsa_code_object_version (Tensile retargets).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

template <typename T>
__device__ __forceinline__ T loadnt(T* addr)
{
    return __builtin_nontemporal_load(addr);
}

using half8 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

#define THRDS 64
#define WvPrGrp 16
#define A_CHUNK 8
#define YTILE 3
#define UNRL 2
#define M 4

// Transpose column-major A into LDS so the readback is bank-conflict free and
// each m row is contiguous: s[m * K + k] = A[k * lda + m]. Walking A in memory
// order keeps each wave's loads on consecutive addresses when lda == MR; a
// compile-time MR turns the index split into shifts.
template <int MR>
__device__ __forceinline__ void stageA(__half* s, const __half* A, const int K, const int lda)
{
    for(uint32_t t = threadIdx.y * THRDS + threadIdx.x; t < K * MR; t += THRDS * WvPrGrp)
    {
        const uint32_t k = t / MR;
        const uint32_t m = t % MR;
        s[m * K + k]     = A[(size_t)k * lda + m];
    }
}

// Everything the staging and the K loop read comes first, so the preloaded
// kernarg SGPRs cover it and only the epilogue's arguments are fetched.
extern "C" __global__ void wvSpltK_hf_m4(const int    K,
                                         const int    N,
                                         const __half* B,
                                         const __half* __restrict__ A,
                                         const int    cuCount,
                                         const int    Mrt,
                                         const int    ldb,
                                         const int    lda,
                                         const __half* C,
                                         __half*      D,
                                         const float  alpha,
                                         const float  beta,
                                         const int    ldc,
                                         const int    ldd)
{
    union bigType
    {
        __half     h[A_CHUNK];
        float      f[A_CHUNK / 2];
        float2     f2[A_CHUNK / 4];
        double     d[A_CHUNK / 4];
        __int128_t b128;
        half8      h8;
    };

    // 64 KB static LDS: one WG / CU, activation matrix A staged for the WG
    // lifetime. Requires K*M <= 32*1024 halves; custom.config bounds K <= 8192,
    // which covers every M <= 4.
    __shared__ __half s[1024 * 32];

    uint32_t commitColumn[YTILE];
    for(uint32_t i = 0; i < YTILE; i++)
        commitColumn[i] = 1;

    unsigned long n = (blockIdx.x * WvPrGrp + threadIdx.y) * YTILE;

    if(n < N && (n + YTILE) >= N)
    {
        uint32_t startColumn = N - YTILE;
        for(uint32_t i = 0; i < (n - startColumn); i++)
            commitColumn[i] = 0;
        n = startColumn;
    }

    switch(Mrt)
    {
    case 1:
        stageA<1>(s, A, K, lda);
        break;
    case 2:
        stageA<2>(s, A, K, lda);
        break;
    case 3:
        stageA<3>(s, A, K, lda);
        break;
    default:
        stageA<4>(s, A, K, lda);
        break;
    }
    __syncthreads();

    uint32_t row[M];
    for(int m = 0; m < M; m++)
        row[m] = min(m, Mrt - 1);

    float sum[M][YTILE];

    const int step = cuCount * WvPrGrp * YTILE;

    while(n < N)
    {
        for(int i = 0; i < YTILE; i++)
            for(int m = 0; m < M; m++)
                sum[m][i] = 0;

        bigType bigA[M][UNRL];
        bigType bigB0[UNRL];
        bigType bigB1[UNRL];
        bigType bigB2[UNRL];

        for(uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL)
        {
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                const __half* B_ = &B[(n + 0) * ldb + k_];
                bigB0[k2].h8     = (loadnt((half8*)(&B_[(size_t)0 * ldb])));
                bigB1[k2].h8     = (loadnt((half8*)(&B_[(size_t)1 * ldb])));
                bigB2[k2].h8     = (loadnt((half8*)(&B_[(size_t)2 * ldb])));
            }

#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                for(int m = 0; m < M; m++)
                    bigA[m][k2] = *((const bigType*)(&(s[k_ + K * row[m]])));
            }

#pragma unroll
            for(uint32_t m = 0; m < M; m++)
            {
#pragma unroll
                for(uint32_t k2 = 0; k2 < UNRL; k2++)
                {
                    uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                    uint32_t k_ = k + threadIdx.x * A_CHUNK;
                    if(k_ >= K)
                        break;

#pragma unroll
                    for(uint32_t b = 0; b < A_CHUNK / 2; b++)
                    {
                        asm("v_dot2c_f32_f16 %0, %2, %3"
                            : "=v"(sum[m][0])
                            : "0"(sum[m][0]), "v"(bigA[m][k2].f[b]), "v"(bigB0[k2].f[b]));
                        asm("v_dot2c_f32_f16 %0, %2, %3"
                            : "=v"(sum[m][1])
                            : "0"(sum[m][1]), "v"(bigA[m][k2].f[b]), "v"(bigB1[k2].f[b]));
                        asm("v_dot2c_f32_f16 %0, %2, %3"
                            : "=v"(sum[m][2])
                            : "0"(sum[m][2]), "v"(bigA[m][k2].f[b]), "v"(bigB2[k2].f[b]));
                    }
                }
            }
        }

        for(int m = 0; m < M; m++)
        {
            for(int y = 0; y < YTILE; y++)
            {
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
                    : "=v"(sum[m][y])
                    : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
                    : "=v"(sum[m][y])
                    : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
                    : "=v"(sum[m][y])
                    : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
                    : "=v"(sum[m][y])
                    : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
                    : "=v"(sum[m][y])
                    : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
                    : "=v"(sum[m][y])
                    : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
            }
        }

        if(threadIdx.x == 63)
        {
            // Column-major D: D[n * ldd + m].
            __half*       Dn = &D[n * ldd];
            const __half* Cn = &C[n * ldc];
#pragma unroll
            for(int m = 0; m < M; m++)
            {
                if(m < Mrt)
                {
#pragma unroll
                    for(int i = 0; i < YTILE; i++)
                    {
                        if(commitColumn[i])
                        {
                            if(beta == 0)
                                Dn[(size_t)i * ldd + m] = static_cast<__half>(sum[m][i] * alpha);
                            else
                                Dn[(size_t)i * ldd + m] = static_cast<__half>(
                                    sum[m][i] * alpha
                                    + __half2float(Cn[(size_t)i * ldc + m]) * beta);
                        }
                    }
                }
            }
        }

        n += step;

        if(n < N && (n + YTILE) >= N)
        {
            uint32_t startColumn = N - YTILE;
            for(uint32_t i = 0; i < (n - startColumn); i++)
                commitColumn[i] = 0;
            n = startColumn;
        }
    }
}
