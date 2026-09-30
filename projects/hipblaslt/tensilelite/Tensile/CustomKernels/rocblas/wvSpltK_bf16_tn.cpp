// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Wave-split-K skinny GEMM, BF16 in and out with FP32 accumulation, for
// hipBLASLt TN with a skinny n: D (m x n) = alpha * A^T * B + beta * C, n <= 4.
// torch.mm(x, w.t()) and F.linear reach hipBLASLt in this layout, with m the
// output features and n the tokens. rocBLAS-internal served the same layout
// with the TRANSA=false instantiation of wvSpltK_hf_m*_ in
// library/src/blas_ex/rocblas_gemm_ex_kernels.cpp.
//
// Names follow rocBLAS: N is the long side (hipBLASLt m, one weight row per
// output) and M the skinny side (hipBLASLt n). Each wave owns YTILE weight rows
// and splits K across its 64 lanes; the tokens are staged in LDS.
//
// Differences from the rocBLAS original:
//   * BF16 through v_dot2c_f32_bf16. Only gfx950 has it; gfx942 has no BF16
//     dot instruction, so these kernels are gfx950-only.
//   * Leading dimensions are kernargs. torch.mm on a sliced activation, for
//     example q = qkv[:, :q_size], reaches hipBLASLt with ldb > K, and no
//     predicate can compare a stride against a runtime size.
//   * M is a kernarg as well. Tile rows past it read the last real token and
//     are never stored, so the M=4 kernel serves every M <= 4.
//   * CuCount is a kernarg, C and D are separate pointers, and beta is a
//     runtime value, as in the FP16 NN kernels next to this file.
//   * The token stage is 128 KB rather than 64 KB, since gfx950 has 160 KB of
//     LDS per CU. Tokens past it are read from global memory, as in rocBLAS; in
//     this layout that fallback indexes the tokens correctly, so K needs no
//     bound.
//   * M=4 uses YTILE=1 / UNRL=4 instead of rocBLAS's YTILE=7 / UNRL=1. On
//     gfx950's 256 CUs the GLM-5.2 decode shapes (N = 2048 to 16384) give a
//     wider tile too few waves to hide memory latency: with weights streamed
//     from HBM this tile is 1.2x faster than YTILE=3 / UNRL=2 over those shapes
//     and no slower on the larger decode shapes. M=1 and M=2 keep rocBLAS's
//     YTILE=2 / UNRL=2; no other tile in the same sweep beat it across them.
//
// One source builds the three kernels; keep .amdgcn_target and
// .amdhsa_code_object_version in the output:
//   hipcc -S --cuda-device-only --offload-arch=gfx950 -O3 -DWVSPLTK_M=1 \
//     -mllvm -amdgpu-kernarg-preload-count=14 \
//     -o wvSpltK_bf16_tn_m1.s wvSpltK_bf16_tn.cpp
// and likewise -DWVSPLTK_M=2 and -DWVSPLTK_M=4.

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#if WVSPLTK_M == 1
#define YTILE 2
#define UNRL 2
#define WVSPLTK_NAME wvSpltK_bf16_tn_m1
#elif WVSPLTK_M == 2
#define YTILE 2
#define UNRL 2
#define WVSPLTK_NAME wvSpltK_bf16_tn_m2
#elif WVSPLTK_M == 4
#define YTILE 1
#define UNRL 4
#define WVSPLTK_NAME wvSpltK_bf16_tn_m4
#else
#error "build with -DWVSPLTK_M=1, 2 or 4"
#endif

#define M WVSPLTK_M
#define THRDS 64
#define WvPrGrp 16
#define A_CHUNK 8
#define LDS_ELEMS (64 * 1024)

template <typename T>
__device__ __forceinline__ T loadnt(T* addr)
{
    return __builtin_nontemporal_load(addr);
}

using bf16x8 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

// Everything the staging and the K loop read comes first, so the preloaded
// kernarg SGPRs cover it and only the epilogue's arguments are fetched.
extern "C" __global__ void WVSPLTK_NAME(const int                          K,
                                        const int                          N,
                                        const __hip_bfloat16*              W,
                                        const __hip_bfloat16* __restrict__ X,
                                        const int                          cuCount,
                                        const int                          Mrt,
                                        const int                          ldw,
                                        const int                          ldx,
                                        const __hip_bfloat16*              C,
                                        __hip_bfloat16*                    D,
                                        const float                        alpha,
                                        const float                        beta,
                                        const int                          ldc,
                                        const int                          ldd)
{
    union bigType
    {
        uint16_t   h[A_CHUNK];
        float      f[A_CHUNK / 2];
        __int128_t b128;
        bf16x8     h8;
    };

    // 128 KB static LDS (gfx950 has 160 KB per CU): one WG / CU, tokens staged
    // for the WG lifetime.
    __shared__ uint16_t s[LDS_ELEMS];

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

    // s[m * K + k] = X[m * ldx + k]. K % 8 == 0, so no chunk straddles two tokens.
    const uint32_t staged = min(K * Mrt, LDS_ELEMS);
    for(uint32_t e = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK; e < staged;
        e += THRDS * WvPrGrp * A_CHUNK)
    {
        const uint32_t m = e / K;
        const uint32_t k = e - m * K;
        ((bigType*)(&s[e]))->b128 = ((const bigType*)(&X[(size_t)m * ldx + k]))->b128;
    }
    __syncthreads();

    uint32_t tok[M];
    for(int m = 0; m < M; m++)
        tok[m] = min(m, Mrt - 1);

    float sum[M][YTILE];

    const int step = cuCount * WvPrGrp * YTILE;

    while(n < N)
    {
        for(int y = 0; y < YTILE; y++)
            for(int m = 0; m < M; m++)
                sum[m][y] = 0;

        bigType bigA[M][UNRL];
        bigType bigB[YTILE][UNRL];

        for(uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL)
        {
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                const __hip_bfloat16* W_ = &W[n * ldw + k_];
#pragma unroll
                for(int y = 0; y < YTILE; y++)
                    bigB[y][k2].h8 = loadnt((bf16x8*)(&W_[(size_t)y * ldw]));
            }

#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

#pragma unroll
                for(int m = 0; m < M; m++)
                {
                    const uint32_t e = tok[m] * K + k_;
                    if(e < LDS_ELEMS)
                        bigA[m][k2] = *((const bigType*)(&s[e]));
                    else
                        bigA[m][k2] = *((const bigType*)(&X[(size_t)tok[m] * ldx + k_]));
                }
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
#pragma unroll
                        for(int y = 0; y < YTILE; y++)
                            asm("v_dot2c_f32_bf16 %0, %2, %3"
                                : "=v"(sum[m][y])
                                : "0"(sum[m][y]), "v"(bigA[m][k2].f[b]), "v"(bigB[y][k2].f[b]));
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

        if(threadIdx.x == THRDS - 1)
        {
            // Column-major D: D[n + m * ldd].
            __hip_bfloat16*       Dn = &D[n];
            const __hip_bfloat16* Cn = &C[n];
#pragma unroll
            for(int m = 0; m < M; m++)
            {
                if(m < Mrt)
                {
#pragma unroll
                    for(int y = 0; y < YTILE; y++)
                    {
                        if(commitColumn[y])
                        {
                            float v = sum[m][y] * alpha;
                            if(beta != 0)
                                v += __bfloat162float(Cn[(size_t)m * ldc + y]) * beta;
                            Dn[(size_t)m * ldd + y] = __float2bfloat16(v);
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
