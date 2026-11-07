// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * prelim_kernels.hpp
 *
 * GPU kernels — preliminary shift + INT8 extraction (separate per path).
 *
 * Four kernels, named by the transpose value they handle (_T = transposed,
 * _N = non-transposed), each with optimal blockDim and SHMEM for its path:
 *
 *   accu_prelim_A_T_kernel  TRANS_A=true  (k-fast double2 coalesced, blockDim=256)
 *   accu_prelim_A_N_kernel  TRANS_A=false (SHMEM transposition, blockDim=1024)
 *   accu_prelim_B_N_kernel  TRANS_B=false (j-fast double2 coalesced, blockDim=256)
 *   accu_prelim_B_T_kernel  TRANS_B=true  (SHMEM transposition, blockDim=1024)
 *
 * Coalesced kernels use double2 loads (128-bit), tiny LDS (s_wmax only).
 * SHMEM kernels use TILE_M=16 → 4x fewer blocks/syncs.
 *
 * This header must be compiled as HIP.
 */

#include "kernel_common.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace FixedPointEmulation
{
    /* ── A_T: TRANS_A=true, k-fast 128-bit coalesced, blockDim=256, one block per row ──
     * FP64: double2 (2 doubles = 128 bits).  FP32: float4 (4 floats = 128 bits). */
    template <bool CHECK_NAN, typename T>
    __global__ static void accu_prelim_A_T_kernel(const T* __restrict__ A,
                                                  int64_t m,
                                                  int64_t k,
                                                  int64_t lda,
                                                  int8_t* __restrict__ A8i_high,
                                                  size_t lda8i,
                                                  int16_t* __restrict__ sftA,
                                                  uint32_t* __restrict__ nan_flag)
    {
        __shared__ T       s_wmax[OZ2_PRELIM_COALESC_THRS / OZ2_MIN_WARP_SIZE];
        __shared__ int16_t s_sft[1];

        const int64_t  row      = static_cast<int64_t>(blockIdx.x);
        const T* const row_base = A + static_cast<size_t>(row) * static_cast<size_t>(lda);

        /* Pass 1: reduce per-row max via 128-bit vector loads. */
        T local_max = T{0};
        if constexpr(std::is_same_v<T, double>)
        {
            const int64_t k_even = k & ~int64_t{1};
            for(int64_t j = 2LL * threadIdx.x; j < k_even; j += 2LL * blockDim.x)
            {
                const double2 vv = *reinterpret_cast<const double2*>(row_base + j);
                if constexpr(CHECK_NAN)
                {
                    if(!isfinite(vv.x))
                        (void)atomicOr(nan_flag, isinf(vv.x) ? 1u : 2u);
                    if(!isfinite(vv.y))
                        (void)atomicOr(nan_flag, isinf(vv.y) ? 1u : 2u);
                }
                if(fabs(vv.x) > local_max)
                    local_max = fabs(vv.x);
                if(fabs(vv.y) > local_max)
                    local_max = fabs(vv.y);
            }
            if((k & 1) && threadIdx.x == 0)
            {
                const double val = row_base[k - 1];
                if constexpr(CHECK_NAN)
                    if(!isfinite(val))
                        (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                if(fabs(val) > local_max)
                    local_max = fabs(val);
            }
        }
        else /* float */
        {
            const int64_t k_align4 = k & ~int64_t{3};
            for(int64_t j = 4LL * threadIdx.x; j < k_align4; j += 4LL * blockDim.x)
            {
                const float4 vv = *reinterpret_cast<const float4*>(row_base + j);
                if constexpr(CHECK_NAN)
                {
                    if(!isfinite(vv.x))
                        (void)atomicOr(nan_flag, isinf(vv.x) ? 1u : 2u);
                    if(!isfinite(vv.y))
                        (void)atomicOr(nan_flag, isinf(vv.y) ? 1u : 2u);
                    if(!isfinite(vv.z))
                        (void)atomicOr(nan_flag, isinf(vv.z) ? 1u : 2u);
                    if(!isfinite(vv.w))
                        (void)atomicOr(nan_flag, isinf(vv.w) ? 1u : 2u);
                }
                local_max = fmaxf(
                    local_max,
                    fmaxf(fmaxf(fabsf(vv.x), fabsf(vv.y)), fmaxf(fabsf(vv.z), fabsf(vv.w))));
            }
            /* scalar tail for k % 4 != 0 */
            for(int64_t j = (k & ~int64_t{3}) + threadIdx.x; j < k; j += blockDim.x)
            {
                const float val = row_base[j];
                if constexpr(CHECK_NAN)
                    if(!isfinite(val))
                        (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                if(fabsf(val) > local_max)
                    local_max = fabsf(val);
            }
        }

        local_max = warp_reduce_max_abs(local_max);
        local_max = block_reduce_max(local_max, s_wmax);
        if(threadIdx.x == 0)
        {
            if(local_max < std::numeric_limits<T>::min())
                local_max = std::numeric_limits<T>::min();
            s_sft[0]  = static_cast<int16_t>(6 - floor_log2(local_max));
            sftA[row] = s_sft[0];
        }
        __syncthreads();
        const int sft = static_cast<int>(s_sft[0]);

        /* Pass 2: scale and extract INT8.  Same vector-load pattern as Pass 1. */
        const size_t row_out = static_cast<size_t>(row) * lda8i;
        if constexpr(std::is_same_v<T, double>)
        {
            const int64_t k_even = k & ~int64_t{1};
            for(int64_t j = 2LL * threadIdx.x; j < k_even; j += 2LL * blockDim.x)
            {
                const double2 vv = *reinterpret_cast<const double2*>(row_base + j);
                A8i_high[row_out + static_cast<size_t>(j)]
                    = static_cast<int8_t>(static_cast<int32_t>(ceil(ldexp(fabs(vv.x), sft))));
                A8i_high[row_out + static_cast<size_t>(j) + 1]
                    = static_cast<int8_t>(static_cast<int32_t>(ceil(ldexp(fabs(vv.y), sft))));
            }
            if((k & 1) && threadIdx.x == 0)
            {
                const double scaled = ceil(ldexp(fabs(row_base[k - 1]), sft));
                A8i_high[row_out + static_cast<size_t>(k - 1)]
                    = static_cast<int8_t>(static_cast<int32_t>(scaled));
            }
        }
        else /* float */
        {
            const int64_t k_align4 = k & ~int64_t{3};
            for(int64_t j = 4LL * threadIdx.x; j < k_align4; j += 4LL * blockDim.x)
            {
                const float4 vv           = *reinterpret_cast<const float4*>(row_base + j);
                A8i_high[row_out + j + 0] = static_cast<int8_t>(static_cast<int32_t>(
                    ceilf(static_cast<float>(ldexp(static_cast<double>(fabsf(vv.x)), sft)))));
                A8i_high[row_out + j + 1] = static_cast<int8_t>(static_cast<int32_t>(
                    ceilf(static_cast<float>(ldexp(static_cast<double>(fabsf(vv.y)), sft)))));
                A8i_high[row_out + j + 2] = static_cast<int8_t>(static_cast<int32_t>(
                    ceilf(static_cast<float>(ldexp(static_cast<double>(fabsf(vv.z)), sft)))));
                A8i_high[row_out + j + 3] = static_cast<int8_t>(static_cast<int32_t>(
                    ceilf(static_cast<float>(ldexp(static_cast<double>(fabsf(vv.w)), sft)))));
            }
            for(int64_t j = (k & ~int64_t{3}) + threadIdx.x; j < k; j += blockDim.x)
            {
                A8i_high[row_out + j] = static_cast<int8_t>(static_cast<int32_t>(ceilf(
                    static_cast<float>(ldexp(static_cast<double>(fabsf(row_base[j])), sft)))));
            }
        }
    }

    /* ── A_N: TRANS_A=false, SHMEM transposition, blockDim=1024, TILE_M=16 rows/block ──
     * SHMEM always uses double regardless of T — values are widened at load time.
     * This keeps the residue shift arithmetic in double for both FP32 and FP64.  */
    template <bool CHECK_NAN, typename T>
    __global__ static void accu_prelim_A_N_kernel(const T* __restrict__ A,
                                                  int64_t m,
                                                  int64_t k,
                                                  int64_t lda,
                                                  int8_t* __restrict__ A8i_high,
                                                  size_t lda8i,
                                                  int16_t* __restrict__ sftA,
                                                  uint32_t* __restrict__ nan_flag)
    {
        static constexpr int TILE_K = OZ2_PRELIM_TILE_K;
        static constexpr int TILE_M = OZ2_PRELIM_SHMEM_TILE_M;
        __shared__ double    shmem[TILE_K][TILE_M + 1]; /* always double for arithmetic */
        __shared__ int16_t   s_sft[TILE_M];

        const int64_t m_base  = static_cast<int64_t>(blockIdx.x) * TILE_M;
        const int     t       = static_cast<int>(threadIdx.x);
        const int     k_local = t / TILE_M;
        const int     m_local = t % TILE_M;
        const int64_t i       = m_base + m_local;

        /* Pass 1: accumulate per-row max (widen T→double at load). */
        double thr_max = 0.0;
        for(int64_t k_base = 0; k_base < k; k_base += TILE_K)
        {
            const int64_t j = k_base + k_local;
            if(i < m && j < k)
            {
                const double val = static_cast<double>(A[i + j * lda]);
                if constexpr(CHECK_NAN)
                    if(!isfinite(val))
                        (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                const double av = fabs(val);
                if(av > thr_max)
                    thr_max = av;
            }
        }
        shmem[k_local][m_local] = thr_max;
        __syncthreads();
        if(k_local == 0)
        {
            double row_max = 0.0;
            for(int kl = 0; kl < TILE_K; ++kl)
                if(shmem[kl][m_local] > row_max)
                    row_max = shmem[kl][m_local];
            if(row_max < std::numeric_limits<double>::min())
                row_max = std::numeric_limits<double>::min();
            s_sft[m_local] = static_cast<int16_t>(6 - floor_log2(row_max));
            if(i < m)
                sftA[i] = s_sft[m_local];
        }
        __syncthreads();
        const int sft = static_cast<int>(s_sft[m_local]);

        /* Pass 2: scale → SHMEM → write INT8. */
        for(int64_t k_base = 0; k_base < k; k_base += TILE_K)
        {
            const int64_t j      = k_base + k_local;
            double        scaled = 0.0;
            if(i < m && j < k)
                scaled = ceil(ldexp(fabs(static_cast<double>(A[i + j * lda])), sft));
            shmem[k_local][m_local] = scaled;
            __syncthreads();
            const int     k_write = t % TILE_K;
            const int     m_write = t / TILE_K;
            const int64_t j_out   = k_base + k_write;
            const int64_t i_out   = m_base + m_write;
            if(i_out < m && j_out < k)
                A8i_high[static_cast<size_t>(j_out) + static_cast<size_t>(i_out) * lda8i]
                    = static_cast<int8_t>(static_cast<int32_t>(shmem[k_write][m_write]));
            __syncthreads();
        }
    }

    /* ── B_N: TRANS_B=false, k-fast 128-bit coalesced, blockDim=256, one block per col ── */
    template <bool CHECK_NAN, typename T>
    __global__ static void accu_prelim_B_N_kernel(const T* __restrict__ B,
                                                  int64_t n,
                                                  int64_t k,
                                                  int64_t ldb,
                                                  int8_t* __restrict__ B8i_high,
                                                  size_t ldb8i,
                                                  int16_t* __restrict__ sftB,
                                                  uint32_t* __restrict__ nan_flag)
    {
        __shared__ T       s_wmax[OZ2_PRELIM_COALESC_THRS / OZ2_MIN_WARP_SIZE];
        __shared__ int16_t s_sft[1];

        const int64_t  col      = static_cast<int64_t>(blockIdx.x);
        const T* const col_base = B + static_cast<size_t>(col) * static_cast<size_t>(ldb);

        T local_max = T{0};
        if constexpr(std::is_same_v<T, double>)
        {
            const int64_t k_even = k & ~int64_t{1};
            for(int64_t j = 2LL * threadIdx.x; j < k_even; j += 2LL * blockDim.x)
            {
                const double2 vv = *reinterpret_cast<const double2*>(col_base + j);
                if constexpr(CHECK_NAN)
                {
                    if(!isfinite(vv.x))
                        (void)atomicOr(nan_flag, isinf(vv.x) ? 1u : 2u);
                    if(!isfinite(vv.y))
                        (void)atomicOr(nan_flag, isinf(vv.y) ? 1u : 2u);
                }
                if(fabs(vv.x) > local_max)
                    local_max = fabs(vv.x);
                if(fabs(vv.y) > local_max)
                    local_max = fabs(vv.y);
            }
            if((k & 1) && threadIdx.x == 0)
            {
                const double val = col_base[k - 1];
                if constexpr(CHECK_NAN)
                    if(!isfinite(val))
                        (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                if(fabs(val) > local_max)
                    local_max = fabs(val);
            }
        }
        else /* float */
        {
            const int64_t k_align4 = k & ~int64_t{3};
            for(int64_t j = 4LL * threadIdx.x; j < k_align4; j += 4LL * blockDim.x)
            {
                const float4 vv = *reinterpret_cast<const float4*>(col_base + j);
                if constexpr(CHECK_NAN)
                {
                    if(!isfinite(vv.x))
                        (void)atomicOr(nan_flag, isinf(vv.x) ? 1u : 2u);
                    if(!isfinite(vv.y))
                        (void)atomicOr(nan_flag, isinf(vv.y) ? 1u : 2u);
                    if(!isfinite(vv.z))
                        (void)atomicOr(nan_flag, isinf(vv.z) ? 1u : 2u);
                    if(!isfinite(vv.w))
                        (void)atomicOr(nan_flag, isinf(vv.w) ? 1u : 2u);
                }
                local_max = fmaxf(
                    local_max,
                    fmaxf(fmaxf(fabsf(vv.x), fabsf(vv.y)), fmaxf(fabsf(vv.z), fabsf(vv.w))));
            }
            for(int64_t j = (k & ~int64_t{3}) + threadIdx.x; j < k; j += blockDim.x)
            {
                const float val = col_base[j];
                if constexpr(CHECK_NAN)
                    if(!isfinite(val))
                        (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                if(fabsf(val) > local_max)
                    local_max = fabsf(val);
            }
        }

        local_max = warp_reduce_max_abs(local_max);
        local_max = block_reduce_max(local_max, s_wmax);
        if(threadIdx.x == 0)
        {
            if(local_max < std::numeric_limits<T>::min())
                local_max = std::numeric_limits<T>::min();
            s_sft[0]  = static_cast<int16_t>(6 - floor_log2(local_max));
            sftB[col] = s_sft[0];
        }
        __syncthreads();
        const int sft = static_cast<int>(s_sft[0]);

        const size_t col_out = static_cast<size_t>(col) * ldb8i;
        if constexpr(std::is_same_v<T, double>)
        {
            const int64_t k_even = k & ~int64_t{1};
            for(int64_t j = 2LL * threadIdx.x; j < k_even; j += 2LL * blockDim.x)
            {
                const double2 vv = *reinterpret_cast<const double2*>(col_base + j);
                B8i_high[col_out + j]
                    = static_cast<int8_t>(static_cast<int32_t>(ceil(ldexp(fabs(vv.x), sft))));
                B8i_high[col_out + j + 1]
                    = static_cast<int8_t>(static_cast<int32_t>(ceil(ldexp(fabs(vv.y), sft))));
            }
            if((k & 1) && threadIdx.x == 0)
            {
                const double scaled       = ceil(ldexp(fabs(col_base[k - 1]), sft));
                B8i_high[col_out + k - 1] = static_cast<int8_t>(static_cast<int32_t>(scaled));
            }
        }
        else /* float */
        {
            const int64_t k_align4 = k & ~int64_t{3};
            for(int64_t j = 4LL * threadIdx.x; j < k_align4; j += 4LL * blockDim.x)
            {
                const float4 vv           = *reinterpret_cast<const float4*>(col_base + j);
                B8i_high[col_out + j + 0] = static_cast<int8_t>(static_cast<int32_t>(
                    ceilf(static_cast<float>(ldexp(static_cast<double>(fabsf(vv.x)), sft)))));
                B8i_high[col_out + j + 1] = static_cast<int8_t>(static_cast<int32_t>(
                    ceilf(static_cast<float>(ldexp(static_cast<double>(fabsf(vv.y)), sft)))));
                B8i_high[col_out + j + 2] = static_cast<int8_t>(static_cast<int32_t>(
                    ceilf(static_cast<float>(ldexp(static_cast<double>(fabsf(vv.z)), sft)))));
                B8i_high[col_out + j + 3] = static_cast<int8_t>(static_cast<int32_t>(
                    ceilf(static_cast<float>(ldexp(static_cast<double>(fabsf(vv.w)), sft)))));
            }
            for(int64_t j = (k & ~int64_t{3}) + threadIdx.x; j < k; j += blockDim.x)
            {
                B8i_high[col_out + j] = static_cast<int8_t>(static_cast<int32_t>(ceilf(
                    static_cast<float>(ldexp(static_cast<double>(fabsf(col_base[j])), sft)))));
            }
        }
    }

    /* ── B_T: TRANS_B=true, SHMEM transposition, blockDim=1024, TILE_M=16 cols/block ── */
    template <bool CHECK_NAN, typename T>
    __global__ static void accu_prelim_B_T_kernel(const T* __restrict__ B,
                                                  int64_t n,
                                                  int64_t k,
                                                  int64_t ldb,
                                                  int8_t* __restrict__ B8i_high,
                                                  size_t ldb8i,
                                                  int16_t* __restrict__ sftB,
                                                  uint32_t* __restrict__ nan_flag)
    {
        static constexpr int TILE_K = OZ2_PRELIM_TILE_K;
        static constexpr int TILE_M = OZ2_PRELIM_SHMEM_TILE_M;
        __shared__ double    shmem[TILE_K][TILE_M + 1];
        __shared__ int16_t   s_sft[TILE_M];

        const int64_t n_base  = static_cast<int64_t>(blockIdx.x) * TILE_M;
        const int     t       = static_cast<int>(threadIdx.x);
        const int     k_local = t / TILE_M;
        const int     l_local = t % TILE_M;
        const int64_t col     = n_base + l_local;

        double thr_max = 0.0;
        for(int64_t k_base = 0; k_base < k; k_base += TILE_K)
        {
            const int64_t j = k_base + k_local;
            if(col < n && j < k)
            {
                const double val = static_cast<double>(B[col + j * ldb]);
                if constexpr(CHECK_NAN)
                    if(!isfinite(val))
                        (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                const double av = fabs(val);
                if(av > thr_max)
                    thr_max = av;
            }
        }
        shmem[k_local][l_local] = thr_max;
        __syncthreads();
        if(k_local == 0)
        {
            double col_max = 0.0;
            for(int kl = 0; kl < TILE_K; ++kl)
                if(shmem[kl][l_local] > col_max)
                    col_max = shmem[kl][l_local];
            if(col_max < std::numeric_limits<double>::min())
                col_max = std::numeric_limits<double>::min();
            s_sft[l_local] = static_cast<int16_t>(6 - floor_log2(col_max));
            if(col < n)
                sftB[col] = s_sft[l_local];
        }
        __syncthreads();
        const int sft = static_cast<int>(s_sft[l_local]);

        for(int64_t k_base = 0; k_base < k; k_base += TILE_K)
        {
            const int64_t j      = k_base + k_local;
            double        scaled = 0.0;
            if(col < n && j < k)
                scaled = ceil(ldexp(fabs(static_cast<double>(B[col + j * ldb])), sft));
            shmem[k_local][l_local] = scaled;
            __syncthreads();
            const int     k_write = t % TILE_K;
            const int     l_write = t / TILE_K;
            const int64_t j_out   = k_base + k_write;
            const int64_t col_out = n_base + l_write;
            if(col_out < n && j_out < k)
                B8i_high[static_cast<size_t>(j_out) + static_cast<size_t>(col_out) * ldb8i]
                    = static_cast<int8_t>(static_cast<int32_t>(shmem[k_write][l_write]));
            __syncthreads();
        }
    }

} // namespace FixedPointEmulation
