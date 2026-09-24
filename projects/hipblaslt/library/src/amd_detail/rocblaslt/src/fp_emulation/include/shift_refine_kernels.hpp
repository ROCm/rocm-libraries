// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * shift_refine_kernels.hpp
 *
 * GPU kernels — shift refinement (prelim) from the preliminary INT8 GEMM.
 *
 * These kernels compute per-row/per-column maxima of the preliminary GEMM
 * result and refine the per-row/col shift values.  The ADP (Adaptive
 * Precision) reduction kernels determine the minimum number of CRT moduli
 * (effective_s) needed to reach the target accuracy.
 *
 *   refine_sftA_partial_kernel  per-row max of |C32i_prelim|
 *   col_max_kernel              per-column max of |C32i_prelim|
 *   adp_reduce_A_kernel         A-side ADP log2P requirement
 *   refine_sftA_apply_kernel    apply the A-side shift-refinement delta
 *   adp_reduce_B_kernel         B-side ADP log2P requirement (reads col_max)
 *   refine_sftB_kernel          apply the B-side shift-refinement delta
 *
 * This header must be compiled as HIP.
 */

#include "kernel_common.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace FixedPointEmulation
{
    __global__ static void refine_sftA_partial_kernel(const int32_t* __restrict__ C32i,
                                                      int64_t m,
                                                      int64_t n,
                                                      size_t  ldc32i,
                                                      int32_t* __restrict__ row_max)
    {
        const int64_t row
            = static_cast<int64_t>(blockIdx.x) * 64 + static_cast<int64_t>(threadIdx.x);
        const int64_t col_base = static_cast<int64_t>(blockIdx.y) * 64;
        if(row >= m)
            return;
        int32_t       local_max = 0;
        const int64_t col_end   = (col_base + 64 < n) ? col_base + 64 : n;
        for(int64_t col = col_base; col < col_end; ++col)
        {
            int32_t v  = C32i[static_cast<size_t>(row) + static_cast<size_t>(col) * ldc32i];
            int32_t av = v < 0 ? -v : v;
            if(av > local_max)
                local_max = av;
        }
        if(local_max > 0)
            atomicMax(row_max + static_cast<size_t>(row), local_max);
    }

    /* col_max_kernel — computes per-column max of |C32i_prelim| and writes col_max[n].
     * Reads C32i in column-major order (one block per column, 256 threads reduce over rows).
     * Avoids LDS atomic contention by using a standard block reduction.
     * Runs in the same stream as refine_sftA_partial_kernel so both C32i reads
     * are pipelined; downstream adp_reduce_B and refine_sftB read only the tiny col_max[n]
     * array (n×4 bytes) instead of the full m×n C32i matrix.
     * Launch: dim3(n) blocks × dim3(256) threads.                                        */
    __global__ static void col_max_kernel(const int32_t* __restrict__ C32i,
                                          int64_t m,
                                          size_t  ldc32i,
                                          int32_t* __restrict__ col_max)
    {
        __shared__ int32_t s_wmax[8];
        const int64_t      col       = static_cast<int64_t>(blockIdx.x);
        int32_t            local_max = 0;
        for(int64_t i = threadIdx.x; i < m; i += blockDim.x)
        {
            int32_t v  = C32i[static_cast<size_t>(i) + static_cast<size_t>(col) * ldc32i];
            int32_t av = v < 0 ? -v : v;
            if(av > local_max)
                local_max = av;
        }
        local_max = warp_reduce_max_abs(local_max);
        local_max = block_reduce_max(local_max, s_wmax);
        if(threadIdx.x == 0)
            col_max[col] = local_max; /* direct write, no atomicMax needed */
    }

    /* Computes global max of  (52 − sftA_init[i]) + 0.5·log2(row_max[i]) + 200
     * over all rows i ∈ [0, m).  The +200 bias guarantees a non-negative result
     * (log2P_req_unbiased is in ≈ [−50, 170] for any valid FP64 input).
     * Subtract 200 on the host to recover the actual log2P requirement.
     *
     * Must be launched AFTER refine_sftA_partial_kernel (which fills row_max[])
     * and BEFORE refine_sftA_apply_kernel (which overwrites sftA[]).           */
    __global__ static void adp_reduce_A_kernel(const int32_t* __restrict__ row_max,
                                               const int16_t* __restrict__ sftA_init,
                                               int64_t m,
                                               float* __restrict__ adp_A_out,
                                               float adp_bits)
    {
        __shared__ float s_wmax[OZ2_PRELIM_COALESC_THRS / OZ2_MIN_WARP_SIZE]; /* 8 slots */
        const int64_t    row
            = static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);

        /* Biased log2P requirement for this row.
         * Skip rows where row_max == 0: those rows have all-zero preliminary inner
         * products (e.g. zero-matrix inputs) and need zero CRT precision (s=2).
         * Using max(row_max,1) would incorrectly return (adp_bits − sftA_init)
         * bits for zero rows because sftA_init=6 is a dummy value set for zero inputs. */
        float local_val = 0.0f; /* 0.0f = biased −200 = needs no precision */
        if(row < m && row_max[row] > 0)
        {
            const int32_t rm      = row_max[row];
            const float   sftA_f  = static_cast<float>(sftA_init[row]);
            const float   log2_rm = 0.5f * log2f(static_cast<float>(rm));
            /* log2_rm = 0.5*log2(R_i): A-side half of the CRT budget (A/B split).
             * Three independent lower bounds on log2P; the row needs the max:
             *   crt_acc: CRT grid fine enough to resolve to 2^-adp_bits.
             *   crt_rep: |X_true| < M_s/2 representability (no adp_bits; the
             *            wraparound guard for wide-range/cancelling inputs).
             *   trunc:   INT8 operand quantization floor (data-independent). */
            const float crt_acc   = (adp_bits - sftA_f) + log2_rm;
            const float crt_rep   = 2.0f * log2_rm - sftA_f;
            const float trunc_val = adp_bits + 6.0f - 2.0f * sftA_f;
            local_val = fmaxf(fmaxf(crt_acc, crt_rep), trunc_val) + 200.0f;
        }

        /* Warp-level max reduction. */
        for(int off = warpSize >> 1; off > 0; off >>= 1)
        {
            float other = __shfl_down(local_val, off);
            if(other > local_val)
                local_val = other;
        }

        if(threadIdx.x % warpSize == 0)
            s_wmax[threadIdx.x / warpSize] = local_val;
        __syncthreads();

        if(threadIdx.x == 0)
        {
            float     block_max = 0.0f;
            const int nw        = (blockDim.x + warpSize - 1) / warpSize;
            for(int w = 0; w < nw; ++w)
                if(s_wmax[w] > block_max)
                    block_max = s_wmax[w];
            adp_atomicMaxF(adp_A_out, block_max);
        }
    }

    __global__ static void refine_sftA_apply_kernel(const int32_t* __restrict__ row_max,
                                                    int16_t* __restrict__ sftA,
                                                    int64_t m,
                                                    float   log2P)
    {
        const int64_t row
            = static_cast<int64_t>(blockIdx.x) * 64 + static_cast<int64_t>(threadIdx.x);
        if(row >= m)
            return;
        int32_t max_val = row_max[row];
        if(max_val < 1)
            max_val = 1;
        /* mh4u_ru = -round_up(0.5/(1 - 4·2^-24)), slightly more negative than -0.5F,
         * compensates for __log2f's ≤1 ULP rounding error so the shift is never
         * overestimated.  __int2float_ru gives a conservative (rounded-up) float of
         * the INT32 max, preventing underestimation of log2(max). */
        static constexpr float mh4u_ru = -0x1.0000060000000p-1F;
        sftA[row]
            += static_cast<int16_t>(floorf(fmaf(mh4u_ru, __log2f(__int2float_ru(max_val)), log2P)));
    }

    /* adp_reduce_B_kernel — reads the precomputed col_max[j] (NOT the full C32i
     * matrix) to compute the B-side ADP log2P requirement.  col_max[] is filled by
     * refine_sftA_partial_kernel in the same pass that computes row_max[], so
     * C32i is read only ONCE total instead of three times.
     *
     * Each thread handles one column independently — no inter-thread reduction needed.
     * Launch config: ceil(n/256) blocks × 256 threads.                              */
    __global__ static void adp_reduce_B_kernel(const int32_t* __restrict__ col_max,
                                               int64_t n,
                                               const int16_t* __restrict__ sftB_init,
                                               float* __restrict__ adp_B_out,
                                               float adp_bits)
    {
        const int64_t col
            = static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);
        if(col >= n)
            return;
        const int32_t local_max = col_max[col];
        /* Skip zero columns: col_max==0 means all preliminary products for that
         * column are zero, so no CRT precision is needed for it.                 */
        if(local_max > 0)
        {
            const float sftB_f  = static_cast<float>(sftB_init[col]);
            const float log2_cm = 0.5f * log2f(static_cast<float>(local_max));
            /* log2_cm = 0.5*log2(C_j): B-side half of the CRT budget (A/B split).
             * Three bounds, max over them (see adp_reduce_A_kernel):
             *   crt_acc: CRT grid resolves to 2^-adp_bits.
             *   crt_rep: |X_true| < M_s/2 representability (no adp_bits).
             *   trunc:   INT8 quantization floor (data-independent). */
            const float crt_acc    = (adp_bits - sftB_f) + log2_cm;
            const float crt_rep    = 2.0f * log2_cm - sftB_f;
            const float trunc_req  = adp_bits + 6.0f - 2.0f * sftB_f;
            const float req_biased = fmaxf(fmaxf(crt_acc, crt_rep), trunc_req) + 200.0f;
            adp_atomicMaxF(adp_B_out, req_biased);
        }
    }

    /* refine_sftB_kernel — reads the precomputed col_max[j] (NOT the full C32i
     * matrix) to apply the shift-refinement delta to sftB[col].
     * Launch config: ceil(n/256) blocks × 256 threads.                              */
    __global__ static void refine_sftB_kernel(const int32_t* __restrict__ col_max,
                                              int64_t n,
                                              int16_t* __restrict__ sftB,
                                              float log2P)
    {
        const int64_t col
            = static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);
        if(col >= n)
            return;
        int32_t local_max = col_max[col];
        if(local_max < 1)
            local_max = 1;
        static constexpr float mh4u_ru = -0x1.0000060000000p-1F; /* see refine_sftA_apply_kernel */
        sftB[col] += static_cast<int16_t>(
            floorf(fmaf(mh4u_ru, __log2f(__int2float_ru(local_max)), log2P)));
    }

} // namespace FixedPointEmulation
