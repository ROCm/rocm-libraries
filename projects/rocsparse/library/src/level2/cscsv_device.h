/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "rocsparse_common.hpp"

// CSC solve of A * y = alpha * x with A triangular in CSC: once y_c is known, column c
// pushes a_rc * y_c into sum[r] and decrements rem[r]; unknown r is ready when rem[r] is zero.
namespace rocsparse
{
    template <uint32_t NUM_LANES, typename T>
    __device__ __forceinline__ T cscsv_group_broadcast(T v)
    {
        return __shfl(v, 0, NUM_LANES);
    }

    template <uint32_t NUM_LANES>
    __device__ __forceinline__ int cscsv_group_any(int v)
    {
        for(uint32_t o = NUM_LANES / 2; o > 0; o >>= 1)
        {
            v |= __shfl_xor(v, o, NUM_LANES);
        }
        return v;
    }

    // One group of NUM_LANES lanes per column: diagonal entry, dependency counts, longest column.
    template <uint32_t BLOCKSIZE, uint32_t NUM_LANES, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_analysis_kernel(J m,
                               const I* __restrict__ csc_col_ptr,
                               const J* __restrict__ csc_row_ind,
                               I* __restrict__ csc_diag_ind,
                               J* __restrict__ dependency_count,
                               I* __restrict__ max_nnz,
                               J* __restrict__ zero_pivot,
                               rocsparse_index_base idx_base,
                               rocsparse_diag_type  diag_type,
                               rocsparse_fill_mode  fill_mode)
    {
        const uint32_t lid = hipThreadIdx_x & (NUM_LANES - 1);
        const uint32_t wid = hipThreadIdx_x / NUM_LANES;
        const J        col = hipBlockIdx_x * (BLOCKSIZE / NUM_LANES) + wid;

        __shared__ I shared_nnz[BLOCKSIZE / NUM_LANES];

        I nnz = 0;
        if(col < m)
        {
            const I col_begin = csc_col_ptr[col] - idx_base;
            const I col_end   = csc_col_ptr[col + 1] - idx_base;
            nnz               = col_end - col_begin;

            int has_diag = 0;
            for(I k = col_begin + lid; k < col_end; k += NUM_LANES)
            {
                const J row = csc_row_ind[k] - idx_base;
                if(row == col)
                {
                    csc_diag_ind[col] = k;
                    has_diag          = 1;
                }
                else if((fill_mode == rocsparse_fill_mode_lower) ? (row > col) : (row < col))
                {
                    rocsparse::atomic_add(&dependency_count[row], static_cast<J>(1));
                }
            }

            has_diag = rocsparse::cscsv_group_any<NUM_LANES>(has_diag);
            if(lid == 0 && !has_diag && diag_type == rocsparse_diag_type_non_unit)
            {
                rocsparse::atomic_min(zero_pivot, col + idx_base);
            }
        }

        // One atomic per block, a per-column atomic on the same address serializes the kernel.
        if(lid == 0)
        {
            shared_nnz[wid] = nnz;
        }
        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE / NUM_LANES>(hipThreadIdx_x, shared_nnz);

        if(hipThreadIdx_x == 0)
        {
            rocsparse::atomic_max(max_nnz, shared_nnz[0]);
        }
    }

    // Resets the number of unresolved dependencies of every unknown, for each batch instance,
    // from the counts of the analysis. The solve decrements these counters.
    template <uint32_t BLOCKSIZE, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_init_kernel(J       m,
                           int64_t batch_count,
                           const J* __restrict__ dependency_count,
                           J* __restrict__ rem)
    {
        for(int64_t i = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
            i < m * batch_count;
            i += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            rem[i] = dependency_count[i % m];
        }
    }

    // One group of NUM_LANES lanes per unknown, in elimination order. Blocks take an atomic ticket
    // instead of hipBlockIdx_x, so a group only waits on groups that are already running.
    template <uint32_t BLOCKSIZE, uint32_t NUM_LANES, typename I, typename J, typename T>
    ROCSPARSE_DEVICE_ILF void cscsv_solve_device(J m,
                                                 T alpha,
                                                 const I* __restrict__ csc_col_ptr,
                                                 const J* __restrict__ csc_row_ind,
                                                 const T* __restrict__ csc_val,
                                                 const I* __restrict__ csc_diag_ind,
                                                 const T* __restrict__ x,
                                                 int64_t x_inc,
                                                 T*      y,
                                                 int64_t y_inc,
                                                 J* __restrict__ rem,
                                                 T* __restrict__ sum,
                                                 int32_t* __restrict__ ticket,
                                                 J*                   zero_pivot,
                                                 rocsparse_index_base idx_base,
                                                 rocsparse_fill_mode  fill_mode,
                                                 rocsparse_diag_type  diag_type)
    {
        __shared__ int32_t block_id;
        if(hipThreadIdx_x == 0)
        {
            block_id = rocsparse::atomic_add(ticket, 1);
        }
        __syncthreads();

        const uint32_t lid = hipThreadIdx_x & (NUM_LANES - 1);
        const uint32_t wid = hipThreadIdx_x / NUM_LANES;
        const int64_t  w   = static_cast<int64_t>(block_id) * (BLOCKSIZE / NUM_LANES) + wid;

        if(w >= m)
        {
            return;
        }

        const bool lower = (fill_mode == rocsparse_fill_mode_lower);
        const J    col   = lower ? static_cast<J>(w) : static_cast<J>(m - 1 - w);

        const I col_begin = csc_col_ptr[col] - idx_base;
        const I col_end   = csc_col_ptr[col + 1] - idx_base;

        // Off the critical path: everything that does not depend on other unknowns.
        T inv_diag = static_cast<T>(1);
        if(diag_type == rocsparse_diag_type_non_unit)
        {
            const I d = csc_diag_ind[col];
            if(d >= 0)
            {
                T diag = csc_val[d];
                if(diag == static_cast<T>(0))
                {
                    if(lid == 0)
                    {
                        rocsparse::atomic_min(zero_pivot, col + idx_base);
                    }
                    diag = static_cast<T>(1);
                }
                inv_diag = static_cast<T>(1) / diag;
            }
        }
        const T ax = alpha * x[col * x_inc];

        // Loaded before the wait to keep the memory round trip off the dependency chain.
        const I    k0   = col_begin + lid;
        const J    row0 = (k0 < col_end) ? csc_row_ind[k0] - idx_base : col;
        const T    val0 = (k0 < col_end) ? csc_val[k0] : static_cast<T>(0);
        const bool dep0 = lower ? (row0 > col) : (row0 < col);

        bool done = false;
        while(!done)
        {
            J r = (lid == 0)
                      ? rocsparse::atomic_load(&rem[col], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
                      : static_cast<J>(0);
            r   = rocsparse::cscsv_group_broadcast<NUM_LANES>(r);

            if(r == 0)
            {
                rocsparse::acquire_done_rows();

                const T s
                    = rocsparse::atomic_load(&sum[col], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                T yc = ax - s;
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    yc = yc * inv_diag;
                }

                if(lid == 0)
                {
                    y[col * y_inc] = yc;
                }

                if(dep0)
                {
                    rocsparse::atomic_add(&sum[row0], val0 * yc);
                }
                for(I k = k0 + NUM_LANES; k < col_end; k += NUM_LANES)
                {
                    const J row = csc_row_ind[k] - idx_base;
                    if(lower ? (row > col) : (row < col))
                    {
                        rocsparse::atomic_add(&sum[row], csc_val[k] * yc);
                    }
                }

                // One release for all the additions of this lane.
                rocsparse::release_done_row_data();

                if(dep0)
                {
                    __scoped_atomic_fetch_sub(
                        &rem[row0], static_cast<J>(1), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                }
                for(I k = k0 + NUM_LANES; k < col_end; k += NUM_LANES)
                {
                    const J row = csc_row_ind[k] - idx_base;
                    if(lower ? (row > col) : (row < col))
                    {
                        __scoped_atomic_fetch_sub(
                            &rem[row], static_cast<J>(1), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                    }
                }
                done = true;
            }
            else
            {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
}
