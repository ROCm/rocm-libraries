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

// CSC solve of A * y = alpha * x with A triangular in CSC: sum[r] starts at alpha * x_r; once
// y_c is known, column c subtracts a_rc * y_c from sum[r] and decrements rem[r]; unknown r is
// ready when rem[r] is zero.
namespace rocsparse
{
    template <uint32_t NUM_LANES, typename T>
    __device__ __forceinline__ T cscsv_group_broadcast(T v)
    {
        return __shfl(v, 0, NUM_LANES);
    }

    // For each batch instance, resets the dependency counters from the analysis and starts the
    // partial sums at alpha * x. The solve does not read x, so x and y may be the same vector.
    template <uint32_t BLOCKSIZE, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_init_kernel(J       m,
                           int64_t batch_count,
                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                           int64_t alpha_stride,
                           const J* __restrict__ dependency_count,
                           const T* __restrict__ x,
                           int64_t x_inc,
                           int64_t x_stride,
                           J* __restrict__ rem,
                           T* __restrict__ sum,
                           bool is_host_mode)
    {
        for(int64_t i = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
            i < m * batch_count;
            i += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            const int64_t batch = i / m;
            const int64_t row   = i % m;
            const T       alpha
                = is_host_mode ? alpha_union.value : alpha_union.pointer[batch * alpha_stride];
            rem[i] = dependency_count[row];
            sum[i] = alpha * x[batch * x_stride + row * x_inc];
        }
    }

    // Tasks per column: one per chunk of NUM_LANES * ITEMS entries, at least one.
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_count_tasks_kernel(int64_t m,
                                  const I* __restrict__ csc_col_ptr,
                                  int64_t chunk,
                                  I* __restrict__ task_count)
    {
        const int64_t col = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(col < m)
        {
            const int64_t len = csc_col_ptr[col + 1] - csc_col_ptr[col];
            task_count[col]   = static_cast<I>(
                rocsparse::max(static_cast<int64_t>(1), (len + chunk - 1) / chunk));
        }
    }

    // One thread per nonzero, so that a long column does not serialize the analysis.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_analysis_entries_kernel(int64_t m,
                                       const I* __restrict__ csc_col_ptr,
                                       const J* __restrict__ csc_row_ind,
                                       I* __restrict__ csc_diag_ind,
                                       J* __restrict__ dependency_count,
                                       rocsparse_index_base idx_base,
                                       rocsparse_fill_mode  fill_mode)
    {
        const int64_t nnz   = csc_col_ptr[m] - idx_base;
        const bool    lower = (fill_mode == rocsparse_fill_mode_lower);
        for(int64_t k = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x; k < nnz;
            k += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            // Largest column c with csc_col_ptr[c] <= k.
            int64_t lo = 0;
            int64_t hi = m;
            while(hi - lo > 1)
            {
                const int64_t mid = (lo + hi) / 2;
                if(csc_col_ptr[mid] - idx_base <= k)
                {
                    lo = mid;
                }
                else
                {
                    hi = mid;
                }
            }
            const int64_t col = lo;
            const int64_t row = csc_row_ind[k] - idx_base;
            if(row == col)
            {
                csc_diag_ind[col] = static_cast<I>(k);
            }
            else if(lower ? (row > col) : (row < col))
            {
                rocsparse::atomic_add(&dependency_count[row], static_cast<J>(1));
            }
        }
    }

    // Structural zero pivot: the first column without a diagonal entry.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_analysis_pivot_kernel(int64_t m,
                                     const I* __restrict__ csc_diag_ind,
                                     J* __restrict__ zero_pivot,
                                     rocsparse_index_base idx_base,
                                     rocsparse_diag_type  diag_type)
    {
        const int64_t col = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(col < m && diag_type == rocsparse_diag_type_non_unit && csc_diag_ind[col] < 0)
        {
            rocsparse::atomic_min(zero_pivot, static_cast<J>(col + idx_base));
        }
    }

    // Tasks in elimination order: column order, reversed for an upper triangle.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_expand_tasks_kernel(int64_t m,
                                   const I* __restrict__ csc_col_ptr,
                                   const I* __restrict__ task_ptr,
                                   int64_t chunk,
                                   int64_t num_tasks,
                                   bool    reverse,
                                   J* __restrict__ task_col,
                                   I* __restrict__ task_begin,
                                   rocsparse_index_base idx_base)
    {
        const int64_t col = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(col < m)
        {
            const I begin = csc_col_ptr[col] - idx_base;
            for(I t = task_ptr[col]; t < task_ptr[col + 1]; ++t)
            {
                const int64_t pos = reverse ? (num_tasks - 1 - t) : t;
                task_col[pos]     = static_cast<J>(col);
                task_begin[pos]   = begin + static_cast<I>((t - task_ptr[col]) * chunk);
            }
        }
    }

    // One group of NUM_LANES lanes per task, in elimination order; the first task of a column
    // stores y. Blocks take an atomic ticket instead of hipBlockIdx_x, so a group only waits on
    // groups that are already running.
    template <uint32_t BLOCKSIZE,
              uint32_t NUM_LANES,
              uint32_t ITEMS,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void cscsv_solve_device(int64_t num_tasks,
                                                 const I* __restrict__ csc_col_ptr,
                                                 const J* __restrict__ csc_row_ind,
                                                 const T* __restrict__ csc_val,
                                                 const I* __restrict__ csc_diag_ind,
                                                 const J* __restrict__ task_col,
                                                 const I* __restrict__ task_begin,
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
        const int64_t  t   = static_cast<int64_t>(block_id) * (BLOCKSIZE / NUM_LANES) + wid;

        if(t >= num_tasks)
        {
            return;
        }

        const bool lower     = (fill_mode == rocsparse_fill_mode_lower);
        const J    col       = task_col[t];
        const I    kbeg      = task_begin[t];
        const I    col_begin = csc_col_ptr[col] - idx_base;
        const I    col_end   = csc_col_ptr[col + 1] - idx_base;
        const bool owner     = (kbeg == col_begin);

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
                    if(owner && lid == 0)
                    {
                        rocsparse::atomic_min(zero_pivot, col + idx_base);
                    }
                    diag = static_cast<T>(1);
                }
                inv_diag = static_cast<T>(1) / diag;
            }
        }

        J    rows[ITEMS];
        T    vals[ITEMS];
        bool deps[ITEMS];
        for(uint32_t i = 0; i < ITEMS; ++i)
        {
            const I k = kbeg + lid + i * NUM_LANES;
            rows[i]   = (k < col_end) ? csc_row_ind[k] - idx_base : col;
            vals[i]   = (k < col_end) ? csc_val[k] : static_cast<T>(0);
            deps[i]   = lower ? (rows[i] > col) : (rows[i] < col);
        }

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
                T yc = s;
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    yc = yc * inv_diag;
                }

                if(owner && lid == 0)
                {
                    y[col * y_inc] = yc;
                }

                for(uint32_t i = 0; i < ITEMS; ++i)
                {
                    if(deps[i])
                    {
                        rocsparse::atomic_add(&sum[rows[i]], -(vals[i] * yc));
                    }
                }

                // One release for all the additions of this lane.
                rocsparse::release_done_row_data();

                for(uint32_t i = 0; i < ITEMS; ++i)
                {
                    if(deps[i])
                    {
                        __scoped_atomic_fetch_sub(&rem[rows[i]],
                                                  static_cast<J>(1),
                                                  __ATOMIC_RELAXED,
                                                  __MEMORY_SCOPE_DEVICE);
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
