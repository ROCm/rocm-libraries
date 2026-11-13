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
    ROCSPARSE_DEVICE_ILF T cscsv_group_broadcast(T v)
    {
        return __shfl(v, 0, NUM_LANES);
    }

    // Task t of the solve, run by a group of NUM_LANES lanes.
    template <uint32_t NUM_LANES, uint32_t NUM_ITEMS, typename I, typename J, typename T>
    ROCSPARSE_DEVICE_ILF void cscsv_solve_task(int64_t  t,
                                               uint32_t lid,
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
                                               J*                   zero_pivot,
                                               rocsparse_index_base idx_base,
                                               rocsparse_fill_mode  fill_mode,
                                               rocsparse_diag_type  diag_type)
    {
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

        J    rows[NUM_ITEMS];
        T    vals[NUM_ITEMS];
        bool deps[NUM_ITEMS];
        for(uint32_t i = 0; i < NUM_ITEMS; ++i)
        {
            const int64_t k = static_cast<int64_t>(kbeg) + lid + i * NUM_LANES;
            rows[i]         = (k < col_end) ? csc_row_ind[k] - idx_base : col;
            vals[i]         = (k < col_end) ? csc_val[k] : static_cast<T>(0);
            deps[i]         = lower ? (rows[i] > col) : (rows[i] < col);
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

                for(uint32_t i = 0; i < NUM_ITEMS; ++i)
                {
                    if(deps[i])
                    {
                        rocsparse::atomic_add(&sum[rows[i]], -(vals[i] * yc));
                    }
                }

                // One release for all the additions of this lane.
                rocsparse::release_done_row_data();

                for(uint32_t i = 0; i < NUM_ITEMS; ++i)
                {
                    if(deps[i])
                    {
                        rocsparse::atomic_fetch_sub(&rem[rows[i]],
                                                    static_cast<J>(1),
                                                    __ATOMIC_RELAXED,
                                                    __MEMORY_SCOPE_DEVICE);
                    }
                }
                done = true;
            }
        }
    }

    // One group of NUM_LANES lanes per task, in elimination order; the first task of a column
    // stores y. Blocks take an atomic ticket instead of hipBlockIdx_x, so a group only waits on
    // groups that are already running.
    template <uint32_t BLOCKSIZE,
              uint32_t NUM_LANES,
              uint32_t NUM_ITEMS,
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

        const uint32_t lid             = hipThreadIdx_x & (NUM_LANES - 1);
        const uint32_t wid             = hipThreadIdx_x / NUM_LANES;
        const int64_t  tasks_per_block = BLOCKSIZE / NUM_LANES;

        // With a clamped grid, blocks take tickets until all the tasks are done. The tickets
        // keep the elimination order, so this does not change the deadlock argument.
        const bool clamped = static_cast<int64_t>(hipGridDim_x) * tasks_per_block < num_tasks;
        while(true)
        {
            if(hipThreadIdx_x == 0)
            {
                block_id = rocsparse::atomic_add(ticket, 1);
            }
            __syncthreads();

            const int64_t first = static_cast<int64_t>(block_id) * tasks_per_block;
            if(first >= num_tasks)
            {
                break;
            }

            const int64_t t = first + wid;
            if(t < num_tasks)
            {
                rocsparse::cscsv_solve_task<NUM_LANES, NUM_ITEMS>(t,
                                                                  lid,
                                                                  csc_col_ptr,
                                                                  csc_row_ind,
                                                                  csc_val,
                                                                  csc_diag_ind,
                                                                  task_col,
                                                                  task_begin,
                                                                  y,
                                                                  y_inc,
                                                                  rem,
                                                                  sum,
                                                                  zero_pivot,
                                                                  idx_base,
                                                                  fill_mode,
                                                                  diag_type);
            }

            if(!clamped)
            {
                break;
            }
            __syncthreads();
        }
    }
}
