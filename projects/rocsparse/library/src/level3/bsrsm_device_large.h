/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t NCOLS, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsm_upper_large_kernel(rocsparse_int        mb,
                                  rocsparse_int        nrhs,
                                  const rocsparse_int* bsr_row_ptr,
                                  const rocsparse_int* bsr_col_ind,
                                  const T*             bsr_val,
                                  rocsparse_int        block_dim,
                                  T*                   X,
                                  rocsparse_int        ldx,
                                  int*                 done_array,
                                  const rocsparse_int* map,
                                  rocsparse_int*       zero_pivot,
                                  rocsparse_index_base idx_base,
                                  rocsparse_diag_type  diag_type,
                                  rocsparse_direction  dir)
    {
        static constexpr uint32_t WFSIZE = BLOCKSIZE / NCOLS;

        const int lid = threadIdx.x & (WFSIZE - 1);

        // Index into the row map
        const rocsparse_int idx = blockIdx.x % mb;

        // Get the BSR row this thread block will operate on
        const rocsparse_int row = map[idx];

        // Get the id of the rhs, this thread block will operate on
        const rocsparse_int id = blockIdx.x / mb * mb;

        // Current row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Column index (rhs) into X
        const rocsparse_int col_X = blockIdx.x / mb * NCOLS + threadIdx.x / WFSIZE;

        // Initialize local_col with mb
        rocsparse_int local_col = mb;

        // Loop over current row
        rocsparse_int j;
        for(j = row_end - 1; j >= row_begin; --j)
        {
            // Current column index
            local_col = bsr_col_ind[j] - idx_base;

            // Processing upper triangular

            // Ignore all diagonal entries and below
            if(local_col <= row)
            {
                break;
            }

            // Spin loop until dependency has been resolved
            if(threadIdx.x == 0)
            {
                rocsparse::spin_loop<SLEEP>(&done_array[local_col + id], __HIP_MEMORY_SCOPE_AGENT);
            }

            // Make sure updated X is visible globally
            __threadfence();

            // Wait for spin looping thread to finish as the whole block depends on this row
            __syncthreads();

            // Local sum computation

            // Do not run out of bounds
            if(col_X < nrhs)
            {
                // Loop over rows of the BSR block
                for(int bi = lid; bi < block_dim; bi += WFSIZE)
                {
                    // Local sum accumulator
                    T sum = static_cast<T>(0);

                    // Loop over columns of the BSR block
                    for(int bj = 0; bj < block_dim; ++bj)
                    {
                        sum = rocsparse::fma(bsr_val[BSR_IND(j, bi, bj, dir)],
                                             X[(block_dim * local_col + bj) * ldx + col_X],
                                             sum);
                    }

                    // Write local sum to X
                    X[(block_dim * row + bi) * ldx + col_X] -= sum;
                }
            }
        }

        bool pivot = false;

        // Process diagonal
        if(row < mb && row == local_col && col_X < nrhs)
        {
            // Loop over rows of the BSR block
            for(int bi = block_dim - 1; bi >= 0; --bi)
            {
                // Load diagonal matrix entry
                const T diag = (diag_type == rocsparse_diag_type_non_unit)
                                   ? bsr_val[BSR_IND(j, bi, bi, dir)]
                                   : static_cast<T>(1);

                // Load result of bi-th BSR row
                T val = X[(block_dim * row + bi) * ldx + col_X];

                // Check for numerical pivot
                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    // Divide result of bi-th BSR row by diagonal entry
                    X[(block_dim * row + bi) * ldx + col_X] = val /= diag;
                }

                // Update remaining non-diagonal entries
                for(int bj = lid; bj < bi; bj += WFSIZE)
                {
                    X[(block_dim * row + bj) * ldx + col_X]
                        -= val * bsr_val[BSR_IND(j, bj, bi, dir)];
                }
            }
        }

        // Make sure X is written to global memory before setting row is done flag
        __threadfence();

        // Wait for all threads to finish the threadfence before we mark the row "done"
        __syncthreads();

        if(row < mb && threadIdx.x == 0)
        {
            // Write "row is done" flag
            __hip_atomic_store(
                &done_array[row + id], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t NCOLS, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsm_lower_large_kernel(rocsparse_int        mb,
                                  rocsparse_int        nrhs,
                                  const rocsparse_int* bsr_row_ptr,
                                  const rocsparse_int* bsr_col_ind,
                                  const T*             bsr_val,
                                  rocsparse_int        block_dim,
                                  T*                   X,
                                  rocsparse_int        ldx,
                                  int*                 done_array,
                                  const rocsparse_int* map,
                                  rocsparse_int*       zero_pivot,
                                  rocsparse_index_base idx_base,
                                  rocsparse_diag_type  diag_type,
                                  rocsparse_direction  dir)
    {
        static constexpr uint32_t WFSIZE = BLOCKSIZE / NCOLS;

        const int lid = threadIdx.x & (WFSIZE - 1);

        // Index into the row map
        const rocsparse_int idx = blockIdx.x % mb;

        // Get the BSR row this thread block will operate on
        const rocsparse_int row = map[idx];

        // Get the id of the rhs, this thread block will operate on
        const rocsparse_int id = blockIdx.x / mb * mb;

        // Current row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Column index into X
        const rocsparse_int col_X = blockIdx.x / mb * NCOLS + threadIdx.x / WFSIZE;

        // Initialize local_col with mb
        rocsparse_int local_col = mb;

        // Loop over current row
        rocsparse_int j;
        for(j = row_begin; j < row_end; ++j)
        {
            // Current column index
            local_col = bsr_col_ind[j] - idx_base;

            // Processing lower triangular

            // Ignore all diagonal entries and above
            if(local_col >= row)
            {
                break;
            }

            // Spin loop until dependency has been resolved
            if(threadIdx.x == 0)
            {
                rocsparse::spin_loop<SLEEP>(&done_array[local_col + id], __HIP_MEMORY_SCOPE_AGENT);
            }

            // Make sure updated X is visible globally
            __threadfence();

            // Wait for spin looping thread to finish as the whole block depends on this row
            __syncthreads();

            // Local sum computation

            // Do not run out of bounds
            if(col_X < nrhs)
            {
                // Loop over rows of the BSR block
                for(int bi = lid; bi < block_dim; bi += WFSIZE)
                {
                    // Local sum accumulator
                    T sum = static_cast<T>(0);

                    // Loop over columns of the BSR block
                    for(int bj = 0; bj < block_dim; ++bj)
                    {
                        sum = rocsparse::fma(bsr_val[BSR_IND(j, bi, bj, dir)],
                                             X[(block_dim * local_col + bj) * ldx + col_X],
                                             sum);
                    }

                    // Write local sum to X
                    X[(block_dim * row + bi) * ldx + col_X] -= sum;
                }
            }
        }

        bool pivot = false;

        // Process diagonal
        if(row < mb && row == local_col && col_X < nrhs)
        {
            // Loop over rows of the BSR block
            for(int bi = 0; bi < block_dim; ++bi)
            {
                // Load diagonal matrix entry
                T diag = (diag_type == rocsparse_diag_type_non_unit)
                             ? bsr_val[BSR_IND(j, bi, bi, dir)]
                             : static_cast<T>(1);

                // Load result of bi-th BSR row
                T val = X[(block_dim * row + bi) * ldx + col_X];

                // Check for numerical pivot
                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    // Divide result of bi-th BSR row by diagonal entry
                    X[(block_dim * row + bi) * ldx + col_X] = val /= diag;
                }

                // Update remaining non-diagonal entries
                for(int bj = bi + lid + 1; bj < block_dim; bj += WFSIZE)
                {
                    X[(block_dim * row + bj) * ldx + col_X]
                        -= val * bsr_val[BSR_IND(j, bj, bi, dir)];
                }
            }
        }

        // Make sure X is written to global memory before setting row is done flag
        __threadfence();

        // Wait for all threads to finish the threadfence before we mark the row "done"
        __syncthreads();

        if(row < mb && threadIdx.x == 0)
        {
            // Write "row is done" flag
            __hip_atomic_store(
                &done_array[row + id], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }
}
