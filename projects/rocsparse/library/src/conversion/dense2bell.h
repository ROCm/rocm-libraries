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

#include "rocsparse_grid.hpp"

#include <hip/hip_runtime.h>

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void dense2bell_nnz_kernel(int64_t         m,
                               int64_t         n,
                               int64_t         mb,
                               const T*        A,
                               int64_t         ld,
                               rocsparse_order order,
                               int64_t         ell_block_size,
                               int64_t*        nnzb_per_row)
    {
        uint32_t tid = hipThreadIdx_x;

        __shared__ uint32_t shared[BLOCKSIZE];

        // Each window of stride columns holds whole block columns: BLOCKSIZE / ell_block_size
        // of them, or a single block column when it is wider than the thread block.
        const int64_t stride = (ell_block_size <= BLOCKSIZE)
                                   ? (BLOCKSIZE / ell_block_size) * ell_block_size
                                   : ell_block_size;

        for(int64_t bid = hipBlockIdx_x; bid < mb; bid += hipGridDim_x)
        {
            int64_t block_cols_per_block_row = 0;

            const int64_t row_begin = ell_block_size * bid;
            const int64_t rows_in   = rocsparse::min(ell_block_size, m - row_begin);

            for(int64_t i = 0; i < n; i += stride)
            {
                shared[tid] = 0;
                __syncthreads();

                const int64_t cols_in = rocsparse::min(stride, n - i);

                for(int64_t l = tid; l < cols_in; l += BLOCKSIZE)
                {
                    const int64_t j = i + l;

                    for(int64_t k = 0; k < rows_in; k++)
                    {
                        const T val = (order == rocsparse_order_row) ? A[ld * (row_begin + k) + j]
                                                                     : A[ld * j + (row_begin + k)];

                        if(val != static_cast<T>(0))
                        {
                            shared[l / ell_block_size] = 1;
                            break;
                        }
                    }
                }

                __syncthreads();

                rocsparse::blockreduce_sum<BLOCKSIZE>(tid, shared);

                if(tid == 0)
                {
                    block_cols_per_block_row += shared[0];
                }
            }

            if(tid == 0)
            {
                nnzb_per_row[bid] = block_cols_per_block_row;
            }
        }
    }

    // First stage of the max-reduction: each thread block reduces a grid-strided
    // slice of nnzb_per_row to a single value and writes it to workspace[blockIdx].
    // The workspace array must have one entry per launched block (gridDim.x).
    template <uint32_t BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void dense2bell_max_nnzb_part1_kernel(int64_t        mb,
                                          const int64_t* nnzb_per_row,
                                          int64_t*       workspace)
    {
        uint32_t tid = hipThreadIdx_x;

        __shared__ int64_t shared[BLOCKSIZE];

        int64_t gid    = int64_t(hipBlockIdx_x) * BLOCKSIZE + tid;
        int64_t stride = int64_t(BLOCKSIZE) * hipGridDim_x;

        int64_t local_max = 0;
        for(int64_t i = gid; i < mb; i += stride)
        {
            local_max = rocsparse::max(local_max, nnzb_per_row[i]);
        }

        shared[tid] = local_max;
        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(tid, shared);

        if(tid == 0)
        {
            workspace[hipBlockIdx_x] = shared[0];
        }
    }

    // Second stage of the max-reduction: a single block of BLOCKSIZE threads reduces
    // the workspace array (of size BLOCKSIZE) to a single maximum, written to workspace[0].
    template <uint32_t BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void dense2bell_max_nnzb_part2_kernel(int64_t* workspace)
    {
        uint32_t tid = hipThreadIdx_x;

        __shared__ int64_t shared[BLOCKSIZE];

        shared[tid] = workspace[tid];
        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(tid, shared);

        if(tid == 0)
        {
            workspace[0] = shared[0];
        }
    }

    // Fills the BELL value and column-index arrays from the dense matrix.
    //
    // One thread block is responsible for a single block-row (hipBlockIdx_x). It walks the
    // block-columns in increasing order and, for each block-column that contains at least one
    // non-zero, appends it to the next free ELL slot: the block-column index is stored in
    // bell_col_ind and the ell_block_size x ell_block_size block of values is copied into
    // bell_val. Unused (padded) slots are marked with an out-of-range column index.
    //
    // Layout (cuSPARSE/bellmm compatible, ell_cols == ell_block_width * ell_block_size):
    //   bell_col_ind[block_row * ell_block_width + slot]
    //   bell_val[(block_row * ell_block_size + r) * ell_cols + slot * ell_block_size + c]
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void dense2bell_fill_kernel(int64_t              m,
                                int64_t              n,
                                int64_t              mb,
                                const T*             A,
                                int64_t              ld,
                                rocsparse_order      order,
                                int64_t              ell_block_size,
                                int64_t              ell_block_width,
                                rocsparse_index_base base,
                                T*                   bell_val,
                                I*                   bell_col_ind)
    {
        const uint32_t tid = hipThreadIdx_x;

        __shared__ uint32_t s_found[BLOCKSIZE];
        __shared__ int64_t  s_slot;

        const int64_t ell_cols = ell_block_width * ell_block_size;
        const int64_t nb       = rocsparse::ceil_div(n, ell_block_size);

        for(int64_t bid = hipBlockIdx_x; bid < mb; bid += hipGridDim_x)
        {
            if(tid == 0)
            {
                s_slot = 0;
            }
            __syncthreads();

            const int64_t row_begin = bid * ell_block_size;
            const int64_t rows_in   = rocsparse::min(ell_block_size, m - row_begin);

            for(int64_t jb = 0; jb < nb; jb++)
            {
                // Only the part of the block inside the matrix is read; the rest of the
                // ELL slot keeps the zeros the launcher wrote into bell_val.
                const int64_t col_begin = jb * ell_block_size;
                const int64_t cols_in   = rocsparse::min(ell_block_size, n - col_begin);
                const int64_t elems     = rows_in * cols_in;

                // Detect whether this block-column contains any non-zero.
                uint32_t found = 0;
                for(int64_t e = tid; e < elems; e += BLOCKSIZE)
                {
                    const int64_t r  = e / cols_in;
                    const int64_t c  = e - r * cols_in;
                    const int64_t gr = row_begin + r;
                    const int64_t gc = col_begin + c;

                    const T val
                        = (order == rocsparse_order_row) ? A[ld * gr + gc] : A[ld * gc + gr];
                    if(val != static_cast<T>(0))
                    {
                        found = 1;
                    }
                }

                s_found[tid] = found;
                __syncthreads();

                rocsparse::blockreduce_max<BLOCKSIZE>(tid, s_found);

                // A block row with more non-zero blocks than ell_block_width slots (ell_cols
                // smaller than the analysis result) is truncated instead of written past the
                // ends of bell_val and bell_col_ind. s_found[0] and s_slot are block-uniform.
                if(s_found[0] != 0 && s_slot < ell_block_width)
                {
                    const int64_t slot = s_slot;

                    for(int64_t e = tid; e < elems; e += BLOCKSIZE)
                    {
                        const int64_t r  = e / cols_in;
                        const int64_t c  = e - r * cols_in;
                        const int64_t gr = row_begin + r;
                        const int64_t gc = col_begin + c;

                        bell_val[gr * ell_cols + slot * ell_block_size + c]
                            = (order == rocsparse_order_row) ? A[ld * gr + gc] : A[ld * gc + gr];
                    }

                    // Ensure every thread has read s_slot (into slot) before thread 0 advances it.
                    // s_found[0] is uniform across the block, so this barrier is not divergent.
                    __syncthreads();

                    if(tid == 0)
                    {
                        bell_col_ind[bid * ell_block_width + slot] = static_cast<I>(jb + base);
                        s_slot                                     = slot + 1;
                    }
                }

                __syncthreads();
            }

            // Mark the remaining (padded) slots with an out-of-range column index.
            const int64_t occupied = s_slot;
            for(int64_t slot = occupied + tid; slot < ell_block_width; slot += BLOCKSIZE)
            {
                bell_col_ind[bid * ell_block_width + slot] = static_cast<I>(base) - 1;
            }

            // Ensure every thread has read s_slot before it is reset for the next block-row.
            __syncthreads();
        }
    }
}
