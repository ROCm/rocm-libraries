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

#include <hip/hip_runtime.h>

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void dense2bell_nnz_kernel(int64_t         m,
                               int64_t         n,
                               const T*        A,
                               int64_t         ld,
                               rocsparse_order order,
                               int64_t         ell_block_size,
                               int64_t*        nnzb_per_row)
    {
        uint32_t bid = hipBlockIdx_x;
        uint32_t tid = hipThreadIdx_x;

        __shared__ uint32_t shared[BLOCKSIZE];

        int64_t block_cols_per_block_row = 0;

        int64_t stride = (BLOCKSIZE / ell_block_size) * ell_block_size;

        int64_t i = 0;
        while(i < n)
        {
            int64_t j = i + tid;

            shared[tid] = 0;
            __syncthreads();

            if(j < n && tid < stride)
            {
                if(order == rocsparse_order_row)
                {
                    for(uint32_t k = 0; k < ell_block_size; k++)
                    {
                        const T val = (ell_block_size * bid + k) < m
                                          ? A[ld * (ell_block_size * bid + k) + j]
                                          : static_cast<T>(0);

                        if(val != static_cast<T>(0))
                        {
                            shared[tid / ell_block_size] = 1;
                        }
                    }
                }
                else
                {
                    for(uint32_t k = 0; k < ell_block_size; k++)
                    {
                        const T val = (ell_block_size * bid + k) < m
                                          ? A[ld * j + (ell_block_size * bid + k)]
                                          : static_cast<T>(0);

                        if(val != static_cast<T>(0))
                        {
                            shared[tid / ell_block_size] = 1;
                        }
                    }
                }
            }

            __syncthreads();

            rocsparse::blockreduce_sum<BLOCKSIZE>(tid, shared);

            if(tid == 0)
            {
                block_cols_per_block_row += shared[0];
            }

            i += stride;
        }

        if(tid == 0)
        {
            nnzb_per_row[bid] = block_cols_per_block_row;
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
                                const T*             A,
                                int64_t              ld,
                                rocsparse_order      order,
                                int64_t              ell_block_size,
                                int64_t              ell_block_width,
                                rocsparse_index_base base,
                                T*                   bell_val,
                                I*                   bell_col_ind)
    {
        const int64_t  bid = hipBlockIdx_x;
        const uint32_t tid = hipThreadIdx_x;

        __shared__ uint32_t s_found[BLOCKSIZE];
        __shared__ int64_t  s_slot;

        if(tid == 0)
        {
            s_slot = 0;
        }
        __syncthreads();

        const int64_t ell_cols    = ell_block_width * ell_block_size;
        const int64_t block_elems = ell_block_size * ell_block_size;
        const int64_t nb          = (n + ell_block_size - 1) / ell_block_size;

        for(int64_t jb = 0; jb < nb; jb++)
        {
            // Detect whether this block-column contains any non-zero.
            uint32_t found = 0;
            for(int64_t e = tid; e < block_elems; e += BLOCKSIZE)
            {
                const int64_t r  = e / ell_block_size;
                const int64_t c  = e - r * ell_block_size;
                const int64_t gr = bid * ell_block_size + r;
                const int64_t gc = jb * ell_block_size + c;

                if(gr < m && gc < n)
                {
                    const T val
                        = (order == rocsparse_order_row) ? A[ld * gr + gc] : A[ld * gc + gr];
                    if(val != static_cast<T>(0))
                    {
                        found = 1;
                    }
                }
            }

            s_found[tid] = found;
            __syncthreads();

            rocsparse::blockreduce_max<BLOCKSIZE>(tid, s_found);

            if(s_found[0] != 0)
            {
                const int64_t slot = s_slot;

                // Copy the whole block (including its structural zeros) into the ELL slot.
                for(int64_t e = tid; e < block_elems; e += BLOCKSIZE)
                {
                    const int64_t r  = e / ell_block_size;
                    const int64_t c  = e - r * ell_block_size;
                    const int64_t gr = bid * ell_block_size + r;
                    const int64_t gc = jb * ell_block_size + c;

                    if(gr < m)
                    {
                        T val = static_cast<T>(0);
                        if(gc < n)
                        {
                            val = (order == rocsparse_order_row) ? A[ld * gr + gc]
                                                                 : A[ld * gc + gr];
                        }
                        bell_val[gr * ell_cols + slot * ell_block_size + c] = val;
                    }
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
    }
}
