/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    // CSR to COO matrix conversion kernel
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2coo_kernel(J                    m,
                        const I*             csr_row_ptr_begin,
                        const I*             csr_row_ptr_end,
                        J*                   coo_row_ind,
                        rocsparse_index_base idx_base)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");
        J tid = hipThreadIdx_x;
        J lid = tid & (WF_SIZE - 1);
        J wid = tid / WF_SIZE;

        __shared__ int all_short_rows;
        __shared__ int short_rows[BLOCKSIZE / WF_SIZE];

        // AISPARSE-685. The launcher sizes grid.x as ((int64_t)WF_SIZE * m - 1) /
        // BLOCKSIZE + 1. That cast is deliberate and correct -- it is the reference
        // idiom of this epic -- but the 64-bit result was then assigned into a dim3,
        // which narrows it back to unsigned int, and this kernel indexed rows with
        // hipBlockIdx_x alone with no x-stride. The launcher now clamps that same
        // 64-bit expression against the device grid.x limit, and the loop below
        // covers the rows the clamp drops. The long-row path further down does have a
        // stride loop, but it strides over the non-zero index within one row, not
        // over the grid, so it never helped here.
        //
        // Block-uniform stride bound: row_base is built from hipBlockIdx_x,
        // hipGridDim_x, the kernel argument m and compile-time constants only, so
        // every thread of a block runs the same number of iterations. That is what
        // makes the three __syncthreads() below legal. There is no `return` in the
        // body, so no thread can leave the loop early and strand the others at a
        // barrier.
        constexpr int64_t rows_per_block = BLOCKSIZE / WF_SIZE;
        const int64_t     row_stride     = rows_per_block * hipGridDim_x;

        for(int64_t row_base = rows_per_block * hipBlockIdx_x; row_base < m; row_base += row_stride)
        {
            all_short_rows = 1;

            __syncthreads();

            J row = static_cast<J>(row_base) + wid;

            I start = (row < m) ? csr_row_ptr_begin[row] - idx_base : static_cast<I>(0);
            I end   = (row < m) ? csr_row_ptr_end[row] - idx_base : static_cast<I>(0);

            int short_row = (end - start <= 8 * WF_SIZE) ? 1 : 0;

            if(short_row)
            {
                for(I j = start + lid; j < end; j += WF_SIZE)
                {
                    coo_row_ind[j] = row + idx_base;
                }
            }
            else
            {
                all_short_rows = 0;
            }

            short_rows[wid] = short_row;

            __syncthreads();

            // Process any long rows
            if(all_short_rows == 0)
            {
                for(int i = 0; i < (BLOCKSIZE / WF_SIZE); i++)
                {
                    if(short_rows[i] == 0)
                    {
                        J long_row = static_cast<J>(row_base) + i;

                        I start = (long_row < m) ? csr_row_ptr_begin[long_row] - idx_base
                                                 : static_cast<I>(0);
                        I end   = (long_row < m) ? csr_row_ptr_end[long_row] - idx_base
                                                 : static_cast<I>(0);

                        for(I j = start + tid; j < end; j += BLOCKSIZE)
                        {
                            coo_row_ind[j] = long_row + idx_base;
                        }
                    }
                }
            }

            // Separates this iteration's reads of all_short_rows / short_rows from
            // the next iteration's writes to them.
            __syncthreads();
        }
    }
}
