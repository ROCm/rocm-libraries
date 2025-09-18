/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
        J tid = hipThreadIdx_x;
        J lid = tid & (WF_SIZE - 1);
        J wid = tid / WF_SIZE;

        __shared__ int all_short_rows;
        __shared__ int short_rows[BLOCKSIZE / WF_SIZE];

        all_short_rows = 1;

        __syncthreads();

        J row = (BLOCKSIZE / WF_SIZE) * hipBlockIdx_x + wid;

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
                    J long_row = (BLOCKSIZE / WF_SIZE) * hipBlockIdx_x + i;

                    I start = (long_row < m) ? csr_row_ptr_begin[long_row] - idx_base
                                             : static_cast<I>(0);
                    I end
                        = (long_row < m) ? csr_row_ptr_end[long_row] - idx_base : static_cast<I>(0);

                    for(I j = start + tid; j < end; j += BLOCKSIZE)
                    {
                        coo_row_ind[j] = long_row + idx_base;
                    }
                }
            }
        }
    }
}
