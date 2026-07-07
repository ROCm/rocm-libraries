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
        int64_t bid = hipBlockIdx_x;
        int64_t tid = hipThreadIdx_x;

        __shared__ uint32_t shared[BLOCKSIZE];

        uint32_t block_cols_per_block_row = 0;

        uint32_t stride = (BLOCKSIZE / ell_block_size) * ell_block_size;

        uint32_t i = 0;
        while(i < n)
        {
            uint32_t j = stride * i + tid;

            shared[tid] = 0;
            __syncthreads();

            uint32_t nnz_found = 0;

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
                            nnz_found = 1;
                        }
                    }
                }
            }

            shared[tid % ell_block_size] = nnz_found;
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
}
