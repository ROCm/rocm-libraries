/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void coo2dense_kernel(I                    m,
                          I                    n,
                          int64_t              nnz,
                          int64_t              lda,
                          rocsparse_index_base base,
                          const T*             coo_val,
                          const I*             coo_row_ind,
                          const I*             coo_col_ind,
                          T*                   A,
                          rocsparse_order      order)
    {
        int64_t gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= nnz)
        {
            return;
        }

        I row = coo_row_ind[gid] - base;
        I col = coo_col_ind[gid] - base;
        T val = coo_val[gid];

        if(order == rocsparse_order_column)
        {
            A[lda * col + row] = val;
        }
        else
        {
            A[lda * row + col] = val;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void coo2dense_aos_kernel(I                    m,
                              I                    n,
                              int64_t              nnz,
                              int64_t              lda,
                              rocsparse_index_base base,
                              const T*             coo_val,
                              const I*             coo_ind,
                              T*                   A,
                              rocsparse_order      order)
    {
        const auto NUM_THREADS = hipGridDim_x * BLOCKSIZE;

        const auto gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        for(auto idx = gid; idx < nnz; idx += NUM_THREADS)
        {
            I row = coo_ind[2 * idx] - base;
            I col = coo_ind[2 * idx + 1] - base;
            T val = coo_val[idx];

            if(order == rocsparse_order_column)
            {
                A[lda * col + row] = val;
            }
            else
            {
                A[lda * row + col] = val;
            }
        }
    }
}
