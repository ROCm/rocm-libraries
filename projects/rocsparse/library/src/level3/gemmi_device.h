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

#include "common.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void gemmit_device(rocsparse_int m,
                                            rocsparse_int n,
                                            T             alpha,
                                            const T* __restrict__ A,
                                            int64_t lda,
                                            const rocsparse_int* __restrict__ csr_row_ptr,
                                            const rocsparse_int* __restrict__ csr_col_ind,
                                            const T* __restrict__ csr_val,
                                            T beta,
                                            T* __restrict__ C,
                                            int64_t              ldc,
                                            rocsparse_index_base base)
    {
        rocsparse_int row = hipBlockIdx_y;
        rocsparse_int col = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        // Do not run out of bounds
        if(col >= m)
        {
            return;
        }

        for(rocsparse_int i = 0; i < n; i += 65535)
        {
            if((row + i) < n)
            {
                // Row entry into B
                rocsparse_int row_begin = csr_row_ptr[row + i] - base;
                rocsparse_int row_end   = csr_row_ptr[row + i + 1] - base;

                // Accumulator
                T sum = static_cast<T>(0);

                // Loop over the column indices of B of the current row
                for(rocsparse_int k = row_begin; k < row_end; ++k)
                {
                    rocsparse_int col_B = csr_col_ind[k] - base;
                    T             val_B = csr_val[k];
                    T             val_A = A[col_B * lda + col];

                    sum = rocsparse::fma(val_A, val_B, sum);
                }

                // Write result back to C
                if(beta != static_cast<T>(0))
                {
                    C[(row + i) * ldc + col]
                        = rocsparse::fma(beta, C[(row + i) * ldc + col], alpha * sum);
                }
                else
                {
                    C[(row + i) * ldc + col] = alpha * sum;
                }
            }
        }
    }
}
