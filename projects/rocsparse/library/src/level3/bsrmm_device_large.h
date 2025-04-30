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

#include <hip/hip_runtime.h>

namespace rocsparse
{
    template <rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrmm_large_blockdim_device(rocsparse_direction direction,
                                    rocsparse_operation trans_B,
                                    rocsparse_int       Mb,
                                    rocsparse_int       N,
                                    T                   alpha,
                                    const rocsparse_int* __restrict__ bsr_row_ptr,
                                    const rocsparse_int* __restrict__ bsr_col_ind,
                                    const T* __restrict__ bsr_val,
                                    rocsparse_int block_dim,
                                    const T* __restrict__ B,
                                    int64_t ldb,
                                    T       beta,
                                    T* __restrict__ C,
                                    int64_t              ldc,
                                    rocsparse_index_base idx_base)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int tidy = hipThreadIdx_y;

        rocsparse_int global_row = tidx + hipBlockIdx_x * block_dim;
        rocsparse_int global_col = tidy + hipBlockIdx_y * BLK_SIZE_Y;

        rocsparse_int block_row = hipBlockIdx_x;

        rocsparse_int block_row_start = 0;
        rocsparse_int block_row_end   = 0;
        if(block_row < Mb)
        {
            block_row_start = bsr_row_ptr[block_row] - idx_base;
            block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;
        }

        int64_t colB = global_col * ldb;
        int64_t colC = global_col * ldc;

        __shared__ T shared_B[BSR_BLOCK_DIM * BLK_SIZE_Y];
        __shared__ T shared_A[BSR_BLOCK_DIM * BSR_BLOCK_DIM];

        T sum = static_cast<T>(0);

        rocsparse_int index         = BSR_BLOCK_DIM * tidy + tidx;
        rocsparse_int block_dim_sqr = block_dim * block_dim;

        for(rocsparse_int k = block_row_start; k < block_row_end; k++)
        {
            rocsparse_int block_col = (bsr_col_ind[k] - idx_base);

            if(trans_B == rocsparse_operation_none)
            {
                shared_B[index] = (global_col < N && tidx < block_dim)
                                      ? B[block_dim * block_col + tidx + colB]
                                      : static_cast<T>(0);
            }
            else
            {
                shared_B[index] = (global_col < N && tidx < block_dim)
                                      ? B[global_col + ldb * (block_dim * block_col + tidx)]
                                      : static_cast<T>(0);
            }

            if(direction == rocsparse_direction_row)
            {
                if(tidx < block_dim && tidy < block_dim)
                {
                    shared_A[index] = bsr_val[block_dim_sqr * k + block_dim * tidx + tidy];
                }
            }
            else
            {
                if(tidx < block_dim && tidy < block_dim)
                {
                    shared_A[index] = bsr_val[block_dim_sqr * k + block_dim * tidy + tidx];
                }
            }

            __syncthreads();

            for(rocsparse_int j = 0; j < block_dim; j++)
            {
                sum = rocsparse::fma(
                    shared_A[BSR_BLOCK_DIM * j + tidx], shared_B[BSR_BLOCK_DIM * tidy + j], sum);
            }

            __syncthreads();
        }

        if(block_row < Mb && global_col < N && tidx < block_dim)
        {
            if(beta == static_cast<T>(0))
            {
                C[global_row + colC] = alpha * sum;
            }
            else
            {
                C[global_row + colC] = rocsparse::fma(beta, C[global_row + colC], alpha * sum);
            }
        }
    }
}
