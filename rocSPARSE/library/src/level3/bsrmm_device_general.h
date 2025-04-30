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
    template <uint32_t BSR_BLOCK_DIM,
              uint32_t BLK_SIZE_Y,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void bsrmm_general_blockdim_device(bool                nn,
                                                            rocsparse_direction direction,
                                                            J                   Mb,
                                                            J                   N,
                                                            int64_t offsets_batch_stride_A,
                                                            int64_t columns_values_batch_stride_A,
                                                            T       alpha,
                                                            const I* __restrict__ bsr_row_ptr,
                                                            const J* __restrict__ bsr_col_ind,
                                                            const A* __restrict__ bsr_val,
                                                            J block_dim,
                                                            const B* __restrict__ dense_B,
                                                            int64_t ldb,
                                                            int64_t batch_stride_B,
                                                            T       beta,
                                                            C* __restrict__ dense_C,
                                                            int64_t              ldc,
                                                            int64_t              batch_stride_C,
                                                            rocsparse_order      order_C,
                                                            rocsparse_index_base idx_base)
    {
        const int32_t tidx = hipThreadIdx_x;
        const int32_t tidy = hipThreadIdx_y;

        const int32_t block_row = hipBlockIdx_x;

        const I block_row_start = (block_row < Mb) ? (bsr_row_ptr[block_row] - idx_base) : 0;
        const I block_row_end   = (block_row < Mb) ? (bsr_row_ptr[block_row + 1] - idx_base) : 0;

        __shared__ T shared_B[BSR_BLOCK_DIM * BLK_SIZE_Y];
        __shared__ T shared_A[BSR_BLOCK_DIM * BSR_BLOCK_DIM];

        const J global_col = tidy + hipBlockIdx_y * BLK_SIZE_Y;

        for(J x = 0; x < block_dim; x += BSR_BLOCK_DIM)
        {
            const J global_row = tidx + x + hipBlockIdx_x * block_dim;

            T sum = static_cast<T>(0);

            for(I k = block_row_start; k < block_row_end; k++)
            {
                const J block_col = (bsr_col_ind[k] - idx_base);

                for(J y = 0; y < block_dim; y += BLK_SIZE_Y)
                {
                    if(nn)
                    {
                        shared_B[BSR_BLOCK_DIM * tidy + tidx]
                            = (global_col < N && (tidx + y) < block_dim)
                                  ? dense_B[block_dim * block_col + (tidx + y) + global_col * ldb]
                                  : static_cast<T>(0);
                    }
                    else
                    {
                        shared_B[BSR_BLOCK_DIM * tidy + tidx]
                            = (global_col < N && (tidx + y) < block_dim)
                                  ? dense_B[global_col + ldb * (block_dim * block_col + (tidx + y))]
                                  : static_cast<T>(0);
                    }

                    if(direction == rocsparse_direction_row)
                    {
                        shared_A[BSR_BLOCK_DIM * tidy + tidx]
                            = ((tidx + x) < block_dim && (tidy + y) < block_dim)
                                  ? bsr_val[block_dim * block_dim * k + block_dim * (tidx + x)
                                            + (tidy + y)]
                                  : static_cast<T>(0);
                    }
                    else
                    {
                        shared_A[BSR_BLOCK_DIM * tidy + tidx]
                            = ((tidx + x) < block_dim && (tidy + y) < block_dim)
                                  ? bsr_val[block_dim * block_dim * k + block_dim * (tidy + y)
                                            + (tidx + x)]
                                  : static_cast<T>(0);
                    }

                    __syncthreads();

                    for(uint32_t j = 0; j < BSR_BLOCK_DIM; j++)
                    {
                        sum = rocsparse::fma<T>(shared_A[BSR_BLOCK_DIM * j + tidx],
                                                shared_B[BSR_BLOCK_DIM * tidy + j],
                                                sum);
                    }

                    __syncthreads();
                }
            }

            if(block_row < Mb && global_col < N && (tidx + x) < block_dim)
            {
                if(beta == static_cast<T>(0))
                {
                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[global_row + ldc * global_col] = alpha * sum;
                    }
                    else
                    {
                        dense_C[global_row * ldc + global_col] = alpha * sum;
                    }
                }
                else
                {
                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[global_row + ldc * global_col] = rocsparse::fma<T>(
                            beta, dense_C[global_row + ldc * global_col], alpha * sum);
                    }
                    else
                    {
                        dense_C[global_row * ldc + global_col] = rocsparse::fma<T>(
                            beta, dense_C[global_row * ldc + global_col], alpha * sum);
                    }
                }
            }
        }
    }
}
