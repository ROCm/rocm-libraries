/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.hpp"

namespace rocsparse
{
    template <uint32_t BSR_BLOCK_DIM,
              uint32_t BLK_SIZE_Y,
              uint32_t UNROLL_SIZE_Y,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void bsrmm_large_blockdim_device_ext(bool                nn,
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
        const int32_t tidx = hipThreadIdx_x, tidy = hipThreadIdx_y;

        const J global_row = tidx + hipBlockIdx_x * block_dim;

        __shared__ T shared_B[BSR_BLOCK_DIM * (BLK_SIZE_Y * UNROLL_SIZE_Y)];
        __shared__ T shared_A[BSR_BLOCK_DIM * BSR_BLOCK_DIM];

        T sum[UNROLL_SIZE_Y];
        J cols[UNROLL_SIZE_Y];
        for(uint32_t l = 0; l < UNROLL_SIZE_Y; ++l)
        {
            cols[l] = (tidy + BLK_SIZE_Y * l) + hipBlockIdx_y * (BLK_SIZE_Y * UNROLL_SIZE_Y);
        }

        const int block_row       = hipBlockIdx_x;
        const I   block_row_start = (block_row < Mb) ? (bsr_row_ptr[block_row] - idx_base) : 0;
        const I   block_row_end   = (block_row < Mb) ? (bsr_row_ptr[block_row + 1] - idx_base) : 0;

        for(uint32_t l = 0; l < UNROLL_SIZE_Y; ++l)
        {
            sum[l] = static_cast<T>(0);
        }

        const J    block_dim_sqr = block_dim * block_dim;
        const bool is_tidx       = tidx < block_dim;
        const bool is_tidy       = tidy < block_dim;
        const bool is_tidx_tidy  = is_tidx && is_tidy;

        for(I k = block_row_start; k < block_row_end; k++)
        {
            const J block_col = (bsr_col_ind[k] - idx_base);
            if(is_tidx)
            {
                for(uint32_t l = 0; l < UNROLL_SIZE_Y; ++l)
                {
                    if(cols[l] < N)
                    {
                        if(nn)
                        {
                            shared_B[BSR_BLOCK_DIM * (BLK_SIZE_Y * l + tidy) + tidx]
                                = dense_B[block_dim * block_col + tidx + cols[l] * ldb];
                        }
                        else
                        {
                            shared_B[BSR_BLOCK_DIM * (BLK_SIZE_Y * l + tidy) + tidx]
                                = dense_B[cols[l] + ldb * (block_dim * block_col + tidx)];
                        }
                    }
                }
            }

            if(is_tidx_tidy)
            {
                if(direction == rocsparse_direction_row)
                {
                    shared_A[BSR_BLOCK_DIM * tidx + tidy]
                        = bsr_val[block_dim_sqr * k + block_dim * tidx + tidy];
                }
                else
                {
                    shared_A[BSR_BLOCK_DIM * tidx + tidy]
                        = bsr_val[block_dim_sqr * k + block_dim * tidy + tidx];
                }
            }

            __syncthreads();

            if(is_tidx)
            {
                for(uint32_t l = 0; l < UNROLL_SIZE_Y; ++l)
                {
                    if(cols[l] < N)
                    {
                        for(J j = 0; j < block_dim; j++)
                        {
                            sum[l] = rocsparse::fma(
                                shared_A[BSR_BLOCK_DIM * tidx + j],
                                shared_B[BSR_BLOCK_DIM * (BLK_SIZE_Y * l + tidy) + j],
                                sum[l]);
                        }
                    }
                }
            }

            __syncthreads();
        }

        if(block_row < Mb && is_tidx)
        {
            for(uint32_t l = 0; l < UNROLL_SIZE_Y; ++l)
            {
                if(cols[l] < N)
                {
                    if(beta == static_cast<T>(0))
                    {
                        if(order_C == rocsparse_order_column)
                        {
                            dense_C[global_row + ldc * cols[l]] = alpha * sum[l];
                        }
                        else
                        {
                            dense_C[global_row * ldc + cols[l]] = alpha * sum[l];
                        }
                    }
                    else
                    {
                        if(order_C == rocsparse_order_column)
                        {
                            dense_C[global_row + ldc * cols[l]] = rocsparse::fma<T>(
                                beta, dense_C[global_row + ldc * cols[l]], alpha * sum[l]);
                        }
                        else
                        {
                            dense_C[global_row * ldc + cols[l]] = rocsparse::fma<T>(
                                beta, dense_C[global_row * ldc + cols[l]], alpha * sum[l]);
                        }
                    }
                }
            }
        }
    }
}
