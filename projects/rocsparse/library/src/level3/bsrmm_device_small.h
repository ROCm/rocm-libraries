/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t BSR_BLOCK_DIM,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void bsrmmnn_small_blockdim_device(rocsparse_direction direction,
                                                            J                   Mb,
                                                            J                   N,
                                                            int64_t offsets_batch_stride_A,
                                                            int64_t columns_values_batch_stride_A,
                                                            T       alpha,
                                                            const I* __restrict__ bsr_row_ptr,
                                                            const J* __restrict__ bsr_col_ind,
                                                            const A* __restrict__ bsr_val,
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
        const int32_t tid  = hipThreadIdx_x;
        const J       gid  = hipBlockIdx_x * hipBlockDim_x + tid;
        const int32_t lid  = gid & (WF_SIZE - 1);
        const J       nwfb = hipGridDim_x * hipBlockDim_x / (WF_SIZE * BSR_BLOCK_DIM);
        const J       col  = lid + hipBlockIdx_y * WF_SIZE;

        const int64_t colB = col * ldb;

        // global row
        const J global_row = (gid / WF_SIZE);

        // local row within block row
        const J local_row = (gid / WF_SIZE) % BSR_BLOCK_DIM;

        for(J block_row = gid / (WF_SIZE * BSR_BLOCK_DIM); block_row < Mb; block_row += nwfb)
        {
            const I block_row_start = bsr_row_ptr[block_row] - idx_base;
            const I block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;

            T sum = static_cast<T>(0);

            for(I j = block_row_start; j < block_row_end; j += WF_SIZE)
            {
                const I k = j + lid;

                const J my_col
                    = (k < block_row_end) ? BSR_BLOCK_DIM * (bsr_col_ind[k] - idx_base) : 0;

                T my_val[BSR_BLOCK_DIM];
                if(direction == rocsparse_direction_row)
                {
                    my_val[0] = (k < block_row_end)
                                    ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                             + BSR_BLOCK_DIM * local_row])
                                    : static_cast<T>(0);
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        my_val[1] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * local_row + 1])
                                        : static_cast<T>(0);
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        my_val[2] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * local_row + 2])
                                        : static_cast<T>(0);
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        my_val[3] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * local_row + 3])
                                        : static_cast<T>(0);
                    }
                }
                else
                {
                    my_val[0] = (k < block_row_end) ? static_cast<T>(
                                    bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k + local_row])
                                                    : static_cast<T>(0);
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        my_val[1] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * 1 + local_row])
                                        : static_cast<T>(0);
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        my_val[2] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * 2 + local_row])
                                        : static_cast<T>(0);
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        my_val[3] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * 3 + local_row])
                                        : static_cast<T>(0);
                    }
                }

                for(uint32_t i = 0; i < WF_SIZE; ++i)
                {
                    const J sc  = rocsparse::shfl(my_col, i, WF_SIZE);
                    const T sv0 = rocsparse::shfl(my_val[0], i, WF_SIZE);
                    if(col < N)
                    {
                        sum = rocsparse::fma<T>(sv0, dense_B[sc + colB], sum);
                    }
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        const T sv1 = rocsparse::shfl(my_val[1], i, WF_SIZE);
                        if(col < N)
                        {
                            sum = rocsparse::fma<T>(sv1, dense_B[sc + 1 + colB], sum);
                        }
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        const T sv2 = rocsparse::shfl(my_val[2], i, WF_SIZE);
                        if(col < N)
                        {
                            sum = rocsparse::fma<T>(sv2, dense_B[sc + 2 + colB], sum);
                        }
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        const T sv3 = rocsparse::shfl(my_val[3], i, WF_SIZE);
                        if(col < N)
                        {
                            sum = rocsparse::fma<T>(sv3, dense_B[sc + 3 + colB], sum);
                        }
                    }
                }
            }

            if(col < N)
            {
                if(beta == static_cast<T>(0))
                {
                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[global_row + ldc * col] = alpha * sum;
                    }
                    else
                    {
                        dense_C[global_row * ldc + col] = alpha * sum;
                    }
                }
                else
                {
                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[global_row + ldc * col]
                            = rocsparse::fma<T>(beta, dense_C[global_row + ldc * col], alpha * sum);
                    }
                    else
                    {
                        dense_C[ldc * global_row + col]
                            = rocsparse::fma<T>(beta, dense_C[global_row * ldc + col], alpha * sum);
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t BSR_BLOCK_DIM,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void bsrmmnt_small_blockdim_device(rocsparse_direction direction,
                                                            J                   Mb,
                                                            J                   N,
                                                            int64_t offsets_batch_stride_A,
                                                            int64_t columns_values_batch_stride_A,
                                                            T       alpha,
                                                            const I* __restrict__ bsr_row_ptr,
                                                            const J* __restrict__ bsr_col_ind,
                                                            const A* __restrict__ bsr_val,
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
        const int tid        = hipThreadIdx_x;
        const J   gid        = hipBlockIdx_x * hipBlockDim_x + tid;
        const J   block_row  = gid / (WF_SIZE * BSR_BLOCK_DIM);
        const J   global_row = gid / WF_SIZE;
        const J   local_row  = (gid / WF_SIZE) % BSR_BLOCK_DIM;
        const int lid        = tid & (WF_SIZE - 1);

        if(block_row >= Mb)
        {
            return;
        }

        const I block_row_start = bsr_row_ptr[block_row] - idx_base;
        const I block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;

        for(J l = 0; l < N; l += WF_SIZE)
        {
            const J col = l + lid;
            T       sum = static_cast<T>(0);

            for(I j = block_row_start; j < block_row_end; j += WF_SIZE)
            {
                const I k = j + lid;

                const J my_col
                    = (k < block_row_end) ? BSR_BLOCK_DIM * (bsr_col_ind[k] - idx_base) : 0;

                T my_val[BSR_BLOCK_DIM];
                if(direction == rocsparse_direction_row)
                {
                    my_val[0] = (k < block_row_end)
                                    ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                             + BSR_BLOCK_DIM * local_row])
                                    : static_cast<T>(0);
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        my_val[1] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * local_row + 1])
                                        : static_cast<T>(0);
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        my_val[2] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * local_row + 2])
                                        : static_cast<T>(0);
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        my_val[3] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * local_row + 3])
                                        : static_cast<T>(0);
                    }
                }
                else
                {
                    my_val[0] = (k < block_row_end) ? static_cast<T>(
                                    bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k + local_row])
                                                    : static_cast<T>(0);
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        my_val[1] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * 1 + local_row])
                                        : static_cast<T>(0);
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        my_val[2] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * 2 + local_row])
                                        : static_cast<T>(0);
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        my_val[3] = (k < block_row_end)
                                        ? static_cast<T>(bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                                 + BSR_BLOCK_DIM * 3 + local_row])
                                        : static_cast<T>(0);
                    }
                }

                for(uint32_t i = 0; i < WF_SIZE; ++i)
                {
                    const J sc  = rocsparse::shfl(my_col, i, WF_SIZE);
                    const T sv0 = rocsparse::shfl(my_val[0], i, WF_SIZE);
                    if(col < N)
                    {
                        T val_B = rocsparse::ldg(dense_B + col + ldb * sc);
                        sum     = rocsparse::fma<T>(sv0, val_B, sum);
                    }
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        const T sv1 = rocsparse::shfl(my_val[1], i, WF_SIZE);
                        if(col < N)
                        {
                            T val_B = rocsparse::ldg(dense_B + col + ldb * (1 + sc));
                            sum     = rocsparse::fma<T>(sv1, val_B, sum);
                        }
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        const T sv2 = rocsparse::shfl(my_val[2], i, WF_SIZE);
                        if(col < N)
                        {
                            T val_B = rocsparse::ldg(dense_B + col + ldb * (2 + sc));
                            sum     = rocsparse::fma<T>(sv2, val_B, sum);
                        }
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        const T sv3 = rocsparse::shfl(my_val[3], i, WF_SIZE);
                        if(col < N)
                        {
                            T val_B = rocsparse::ldg(dense_B + col + ldb * (3 + sc));
                            sum     = rocsparse::fma<T>(sv3, val_B, sum);
                        }
                    }
                }
            }

            if(col < N)
            {
                if(beta == static_cast<T>(0))
                {
                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[global_row + ldc * col] = alpha * sum;
                    }
                    else
                    {
                        dense_C[global_row * ldc + col] = alpha * sum;
                    }
                }
                else
                {
                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[global_row + ldc * col]
                            = rocsparse::fma<T>(beta, dense_C[global_row + ldc * col], alpha * sum);
                    }
                    else
                    {
                        dense_C[ldc * global_row + col]
                            = rocsparse::fma<T>(beta, dense_C[global_row * ldc + col], alpha * sum);
                    }
                }
            }
        }
    }
}
