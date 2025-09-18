/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
    // Sliced ELL SpMV for general, non-transposed matrices
    template <uint32_t THREADS_PER_ROW,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void sellmvn_device(J                    m,
                                             J                    n,
                                             I                    nnz,
                                             J                    slice_size,
                                             I                    sell_colval_size,
                                             T                    alpha,
                                             const I*             sell_slice_offsets,
                                             const J*             sell_col_ind,
                                             const A*             sell_val,
                                             const X*             x,
                                             T                    beta,
                                             Y*                   y,
                                             rocsparse_index_base idx_base)
    {
        const uint32_t tidx = hipThreadIdx_x; // 0....slice_size
        const uint32_t tidy = hipThreadIdx_y; // 0....THREADS_PER_ROW

        const uint32_t idx = slice_size * tidy + tidx;

        extern __shared__ char shared_memory[]; // THREADS_PER_ROW == hipBlockDim_y
        T*                     shared = (T*)shared_memory;

        const uint32_t sliceid = hipGridDim_x * hipBlockIdx_y + hipBlockIdx_x;

        const J row = slice_size * sliceid + tidx;

        const I start = (row < m) ? sell_slice_offsets[sliceid] - idx_base : 0;
        const I end   = (row < m) ? sell_slice_offsets[sliceid + 1] - idx_base : 0;

        T sum = static_cast<T>(0);

        for(I j = start + idx; j < end; j += (slice_size * THREADS_PER_ROW))
        {
            const J col = sell_col_ind[j] - idx_base;

            if(col >= 0)
            {
                sum = rocsparse::fma<T>(sell_val[j], x[col], sum);
            }
        }

        shared[idx] = sum;
        __syncthreads();

        if(THREADS_PER_ROW > 4)
        {
            if(tidy < 4 && tidy + 4 < THREADS_PER_ROW)
            {
                shared[idx] = shared[idx] + shared[idx + slice_size * 4];
            }
            __syncthreads();
        }
        if(THREADS_PER_ROW > 2)
        {
            if(tidy < 2 && tidy + 2 < THREADS_PER_ROW)
            {
                shared[idx] = shared[idx] + shared[idx + slice_size * 2];
            }
            __syncthreads();
        }
        if(THREADS_PER_ROW > 1)
        {
            if(tidy < 1 && tidy + 1 < THREADS_PER_ROW)
            {
                shared[idx] = shared[idx] + shared[idx + slice_size * 1];
            }
            __syncthreads();
        }

        if(row < m && tidy == 0)
        {
            if(beta == static_cast<T>(0))
            {
                y[row] = alpha * shared[tidx];
            }
            else
            {
                y[row] = rocsparse::fma<T>(beta, y[row], alpha * shared[tidx]);
            }
        }
    }

    // Sliced ELL SpMV for general, non-transposed matrices, large slice size
    template <uint32_t BLOCKSIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void sellmvn_large_slice_device(J                    m,
                                                         J                    n,
                                                         I                    nnz,
                                                         J                    slice_size,
                                                         I                    sell_colval_size,
                                                         T                    alpha,
                                                         const I*             sell_slice_offsets,
                                                         const J*             sell_col_ind,
                                                         const A*             sell_val,
                                                         const X*             x,
                                                         T                    beta,
                                                         Y*                   y,
                                                         rocsparse_index_base idx_base)
    {
        const uint32_t tid     = hipThreadIdx_x;
        const uint32_t sliceid = hipBlockIdx_x;

        const J iter = (slice_size - 1) / BLOCKSIZE + 1;

        for(J p = 0; p < iter; p++)
        {
            const J local_row = (BLOCKSIZE * p + tid);

            const J row = slice_size * sliceid + (BLOCKSIZE * p + tid);

            const I start
                = (row < m && local_row < slice_size) ? sell_slice_offsets[sliceid] - idx_base : 0;
            const I end = (row < m && local_row < slice_size)
                              ? sell_slice_offsets[sliceid + 1] - idx_base
                              : 0;

            T sum = static_cast<T>(0);

            for(I j = start + local_row; j < end; j += slice_size)
            {
                const J col = sell_col_ind[j] - idx_base;

                if(col >= 0)
                {
                    sum = rocsparse::fma<T>(sell_val[j], x[col], sum);
                }
            }

            if(row < m && local_row < slice_size)
            {
                if(beta == static_cast<T>(0))
                {
                    y[row] = alpha * sum;
                }
                else
                {
                    y[row] = rocsparse::fma<T>(beta, y[row], alpha * sum);
                }
            }
        }
    }

    // Sliced ELL SpMV for general, transposed matrices
    template <uint32_t THREADS_PER_ROW,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void sellmvt_device(rocsparse_operation  trans,
                                             J                    m,
                                             J                    n,
                                             I                    nnz,
                                             J                    slice_size,
                                             I                    sell_colval_size,
                                             T                    alpha,
                                             const I*             sell_slice_offsets,
                                             const J*             sell_col_ind,
                                             const A*             sell_val,
                                             const X*             x,
                                             Y*                   y,
                                             rocsparse_index_base idx_base)
    {
        const uint32_t tidx = hipThreadIdx_x; // 0....slice_size
        const uint32_t tidy = hipThreadIdx_y; // 0....THREADS_PER_ROW

        const uint32_t idx = slice_size * tidy + tidx;

        const uint32_t sliceid = hipGridDim_x * hipBlockIdx_y + hipBlockIdx_x;

        const J row = slice_size * sliceid + tidx;

        const I start = (row < m) ? sell_slice_offsets[sliceid] - idx_base : 0;
        const I end   = (row < m) ? sell_slice_offsets[sliceid + 1] - idx_base : 0;

        T row_val = alpha * x[row];

        for(I j = start + idx; j < end; j += (slice_size * THREADS_PER_ROW))
        {
            const J col = sell_col_ind[j] - idx_base;

            if(col >= 0)
            {
                A val = sell_val[j];

                if(trans == rocsparse_operation_conjugate_transpose)
                {
                    val = rocsparse::conj(val);
                }

                rocsparse::atomic_add(&y[col], static_cast<T>(val) * row_val);
            }
        }
    }

    // Sliced ELL SpMV for general, transposed matrices, large slice size
    template <uint32_t BLOCKSIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void sellmvt_large_slice_device(rocsparse_operation  trans,
                                                         J                    m,
                                                         J                    n,
                                                         I                    nnz,
                                                         J                    slice_size,
                                                         I                    sell_colval_size,
                                                         T                    alpha,
                                                         const I*             sell_slice_offsets,
                                                         const J*             sell_col_ind,
                                                         const A*             sell_val,
                                                         const X*             x,
                                                         Y*                   y,
                                                         rocsparse_index_base idx_base)
    {
        const uint32_t tid     = hipThreadIdx_x;
        const uint32_t sliceid = hipBlockIdx_x;

        const J iter = (slice_size - 1) / BLOCKSIZE + 1;

        for(J p = 0; p < iter; p++)
        {
            const J local_row = (BLOCKSIZE * p + tid);

            const J row = slice_size * sliceid + (BLOCKSIZE * p + tid);

            const I start
                = (row < m && local_row < slice_size) ? sell_slice_offsets[sliceid] - idx_base : 0;
            const I end = (row < m && local_row < slice_size)
                              ? sell_slice_offsets[sliceid + 1] - idx_base
                              : 0;

            T row_val = (row < m && local_row < slice_size) ? alpha * x[row] : static_cast<T>(0);

            for(I j = start + local_row; j < end; j += slice_size)
            {
                const J col = sell_col_ind[j] - idx_base;

                if(col >= 0)
                {
                    A val = sell_val[j];

                    if(trans == rocsparse_operation_conjugate_transpose)
                    {
                        val = rocsparse::conj(val);
                    }

                    rocsparse::atomic_add(&y[col], static_cast<T>(val) * row_val);
                }
            }
        }
    }
}