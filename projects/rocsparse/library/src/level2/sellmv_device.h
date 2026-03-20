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
    ROCSPARSE_DEVICE_ILF void sellmvn_device(J m,
                                             J n,
                                             I nnz,
                                             J sell_slice_size,
                                             I sell_colval_size,
                                             T alpha,
                                             const I* __restrict__ sell_slice_offsets,
                                             const J* __restrict__ sell_col_ind,
                                             const A* __restrict__ sell_val,
                                             const X* __restrict__ x,
                                             T beta,
                                             Y* __restrict__ y,
                                             rocsparse_index_base idx_base)
    {
        const uint32_t tidx = hipThreadIdx_x; // 0....sell_slice_size
        const uint32_t tidy = hipThreadIdx_y; // 0....THREADS_PER_ROW

        const uint32_t idx = sell_slice_size * tidy + tidx;

        extern __shared__ char shared_memory[]; // THREADS_PER_ROW == hipBlockDim_y
        T*                     shared = (T*)shared_memory;

        const uint32_t sliceid = hipGridDim_x * hipBlockIdx_y + hipBlockIdx_x;

        const J row = sell_slice_size * sliceid + tidx;

        const I start = (row < m) ? sell_slice_offsets[sliceid] - idx_base : 0;
        const I end   = (row < m) ? sell_slice_offsets[sliceid + 1] - idx_base : 0;

        T sum = static_cast<T>(0);

        for(I j = start + idx; j < end; j += (sell_slice_size * THREADS_PER_ROW))
        {
            const J col = sell_col_ind[j] - idx_base;

            if(col >= 0)
            {
                sum = rocsparse::fma<T>(sell_val[j], x[col], sum);
            }
        }

        shared[idx] = sum;
        __syncthreads();

        for(int32_t level = 4; level > 0; level /= 2)
        {
            if(tidy < level && tidy + level < THREADS_PER_ROW)
            {
                shared[idx] = shared[idx] + shared[idx + sell_slice_size * level];
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
    ROCSPARSE_DEVICE_ILF void sellmvn_large_slice_device(J m,
                                                         J n,
                                                         I nnz,
                                                         J sell_slice_size,
                                                         I sell_colval_size,
                                                         T alpha,
                                                         const I* __restrict__ sell_slice_offsets,
                                                         const J* __restrict__ sell_col_ind,
                                                         const A* __restrict__ sell_val,
                                                         const X* __restrict__ x,
                                                         T beta,
                                                         Y* __restrict__ y,
                                                         rocsparse_index_base idx_base)
    {
        const uint32_t tid     = hipThreadIdx_x;
        const uint32_t sliceid = hipBlockIdx_x;

        const J iter = (sell_slice_size - 1) / BLOCKSIZE + 1;

        for(J p = 0; p < iter; p++)
        {
            const J local_row = (BLOCKSIZE * p + tid);

            const J row = sell_slice_size * sliceid + (BLOCKSIZE * p + tid);

            const bool row_in_range = (row < m && local_row < sell_slice_size);

            const I start = row_in_range ? sell_slice_offsets[sliceid] - idx_base : 0;
            const I end   = row_in_range ? sell_slice_offsets[sliceid + 1] - idx_base : 0;

            T sum = static_cast<T>(0);

            for(I j = start + local_row; j < end; j += sell_slice_size)
            {
                const J col = sell_col_ind[j] - idx_base;

                if(col >= 0)
                {
                    sum = rocsparse::fma<T>(sell_val[j], x[col], sum);
                }
            }

            if(row_in_range)
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
    ROCSPARSE_DEVICE_ILF void sellmvt_device(rocsparse_operation trans,
                                             J                   m,
                                             J                   n,
                                             I                   nnz,
                                             J                   sell_slice_size,
                                             I                   sell_colval_size,
                                             T                   alpha,
                                             const I* __restrict__ sell_slice_offsets,
                                             const J* __restrict__ sell_col_ind,
                                             const A* __restrict__ sell_val,
                                             const X* __restrict__ x,
                                             Y* __restrict__ y,
                                             rocsparse_index_base idx_base)
    {
        const uint32_t tidx = hipThreadIdx_x; // 0....sell_slice_size
        const uint32_t tidy = hipThreadIdx_y; // 0....THREADS_PER_ROW

        const uint32_t idx = sell_slice_size * tidy + tidx;

        const uint32_t sliceid = hipGridDim_x * hipBlockIdx_y + hipBlockIdx_x;

        const J row = sell_slice_size * sliceid + tidx;

        const I start = (row < m) ? sell_slice_offsets[sliceid] - idx_base : 0;
        const I end   = (row < m) ? sell_slice_offsets[sliceid + 1] - idx_base : 0;

        T row_val = alpha * x[row];

        for(I j = start + idx; j < end; j += (sell_slice_size * THREADS_PER_ROW))
        {
            const J col = sell_col_ind[j] - idx_base;

            if(col >= 0)
            {
                A val = sell_val[j];

                if(trans == rocsparse_operation_conjugate_transpose)
                {
                    val = rocsparse::conj(val);
                }

                rocsparse::atomic_add(y, col, n, static_cast<T>(val) * row_val);
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
    ROCSPARSE_DEVICE_ILF void sellmvt_large_slice_device(rocsparse_operation trans,
                                                         J                   m,
                                                         J                   n,
                                                         I                   nnz,
                                                         J                   sell_slice_size,
                                                         I                   sell_colval_size,
                                                         T                   alpha,
                                                         const I* __restrict__ sell_slice_offsets,
                                                         const J* __restrict__ sell_col_ind,
                                                         const A* __restrict__ sell_val,
                                                         const X* __restrict__ x,
                                                         Y* __restrict__ y,
                                                         rocsparse_index_base idx_base)
    {
        const uint32_t tid     = hipThreadIdx_x;
        const uint32_t sliceid = hipBlockIdx_x;

        const J iter = (sell_slice_size - 1) / BLOCKSIZE + 1;

        for(J p = 0; p < iter; p++)
        {
            const J local_row = (BLOCKSIZE * p + tid);

            const J row = sell_slice_size * sliceid + (BLOCKSIZE * p + tid);

            const bool row_in_range = (row < m && local_row < sell_slice_size);

            const I start = row_in_range ? sell_slice_offsets[sliceid] - idx_base : 0;
            const I end   = row_in_range ? sell_slice_offsets[sliceid + 1] - idx_base : 0;

            T row_val = row_in_range ? alpha * x[row] : static_cast<T>(0);

            for(I j = start + local_row; j < end; j += sell_slice_size)
            {
                const J col = sell_col_ind[j] - idx_base;

                if(col >= 0)
                {
                    A val = sell_val[j];

                    if(trans == rocsparse_operation_conjugate_transpose)
                    {
                        val = rocsparse::conj(val);
                    }

                    rocsparse::atomic_add(y, col, n, static_cast<T>(val) * row_val);
                }
            }
        }
    }
}