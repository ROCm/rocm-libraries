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

#include "handle.h"
#include <hip/hip_runtime.h>

namespace rocsparse
{
    //!
    //! @brief Recursive Compile Time Device Reduction.
    //! @param tx        The local thread id.
    //! @param sdata     The array of data.
    //!
    template <rocsparse_int n, typename I>
    ROCSPARSE_DEVICE_ILF void nnz_device_reduce(rocsparse_int tx, I* sdata)
    {
        __syncthreads();
        if(tx < n / 2)
        {
            sdata[tx] += sdata[tx + n / 2];
        }
        nnz_device_reduce<n / 2>(tx, sdata);
    }

    template <>
    __device__ __forceinline__ void nnz_device_reduce<0, int32_t>(rocsparse_int tx, int32_t* sdata)
    {
    }

    template <>
    __device__ __forceinline__ void nnz_device_reduce<0, int64_t>(rocsparse_int tx, int64_t* sdata)
    {
    }

    //!
    //! @brief Kernel for counting the number of non-zeros per column.
    //! @param m         		The number of rows.
    //! @param n         		The number of columns.
    //! @param A       		    The pointer to the values.
    //! @param lda       		The leading dimension.
    //! @param nnz_per_column   The array storing the results of the nnz per column.
    //!
    template <rocsparse_int NB_X, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(NB_X)
    void nnz_kernel_col(rocsparse_order order,
                        J               m,
                        J               n,
                        const T* __restrict__ A,
                        int64_t lda,
                        I* __restrict__ nnz_per_column)
    {
        static constexpr T s_zero = {};

        J tx  = hipThreadIdx_x;
        J col = hipBlockIdx_x;

        J m_full = (m / NB_X) * NB_X;
        I res    = 0;

        __shared__ I sdata[NB_X];

        if(order == rocsparse_order_column)
        {
            A += col * lda + ((tx < m) ? tx : 0);

            for(J i = 0; i < m_full; i += NB_X)
                res += (A[i] != s_zero) ? 1 : 0;

            if(tx + m_full < m)
                res += (A[m_full] != s_zero) ? 1 : 0;
        }
        else
        {
            for(J i = 0; i < m_full; i += NB_X)
            {
                if((tx + i) < m)
                {
                    res += (A[col + (tx + i) * lda] != s_zero) ? 1 : 0;
                }
            }

            if(tx + m_full < m)
            {
                res += (A[col + (tx + m_full) * lda] != s_zero) ? 1 : 0;
            }
        }

        sdata[tx] = res;

        if(NB_X > 16 && m >= NB_X)
        {
            nnz_device_reduce<NB_X>(tx, sdata);
        }
        else
        {
            __syncthreads();

            if(tx == 0)
            {
                for(J i = 1; i < m && i < NB_X; i++)
                    sdata[0] += sdata[i];
            }

            __syncthreads();
        }

        if(tx == 0)
        {
            nnz_per_column[col] = sdata[0];
        }
    }

    //!
    //! @brief Kernel for counting the number of non-zeros per row.
    //! @param m         		The number of rows.
    //! @param n         		The number of columns.
    //! @param A       		    The pointer to the values.
    //! @param lda       		The leading dimension.
    //! @param nnz_per_row      The array storing the results of the nnz per row.
    //!
    template <rocsparse_int DIM_X, rocsparse_int DIM_Y, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(DIM_X* DIM_Y)
    void nnz_kernel_row(rocsparse_order order,
                        J               m,
                        J               n,
                        const T* __restrict__ A,
                        int64_t lda,
                        I* __restrict__ nnz_per_row)
    {
        static constexpr T s_zero = {};

        J thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
        J tx        = thread_id % DIM_X;
        J ty        = thread_id / DIM_X;
        J ind       = hipBlockIdx_x * DIM_X * 4 + tx;
        J n_tail    = n % (4 * DIM_Y);
        J col       = ty * 4;
        I res_A[4];

        __shared__ I sdata[DIM_X * 4 * DIM_Y];

        for(int k = 0; k < 4; ++k)
        {
            res_A[k] = 0;
        }

        for(col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y)
        {
            for(int k = 0; k < 4; ++k)
            {
                if(ind + k * DIM_X < m)
                {
                    if(order == rocsparse_order_column)
                    {
                        for(int j = 0; j < 4; ++j)
                        {
                            if(A[ind + k * DIM_X + (col + j) * lda] != s_zero)
                                res_A[k] += 1;
                        }
                    }
                    else
                    {
                        for(int j = 0; j < 4; ++j)
                        {
                            if(A[(ind + k * DIM_X) * lda + col + j] != s_zero)
                                res_A[k] += 1;
                        }
                    }
                }
            }
        }

        if(n_tail > 0)
        {
            for(int k = 0; k < 4; ++k)
            {
                if(ind + k * DIM_X < m)
                {
                    for(int j = 0; j < 4; ++j)
                    {
                        if(col + j < n)
                        {
                            if(order == rocsparse_order_column)
                            {
                                res_A[k]
                                    += (A[ind + k * DIM_X + (col + j) * lda] != s_zero) ? 1 : 0;
                            }
                            else
                            {
                                res_A[k]
                                    += (A[(ind + k * DIM_X) * lda + col + j] != s_zero) ? 1 : 0;
                            }
                        }
                    }
                }
            }
        }

        for(int k = 0; k < 4; ++k)
        {
            sdata[tx + k * DIM_X + ty * DIM_X * 4] = res_A[k];
        }

        __syncthreads();

        ind = hipBlockIdx_x * DIM_X * 4 + thread_id;
        if(thread_id < DIM_X * 4)
        {
            for(int j = 1; j < DIM_Y; j++)
            {
                sdata[thread_id] += sdata[thread_id + DIM_X * 4 * j];
            }

            if(ind < m)
            {
                nnz_per_row[ind] = sdata[thread_id];
            }
        }
    }
}
