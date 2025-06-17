/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <uint32_t WF_SIZE,
              uint32_t LOOPS,
              uint32_t COLS,
              bool     NT,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void coommnn_segmented_atomic_device(rocsparse_operation  trans_B,
                                                              int64_t              nnz,
                                                              I                    nstart,
                                                              int64_t              batch_stride_A,
                                                              T                    alpha,
                                                              const I*             coo_row_ind,
                                                              const I*             coo_col_ind,
                                                              const A*             coo_val,
                                                              const B*             dense_B,
                                                              int64_t              ldb,
                                                              int64_t              batch_stride_B,
                                                              C*                   dense_C,
                                                              int64_t              ldc,
                                                              int64_t              batch_stride_C,
                                                              rocsparse_order      order_C,
                                                              rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;
        const int lid = tid & (WF_SIZE - 1);

        const int batch = hipBlockIdx_z;

        // Shared memory to hold row indices and values for segmented reduction
        __shared__ I shared_row[WF_SIZE];
        __shared__ T shared_val[COLS][WF_SIZE];

        const I       col_offset = nstart + COLS * hipBlockIdx_y;
        const int64_t offset     = hipBlockIdx_x * LOOPS * WF_SIZE;

        if(offset >= nnz)
        {
            return;
        }

        I row;
        T val[COLS];

        // Current threads index into COO structure
        // Each thread processes 'loop' COO entries
        for(int64_t idx = offset + lid; idx < offset + LOOPS * WF_SIZE; idx += WF_SIZE)
        {
            // Get corresponding COO entry

            const I r
                = (idx < nnz)
                      ? rocsparse::nontemporal_load(coo_row_ind + idx + batch_stride_A * batch)
                            - idx_base
                      : -1;
            const I c
                = (idx < nnz)
                      ? rocsparse::nontemporal_load(coo_col_ind + idx + batch_stride_A * batch)
                            - idx_base
                      : 0;
            const T v
                = (idx < nnz)
                      ? alpha * rocsparse::nontemporal_load(coo_val + idx + batch_stride_A * batch)
                      : static_cast<T>(0);

            row = r;

            if(NT)
            {
                if(trans_B == rocsparse_operation_conjugate_transpose)
                {
                    for(uint32_t p = 0; p < COLS; p++)
                    {
                        val[p] = v
                                 * rocsparse::conj(
                                     dense_B[c * ldb + (col_offset + p) + batch_stride_B * batch]);
                    }
                }
                else
                {
                    for(uint32_t p = 0; p < COLS; p++)
                    {
                        val[p] = v * dense_B[c * ldb + (col_offset + p) + batch_stride_B * batch];
                    }
                }
            }
            else
            {
                if(trans_B == rocsparse_operation_conjugate_transpose)
                {
                    for(uint32_t p = 0; p < COLS; p++)
                    {
                        val[p] = v
                                 * rocsparse::conj(
                                     dense_B[(col_offset + p) * ldb + c + batch_stride_B * batch]);
                    }
                }
                else
                {
                    for(uint32_t p = 0; p < COLS; p++)
                    {
                        val[p] = v * dense_B[(col_offset + p) * ldb + c + batch_stride_B * batch];
                    }
                }
            }

            // First thread in wavefront checks row index from previous loop
            // if it has been completed or if additional rows have to be
            // appended.
            if(idx > offset && lid == 0)
            {
                const I prevrow = shared_row[WF_SIZE - 1];
                if(row == prevrow)
                {
                    for(uint32_t p = 0; p < COLS; p++)
                    {
                        val[p] += shared_val[p][WF_SIZE - 1];
                    }
                }
                else if(prevrow >= 0)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(uint32_t p = 0; p < COLS; p++)
                        {
                            rocsparse::atomic_add(
                                &dense_C[prevrow + (col_offset + p) * ldc + batch_stride_C * batch],
                                static_cast<C>(shared_val[p][WF_SIZE - 1]));
                        }
                    }
                    else
                    {
                        for(uint32_t p = 0; p < COLS; p++)
                        {
                            rocsparse::atomic_add(
                                &dense_C[(col_offset + p) + prevrow * ldc + batch_stride_C * batch],
                                static_cast<C>(shared_val[p][WF_SIZE - 1]));
                        }
                    }
                }
            }

            __syncthreads();

            for(uint32_t p = 0; p < COLS; p++)
            {
                shared_val[p][lid] = val[p];
            }
            shared_row[lid] = row;

            __syncthreads();

#pragma unroll
            // Segmented wavefront reduction
            for(uint32_t j = 1; j < WF_SIZE; j <<= 1)
            {
                if(lid >= j)
                {
                    if(row == shared_row[lid - j])
                    {
                        for(uint32_t p = 0; p < COLS; p++)
                        {
                            val[p] += shared_val[p][lid - j];
                        }
                    }
                }
                __syncthreads();

                for(uint32_t p = 0; p < COLS; p++)
                {
                    shared_val[p][lid] = val[p];
                }

                __syncthreads();
            }

            // All lanes but the last one write their result in C.
            // The last value might need to be appended by the next iteration.
            if(lid < WF_SIZE - 1)
            {
                if(row != shared_row[lid + 1] && row >= 0)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(uint32_t p = 0; p < COLS; p++)
                        {
                            rocsparse::atomic_add(
                                &dense_C[row + (col_offset + p) * ldc + batch_stride_C * batch],
                                static_cast<C>(val[p]));
                        }
                    }
                    else
                    {
                        for(uint32_t p = 0; p < COLS; p++)
                        {
                            rocsparse::atomic_add(
                                &dense_C[(col_offset + p) + row * ldc + batch_stride_C * batch],
                                static_cast<C>(val[p]));
                        }
                    }
                }
            }
        }

        // Write last entries into buffers for segmented block reduction
        if(lid == WF_SIZE - 1 && row >= 0)
        {
            if(order_C == rocsparse_order_column)
            {
                for(uint32_t p = 0; p < COLS; p++)
                {
                    rocsparse::atomic_add(
                        &dense_C[row + (col_offset + p) * ldc + batch_stride_C * batch],
                        static_cast<C>(val[p]));
                }
            }
            else
            {
                for(uint32_t p = 0; p < COLS; p++)
                {
                    rocsparse::atomic_add(
                        &dense_C[(col_offset + p) + row * ldc + batch_stride_C * batch],
                        static_cast<C>(val[p]));
                }
            }
        }
    }
}
