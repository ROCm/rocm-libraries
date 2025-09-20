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
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t LOOPS,
              bool     TRANSB,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void coommnn_atomic_main_device(bool    conj_A,
                                                         bool    conj_B,
                                                         I       ncol,
                                                         int64_t nnz,
                                                         I       n,
                                                         T       alpha,
                                                         const I* __restrict__ coo_row_ind,
                                                         const I* __restrict__ coo_col_ind,
                                                         const A* __restrict__ coo_val,
                                                         const B* __restrict__ dense_B,
                                                         int64_t ldb,
                                                         C* __restrict__ dense_C,
                                                         int64_t              ldc,
                                                         rocsparse_order      order_C,
                                                         rocsparse_index_base idx_base)
    {
        const int     tid = hipThreadIdx_x;
        const int64_t gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const int     lid = tid & (WF_SIZE - 1);

        const I row = (gid < nnz) ? rocsparse::nontemporal_load(coo_row_ind + gid) - idx_base : 0;
        const I col = (gid < nnz) ? rocsparse::nontemporal_load(coo_col_ind + gid) - idx_base : 0;
        const T val = (gid < nnz) ? static_cast<T>(rocsparse::nontemporal_load(coo_val + gid))
                                  : static_cast<T>(0);

        for(I l = 0; l < ncol; l += WF_SIZE * LOOPS)
        {
            const I colB = l + lid;

            T sum[LOOPS]{};

            I current_row = rocsparse::shfl(row, 0, WF_SIZE);
            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                const T v = rocsparse::shfl(val, i, WF_SIZE);
                const I c = rocsparse::shfl(col, i, WF_SIZE);
                const I r = rocsparse::shfl(row, i, WF_SIZE);

                if(r != current_row)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            rocsparse::atomic_add(
                                &dense_C[(colB + p * WF_SIZE) * ldc + current_row],
                                static_cast<C>(alpha * sum[p]));
                        }
                    }
                    else
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            rocsparse::atomic_add(&dense_C[current_row * ldc + colB + p * WF_SIZE],
                                                  static_cast<C>(alpha * sum[p]));
                        }
                    }

                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        sum[p] = static_cast<T>(0);
                    }

                    current_row = r;
                }

                if(TRANSB)
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse::fma<T>(
                            v,
                            rocsparse::conj_val(dense_B[c * ldb + colB + p * WF_SIZE], conj_B),
                            sum[p]);
                    }
                }
                else
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse::fma<T>(
                            v,
                            rocsparse::conj_val(dense_B[(colB + p * WF_SIZE) * ldb + c], conj_B),
                            sum[p]);
                    }
                }
            }

            if(order_C == rocsparse_order_column)
            {
                for(uint32_t p = 0; p < LOOPS; p++)
                {
                    rocsparse::atomic_add(&dense_C[(colB + p * WF_SIZE) * ldc + current_row],
                                          static_cast<C>(alpha * sum[p]));
                }
            }
            else
            {
                for(uint32_t p = 0; p < LOOPS; p++)
                {
                    rocsparse::atomic_add(&dense_C[current_row * ldc + colB + p * WF_SIZE],
                                          static_cast<C>(alpha * sum[p]));
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              bool     TRANSB,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void coommnn_atomic_remainder_device(bool    conj_A,
                                                              bool    conj_B,
                                                              I       ncol_offset,
                                                              I       n,
                                                              int64_t nnz,
                                                              T       alpha,
                                                              const I* __restrict__ coo_row_ind,
                                                              const I* __restrict__ coo_col_ind,
                                                              const A* __restrict__ coo_val,
                                                              const B* __restrict__ dense_B,
                                                              int64_t ldb,
                                                              C* __restrict__ dense_C,
                                                              int64_t              ldc,
                                                              rocsparse_order      order_C,
                                                              rocsparse_index_base idx_base)
    {
        const int     tid = hipThreadIdx_x;
        const int     lid = tid & (WF_SIZE - 1);
        const int     wid = tid / WF_SIZE;
        const int64_t gid = BLOCKSIZE * hipBlockIdx_x + tid;

        __shared__ I shared_row[(BLOCKSIZE / WF_SIZE) * WF_SIZE];
        __shared__ T shared_val[(BLOCKSIZE / WF_SIZE) * WF_SIZE];

        const I row = (gid < nnz) ? rocsparse::nontemporal_load(&coo_row_ind[gid]) - idx_base : -1;
        const I col = (gid < nnz) ? rocsparse::nontemporal_load(&coo_col_ind[gid]) - idx_base : 0;
        const T val
            = (gid < nnz) ? alpha * rocsparse::nontemporal_load(&coo_val[gid]) : static_cast<T>(0);

        for(I l = ncol_offset; l < n; l += WF_SIZE)
        {
            const I colB = l + lid;

            T sum         = static_cast<T>(0);
            I current_row = rocsparse::shfl(row, 0, WF_SIZE);

            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                T v = rocsparse::shfl(val, i, WF_SIZE);
                I c = rocsparse::shfl(col, i, WF_SIZE);
                I r = rocsparse::shfl(row, i, WF_SIZE);

                if(r != current_row)
                {
                    if(colB < n)
                    {
                        if(order_C == rocsparse_order_column)
                        {
                            rocsparse::atomic_add(&dense_C[colB * ldc + current_row],
                                                  static_cast<C>(sum));
                        }
                        else
                        {
                            rocsparse::atomic_add(&dense_C[current_row * ldc + colB],
                                                  static_cast<C>(sum));
                        }
                    }

                    sum = static_cast<T>(0);

                    current_row = r;
                }

                if(colB < n)
                {
                    if(TRANSB)
                    {
                        sum = rocsparse::fma<T>(
                            v, rocsparse::conj_val(dense_B[c * ldb + colB], conj_B), sum);
                    }
                    else
                    {
                        sum = rocsparse::fma<T>(
                            v, rocsparse::conj_val(dense_B[colB * ldb + c], conj_B), sum);
                    }
                }
            }

            __syncthreads();
            shared_row[(BLOCKSIZE / WF_SIZE) * lid + wid] = current_row;
            shared_val[(BLOCKSIZE / WF_SIZE) * lid + wid] = sum;
            __syncthreads();

            current_row = shared_row[tid];
            sum         = shared_val[tid];

            const int slid = tid & ((BLOCKSIZE / WF_SIZE) - 1);
            const int swid = tid / (BLOCKSIZE / WF_SIZE);

            // segmented reduction
            for(uint32_t j = 1; j < (BLOCKSIZE / WF_SIZE); j <<= 1)
            {
                if(slid >= j)
                {
                    if(current_row == shared_row[slid - j])
                    {
                        sum = sum + shared_val[(BLOCKSIZE / WF_SIZE) * swid + slid - j];
                    }
                }
                __syncthreads();
                shared_val[(BLOCKSIZE / WF_SIZE) * swid + slid] = sum;
                __syncthreads();
            }

            if(slid < ((BLOCKSIZE / WF_SIZE) - 1))
            {
                if(current_row != shared_row[slid + 1] && current_row >= 0)
                {
                    if((l + swid) < n)
                    {
                        if(order_C == rocsparse_order_column)
                        {
                            rocsparse::atomic_add(&dense_C[(l + swid) * ldc + current_row],
                                                  static_cast<C>(sum));
                        }
                        else
                        {
                            rocsparse::atomic_add(&dense_C[current_row * ldc + (l + swid)],
                                                  static_cast<C>(sum));
                        }
                    }
                }
            }

            if(slid == ((BLOCKSIZE / WF_SIZE) - 1))
            {
                if(current_row >= 0)
                {
                    if((l + swid) < n)
                    {
                        if(order_C == rocsparse_order_column)
                        {
                            rocsparse::atomic_add(&dense_C[(l + swid) * ldc + current_row],
                                                  static_cast<C>(sum));
                        }
                        else
                        {
                            rocsparse::atomic_add(&dense_C[current_row * ldc + (l + swid)],
                                                  static_cast<C>(sum));
                        }
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              bool     TRANSB,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void coommtn_atomic_device(bool                 conj_A,
                                                    bool                 conj_B,
                                                    int64_t              nnz,
                                                    I                    n,
                                                    T                    alpha,
                                                    const I*             coo_row_ind,
                                                    const I*             coo_col_ind,
                                                    const A*             coo_val,
                                                    const B*             dense_B,
                                                    int64_t              ldb,
                                                    C*                   dense_C,
                                                    int64_t              ldc,
                                                    rocsparse_order      order_C,
                                                    rocsparse_index_base idx_base)
    {
        const int64_t gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= nnz)
        {
            return;
        }

        const I row = coo_row_ind[gid] - idx_base;
        const I col = coo_col_ind[gid] - idx_base;
        const T val = rocsparse::conj_val(coo_val[gid], conj_A);

        const T bval = (TRANSB) ? rocsparse::conj_val(dense_B[ldb * row + hipBlockIdx_y], conj_B)
                                : rocsparse::conj_val(dense_B[hipBlockIdx_y * ldb + row], conj_B);

        if(order_C == rocsparse_order_column)
        {
            rocsparse::atomic_add(&dense_C[hipBlockIdx_y * ldc + col],
                                  static_cast<C>(alpha * (val * bval)));
        }
        else
        {
            rocsparse::atomic_add(&dense_C[col * ldc + hipBlockIdx_y],
                                  static_cast<C>(alpha * (val * bval)));
        }
    }
}
