/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnn_row_split_shared_device(T       alpha,
                                                              T       beta,
                                                              bool    conj_A,
                                                              bool    conj_B,
                                                              J       M,
                                                              J       N,
                                                              int64_t offsets_batch_stride_A,
                                                              int64_t columns_values_batch_stride_A,
                                                              const I* __restrict__ csr_row_ptr,
                                                              const J* __restrict__ csr_col_ind,
                                                              const A* __restrict__ csr_val,
                                                              const B* __restrict__ dense_B,
                                                              int64_t ldb,
                                                              int64_t batch_stride_B,
                                                              C* __restrict__ dense_C,
                                                              int64_t              ldc,
                                                              int64_t              batch_stride_C,
                                                              rocsparse_order      order_C,
                                                              rocsparse_index_base idx_base)
    {
        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const uint32_t lid = gid & (WF_SIZE - 1);
        const uint32_t wid = tid / WF_SIZE;
        const J        row = gid / WF_SIZE;
        const J        col = lid + hipBlockIdx_y * WF_SIZE;

        const J batch = hipBlockIdx_z;

        if(row >= M)
        {
            return;
        }

        const int64_t colB = col * ldb;

        __shared__ J shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
        __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

        const I row_start = csr_row_ptr[row + offsets_batch_stride_A * batch] - idx_base;
        const I row_end   = csr_row_ptr[row + 1 + offsets_batch_stride_A * batch] - idx_base;

        T sum = static_cast<T>(0);

        for(I j = row_start; j < row_end; j += WF_SIZE)
        {
            const I k = j + lid;

            __syncthreads();

            if(k < row_end)
            {
                shared_col[wid][lid]
                    = csr_col_ind[k + columns_values_batch_stride_A * batch] - idx_base;
                shared_val[wid][lid] = rocsparse::conj_val(
                    csr_val[k + columns_values_batch_stride_A * batch], conj_A);
            }
            else
            {
                shared_col[wid][lid] = 0;
                shared_val[wid][lid] = static_cast<T>(0);
            }

            __syncthreads();

            if(col < N)
            {
                for(uint32_t i = 0; i < WF_SIZE; ++i)
                {
                    sum = rocsparse::fma<T>(
                        shared_val[wid][i],
                        rocsparse::conj_val(
                            dense_B[shared_col[wid][i] + colB + batch_stride_B * batch], conj_B),
                        sum);
                }
            }
        }

        if(col < N)
        {
            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
                    dense_C[row + col * ldc + batch_stride_C * batch] = alpha * sum;
                }
                else
                {
                    dense_C[row * ldc + col + batch_stride_C * batch] = alpha * sum;
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    dense_C[row + col * ldc + batch_stride_C * batch] = rocsparse::fma<T>(
                        beta, dense_C[row + col * ldc + batch_stride_C * batch], alpha * sum);
                }
                else
                {
                    dense_C[row * ldc + col + batch_stride_C * batch] = rocsparse::fma<T>(
                        beta, dense_C[row * ldc + col + batch_stride_C * batch], alpha * sum);
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnn_row_split_device(T       alpha,
                                                       T       beta,
                                                       bool    conj_A,
                                                       bool    conj_B,
                                                       J       offset,
                                                       J       M,
                                                       J       N,
                                                       int64_t offsets_batch_stride_A,
                                                       int64_t columns_values_batch_stride_A,
                                                       const I* __restrict__ csr_row_ptr,
                                                       const J* __restrict__ csr_col_ind,
                                                       const A* __restrict__ csr_val,
                                                       const B* __restrict__ dense_B,
                                                       int64_t ldb,
                                                       int64_t batch_stride_B,
                                                       C* __restrict__ dense_C,
                                                       int64_t              ldc,
                                                       int64_t              batch_stride_C,
                                                       rocsparse_order      order_C,
                                                       rocsparse_index_base idx_base)
    {
        const uint32_t tid  = hipThreadIdx_x;
        const J        gid  = hipBlockIdx_x * BLOCKSIZE + tid;
        const uint32_t lid  = gid & (WF_SIZE - 1);
        const J        row  = gid / WF_SIZE;
        const J        colB = offset + LOOPS * hipBlockIdx_y;

        const J batch = hipBlockIdx_z;

        if(row >= M)
        {
            return;
        }

        const I row_start = csr_row_ptr[row + offsets_batch_stride_A * batch] - idx_base;
        const I row_end   = csr_row_ptr[row + 1 + offsets_batch_stride_A * batch] - idx_base;

        T sum[LOOPS]{};

        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            const J col = csr_col_ind[j + columns_values_batch_stride_A * batch] - idx_base;
            const T val
                = rocsparse::conj_val(csr_val[j + columns_values_batch_stride_A * batch], conj_A);

            for(uint32_t p = 0; p < LOOPS; p++)
            {
                sum[p] = rocsparse::fma<T>(
                    val,
                    rocsparse::conj_val(
                        rocsparse::ldg(dense_B + col + (colB + p) * ldb + batch_stride_B * batch),
                        conj_B),
                    sum[p]);
            }
        }
        for(uint32_t p = 0; p < LOOPS; p++)
        {
            sum[p] = rocsparse::wfreduce_sum<WF_SIZE>(sum[p]);
        }

        if(lid == WF_SIZE - 1)
        {
            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        dense_C[row + (colB + p) * ldc + batch_stride_C * batch] = alpha * sum[p];
                    }
                }
                else
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        dense_C[row * ldc + (colB + p) + batch_stride_C * batch] = alpha * sum[p];
                    }
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        dense_C[row + (colB + p) * ldc + batch_stride_C * batch]
                            = rocsparse::fma<T>(
                                beta,
                                dense_C[row + (colB + p) * ldc + batch_stride_C * batch],
                                alpha * sum[p]);
                    }
                }
                else
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        dense_C[row * ldc + (colB + p) + batch_stride_C * batch]
                            = rocsparse::fma<T>(
                                beta,
                                dense_C[row * ldc + (colB + p) + batch_stride_C * batch],
                                alpha * sum[p]);
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t SUBWFSIZE,
              uint32_t LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void
        csrmmnt_row_split_subwfsize_x_loop_columns_device(T       alpha,
                                                          T       beta,
                                                          J       col_start,
                                                          J       col_end,
                                                          J       M,
                                                          J       N,
                                                          int64_t offsets_batch_stride_A,
                                                          int64_t columns_values_batch_stride_A,
                                                          int64_t ldb,
                                                          int64_t batch_stride_B,
                                                          int64_t ldc,
                                                          int64_t batch_stride_C,
                                                          const I* __restrict__ csr_row_ptr,
                                                          const J* __restrict__ csr_col_ind,
                                                          const A* __restrict__ csr_val,
                                                          const B* __restrict__ dense_B,
                                                          C* __restrict__ dense_C,
                                                          rocsparse_order      order_C,
                                                          rocsparse_index_base idx_base,
                                                          bool                 conj_A,
                                                          bool                 conj_B)
    {
        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const J        row = gid / WFSIZE;
        const uint32_t lid = tid & (WFSIZE - 1);

        const uint32_t slid = lid & (SUBWFSIZE - 1);
        const uint32_t swid = lid / SUBWFSIZE;

        const J batch = hipBlockIdx_y;

        if(row >= M)
        {
            return;
        }

        const I row_start
            = rocsparse::nontemporal_load(csr_row_ptr + row + offsets_batch_stride_A * batch)
              - idx_base;
        const I row_end
            = rocsparse::nontemporal_load(csr_row_ptr + row + 1 + offsets_batch_stride_A * batch)
              - idx_base;

        for(J l = col_start; l < col_end; l += SUBWFSIZE * LOOPS)
        {
            const J col = l + slid;

            T sum[LOOPS]{};

            for(I j = row_start; j < row_end; j += WFSIZE)
            {
                const I k = j + lid;

                const J c = (k < row_end)
                                ? (rocsparse::nontemporal_load(
                                       csr_col_ind + k + columns_values_batch_stride_A * batch)
                                   - idx_base)
                                : 0;

                const T v = (k < row_end) ? static_cast<T>(rocsparse::conj_val(
                                rocsparse::nontemporal_load(
                                    csr_val + k + columns_values_batch_stride_A * batch),
                                conj_A))
                                          : static_cast<T>(0);

                for(uint32_t i = 0; i < SUBWFSIZE; ++i)
                {
                    const int64_t sc = ldb * rocsparse::shfl(c, SUBWFSIZE * swid + i, SUBWFSIZE);
                    const T       sv = rocsparse::shfl(v, SUBWFSIZE * swid + i, SUBWFSIZE);

                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse::fma<T>(
                            sv,
                            rocsparse::conj_val(rocsparse::ldg(dense_B + col + p * SUBWFSIZE + sc
                                                               + batch_stride_B * batch),
                                                conj_B),
                            sum[p]);
                    }
                }
            }

            if(SUBWFSIZE < WFSIZE)
            {
                for(uint32_t p = 0; p < LOOPS; p++)
                {
                    sum[p] = rocsparse::wfreduce_partial_sum<WFSIZE, SUBWFSIZE>(sum[p]);
                }
            }

            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
                    if(lid >= (WFSIZE - SUBWFSIZE))
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            dense_C[row + (col + p * SUBWFSIZE) * ldc + batch_stride_C * batch]
                                = alpha * sum[p];
                        }
                    }
                }
                else
                {
                    if(lid >= (WFSIZE - SUBWFSIZE))
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            dense_C[row * ldc + col + p * SUBWFSIZE + batch_stride_C * batch]
                                = alpha * sum[p];
                        }
                    }
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    if(lid >= (WFSIZE - SUBWFSIZE))
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            dense_C[row + (col + p * SUBWFSIZE) * ldc + batch_stride_C * batch]
                                = rocsparse::fma<T>(beta,
                                                    dense_C[row + (col + p * SUBWFSIZE) * ldc
                                                            + batch_stride_C * batch],
                                                    alpha * sum[p]);
                        }
                    }
                }
                else
                {
                    if(lid >= (WFSIZE - SUBWFSIZE))
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            dense_C[row * ldc + col + p * SUBWFSIZE + batch_stride_C * batch]
                                = rocsparse::fma<T>(beta,
                                                    dense_C[row * ldc + col + p * SUBWFSIZE
                                                            + batch_stride_C * batch],
                                                    alpha * sum[p]);
                        }
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t SUB_WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void
        csrmmnt_row_split_shared_remainder_device(T       alpha,
                                                  T       beta,
                                                  bool    conj_A,
                                                  bool    conj_B,
                                                  J       col_start,
                                                  J       col_end,
                                                  J       M,
                                                  J       N,
                                                  int64_t offsets_batch_stride_A,
                                                  int64_t columns_values_batch_stride_A,
                                                  const I* __restrict__ csr_row_ptr,
                                                  const J* __restrict__ csr_col_ind,
                                                  const A* __restrict__ csr_val,
                                                  const B* __restrict__ dense_B,
                                                  int64_t ldb,
                                                  int64_t batch_stride_B,
                                                  C* __restrict__ dense_C,
                                                  int64_t              ldc,
                                                  int64_t              batch_stride_C,
                                                  rocsparse_order      order_C,
                                                  rocsparse_index_base idx_base)
    {
        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const J        row = gid / WF_SIZE;
        const uint32_t lid = tid & (WF_SIZE - 1);
        const uint32_t wid = tid / WF_SIZE;

        const uint32_t slid = lid & (SUB_WF_SIZE - 1);
        const uint32_t swid = lid / SUB_WF_SIZE;

        const J batch = hipBlockIdx_y;

        if(row >= M)
        {
            return;
        }

        __shared__ int64_t shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
        __shared__ T       shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

        const I row_start
            = rocsparse::nontemporal_load(csr_row_ptr + row + offsets_batch_stride_A * batch)
              - idx_base;
        const I row_end
            = rocsparse::nontemporal_load(csr_row_ptr + row + 1 + offsets_batch_stride_A * batch)
              - idx_base;

        const J col = col_start + slid;
        T       sum = static_cast<T>(0);

        for(I j = row_start; j < row_end; j += WF_SIZE)
        {
            const I k = j + lid;

            __syncthreads();

            if(k < row_end)
            {
                shared_col[wid][lid]
                    = ldb
                          * (rocsparse::nontemporal_load(csr_col_ind + k
                                                         + columns_values_batch_stride_A * batch)
                             - idx_base)
                      + batch_stride_B * batch;
                shared_val[wid][lid]
                    = rocsparse::conj_val(rocsparse::nontemporal_load(
                                              csr_val + k + columns_values_batch_stride_A * batch),
                                          conj_A);
            }
            else
            {
                shared_col[wid][lid] = 0;
                shared_val[wid][lid] = static_cast<T>(0);
            }

            __syncthreads();

            if(col < col_end)
            {
                for(uint32_t i = 0; i < SUB_WF_SIZE; ++i)
                {
                    sum = rocsparse::fma<T>(
                        shared_val[wid][SUB_WF_SIZE * swid + i],
                        rocsparse::conj_val(
                            rocsparse::ldg(dense_B + col + shared_col[wid][SUB_WF_SIZE * swid + i]),
                            conj_B),
                        sum);
                }
            }
        }

        if(SUB_WF_SIZE < WF_SIZE)
        {
            sum = rocsparse::wfreduce_partial_sum<WF_SIZE, SUB_WF_SIZE>(sum);
        }

        if(col < col_end && lid < SUB_WF_SIZE)
        {
            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
                    dense_C[row + col * ldc + batch_stride_C * batch] = alpha * sum;
                }
                else
                {
                    dense_C[row * ldc + col + batch_stride_C * batch] = alpha * sum;
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    dense_C[row + col * ldc + batch_stride_C * batch] = rocsparse::fma<T>(
                        beta, dense_C[row + col * ldc + batch_stride_C * batch], alpha * sum);
                }
                else
                {
                    dense_C[row * ldc + col + batch_stride_C * batch] = rocsparse::fma<T>(
                        beta, dense_C[row * ldc + col + batch_stride_C * batch], alpha * sum);
                }
            }
        }
    }

    template <typename T, uint32_t WFSIZE>
    ROCSPARSE_DEVICE_ILF void device_partial_sum(T* sum){};

    template <typename T, uint32_t WFSIZE, uint32_t N, uint32_t... R>
    ROCSPARSE_DEVICE_ILF void device_partial_sum(T* sum)
    {
        if(N < WFSIZE)
        {
            sum[0] = rocsparse::wfreduce_partial_sum<WFSIZE, N>(sum[0]);
        }
        device_partial_sum<T, WFSIZE, R...>(++sum);
    };

    template <std::size_t N, typename T>
    ROCSPARSE_DEVICE_ILF constexpr std::size_t sizeofarray(T (&)[N])
    {
        return N;
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t SUBWFSIZE,
              uint32_t LOOP,
              uint32_t... SUBWFSIZES_LIST,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnt_row_split_subwfsize_x_loop_plus_swfs_columns_device(
        T       alpha,
        T       beta,
        J       col_start,
        J       col_end,
        J       M,
        J       N,
        int64_t offsets_batch_stride_A,
        int64_t columns_values_batch_stride_A,
        const I* __restrict__ csr_row_ptr,
        const J* __restrict__ csr_col_ind,
        const A* __restrict__ csr_val,
        const B* __restrict__ dense_B,
        int64_t ldb,
        int64_t batch_stride_B,
        C* __restrict__ dense_C,
        int64_t              ldc,
        int64_t              batch_stride_C,
        rocsparse_order      order_C,
        rocsparse_index_base idx_base,
        bool                 conj_A,
        bool                 conj_B)
    {
        static constexpr uint32_t SUBWFSIZES[]   = {SUBWFSIZES_LIST...};
        static constexpr uint32_t NUM_SUBWFSIZES = sizeofarray(SUBWFSIZES);

        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const J        row = gid / WFSIZE;
        const uint32_t lid = tid & (WFSIZE - 1);

        const uint32_t slid = lid & (SUBWFSIZE - 1);
        const uint32_t swid = lid / SUBWFSIZE;

        uint32_t slid_swf[NUM_SUBWFSIZES];
        uint32_t swid_swf[NUM_SUBWFSIZES];

        for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
        {
            slid_swf[iswf] = lid & (SUBWFSIZES[iswf] - 1);
            swid_swf[iswf] = lid / SUBWFSIZES[iswf];
        }

        const J batch = hipBlockIdx_y;

        if(row >= M)
        {
            return;
        }

        const I row_start
            = rocsparse::nontemporal_load(csr_row_ptr + row + offsets_batch_stride_A * batch)
              - idx_base;
        const I row_end
            = rocsparse::nontemporal_load(csr_row_ptr + row + 1 + offsets_batch_stride_A * batch)
              - idx_base;

        const J col_loop = col_start + slid;

        J col_swf[NUM_SUBWFSIZES];

        if(NUM_SUBWFSIZES > 0)
        {
            col_swf[0] = col_start + SUBWFSIZE * LOOP;
            for(uint32_t iswf = 1; iswf < NUM_SUBWFSIZES; ++iswf)
            {
                col_swf[iswf] = col_swf[iswf - 1] + SUBWFSIZES[iswf - 1];
            }
            for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
            {
                col_swf[iswf] += slid_swf[iswf];
            }
        }

        T sum_loop[LOOP]{};
        T sum_swf[NUM_SUBWFSIZES]{};

        for(I j = row_start; j < row_end; j += WFSIZE)
        {
            const I k = j + lid;

            const J c = (k < row_end)
                            ? (rocsparse::nontemporal_load(csr_col_ind + k
                                                           + columns_values_batch_stride_A * batch)
                               - idx_base)
                            : 0;

            const T v = (k < row_end) ? static_cast<T>(rocsparse::conj_val(
                            rocsparse::nontemporal_load(csr_val + k
                                                        + columns_values_batch_stride_A * batch),
                            conj_A))
                                      : static_cast<T>(0);

            for(uint32_t i = 0; i < SUBWFSIZE; ++i)
            {
                const int64_t sc = ldb * rocsparse::shfl(c, SUBWFSIZE * swid + i, SUBWFSIZE);
                const T       sv = rocsparse::shfl(v, SUBWFSIZE * swid + i, SUBWFSIZE);

                for(uint32_t p = 0; p < LOOP; p++)
                {
                    sum_loop[p] = rocsparse::fma<T>(
                        sv,
                        rocsparse::conj_val(rocsparse::ldg(dense_B + col_loop + p * SUBWFSIZE + sc
                                                           + batch_stride_B * batch),
                                            conj_B),
                        sum_loop[p]);
                }
            }

            for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
            {
                for(uint32_t i = 0; i < SUBWFSIZES[iswf]; ++i)
                {
                    const J sc = rocsparse::shfl(
                        c, SUBWFSIZES[iswf] * swid_swf[iswf] + i, SUBWFSIZES[iswf]);
                    const T sv = rocsparse::shfl(
                        v, SUBWFSIZES[iswf] * swid_swf[iswf] + i, SUBWFSIZES[iswf]);

                    sum_swf[iswf] = rocsparse::fma<T>(
                        sv,
                        rocsparse::conj_val(rocsparse::ldg(dense_B + col_swf[iswf] + ldb * sc
                                                           + batch_stride_B * batch),
                                            conj_B),
                        sum_swf[iswf]);
                }
            }
        }

        if(SUBWFSIZE < WFSIZE)
        {
            for(uint32_t p = 0; p < LOOP; p++)
            {
                sum_loop[p] = rocsparse::wfreduce_partial_sum<WFSIZE, SUBWFSIZE>(sum_loop[p]);
            }
        }

        device_partial_sum<T, WFSIZE, SUBWFSIZES_LIST...>(sum_swf);

        if(beta == static_cast<T>(0))
        {
            if(order_C == rocsparse_order_column)
            {
                if(lid >= (WFSIZE - SUBWFSIZE))
                {
                    for(uint32_t p = 0; p < LOOP; p++)
                    {
                        dense_C[row + (col_loop + p * SUBWFSIZE) * ldc + batch_stride_C * batch]
                            = alpha * sum_loop[p];
                    }
                }

                for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
                {
                    if(lid >= (WFSIZE - SUBWFSIZES[iswf]))
                    {
                        dense_C[row + col_swf[iswf] * ldc + batch_stride_C * batch]
                            = alpha * sum_swf[iswf];
                    }
                }
            }
            else
            {
                if(lid >= (WFSIZE - SUBWFSIZE))
                {
                    for(uint32_t p = 0; p < LOOP; p++)
                    {
                        dense_C[row * ldc + (col_loop + p * SUBWFSIZE) + batch_stride_C * batch]
                            = alpha * sum_loop[p];
                    }
                }

                for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
                {
                    if(lid >= (WFSIZE - SUBWFSIZES[iswf]))
                    {
                        dense_C[row * ldc + col_swf[iswf] + batch_stride_C * batch]
                            = alpha * sum_swf[iswf];
                    }
                }
            }
        }
        else
        {
            if(order_C == rocsparse_order_column)
            {
                if(lid >= (WFSIZE - SUBWFSIZE))
                {
                    for(uint32_t p = 0; p < LOOP; p++)
                    {
                        dense_C[row + (col_loop + p * SUBWFSIZE) * ldc + batch_stride_C * batch]
                            = rocsparse::fma<T>(beta,
                                                dense_C[row + (col_loop + p * SUBWFSIZE) * ldc
                                                        + batch_stride_C * batch],
                                                alpha * sum_loop[p]);
                    }
                }

                for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
                {
                    if(lid >= (WFSIZE - SUBWFSIZES[iswf]))
                    {
                        dense_C[row + col_swf[iswf] * ldc + batch_stride_C * batch]
                            = rocsparse::fma<T>(
                                beta,
                                dense_C[row + col_swf[iswf] * ldc + batch_stride_C * batch],
                                alpha * sum_swf[iswf]);
                    }
                }
            }
            else
            {
                if(lid >= (WFSIZE - SUBWFSIZE))
                {
                    for(uint32_t p = 0; p < LOOP; p++)
                    {
                        dense_C[row * ldc + (col_loop + p * SUBWFSIZE) + batch_stride_C * batch]
                            = rocsparse::fma<T>(beta,
                                                dense_C[row * ldc + (col_loop + p * SUBWFSIZE)
                                                        + batch_stride_C * batch],
                                                alpha * sum_loop[p]);
                    }
                }

                for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
                {
                    if(lid >= (WFSIZE - SUBWFSIZES[iswf]))
                    {
                        dense_C[row * ldc + col_swf[iswf] + batch_stride_C * batch]
                            = rocsparse::fma<T>(
                                beta,
                                dense_C[row * ldc + col_swf[iswf] + batch_stride_C * batch],
                                alpha * sum_swf[iswf]);
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmtn_row_split_device(T       alpha,
                                                       bool    conj_A,
                                                       bool    conj_B,
                                                       J       M,
                                                       J       N,
                                                       int64_t offsets_batch_stride_A,
                                                       int64_t columns_values_batch_stride_A,
                                                       const I* __restrict__ csr_row_ptr,
                                                       const J* __restrict__ csr_col_ind,
                                                       const A* __restrict__ csr_val,
                                                       const B* __restrict__ dense_B,
                                                       int64_t ldb,
                                                       int64_t batch_stride_B,
                                                       C* __restrict__ dense_C,
                                                       int64_t              ldc,
                                                       int64_t              batch_stride_C,
                                                       rocsparse_order      order_C,
                                                       rocsparse_index_base idx_base)
    {
        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const uint32_t lid = gid & (WF_SIZE - 1);
        const uint32_t wid = tid / WF_SIZE;

        const J row = gid / WF_SIZE;

        const J batch = hipBlockIdx_z;

        if(row >= M)
        {
            return;
        }

        const J       cid  = lid + hipBlockIdx_y * WF_SIZE;
        const int64_t colB = cid * ldb;

        __shared__ T shared_B[BLOCKSIZE / WF_SIZE][WF_SIZE];
        shared_B[wid][lid]
            = (cid < N) ? rocsparse::conj_val(dense_B[row + colB + batch_stride_B * batch], conj_B)
                        : static_cast<B>(0);

        __threadfence_block();
        const I row_start = csr_row_ptr[row + offsets_batch_stride_A * batch] - idx_base;
        const I row_end   = csr_row_ptr[row + 1 + offsets_batch_stride_A * batch] - idx_base;

        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            const J col = csr_col_ind[j + columns_values_batch_stride_A * batch] - idx_base;
            const T val
                = alpha
                  * rocsparse::conj_val(csr_val[j + columns_values_batch_stride_A * batch], conj_A);

            if(order_C == rocsparse_order_column)
            {
                for(J i = 0; i < WF_SIZE && (i + hipBlockIdx_y * WF_SIZE) < N; ++i)
                {
                    rocsparse::atomic_add(&dense_C[col + (i + hipBlockIdx_y * WF_SIZE) * ldc
                                                   + batch_stride_C * batch],
                                          static_cast<C>(val * shared_B[wid][i]));
                }
            }
            else
            {
                for(J i = 0; i < WF_SIZE && (i + hipBlockIdx_y * WF_SIZE) < N; ++i)
                {
                    rocsparse::atomic_add(&dense_C[col * ldc + (i + hipBlockIdx_y * WF_SIZE)
                                                   + batch_stride_C * batch],
                                          static_cast<C>(val * shared_B[wid][i]));
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmtt_row_split_device(T       alpha,
                                                       bool    conj_A,
                                                       bool    conj_B,
                                                       J       M,
                                                       J       N,
                                                       int64_t offsets_batch_stride_A,
                                                       int64_t columns_values_batch_stride_A,
                                                       const I* __restrict__ csr_row_ptr,
                                                       const J* __restrict__ csr_col_ind,
                                                       const A* __restrict__ csr_val,
                                                       const B* __restrict__ dense_B,
                                                       int64_t ldb,
                                                       int64_t batch_stride_B,
                                                       C* __restrict__ dense_C,
                                                       int64_t              ldc,
                                                       int64_t              batch_stride_C,
                                                       rocsparse_order      order_C,
                                                       rocsparse_index_base idx_base)
    {
        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const uint32_t lid = gid & (WF_SIZE - 1);
        const uint32_t wid = tid / WF_SIZE;

        const J row = gid / WF_SIZE;
        const J cid = lid + hipBlockIdx_y * WF_SIZE;

        const J batch = hipBlockIdx_z;

        if(row >= M)
        {
            return;
        }

        __shared__ T shared_B[BLOCKSIZE / WF_SIZE][WF_SIZE];

        shared_B[wid][lid]
            = (cid < N)
                  ? rocsparse::conj_val(dense_B[ldb * row + cid + batch_stride_B * batch], conj_B)
                  : static_cast<B>(0);

        __threadfence_block();
        const I row_start = csr_row_ptr[row + offsets_batch_stride_A * batch] - idx_base;
        const I row_end   = csr_row_ptr[row + 1 + offsets_batch_stride_A * batch] - idx_base;

        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            const J col = csr_col_ind[j + columns_values_batch_stride_A * batch] - idx_base;
            const T val
                = alpha
                  * rocsparse::conj_val(csr_val[j + columns_values_batch_stride_A * batch], conj_A);

            if(order_C == rocsparse_order_column)
            {
                for(J i = 0; i < WF_SIZE && (i + hipBlockIdx_y * WF_SIZE) < N; ++i)
                {
                    rocsparse::atomic_add(&dense_C[col + (i + hipBlockIdx_y * WF_SIZE) * ldc
                                                   + batch_stride_C * batch],
                                          static_cast<C>(val * shared_B[wid][i]));
                }
            }
            else
            {
                for(J i = 0; i < WF_SIZE && (i + hipBlockIdx_y * WF_SIZE) < N; ++i)
                {
                    rocsparse::atomic_add(&dense_C[col * ldc + (i + hipBlockIdx_y * WF_SIZE)
                                                   + batch_stride_C * batch],
                                          static_cast<C>(val * shared_B[wid][i]));
                }
            }
        }
    }
}
