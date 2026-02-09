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
    ROCSPARSE_DEVICE_ILF void coommnn_segmented_main_device(bool    conj_A,
                                                            bool    conj_B,
                                                            I       M,
                                                            I       N,
                                                            I       K,
                                                            int64_t nnz,
                                                            int64_t batch_stride_A,
                                                            T       alpha,
                                                            I* __restrict__ row_block_red,
                                                            T* __restrict__ val_block_red,
                                                            const I* __restrict__ coo_row_ind,
                                                            const I* __restrict__ coo_col_ind,
                                                            const A* __restrict__ coo_val,
                                                            const B* __restrict__ dense_B,
                                                            int64_t ldb,
                                                            int64_t batch_stride_B,
                                                            C* __restrict__ dense_C,
                                                            int64_t              ldc,
                                                            int64_t              batch_stride_C,
                                                            rocsparse_order      order_C,
                                                            rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        const int batch = hipBlockIdx_z;

        __shared__ I shared_row[BLOCKSIZE];
        __shared__ T shared_val_prev[WF_SIZE];
        __shared__ T shared_val[BLOCKSIZE * WF_SIZE];

        const I       colB   = WF_SIZE * hipBlockIdx_y;
        const int64_t offset = bid * LOOPS * BLOCKSIZE;

        I row_ind;
        T valB[WF_SIZE];
        for(int64_t idx = offset + tid; idx < (offset + LOOPS * BLOCKSIZE); idx += BLOCKSIZE)
        {

            const I row
                = (idx < nnz)
                      ? rocsparse::nontemporal_load(&coo_row_ind[idx + batch_stride_A * batch])
                            - idx_base
                      : -1;
            const I col
                = (idx < nnz)
                      ? rocsparse::nontemporal_load(&coo_col_ind[idx + batch_stride_A * batch])
                            - idx_base
                      : 0;
            const T val
                = (idx < nnz)
                      ? alpha
                            * rocsparse::conj_val(
                                rocsparse::nontemporal_load(&coo_val[idx + batch_stride_A * batch]),
                                conj_A)
                      : static_cast<T>(0);

            row_ind = row;

            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                T v = rocsparse::shfl(val, i, WF_SIZE);
                I c = rocsparse::shfl(col, i, WF_SIZE);

                if(!TRANSB)
                {
                    valB[i] = v
                              * rocsparse::conj_val(
                                  dense_B[c + ldb * (colB + lid) + batch_stride_B * batch], conj_B);
                }
                else
                {
                    valB[i] = v
                              * rocsparse::conj_val(
                                  dense_B[ldb * c + (colB + lid) + batch_stride_B * batch], conj_B);
                }
            }

            // Transpose
            __syncthreads();
            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                shared_val[BLOCKSIZE * lid + WF_SIZE * wid + i] = valB[i];
            }
            __syncthreads();
            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                valB[i] = shared_val[BLOCKSIZE * i + tid];
            }

            // First thread in block checks row index from previous loop
            // if it has been completed or if additional rows have to be
            // appended.
            if(idx > offset && tid == 0)
            {
                const I prevrow = shared_row[BLOCKSIZE - 1];
                if(row_ind == prevrow)
                {
                    for(uint32_t i = 0; i < WF_SIZE; ++i)
                    {
                        valB[i] += shared_val_prev[i];
                    }
                }
                else if(prevrow >= 0)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(uint32_t i = 0; i < WF_SIZE; ++i)
                        {
                            dense_C[prevrow + ldc * (colB + i) + batch_stride_C * batch]
                                += shared_val_prev[i];
                        }
                    }
                    else
                    {
                        for(uint32_t i = 0; i < WF_SIZE; ++i)
                        {
                            dense_C[(colB + i) + ldc * prevrow + batch_stride_C * batch]
                                += shared_val_prev[i];
                        }
                    }
                }
            }

            __syncthreads();
            shared_row[tid] = row_ind;
            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                shared_val[BLOCKSIZE * i + tid] = valB[i];
            }
            __syncthreads();

            // segmented reduction
            for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
            {
                if(tid >= j)
                {
                    if(row_ind == shared_row[tid - j])
                    {
                        for(uint32_t i = 0; i < WF_SIZE; ++i)
                        {
                            valB[i] = valB[i] + shared_val[BLOCKSIZE * i + tid - j];
                        }
                    }
                }
                __syncthreads();
                for(uint32_t i = 0; i < WF_SIZE; ++i)
                {
                    shared_val[BLOCKSIZE * i + tid] = valB[i];
                }
                __syncthreads();
            }

            shared_val_prev[lid] = shared_val[BLOCKSIZE * lid + (BLOCKSIZE - 1)];
            __syncthreads();

            // All lanes but the last one write their result in C.
            // The last value might need to be appended by the next iteration.
            if(tid < BLOCKSIZE - 1)
            {
                if(row_ind != shared_row[tid + 1] && row_ind >= 0)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(uint32_t i = 0; i < WF_SIZE; ++i)
                        {
                            dense_C[row_ind + ldc * (colB + i) + batch_stride_C * batch] += valB[i];
                        }
                    }
                    else
                    {
                        for(uint32_t i = 0; i < WF_SIZE; ++i)
                        {
                            dense_C[(colB + i) + ldc * row_ind + batch_stride_C * batch] += valB[i];
                        }
                    }
                }
            }
        }

        if(tid == BLOCKSIZE - 1)
        {
            row_block_red[bid + hipGridDim_x * batch] = row_ind;
            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                val_block_red[hipGridDim_x * (colB + i) + bid + (hipGridDim_x * N) * batch]
                    = valB[i];
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t LOOPS,
              bool     TRANSB,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void coommnn_segmented_remainder_device(bool    conj_A,
                                                                 bool    conj_B,
                                                                 I       colB_offset,
                                                                 I       M,
                                                                 I       N,
                                                                 I       K,
                                                                 int64_t nnz,
                                                                 int64_t batch_stride_A,
                                                                 T       alpha,
                                                                 I* __restrict__ row_block_red,
                                                                 T* __restrict__ val_block_red,
                                                                 const I* __restrict__ coo_row_ind,
                                                                 const I* __restrict__ coo_col_ind,
                                                                 const A* __restrict__ coo_val,
                                                                 const B* __restrict__ dense_B,
                                                                 int64_t ldb,
                                                                 int64_t batch_stride_B,
                                                                 C* __restrict__ dense_C,
                                                                 int64_t         ldc,
                                                                 int64_t         batch_stride_C,
                                                                 rocsparse_order order_C,
                                                                 rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        const int batch = hipBlockIdx_z;

        __shared__ I shared_row[BLOCKSIZE];
        __shared__ T shared_val_prev[WF_SIZE];
        __shared__ T shared_val[BLOCKSIZE * WF_SIZE];

        const I colB = colB_offset;

        const int64_t offset = bid * LOOPS * BLOCKSIZE;

        I row_ind;
        T valB[WF_SIZE];

        for(int64_t idx = offset + tid; idx < (offset + LOOPS * BLOCKSIZE); idx += BLOCKSIZE)
        {

            const I row
                = (idx < nnz)
                      ? rocsparse::nontemporal_load(&coo_row_ind[idx + batch_stride_A * batch])
                            - idx_base
                      : -1;
            const I col
                = (idx < nnz)
                      ? rocsparse::nontemporal_load(&coo_col_ind[idx + batch_stride_A * batch])
                            - idx_base
                      : 0;
            const T val
                = (idx < nnz)
                      ? alpha
                            * rocsparse::conj_val(
                                rocsparse::nontemporal_load(&coo_val[idx + batch_stride_A * batch]),
                                conj_A)
                      : static_cast<T>(0);

            row_ind = row;

            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {

                const T v = rocsparse::shfl(val, i, WF_SIZE);
                const I c = rocsparse::shfl(col, i, WF_SIZE);

                if(!TRANSB)
                {
                    valB[i]
                        = (colB + lid) < N
                              ? v
                                    * rocsparse::conj_val(
                                        dense_B[c + ldb * (colB + lid) + batch_stride_B * batch],
                                        conj_B)
                              : static_cast<T>(0);
                }
                else
                {
                    valB[i]
                        = (colB + lid) < N
                              ? v
                                    * rocsparse::conj_val(
                                        dense_B[ldb * c + (colB + lid) + batch_stride_B * batch],
                                        conj_B)
                              : static_cast<T>(0);
                }
            }

            // Transpose
            __syncthreads();
            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                shared_val[BLOCKSIZE * lid + WF_SIZE * wid + i] = valB[i];
            }
            __syncthreads();
            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                valB[i] = shared_val[BLOCKSIZE * i + tid];
            }

            // First thread in block checks row index from previous loop
            // if it has been completed or if additional rows have to be
            // appended.
            if(idx > offset && tid == 0)
            {
                const I prevrow = shared_row[BLOCKSIZE - 1];
                if(row_ind == prevrow)
                {
                    for(uint32_t i = 0; i < WF_SIZE; ++i)
                    {
                        valB[i] += shared_val_prev[i];
                    }
                }
                else if(prevrow >= 0)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(uint32_t i = 0; i < WF_SIZE; ++i)
                        {
                            if((colB + i) < N)
                            {
                                dense_C[prevrow + ldc * (colB + i) + batch_stride_C * batch]
                                    += shared_val_prev[i];
                            }
                        }
                    }
                    else
                    {
                        for(uint32_t i = 0; i < WF_SIZE; ++i)
                        {
                            if((colB + i) < N)
                            {
                                dense_C[colB + i + ldc * prevrow + batch_stride_C * batch]
                                    += shared_val_prev[i];
                            }
                        }
                    }
                }
            }

            __syncthreads();
            shared_row[tid] = row_ind;
            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                shared_val[BLOCKSIZE * i + tid] = valB[i];
            }
            __syncthreads();

            // segmented reduction
            for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
            {
                if(tid >= j)
                {
                    if(row_ind == shared_row[tid - j])
                    {
                        for(uint32_t i = 0; i < WF_SIZE; ++i)
                        {
                            valB[i] = valB[i] + shared_val[BLOCKSIZE * i + tid - j];
                        }
                    }
                }
                __syncthreads();
                for(uint32_t i = 0; i < WF_SIZE; ++i)
                {
                    shared_val[BLOCKSIZE * i + tid] = valB[i];
                }
                __syncthreads();
            }

            shared_val_prev[lid] = shared_val[BLOCKSIZE * lid + (BLOCKSIZE - 1)];
            __syncthreads();

            // All lanes but the last one write their result in C.
            // The last value might need to be appended by the next iteration.
            if(tid < BLOCKSIZE - 1)
            {
                if(row_ind != shared_row[tid + 1] && row_ind >= 0)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(uint32_t i = 0; i < WF_SIZE; ++i)
                        {
                            if((colB + i) < N)
                            {
                                dense_C[row_ind + ldc * (colB + i) + batch_stride_C * batch]
                                    += valB[i];
                            }
                        }
                    }
                    else
                    {
                        for(uint32_t i = 0; i < WF_SIZE; ++i)
                        {
                            if((colB + i) < N)
                            {
                                dense_C[(colB + i) + ldc * row_ind + batch_stride_C * batch]
                                    += valB[i];
                            }
                        }
                    }
                }
            }
        }

        if(tid == BLOCKSIZE - 1)
        {
            row_block_red[bid + hipGridDim_x * batch] = row_ind;
            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                if((colB + i) < N)
                {
                    val_block_red[hipGridDim_x * (colB + i) + bid + (hipGridDim_x * N) * batch]
                        = valB[i];
                }
            }
        }
    }

    // Segmented block reduction kernel
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void segmented_blockreduce(const I* rows, T* vals)
    {
        const int tid = hipThreadIdx_x;

#pragma unroll
        for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
        {

            const T val
                = ((tid >= j) && (rows[tid] == rows[tid - j])) ? vals[tid - j] : static_cast<T>(0);

            __syncthreads();

            vals[tid] = vals[tid] + val;

            __syncthreads();
        }
    }

    // Do the final block reduction of the block reduction buffers back into global memory
    template <uint32_t BLOCKSIZE, typename T, typename I, typename C>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void coommnn_general_block_reduce(I n,
                                      I nblocks,
                                      const I* __restrict__ row_block_red,
                                      const T* __restrict__ val_block_red,
                                      C*              dense_C,
                                      int64_t         ldc,
                                      int64_t         batch_stride_C,
                                      rocsparse_order order_C)
    {
        const int tid   = hipThreadIdx_x;
        const int batch = hipBlockIdx_z;

        // Shared memory to hold row indices and values for segmented reduction
        __shared__ I shared_row[BLOCKSIZE];
        __shared__ T shared_val[BLOCKSIZE];

        const I col = hipBlockIdx_x;

        for(I i = 0; i < nblocks; i += BLOCKSIZE)
        {
            // Copy data to reduction buffers
            shared_row[tid] = (tid + i < nblocks) ? row_block_red[tid + i + nblocks * batch] : -1;
            shared_val[tid] = (tid + i < nblocks)
                                  ? val_block_red[tid + i + nblocks * col + nblocks * n * batch]
                                  : static_cast<T>(0);

            __syncthreads();

            // Do segmented block reduction
            segmented_blockreduce<BLOCKSIZE>(shared_row, shared_val);

            // Add reduced sum to C if valid
            const I row   = shared_row[tid];
            const I rowp1 = (tid < BLOCKSIZE - 1) ? shared_row[tid + 1] : -1;

            if(row != rowp1 && row >= 0)
            {
                if(order_C == rocsparse_order_column)
                {
                    dense_C[row + ldc * col + batch_stride_C * batch] += shared_val[tid];
                }
                else
                {
                    dense_C[col + ldc * row + batch_stride_C * batch] += shared_val[tid];
                }
            }

            __syncthreads();
        }
    }
}
