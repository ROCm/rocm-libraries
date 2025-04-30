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

#include "common.h"

namespace rocsparse
{
    // Implementation motivated by papers 'Efficient Sparse Matrix-Vector Multiplication on CUDA',
    // 'Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors' and
    // 'Segmented operations for sparse matrix computation on vector multiprocessors'
    template <uint32_t BLOCKSIZE, typename I, typename A, typename X, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void coomvn_segmented_loops_device(int64_t nnz,
                                                            I       nloops,
                                                            T       alpha,
                                                            const I* __restrict__ coo_row_ind,
                                                            const I* __restrict__ coo_col_ind,
                                                            const A* __restrict__ coo_val,
                                                            const X* __restrict__ x,
                                                            Y* __restrict__ y,
                                                            I* __restrict__ row_block_red,
                                                            T* __restrict__ val_block_red,
                                                            rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;

        // Shared memory to hold row indices and values for segmented reduction
        __shared__ I shared_row[BLOCKSIZE];
        __shared__ T shared_val[BLOCKSIZE];

        // Current threads index into COO structure
        int64_t idx = hipBlockIdx_x * nloops * BLOCKSIZE + tid;

        I row;
        T val;

        if(idx < nnz)
        {
            row = rocsparse::nontemporal_load(coo_row_ind + idx) - idx_base;
            val = static_cast<T>(rocsparse::nontemporal_load(coo_val + idx))
                  * static_cast<T>(rocsparse::ldg(x + rocsparse::nontemporal_load(coo_col_ind + idx)
                                                  - idx_base));
        }
        else
        {
            row = -1;
            val = static_cast<T>(0);
        }

        shared_row[tid] = row;
        shared_val[tid] = val;
        __syncthreads();

        // Segmented wavefront reduction
        for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
        {
            if(tid >= j)
            {
                if(row == shared_row[tid - j])
                {
                    val = val + shared_val[tid - j];
                }
            }
            __syncthreads();
            shared_val[tid] = val;
            __syncthreads();
        }

        // All lanes but the last one write their result in y.
        // The last value might need to be appended by the next iteration.
        if(tid < BLOCKSIZE - 1)
        {
            if(row != shared_row[tid + 1] && row >= 0)
            {
                y[row] = rocsparse::fma<T>(alpha, val, y[row]);
            }
        }

        for(int i = 0; i < nloops - 1; i++)
        {
            // Keep going for the next iteration
            idx += BLOCKSIZE;

            // Get corresponding COO entry, if not out of bounds.
            // This can happen when processing more than 1 entry if
            // nnz % BLOCKSIZE != 0
            if(idx < nnz)
            {
                row = rocsparse::nontemporal_load(coo_row_ind + idx) - idx_base;
                val = static_cast<T>(rocsparse::nontemporal_load(coo_val + idx))
                      * static_cast<T>(rocsparse::ldg(
                          x + rocsparse::nontemporal_load(coo_col_ind + idx) - idx_base));
            }
            else
            {
                row = -1;
                val = static_cast<T>(0);
            }

            // First thread in wavefront checks row index from previous loop
            // if it has been completed or if additional rows have to be
            // appended.
            if(tid == 0)
            {
                const I prevrow = shared_row[BLOCKSIZE - 1];
                if(row == prevrow)
                {
                    val = val + shared_val[BLOCKSIZE - 1];
                }
                else if(prevrow >= 0)
                {
                    y[prevrow] = rocsparse::fma<T>(alpha, shared_val[BLOCKSIZE - 1], y[prevrow]);
                }
            }

            __syncthreads();
            shared_row[tid] = row;
            shared_val[tid] = val;
            __syncthreads();

            // Segmented wavefront reduction
            for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
            {
                if(tid >= j)
                {
                    if(row == shared_row[tid - j])
                    {
                        val = val + shared_val[tid - j];
                    }
                }
                __syncthreads();
                shared_val[tid] = val;
                __syncthreads();
            }

            // All lanes but the last one write their result in y.
            // The last value might need to be appended by the next iteration.
            if(tid < BLOCKSIZE - 1)
            {
                if(row != shared_row[tid + 1] && row >= 0)
                {
                    y[row] = rocsparse::fma<T>(alpha, val, y[row]);
                }
            }
        }

        // Write last entries into buffers for segmented block reduction
        if(tid == BLOCKSIZE - 1)
        {
            rocsparse::nontemporal_store(row, row_block_red + hipBlockIdx_x);
            rocsparse::nontemporal_store(alpha * val, val_block_red + hipBlockIdx_x);
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
            T val = static_cast<T>(0);
            if(tid >= j)
            {
                if(rows[tid] == rows[tid - j])
                {
                    val = vals[tid - j];
                }
            }
            __syncthreads();

            vals[tid] = vals[tid] + val;
            __syncthreads();
        }
    }

    // Do the final block reduction of the block reduction buffers back into global memory
    template <uint32_t BLOCKSIZE, typename I, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void
        coomvn_segmented_loops_reduce_device(I nblocks,
                                             const I* __restrict__ row_block_red,
                                             const T* __restrict__ val_block_red,
                                             Y* __restrict__ y)
    {
        const int tid = hipThreadIdx_x;

        // Shared memory to hold row indices and values for segmented reduction
        __shared__ I shared_row[BLOCKSIZE];
        __shared__ T shared_val[BLOCKSIZE];

        // Loop over blocks that are subject for segmented reduction
        for(I i = 0; i < nblocks; i += BLOCKSIZE)
        {
            // Copy data to reduction buffers
            shared_row[tid] = (tid + i < nblocks) ? row_block_red[tid + i] : -1;
            shared_val[tid] = (tid + i < nblocks) ? val_block_red[tid + i] : static_cast<T>(0);

            __syncthreads();

            // Do segmented block reduction
            segmented_blockreduce<BLOCKSIZE>(shared_row, shared_val);

            // Add reduced sum to y if valid
            const I row   = shared_row[tid];
            const I rowp1 = (tid < BLOCKSIZE - 1) ? shared_row[tid + 1] : -1;

            if(row != rowp1 && row >= 0)
            {
                y[row] = y[row] + shared_val[tid];
            }

            __syncthreads();
        }
    }

    // Implementation motivated by papers 'Efficient Sparse Matrix-Vector Multiplication on CUDA',
    // 'Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors' and
    // 'Segmented operations for sparse matrix computation on vector multiprocessors' for array
    // of structure format (AoS)
    template <uint32_t BLOCKSIZE, typename I, typename A, typename X, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void coomvn_aos_segmented_loops_device(int64_t nnz,
                                                                I       nloops,
                                                                T       alpha,
                                                                const I* __restrict__ coo_ind,
                                                                const A* __restrict__ coo_val,
                                                                const X* __restrict__ x,
                                                                Y* __restrict__ y,
                                                                I* __restrict__ row_block_red,
                                                                T* __restrict__ val_block_red,
                                                                rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;

        // Shared memory to hold row indices and values for segmented reduction
        __shared__ I shared_row[BLOCKSIZE];
        __shared__ T shared_val[BLOCKSIZE];

        // Current threads index into COO structure
        int64_t idx = hipBlockIdx_x * nloops * BLOCKSIZE + tid;

        I row;
        T val;

        if(idx < nnz)
        {
            row = rocsparse::nontemporal_load(coo_ind + 2 * idx) - idx_base;
            val = static_cast<T>(rocsparse::nontemporal_load(coo_val + idx))
                  * static_cast<T>(rocsparse::ldg(
                      x + rocsparse::nontemporal_load(coo_ind + 2 * idx + 1) - idx_base));
        }
        else
        {
            row = -1;
            val = static_cast<T>(0);
        }

        shared_row[tid] = row;
        shared_val[tid] = val;
        __syncthreads();

        // Segmented wavefront reduction
        for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
        {
            if(tid >= j)
            {
                if(row == shared_row[tid - j])
                {
                    val = val + shared_val[tid - j];
                }
            }
            __syncthreads();
            shared_val[tid] = val;
            __syncthreads();
        }

        // All lanes but the last one write their result in y.
        // The last value might need to be appended by the next iteration.
        if(tid < BLOCKSIZE - 1)
        {
            if(row != shared_row[tid + 1] && row >= 0)
            {
                y[row] = rocsparse::fma<T>(alpha, val, y[row]);
            }
        }

        for(int i = 0; i < nloops - 1; i++)
        {
            // Keep going for the next iteration
            idx += BLOCKSIZE;

            // Get corresponding COO entry, if not out of bounds.
            // This can happen when processing more than 1 entry if
            // nnz % BLOCKSIZE != 0
            if(idx < nnz)
            {
                row = rocsparse::nontemporal_load(coo_ind + 2 * idx) - idx_base;
                val = static_cast<T>(rocsparse::nontemporal_load(coo_val + idx))
                      * static_cast<T>(rocsparse::ldg(
                          x + rocsparse::nontemporal_load(coo_ind + 2 * idx + 1) - idx_base));
            }
            else
            {
                row = -1;
                val = static_cast<T>(0);
            }

            // First thread in wavefront checks row index from previous loop
            // if it has been completed or if additional rows have to be
            // appended.
            if(tid == 0)
            {
                const I prevrow = shared_row[BLOCKSIZE - 1];
                if(row == prevrow)
                {
                    val = val + shared_val[BLOCKSIZE - 1];
                }
                else if(prevrow >= 0)
                {
                    y[prevrow] = rocsparse::fma<T>(alpha, shared_val[BLOCKSIZE - 1], y[prevrow]);
                }
            }

            __syncthreads();
            shared_row[tid] = row;
            shared_val[tid] = val;
            __syncthreads();

            // Segmented wavefront reduction
            for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
            {
                if(tid >= j)
                {
                    if(row == shared_row[tid - j])
                    {
                        val = val + shared_val[tid - j];
                    }
                }
                __syncthreads();
                shared_val[tid] = val;
                __syncthreads();
            }

            // All lanes but the last one write their result in y.
            // The last value might need to be appended by the next iteration.
            if(tid < BLOCKSIZE - 1)
            {
                if(row != shared_row[tid + 1] && row >= 0)
                {
                    y[row] = rocsparse::fma<T>(alpha, val, y[row]);
                }
            }
        }

        // Write last entries into buffers for segmented block reduction
        if(tid == BLOCKSIZE - 1)
        {
            rocsparse::nontemporal_store(row, row_block_red + hipBlockIdx_x);
            rocsparse::nontemporal_store(alpha * val, val_block_red + hipBlockIdx_x);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t LOOPS,
              typename I,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void coomvn_atomic_loops_device(int64_t nnz,
                                                         T       alpha,
                                                         const I* __restrict__ coo_row_ind,
                                                         const I* __restrict__ coo_col_ind,
                                                         const A* __restrict__ coo_val,
                                                         const X* __restrict__ x,
                                                         Y* __restrict__ y,
                                                         rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;

        __shared__ I shared_row[BLOCKSIZE];
        __shared__ T shared_val[BLOCKSIZE];

        I row;
        I col;
        T val;

        // Current threads index into COO structure
        int64_t idx = hipBlockIdx_x * LOOPS * BLOCKSIZE + tid;

        if(idx < nnz)
        {
            row = rocsparse::nontemporal_load(&coo_row_ind[idx]) - idx_base;
            col = rocsparse::nontemporal_load(&coo_col_ind[idx]) - idx_base;
            val = static_cast<T>(rocsparse::nontemporal_load(&coo_val[idx]))
                  * static_cast<T>(x[col]);
        }
        else
        {
            row = -1;
            col = 0;
            val = static_cast<T>(0);
        }

        shared_row[tid] = row;
        shared_val[tid] = val;
        __syncthreads();

        // segmented reduction
        for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
        {
            if(tid >= j)
            {
                if(row == shared_row[tid - j])
                {
                    val = val + shared_val[tid - j];
                }
            }
            __syncthreads();
            shared_val[tid] = val;
            __syncthreads();
        }

        if(tid < BLOCKSIZE - 1)
        {
            if(row != shared_row[tid + 1] && row >= 0)
            {
                rocsparse::atomic_add(&y[row], alpha * val);
            }
        }

        if(LOOPS > 1)
        {
            for(uint32_t i = 0; i < LOOPS - 1; i++)
            {
                // Keep going for the next iteration
                idx += BLOCKSIZE;

                if(idx < nnz)
                {
                    row = rocsparse::nontemporal_load(&coo_row_ind[idx]) - idx_base;
                    col = rocsparse::nontemporal_load(&coo_col_ind[idx]) - idx_base;
                    val = static_cast<T>(rocsparse::nontemporal_load(&coo_val[idx]))
                          * static_cast<T>(x[col]);
                }
                else
                {
                    row = -1;
                    col = 0;
                    val = static_cast<T>(0);
                }

                // First thread in wavefront checks row index from previous loop
                // if it has been completed or if additional rows have to be
                // appended.
                if(tid == 0)
                {
                    I prevrow = shared_row[BLOCKSIZE - 1];
                    if(row == prevrow)
                    {
                        val = val + shared_val[BLOCKSIZE - 1];
                    }
                    else if(prevrow >= 0)
                    {
                        rocsparse::atomic_add(&y[prevrow], alpha * shared_val[BLOCKSIZE - 1]);
                    }
                }

                __syncthreads();
                shared_row[tid] = row;
                shared_val[tid] = val;
                __syncthreads();

                // segmented reduction
                for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
                {
                    if(tid >= j)
                    {
                        if(row == shared_row[tid - j])
                        {
                            val = val + shared_val[tid - j];
                        }
                    }
                    __syncthreads();
                    shared_val[tid] = val;
                    __syncthreads();
                }

                if(tid < BLOCKSIZE - 1)
                {
                    if(row != shared_row[tid + 1] && row >= 0)
                    {
                        rocsparse::atomic_add(&y[row], alpha * val);
                    }
                }
            }
        }

        if(tid == BLOCKSIZE - 1)
        {
            if(row >= 0)
            {
                rocsparse::atomic_add(&y[row], alpha * val);
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t LOOPS,
              typename I,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void coomvn_aos_atomic_loops_device(int64_t nnz,
                                                             T       alpha,
                                                             const I* __restrict__ coo_ind,
                                                             const A* __restrict__ coo_val,
                                                             const X* __restrict__ x,
                                                             Y* __restrict__ y,
                                                             rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;

        __shared__ I shared_row[BLOCKSIZE];
        __shared__ T shared_val[BLOCKSIZE];

        I row;
        I col;
        T val;

        // Current threads index into COO structure
        int64_t idx = hipBlockIdx_x * LOOPS * BLOCKSIZE + tid;

        if(idx < nnz)
        {
            row = rocsparse::nontemporal_load(&coo_ind[2 * idx]) - idx_base;
            col = rocsparse::nontemporal_load(&coo_ind[2 * idx + 1]) - idx_base;
            val = static_cast<T>(rocsparse::nontemporal_load(&coo_val[idx]))
                  * static_cast<T>(x[col]);
        }
        else
        {
            row = -1;
            col = 0;
            val = static_cast<T>(0);
        }

        shared_row[tid] = row;
        shared_val[tid] = val;
        __syncthreads();

        // segmented reduction
        for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
        {
            if(tid >= j)
            {
                if(row == shared_row[tid - j])
                {
                    val = val + shared_val[tid - j];
                }
            }
            __syncthreads();
            shared_val[tid] = val;
            __syncthreads();
        }

        if(tid < BLOCKSIZE - 1)
        {
            if(row != shared_row[tid + 1] && row >= 0)
            {
                rocsparse::atomic_add(&y[row], alpha * val);
            }
        }

        for(uint32_t i = 0; i < LOOPS - 1; i++)
        {
            // Keep going for the next iteration
            idx += BLOCKSIZE;

            if(idx < nnz)
            {
                row = rocsparse::nontemporal_load(&coo_ind[2 * idx]) - idx_base;
                col = rocsparse::nontemporal_load(&coo_ind[2 * idx + 1]) - idx_base;
                val = static_cast<T>(rocsparse::nontemporal_load(&coo_val[idx]))
                      * static_cast<T>(x[col]);
            }
            else
            {
                row = -1;
                col = 0;
                val = static_cast<T>(0);
            }

            // First thread in wavefront checks row index from previous loop
            // if it has been completed or if additional rows have to be
            // appended.
            if(tid == 0)
            {
                const I prevrow = shared_row[BLOCKSIZE - 1];
                if(row == prevrow)
                {
                    val = val + shared_val[BLOCKSIZE - 1];
                }
                else if(prevrow >= 0)
                {
                    rocsparse::atomic_add(&y[prevrow], alpha * shared_val[BLOCKSIZE - 1]);
                }
            }

            __syncthreads();
            shared_row[tid] = row;
            shared_val[tid] = val;
            __syncthreads();

            // segmented reduction
            for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
            {
                if(tid >= j)
                {
                    if(row == shared_row[tid - j])
                    {
                        val = val + shared_val[tid - j];
                    }
                }
                __syncthreads();
                shared_val[tid] = val;
                __syncthreads();
            }

            if(tid < BLOCKSIZE - 1)
            {
                if(row != shared_row[tid + 1] && row >= 0)
                {
                    rocsparse::atomic_add(&y[row], alpha * val);
                }
            }
        }

        if(tid == BLOCKSIZE - 1)
        {
            if(row >= 0)
            {
                rocsparse::atomic_add(&y[row], alpha * val);
            }
        }
    }

    template <typename I, typename A, typename X, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void coomvt_device(rocsparse_operation  trans,
                                            int64_t              nnz,
                                            T                    alpha,
                                            const I*             coo_row_ind,
                                            const I*             coo_col_ind,
                                            const A*             coo_val,
                                            const X*             x,
                                            Y*                   y,
                                            rocsparse_index_base idx_base)
    {
        const int64_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(gid >= nnz)
        {
            return;
        }

        const I row = coo_row_ind[gid] - idx_base;
        const I col = coo_col_ind[gid] - idx_base;
        const A val = (trans == rocsparse_operation_conjugate_transpose)
                          ? rocsparse::conj(coo_val[gid])
                          : coo_val[gid];

        rocsparse::atomic_add(&y[col], alpha * val * x[row]);
    }

    template <typename I, typename A, typename X, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void coomvt_aos_device(rocsparse_operation  trans,
                                                int64_t              nnz,
                                                T                    alpha,
                                                const I*             coo_ind,
                                                const A*             coo_val,
                                                const X*             x,
                                                Y*                   y,
                                                rocsparse_index_base idx_base)
    {
        const int64_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(gid >= nnz)
        {
            return;
        }

        const I row = coo_ind[2 * gid] - idx_base;
        const I col = coo_ind[2 * gid + 1] - idx_base;
        const A val = (trans == rocsparse_operation_conjugate_transpose)
                          ? rocsparse::conj(coo_val[gid])
                          : coo_val[gid];

        rocsparse::atomic_add(&y[col], alpha * val * x[row]);
    }
}
