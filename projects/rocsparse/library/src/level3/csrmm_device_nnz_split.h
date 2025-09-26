/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_dichotomic_search.hpp"

namespace rocsparse
{
    // See Yang C., Bulu? A., Owens J.D. (2018) Design Principles for Sparse Matrix Multiplication on the GPU.
    // In: Aldinucci M., Padovani L., Torquati M. (eds) Euro-Par 2018: Parallel Processing. Euro-Par 2018.
    // Lecture Notes in Computer Science, vol 11014. Springer, Cham. https://doi.org/10.1007/978-3-319-96983-1_48
    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmmnn_nnz_split_main_device(bool conj_A,
                                                            bool conj_B,
                                                            J    ncol,
                                                            J    M,
                                                            J    N,
                                                            J    K,
                                                            I    nnz,
                                                            T    alpha,
                                                            J* __restrict__ row_block_red,
                                                            T* __restrict__ val_block_red,
                                                            const J* __restrict__ row_limits,
                                                            const I* __restrict__ csr_row_ptr,
                                                            const J* __restrict__ csr_col_ind,
                                                            const A* __restrict__ csr_val,
                                                            const B* __restrict__ dense_B,
                                                            int64_t ldb,
                                                            T       beta,
                                                            C* __restrict__ dense_C,
                                                            int64_t              ldc,
                                                            rocsparse_order      order_C,
                                                            rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        __shared__ J shared_row[BLOCKSIZE];
        __shared__ T shared_val[BLOCKSIZE * WF_SIZE];

        J row_ind = -1;
        J col_ind = 0;
        T val     = static_cast<T>(0);

        if((BLOCKSIZE * bid + tid) < nnz)
        {
            row_ind = dichotomic_search<I, J>(row_limits[bid],
                                              row_limits[bid + 1],
                                              BLOCKSIZE * bid + tid + idx_base,
                                              nnz + idx_base,
                                              csr_row_ptr);
            col_ind = rocsparse::nontemporal_load(&csr_col_ind[BLOCKSIZE * bid + tid]) - idx_base;
            val     = alpha
                  * conj_val(rocsparse::nontemporal_load(&csr_val[BLOCKSIZE * bid + tid]), conj_A);
        }

        shared_row[tid] = row_ind;
        __syncthreads();

        for(J colB = 0; colB < ncol; colB += WF_SIZE)
        {
            T valB[WF_SIZE];
            for(unsigned int i = 0; i < WF_SIZE; ++i)
            {
                const T v = rocsparse::shfl(val, i, WF_SIZE);
                const J c = rocsparse::shfl(col_ind, i, WF_SIZE);

                valB[i] = v * conj_val(dense_B[c + ldb * (colB + lid)], conj_B);
            }

            for(unsigned int i = 0; i < WF_SIZE; ++i)
            {
                shared_val[BLOCKSIZE * lid + WF_SIZE * wid + i] = valB[i];
            }
            __syncthreads();

            for(unsigned int i = 0; i < WF_SIZE; ++i)
            {
                valB[i] = shared_val[BLOCKSIZE * i + tid];
            }

            // segmented reduction
            for(J j = 1; j < BLOCKSIZE; j <<= 1)
            {
                if(tid >= j)
                {
                    if(row_ind == shared_row[tid - j])
                    {
                        for(unsigned int i = 0; i < WF_SIZE; ++i)
                        {
                            valB[i] = valB[i] + shared_val[BLOCKSIZE * i + tid - j];
                        }
                    }
                }
                __syncthreads();
                for(unsigned int i = 0; i < WF_SIZE; ++i)
                {
                    shared_val[BLOCKSIZE * i + tid] = valB[i];
                }
                __syncthreads();
            }

            // All lanes but the last one write their result in C.
            if(tid < (BLOCKSIZE - 1))
            {
                if(row_ind != shared_row[tid + 1] && row_ind >= 0)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(unsigned int i = 0; i < WF_SIZE; ++i)
                        {
                            dense_C[row_ind + ldc * (colB + i)] += valB[i];
                        }
                    }
                    else
                    {
                        for(unsigned int i = 0; i < WF_SIZE; ++i)
                        {
                            dense_C[colB + i + ldc * row_ind] += valB[i];
                        }
                    }
                }
            }

            if(tid == (BLOCKSIZE - 1))
            {
                for(unsigned int i = 0; i < WF_SIZE; ++i)
                {
                    val_block_red[hipGridDim_x * (colB + i) + bid] = valB[i];
                }
            }
        }

        if(tid == (BLOCKSIZE - 1))
        {
            row_block_red[bid] = row_ind;
        }
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmmnn_nnz_split_remainder_device(bool conj_A,
                                                                 bool conj_B,
                                                                 J    offset,
                                                                 J    M,
                                                                 J    N,
                                                                 J    K,
                                                                 I    nnz,
                                                                 T    alpha,
                                                                 J* __restrict__ row_block_red,
                                                                 T* __restrict__ val_block_red,
                                                                 const J* __restrict__ row_limits,
                                                                 const I* __restrict__ csr_row_ptr,
                                                                 const J* __restrict__ csr_col_ind,
                                                                 const A* __restrict__ csr_val,
                                                                 const B* __restrict__ dense_B,
                                                                 int64_t ldb,
                                                                 T       beta,
                                                                 C* __restrict__ dense_C,
                                                                 int64_t              ldc,
                                                                 rocsparse_order      order_C,
                                                                 rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        __shared__ J shared_row[BLOCKSIZE];
        __shared__ T shared_val[BLOCKSIZE * WF_SIZE];

        J row_ind = -1;
        J col_ind = 0;
        T val     = static_cast<T>(0);

        if(BLOCKSIZE * bid + tid < nnz)
        {
            row_ind = dichotomic_search<I, J>(row_limits[bid],
                                              row_limits[bid + 1],
                                              BLOCKSIZE * bid + tid + idx_base,
                                              nnz + idx_base,
                                              csr_row_ptr);
            col_ind = rocsparse::nontemporal_load(&csr_col_ind[BLOCKSIZE * bid + tid]) - idx_base;
            val     = alpha
                  * conj_val(rocsparse::nontemporal_load(&csr_val[BLOCKSIZE * bid + tid]), conj_A);
        }

        const J colB = offset;

        T valB[WF_SIZE];
        for(unsigned int i = 0; i < WF_SIZE; ++i)
        {
            const T v = rocsparse::shfl(val, i, WF_SIZE);
            const J c = rocsparse::shfl(col_ind, i, WF_SIZE);

            valB[i] = (colB + lid) < N ? v * conj_val(dense_B[c + ldb * (colB + lid)], conj_B)
                                       : static_cast<T>(0);
        }

        __syncthreads();
        shared_row[tid] = row_ind;
        for(unsigned int i = 0; i < WF_SIZE; ++i)
        {
            shared_val[BLOCKSIZE * lid + WF_SIZE * wid + i] = valB[i];
        }
        __syncthreads();

        for(unsigned int i = 0; i < WF_SIZE; ++i)
        {
            valB[i] = shared_val[BLOCKSIZE * i + tid];
        }

        // segmented reduction
        for(J j = 1; j < BLOCKSIZE; j <<= 1)
        {
            if(tid >= j)
            {
                if(row_ind == shared_row[tid - j])
                {
                    for(unsigned int i = 0; i < WF_SIZE; ++i)
                    {
                        valB[i] = valB[i] + shared_val[BLOCKSIZE * i + tid - j];
                    }
                }
            }
            __syncthreads();
            for(unsigned int i = 0; i < WF_SIZE; ++i)
            {
                shared_val[BLOCKSIZE * i + tid] = valB[i];
            }
            __syncthreads();
        }

        // All lanes but the last one write their result in C.
        if(tid < (BLOCKSIZE - 1))
        {
            if(row_ind != shared_row[tid + 1] && row_ind >= 0)
            {
                if(order_C == rocsparse_order_column)
                {
                    for(unsigned int i = 0; i < WF_SIZE; ++i)
                    {
                        if((colB + i) < N)
                        {
                            dense_C[row_ind + ldc * (colB + i)] += valB[i];
                        }
                    }
                }
                else
                {
                    for(unsigned int i = 0; i < WF_SIZE; ++i)
                    {
                        if((colB + i) < N)
                        {
                            dense_C[colB + i + ldc * row_ind] += valB[i];
                        }
                    }
                }
            }
        }

        if(tid == (BLOCKSIZE - 1))
        {
            row_block_red[bid] = row_ind;
            for(unsigned int i = 0; i < WF_SIZE; ++i)
            {
                if((colB + i) < N)
                {
                    val_block_red[hipGridDim_x * (colB + i) + bid] = valB[i];
                }
            }
        }
    }

    // Segmented block reduction kernel
    template <unsigned int BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void segmented_blockreduce(const I* __restrict__ rows,
                                                    T* __restrict__ vals)
    {
        const int tid = hipThreadIdx_x;

        for(unsigned int j = 1; j < BLOCKSIZE; j <<= 1)
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
    template <unsigned int BLOCKSIZE, typename I, typename J, typename C, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmmnn_general_block_reduce(I nblocks,
                                      const J* __restrict__ row_block_red,
                                      const T* __restrict__ val_block_red,
                                      C*              dense_C,
                                      int64_t         ldc,
                                      rocsparse_order order_C)
    {
        const int tid = hipThreadIdx_x;

        // Shared memory to hold row indices and values for segmented reduction
        __shared__ I shared_row[BLOCKSIZE];
        __shared__ T shared_val[BLOCKSIZE];

        shared_row[tid] = -1;
        shared_val[tid] = static_cast<T>(0);

        __syncthreads();

        const I col = hipBlockIdx_x;

        for(I i = tid; i < nblocks; i += BLOCKSIZE)
        {
            // Copy data to reduction buffers
            shared_row[tid] = row_block_red[i];
            shared_val[tid] = val_block_red[i + nblocks * col];

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
                    dense_C[row + ldc * col] = dense_C[row + ldc * col] + shared_val[tid];
                }
                else
                {
                    dense_C[col + ldc * row] = dense_C[col + ldc * row] + shared_val[tid];
                }
            }

            __syncthreads();
        }
    }

    template <unsigned int BLOCKSIZE, unsigned int NNZ_PER_BLOCK, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmmnn_nnz_split_compute_row_limits(J m,
                                              I nblocks,
                                              I nnz,
                                              const I* __restrict__ csr_row_ptr,
                                              J* __restrict__ row_limits,
                                              rocsparse_index_base idx_base)
    {
        const I gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= nblocks)
        {
            return;
        }

        const I s0 = NNZ_PER_BLOCK * gid;

        J left  = 0;
        J right = m;
        J mid   = (left + right) / 2;
        while((csr_row_ptr[left] - idx_base) < s0 && left < mid && right > mid)
        {
            if((csr_row_ptr[mid] - idx_base) <= s0)
            {
                left = mid;
            }
            else
            {
                right = mid;
            }
            mid = (left + right) / 2;
        }

        row_limits[gid] = left;

        if(gid == nblocks - 1)
        {
            row_limits[gid + 1] = m;
        }
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              unsigned int LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnt_nnz_split_main_device(bool conj_A,
                                                            bool conj_B,
                                                            J    ncol,
                                                            J    M,
                                                            J    N,
                                                            J    K,
                                                            I    nnz,
                                                            T    alpha,
                                                            const J* __restrict__ row_limits,
                                                            const I* __restrict__ csr_row_ptr,
                                                            const J* __restrict__ csr_col_ind,
                                                            const A* __restrict__ csr_val,
                                                            const B* __restrict__ dense_B,
                                                            int64_t ldb,
                                                            C* __restrict__ dense_C,
                                                            int64_t              ldc,
                                                            rocsparse_order      order_C,
                                                            rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int lid = tid & (WF_SIZE - 1);

        J row = 0;
        J col = 0;
        T val = static_cast<T>(0);

        if((BLOCKSIZE * bid + tid) < nnz)
        {
            row = dichotomic_search<I, J>(row_limits[bid],
                                          row_limits[bid + 1],
                                          BLOCKSIZE * bid + tid + idx_base,
                                          nnz + idx_base,
                                          csr_row_ptr);
            col = rocsparse::nontemporal_load(&csr_col_ind[BLOCKSIZE * bid + tid]) - idx_base;
            val = conj_val(rocsparse::nontemporal_load(&csr_val[BLOCKSIZE * bid + tid]), conj_A);
        }

        for(J l = 0; l < ncol; l += WF_SIZE * LOOPS)
        {
            const J colB = l + lid;

            T sum[LOOPS]{};

            J current_row = rocsparse::shfl(row, 0, WF_SIZE);

            for(unsigned int i = 0; i < WF_SIZE; ++i)
            {
                const T v = rocsparse::shfl(val, i, WF_SIZE);
                const J c = rocsparse::shfl(col, i, WF_SIZE);
                const J r = rocsparse::shfl(row, i, WF_SIZE);

                if(r != current_row)
                {
                    if(order_C == rocsparse_order_column)
                    {
                        for(unsigned int p = 0; p < LOOPS; p++)
                        {
                            rocsparse::atomic_add(
                                &dense_C[(colB + p * WF_SIZE) * ldc + current_row],
                                static_cast<C>(alpha * sum[p]));
                        }
                    }
                    else
                    {
                        for(unsigned int p = 0; p < LOOPS; p++)
                        {
                            rocsparse::atomic_add(&dense_C[current_row * ldc + colB + p * WF_SIZE],
                                                  static_cast<C>(alpha * sum[p]));
                        }
                    }

                    for(unsigned int p = 0; p < LOOPS; p++)
                    {
                        sum[p] = static_cast<T>(0);
                    }

                    current_row = r;
                }

                for(unsigned int p = 0; p < LOOPS; p++)
                {
                    sum[p] = rocsparse::fma<T>(
                        v, conj_val(dense_B[c * ldb + colB + p * WF_SIZE], conj_B), sum[p]);
                }
            }

            if(order_C == rocsparse_order_column)
            {
                for(unsigned int p = 0; p < LOOPS; p++)
                {
                    rocsparse::atomic_add(&dense_C[(colB + p * WF_SIZE) * ldc + current_row],
                                          static_cast<C>(alpha * sum[p]));
                }
            }
            else
            {
                for(unsigned int p = 0; p < LOOPS; p++)
                {
                    rocsparse::atomic_add(&dense_C[current_row * ldc + colB + p * WF_SIZE],
                                          static_cast<C>(alpha * sum[p]));
                }
            }
        }
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnt_nnz_split_remainder_device(bool conj_A,
                                                                 bool conj_B,
                                                                 J    ncol_offset,
                                                                 J    M,
                                                                 J    N,
                                                                 J    K,
                                                                 I    nnz,
                                                                 T    alpha,
                                                                 const J* __restrict__ row_limits,
                                                                 const I* __restrict__ csr_row_ptr,
                                                                 const J* __restrict__ csr_col_ind,
                                                                 const A* __restrict__ csr_val,
                                                                 const B* __restrict__ dense_B,
                                                                 int64_t ldb,
                                                                 C* __restrict__ dense_C,
                                                                 int64_t              ldc,
                                                                 rocsparse_order      order_C,
                                                                 rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        __shared__ J shared_row[(BLOCKSIZE / WF_SIZE) * WF_SIZE];
        __shared__ T shared_val[(BLOCKSIZE / WF_SIZE) * WF_SIZE];

        J row = 0;
        J col = 0;
        T val = static_cast<T>(0);

        if((BLOCKSIZE * bid + tid) < nnz)
        {
            // Compute COO row index on the fly
            row = dichotomic_search<I, J>(row_limits[bid],
                                          row_limits[bid + 1],
                                          BLOCKSIZE * bid + tid + idx_base,
                                          nnz + idx_base,
                                          csr_row_ptr);
            col = rocsparse::nontemporal_load(&csr_col_ind[BLOCKSIZE * bid + tid]) - idx_base;
            val = conj_val(rocsparse::nontemporal_load(&csr_val[BLOCKSIZE * bid + tid]), conj_A);
        }

        for(J l = ncol_offset; l < N; l += WF_SIZE)
        {
            const J colB = l + lid;

            T sum         = static_cast<T>(0);
            J current_row = rocsparse::shfl(row, 0, WF_SIZE);

            for(unsigned int i = 0; i < WF_SIZE; ++i)
            {
                const T v = rocsparse::shfl(val, i, WF_SIZE);
                const J c = rocsparse::shfl(col, i, WF_SIZE);
                const J r = rocsparse::shfl(row, i, WF_SIZE);

                if(r != current_row)
                {
                    if(colB < N)
                    {
                        if(order_C == rocsparse_order_column)
                        {
                            rocsparse::atomic_add(&dense_C[colB * ldc + current_row],
                                                  static_cast<C>(alpha * sum));
                        }
                        else
                        {
                            rocsparse::atomic_add(&dense_C[current_row * ldc + colB],
                                                  static_cast<C>(alpha * sum));
                        }
                    }

                    sum = static_cast<T>(0);

                    current_row = r;
                }

                if(colB < N)
                {
                    sum = rocsparse::fma<T>(v, conj_val(dense_B[c * ldb + colB], conj_B), sum);
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
            for(J j = 1; j < (BLOCKSIZE / WF_SIZE); j <<= 1)
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
                    if((l + swid) < N)
                    {
                        if(order_C == rocsparse_order_column)
                        {
                            rocsparse::atomic_add(&dense_C[(l + swid) * ldc + current_row],
                                                  static_cast<C>(alpha * sum));
                        }
                        else
                        {
                            rocsparse::atomic_add(&dense_C[current_row * ldc + (l + swid)],
                                                  static_cast<C>(alpha * sum));
                        }
                    }
                }
            }

            if(slid == ((BLOCKSIZE / WF_SIZE) - 1))
            {
                if(current_row >= 0)
                {
                    if((l + swid) < N)
                    {
                        if(order_C == rocsparse_order_column)
                        {
                            rocsparse::atomic_add(&dense_C[(l + swid) * ldc + current_row],
                                                  static_cast<C>(alpha * sum));
                        }
                        else
                        {
                            rocsparse::atomic_add(&dense_C[current_row * ldc + (l + swid)],
                                                  static_cast<C>(alpha * sum));
                        }
                    }
                }
            }
        }
    }
}
