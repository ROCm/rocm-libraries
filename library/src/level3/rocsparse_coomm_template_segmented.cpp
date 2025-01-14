/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"
#include "control.h"
#include "utility.h"

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
                const I c = __shfl(col, i, WF_SIZE);

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

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t LOOPS,
              bool     TRANSB,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void coommnn_segmented_main_kernel(bool    conj_A,
                                       bool    conj_B,
                                       I       M,
                                       I       N,
                                       I       K,
                                       int64_t nnz,
                                       int64_t batch_stride_A,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
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
                                       rocsparse_index_base idx_base,
                                       bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        if(alpha != static_cast<T>(0))
        {
            rocsparse::coommnn_segmented_main_device<BLOCKSIZE, WF_SIZE, LOOPS, TRANSB>(
                conj_A,
                conj_B,
                M,
                N,
                K,
                nnz,
                batch_stride_A,
                alpha,
                row_block_red,
                val_block_red,
                coo_row_ind,
                coo_col_ind,
                coo_val,
                dense_B,
                ldb,
                batch_stride_B,
                dense_C,
                ldc,
                batch_stride_C,
                order_C,
                idx_base);
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
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void coommnn_segmented_remainder_kernel(bool    conj_A,
                                            bool    conj_B,
                                            I       colB_offset,
                                            I       M,
                                            I       N,
                                            I       K,
                                            int64_t nnz,
                                            int64_t batch_stride_A,
                                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
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
                                            rocsparse_index_base idx_base,
                                            bool                 is_host_mode)
    {

        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        if(alpha != static_cast<T>(0))
        {
            rocsparse::coommnn_segmented_remainder_device<BLOCKSIZE, WF_SIZE, LOOPS, TRANSB>(
                conj_A,
                conj_B,
                colB_offset,
                M,
                N,
                K,
                nnz,
                batch_stride_A,
                alpha,
                row_block_red,
                val_block_red,
                coo_row_ind,
                coo_col_ind,
                coo_val,
                dense_B,
                ldb,
                batch_stride_B,
                dense_C,
                ldc,
                batch_stride_C,
                order_C,
                idx_base);
        }
    }

    template <typename T, typename I, typename A>
    rocsparse_status coomm_buffer_size_template_segmented(rocsparse_handle          handle,
                                                          rocsparse_operation       trans_A,
                                                          I                         m,
                                                          I                         n,
                                                          I                         k,
                                                          int64_t                   nnz,
                                                          I                         batch_count,
                                                          const rocsparse_mat_descr descr,
                                                          const A*                  coo_val,
                                                          const I*                  coo_row_ind,
                                                          const I*                  coo_col_ind,
                                                          size_t*                   buffer_size)
    {
#define LOOPS 4
#define COOMMN_DIM 256
        const I nblocks = (nnz - 1) / (COOMMN_DIM * LOOPS) + 1;
        *buffer_size
            = size_t(256) + ((sizeof(I) * nblocks * batch_count - 1) / COOMMN_DIM + 1) * COOMMN_DIM
              + ((sizeof(T) * nblocks * n * batch_count - 1) / COOMMN_DIM + 1) * COOMMN_DIM;
#undef COOMMN_DIM
#undef LOOPS

        return rocsparse_status_success;
    }

#define LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                  \
        (rocsparse::coommnn_segmented_main_kernel<COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB>), \
        dim3(nblocks, (main - 1) / WF_SIZE + 1, batch_count_C),                          \
        dim3(COOMMNN_DIM),                                                               \
        0,                                                                               \
        stream,                                                                          \
        conj_A,                                                                          \
        conj_B,                                                                          \
        m,                                                                               \
        n,                                                                               \
        k,                                                                               \
        nnz,                                                                             \
        batch_stride_A,                                                                  \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                    \
        row_block_red,                                                                   \
        val_block_red,                                                                   \
        coo_row_ind,                                                                     \
        coo_col_ind,                                                                     \
        coo_val,                                                                         \
        dense_B,                                                                         \
        ldb,                                                                             \
        batch_stride_B,                                                                  \
        dense_C,                                                                         \
        ldc,                                                                             \
        batch_stride_C,                                                                  \
        order_C,                                                                         \
        descr->base,                                                                     \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                       \
        (rocsparse::coommnn_segmented_remainder_kernel<COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB>), \
        dim3(nblocks, 1, batch_count_C),                                                      \
        dim3(COOMMNN_DIM),                                                                    \
        0,                                                                                    \
        stream,                                                                               \
        conj_A,                                                                               \
        conj_B,                                                                               \
        main,                                                                                 \
        m,                                                                                    \
        n,                                                                                    \
        k,                                                                                    \
        nnz,                                                                                  \
        batch_stride_A,                                                                       \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                         \
        row_block_red,                                                                        \
        val_block_red,                                                                        \
        coo_row_ind,                                                                          \
        coo_col_ind,                                                                          \
        coo_val,                                                                              \
        dense_B,                                                                              \
        ldb,                                                                                  \
        batch_stride_B,                                                                       \
        dense_C,                                                                              \
        ldc,                                                                                  \
        batch_stride_C,                                                                       \
        order_C,                                                                              \
        descr->base,                                                                          \
        handle->pointer_mode == rocsparse_pointer_mode_host)

    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status coomm_template_segmented(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              I                         m,
                                              I                         n,
                                              I                         k,
                                              int64_t                   nnz,
                                              I                         batch_count_A,
                                              int64_t                   batch_stride_A,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const A*                  coo_val,
                                              const I*                  coo_row_ind,
                                              const I*                  coo_col_ind,
                                              const B*                  dense_B,
                                              int64_t                   ldb,
                                              I                         batch_count_B,
                                              int64_t                   batch_stride_B,
                                              rocsparse_order           order_B,
                                              const T*                  beta_device_host,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              I                         batch_count_C,
                                              int64_t                   batch_stride_C,
                                              rocsparse_order           order_C,
                                              void*                     temp_buffer)
    {
        const bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose);
        const bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

        // Stream
        hipStream_t stream = handle->stream;

        // Run different coomm kernels
        if(trans_A == rocsparse_operation_none)
        {
#define LOOPS 4
#define COOMMN_DIM 256
            const I nblocks = (nnz - 1) / (COOMMN_DIM * LOOPS) + 1;

            // row and val block reduction buffer
            char* ptr = reinterpret_cast<char*>(temp_buffer);
            ptr += 256;
            I* row_block_red = reinterpret_cast<I*>(reinterpret_cast<void*>(ptr));
            ptr += ((sizeof(I) * nblocks * batch_count_C - 1) / COOMMN_DIM + 1) * COOMMN_DIM;
            T* val_block_red = reinterpret_cast<T*>(reinterpret_cast<void*>(ptr));
            // ptr += ((sizeof(T) * nblocks * n * batch_count_C - 1) / COOMMN_DIM + 1) * COOMMN_DIM;

            RETURN_IF_HIP_ERROR(hipMemsetAsync(
                row_block_red,
                0XFF,
                ((sizeof(I) * nblocks * batch_count_C - 1) / COOMMN_DIM + 1) * COOMMN_DIM,
                stream));

            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                I main      = 0;
                I remainder = 0;

                if(n >= 8)
                {
                    remainder = n % 8;
                    main      = n - remainder;
                    LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 8, LOOPS, false);
                }
                else if(n >= 4)
                {
                    remainder = n % 4;
                    main      = n - remainder;
                    LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 4, LOOPS, false);
                }
                else if(n >= 2)
                {
                    remainder = n % 2;
                    main      = n - remainder;
                    LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 2, LOOPS, false);
                }
                else if(n >= 1)
                {
                    remainder = n % 1;
                    main      = n - remainder;
                    LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 1, LOOPS, false);
                }
                else
                {
                    remainder = n;
                }

                if(remainder > 0)
                {
                    if(remainder <= 1)
                    {
                        LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 1, LOOPS, false);
                    }
                    else if(remainder <= 2)
                    {
                        LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 2, LOOPS, false);
                    }
                    else if(remainder <= 4)
                    {
                        LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 4, LOOPS, false);
                    }
                    else if(remainder <= 8)
                    {
                        LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 8, LOOPS, false);
                    }
                }
            }
            else if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                I main      = 0;
                I remainder = 0;

                if(n >= 8)
                {
                    remainder = n % 8;
                    main      = n - remainder;
                    LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 8, LOOPS, true);
                }
                else if(n >= 4)
                {
                    remainder = n % 4;
                    main      = n - remainder;
                    LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 4, LOOPS, true);
                }
                else if(n >= 2)
                {
                    remainder = n % 2;
                    main      = n - remainder;
                    LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 2, LOOPS, true);
                }
                else if(n >= 1)
                {
                    remainder = n % 1;
                    main      = n - remainder;
                    LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 1, LOOPS, true);
                }
                else
                {
                    remainder = n;
                }

                if(remainder > 0)
                {
                    if(remainder <= 1)
                    {
                        LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 1, LOOPS, true);
                    }
                    else if(remainder <= 2)
                    {
                        LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 2, LOOPS, true);
                    }
                    else if(remainder <= 4)
                    {
                        LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 4, LOOPS, true);
                    }
                    else if(remainder <= 8)
                    {
                        LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 8, LOOPS, true);
                    }
                }
            }
#undef COOMMN_DIM
#undef LOOPS

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::coommnn_general_block_reduce<1024>),
                                               dim3(n, 1, batch_count_C),
                                               1024,
                                               0,
                                               stream,
                                               n,
                                               nblocks,
                                               row_block_red,
                                               val_block_red,
                                               dense_C,
                                               ldc,
                                               batch_stride_C,
                                               order_C);
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        return rocsparse_status_success;
    }
}

#define INSTANTIATE_BUFFER_SIZE(TTYPE, ITYPE, ATYPE)                                  \
    template rocsparse_status rocsparse::coomm_buffer_size_template_segmented<TTYPE>( \
        rocsparse_handle          handle,                                             \
        rocsparse_operation       trans_A,                                            \
        ITYPE                     m,                                                  \
        ITYPE                     n,                                                  \
        ITYPE                     k,                                                  \
        int64_t                   nnz,                                                \
        ITYPE                     batch_count,                                        \
        const rocsparse_mat_descr descr,                                              \
        const ATYPE*              coo_val,                                            \
        const ITYPE*              coo_row_ind,                                        \
        const ITYPE*              coo_col_ind,                                        \
        size_t*                   buffer_size);

// Uniform precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, float);
INSTANTIATE_BUFFER_SIZE(double, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, double);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, _Float16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, _Float16);
INSTANTIATE_BUFFER_SIZE(int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int8_t);
#undef INSTANTIATE_BUFFER_SIZE

#define INSTANTIATE(TTYPE, ITYPE, ATYPE, BTYPE, CTYPE)                                               \
    template rocsparse_status rocsparse::coomm_template_segmented(rocsparse_handle    handle,        \
                                                                  rocsparse_operation trans_A,       \
                                                                  rocsparse_operation trans_B,       \
                                                                  ITYPE               m,             \
                                                                  ITYPE               n,             \
                                                                  ITYPE               k,             \
                                                                  int64_t             nnz,           \
                                                                  ITYPE               batch_count_A, \
                                                                  int64_t      batch_stride_A,       \
                                                                  const TTYPE* alpha_device_host,    \
                                                                  const rocsparse_mat_descr descr,   \
                                                                  const ATYPE*              coo_val, \
                                                                  const ITYPE*    coo_row_ind,       \
                                                                  const ITYPE*    coo_col_ind,       \
                                                                  const BTYPE*    dense_B,           \
                                                                  int64_t         ldb,               \
                                                                  ITYPE           batch_count_B,     \
                                                                  int64_t         batch_stride_B,    \
                                                                  rocsparse_order order_B,           \
                                                                  const TTYPE*    beta_device_host,  \
                                                                  CTYPE*          dense_C,           \
                                                                  int64_t         ldc,               \
                                                                  ITYPE           batch_count_C,     \
                                                                  int64_t         batch_stride_C,    \
                                                                  rocsparse_order order_C,           \
                                                                  void*           temp_buffer);

// Uniform precisions
INSTANTIATE(float, int32_t, float, float, float);
INSTANTIATE(float, int64_t, float, float, float);
INSTANTIATE(double, int32_t, double, double, double);
INSTANTIATE(double, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

// Mixed Precisions
INSTANTIATE(float, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, _Float16, _Float16, float);
INSTANTIATE(int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int8_t, int8_t, float);

#undef INSTANTIATE
