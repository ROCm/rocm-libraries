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

#include "common.h"
#include "control.h"
#include "handle.h"
#include "rocsparse.h"
#include "rocsparse_sddmm.hpp"
#include "utility.h"

namespace rocsparse
{
    template <rocsparse_int       BLOCKSIZE,
              rocsparse_int       NTHREADS_PER_DOTPRODUCT,
              rocsparse_direction DIRECTION,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL_W(BLOCKSIZE, 1)
    void sddmm_csx_kernel(rocsparse_operation transA,
                          rocsparse_operation transB,
                          rocsparse_order     orderA,
                          rocsparse_order     orderB,
                          J                   M,
                          J                   N,
                          J                   K,
                          I                   nnz,
                          ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                          const T* __restrict__ A,
                          int64_t lda,
                          const T* __restrict__ B,
                          int64_t ldb,
                          ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                          T* __restrict__ csx_val,
                          const I* __restrict__ csx_ptr,
                          const J* __restrict__ csx_ind,
                          rocsparse_index_base csx_base,
                          T* __restrict__ workspace,
                          bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        //
        // Each group treats one row/column.
        //
        static constexpr rocsparse_int NUM_SEQS        = (BLOCKSIZE / NTHREADS_PER_DOTPRODUCT);
        const I                        local_seq_index = hipThreadIdx_x / NTHREADS_PER_DOTPRODUCT;
        const I               local_thread_index       = hipThreadIdx_x % NTHREADS_PER_DOTPRODUCT;
        const I               tid                      = hipBlockIdx_x * NUM_SEQS + local_seq_index;
        static constexpr bool row_oriented             = (DIRECTION == rocsparse_direction_row);
#define BOUND ((row_oriented) ? M : N)
        if(tid >= BOUND)
        {
            return;
        }

        const int64_t incx = (orderA == rocsparse_order_column)
                                 ? ((transA == rocsparse_operation_none) ? lda : 1)
                                 : ((transA == rocsparse_operation_none) ? 1 : lda);

        const int64_t incy = (orderB == rocsparse_order_column)
                                 ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                                 : ((transB == rocsparse_operation_none) ? ldb : 1);

        const int64_t xinc = (row_oriented) ? incx : incy;
        const int64_t yinc = (row_oriented) ? incy : incx;

        __shared__ T s[NUM_SEQS][NTHREADS_PER_DOTPRODUCT];

        const T* x
            = (row_oriented)
                  ? ((orderA == rocsparse_order_column)
                         ? ((transA == rocsparse_operation_none) ? (A + tid) : (A + lda * tid))
                         : ((transA == rocsparse_operation_none) ? (A + lda * tid) : (A + tid)))
                  : ((orderB == rocsparse_order_column)
                         ? ((transB == rocsparse_operation_none) ? (B + ldb * tid) : (B + tid))
                         : ((transB == rocsparse_operation_none) ? (B + tid) : (B + ldb * tid)));

        for(I at = csx_ptr[tid] - csx_base; at < csx_ptr[tid + 1] - csx_base; ++at)
        {
            I        ind = csx_ind[at] - csx_base;
            const T* y
                = (row_oriented)
                      ? ((orderB == rocsparse_order_column)
                             ? ((transB == rocsparse_operation_none) ? (B + ldb * ind) : (B + ind))
                             : ((transB == rocsparse_operation_none) ? (B + ind) : (B + ldb * ind)))
                      : ((orderA == rocsparse_order_column)
                             ? ((transA == rocsparse_operation_none) ? (A + ind) : (A + lda * ind))
                             : ((transA == rocsparse_operation_none) ? (A + lda * ind)
                                                                     : (A + ind)));

            T sum = static_cast<T>(0);
            for(J k = local_thread_index; k < K; k += NTHREADS_PER_DOTPRODUCT)
            {
                sum += x[k * xinc] * y[k * yinc];
            }
            s[local_seq_index][local_thread_index] = sum;
            __syncthreads();

#pragma unroll
            for(int ipow2_ = 2; ipow2_ <= NTHREADS_PER_DOTPRODUCT; ipow2_ *= 2)
            {
                if(local_thread_index < NTHREADS_PER_DOTPRODUCT / ipow2_)
                {
                    s[local_seq_index][local_thread_index]
                        += s[local_seq_index]
                            [local_thread_index + NTHREADS_PER_DOTPRODUCT / ipow2_];
                }
                __syncthreads();
            }

            if(local_thread_index == 0)
            {
                csx_val[at] = csx_val[at] * beta + alpha * s[local_seq_index][0];
            }
        }
    }

    template <rocsparse_int       BLOCKSIZE,
              rocsparse_int       NTHREADS_PER_GROUP,
              rocsparse_direction DIRECTION,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void sddmm_csx_sample_kernel(J M,
                                 J N,
                                 I nnz,
                                 const T* __restrict__ A,
                                 J lda,
                                 T* __restrict__ csx_val,
                                 const I* __restrict__ csx_ptr,
                                 const J* __restrict__ csx_ind,
                                 rocsparse_index_base csx_base)
    {
        static constexpr auto GROUPS_PER_BLOCK = BLOCKSIZE / NTHREADS_PER_GROUP;

        const auto lid  = hipThreadIdx_x & (NTHREADS_PER_GROUP - 1);
        const auto wid  = hipThreadIdx_x / NTHREADS_PER_GROUP;
        const auto gwid = wid + hipBlockIdx_x * GROUPS_PER_BLOCK;

        static constexpr bool row_oriented = (DIRECTION == rocsparse_direction_row);

#define BOUND ((row_oriented) ? M : N)
        if(gwid >= BOUND)
        {
            return;
        }

        for(I at = csx_ptr[gwid] - csx_base + lid; at < csx_ptr[gwid + 1] - csx_base;
            at += NTHREADS_PER_GROUP)
        {
            const I ind = csx_ind[at] - csx_base;

            const J row = (row_oriented) ? gwid : ind;
            const J col = (row_oriented) ? ind : gwid;

            csx_val[at] = A[col * lda + row];
        }
    }
}
