/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_sddmm.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <rocsparse_int BLOCKSIZE,
              rocsparse_int NTHREADS_PER_DOTPRODUCT,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void sddmm_ell_device(rocsparse_operation transA,
                                               rocsparse_operation transB,
                                               rocsparse_order     orderA,
                                               rocsparse_order     orderB,
                                               J                   M,
                                               J                   N,
                                               J                   K,
                                               I                   nnz,
                                               T                   alpha,
                                               const A* __restrict__ dense_A,
                                               int64_t lda,
                                               const B* __restrict__ dense_B,
                                               int64_t ldb,
                                               T       beta,
                                               C* __restrict__ val,
                                               const J* __restrict__ ind,
                                               rocsparse_index_base base)
    {
        //
        // Each group treats one row.
        //
        static constexpr rocsparse_int NUM_COEFF         = (BLOCKSIZE / NTHREADS_PER_DOTPRODUCT);
        const I                        local_coeff_index = hipThreadIdx_x / NTHREADS_PER_DOTPRODUCT;
        const I       local_thread_index                 = hipThreadIdx_x % NTHREADS_PER_DOTPRODUCT;
        const int64_t incx                               = (orderA == rocsparse_order_column)
                                                               ? ((transA == rocsparse_operation_none) ? lda : 1)
                                                               : ((transA == rocsparse_operation_none) ? 1 : lda);

        const int64_t incy = (orderB == rocsparse_order_column)
                                 ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                                 : ((transB == rocsparse_operation_none) ? ldb : 1);

        const I innz = hipBlockIdx_x * NUM_COEFF + local_coeff_index;
        if(innz >= nnz)
        {
            return;
        }

        const J i = innz % M;
        const J j = ind[innz] - base;
        if(j < 0)
        {
            return;
        }
        const A* x
            = (orderA == rocsparse_order_column)
                  ? ((transA == rocsparse_operation_none) ? (dense_A + i) : (dense_A + lda * i))
                  : ((transA == rocsparse_operation_none) ? (dense_A + lda * i) : (dense_A + i));

        const B* y
            = (orderB == rocsparse_order_column)
                  ? ((transB == rocsparse_operation_none) ? (dense_B + ldb * j) : (dense_B + j))
                  : ((transB == rocsparse_operation_none) ? (dense_B + j) : (dense_B + ldb * j));

        T sum = static_cast<T>(0);
        for(J k = local_thread_index; k < K; k += NTHREADS_PER_DOTPRODUCT)
        {
            sum = rocsparse::fma<T>(x[k * incx], y[k * incy], sum);
        }

        sum = rocsparse::wfreduce_sum<NTHREADS_PER_DOTPRODUCT>(sum);

        if(local_thread_index == NTHREADS_PER_DOTPRODUCT - 1)
        {
            val[innz] = beta * val[innz] + alpha * sum;
        }
    }

    template <rocsparse_int BLOCKSIZE,
              rocsparse_int NTHREADS_PER_DOTPRODUCT,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void sddmm_ell_kernel(rocsparse_operation transA,
                          rocsparse_operation transB,
                          rocsparse_order     orderA,
                          rocsparse_order     orderB,
                          J                   M,
                          J                   N,
                          J                   K,
                          I                   nnz,
                          int64_t             batch_count,
                          ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                          const A* __restrict__ dense_A,
                          int64_t lda,
                          int64_t batch_stride_A,
                          const B* __restrict__ dense_B,
                          int64_t ldb,
                          int64_t batch_stride_B,
                          ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                          C* __restrict__ val,
                          int64_t values_batch_stride_C,
                          const J* __restrict__ ind,
                          int64_t              indices_batch_stride_C,
                          rocsparse_index_base base,
                          bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        // ELL has a single per-batch stride (configured by the user via
        // rocsparse_ell_set_strided_batch) which applies to both ELL
        // buffers (column indices and values). The caller is
        // therefore required to lay out the two buffers with that same
        // stride, and to broadcast A or B across batches the caller passes
        // batch_stride_A == 0 or batch_stride_B == 0.
        for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
        {
            rocsparse::sddmm_ell_device<BLOCKSIZE, NTHREADS_PER_DOTPRODUCT>(
                transA,
                transB,
                orderA,
                orderB,
                M,
                N,
                K,
                nnz,
                alpha,
                load_pointer(dense_A, batch, batch_stride_A),
                lda,
                load_pointer(dense_B, batch, batch_stride_B),
                ldb,
                beta,
                load_pointer(val, batch, values_batch_stride_C),
                load_pointer(ind, batch, indices_batch_stride_C),
                base);
        }
    }

    template <rocsparse_int NUM_ELL_COLUMNS_PER_BLOCK,
              rocsparse_int WF_SIZE,
              typename T,
              typename I,
              typename C>
    ROCSPARSE_KERNEL(WF_SIZE* NUM_ELL_COLUMNS_PER_BLOCK)
    void sddmm_ell_sample_kernel(I m,
                                 I n,
                                 const C* __restrict__ dense_val,
                                 int64_t ld,
                                 I       ell_width,
                                 C* __restrict__ ell_val,
                                 const I* __restrict__ ell_col_ind,
                                 rocsparse_index_base ell_base)
    {
        const auto wavefront_index  = hipThreadIdx_x / WF_SIZE;
        const auto lane_index       = hipThreadIdx_x % WF_SIZE;
        const auto ell_column_index = NUM_ELL_COLUMNS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

        if(ell_column_index < ell_width)
        {
            //
            // One wavefront executes one ell column.
            //
            for(I row_index = lane_index; row_index < m; row_index += WF_SIZE)
            {
                const auto ell_idx      = ELL_IND(row_index, ell_column_index, m, ell_width);
                const auto column_index = ell_col_ind[ell_idx] - ell_base;

                if(column_index >= 0 && column_index < n)
                {
                    ell_val[ell_idx] = dense_val[column_index * ld + row_index];
                }
            }
        }
    }
}
