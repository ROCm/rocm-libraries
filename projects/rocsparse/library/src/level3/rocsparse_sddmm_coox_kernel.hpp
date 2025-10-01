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
              bool          AOS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_KERNEL_W(BLOCKSIZE, 1)
    void sddmm_coox_kernel(rocsparse_operation transA,
                           rocsparse_operation transB,
                           rocsparse_order     orderA,
                           rocsparse_order     orderB,
                           J                   M,
                           J                   N,
                           J                   K,
                           I                   nnz,
                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                           const A* __restrict__ dense_A,
                           int64_t lda,
                           const B* __restrict__ dense_B,
                           int64_t ldb,
                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                           C* __restrict__ coo_val,
                           const I* __restrict__ coo_row_ind,
                           const I* __restrict__ coo_col_ind,
                           rocsparse_index_base coo_base,
                           bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }
        //
        // Each group treats one row / column
        //
        static constexpr rocsparse_int NUM_COEFF         = (BLOCKSIZE / NTHREADS_PER_DOTPRODUCT);
        const I                        local_coeff_index = hipThreadIdx_x / NTHREADS_PER_DOTPRODUCT;
        const I local_thread_index                       = hipThreadIdx_x % NTHREADS_PER_DOTPRODUCT;
        const J incx                                     = (orderA == rocsparse_order_column)
                                                               ? ((transA == rocsparse_operation_none) ? lda : 1)
                                                               : ((transA == rocsparse_operation_none) ? 1 : lda);

        const J incy = (orderB == rocsparse_order_column)
                           ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                           : ((transB == rocsparse_operation_none) ? ldb : 1);

        const I innz = hipBlockIdx_x * NUM_COEFF + local_coeff_index;
        if(innz >= nnz)
        {
            return;
        }

        const I i = coo_row_ind[innz * ((AOS) ? 2 : 1)] - coo_base;
        const I j = coo_col_ind[innz * ((AOS) ? 2 : 1)] - coo_base;

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
            coo_val[innz] = rocsparse::fma<T>(beta, coo_val[innz], alpha * sum);
        }
    }

    template <rocsparse_int BLOCKSIZE, bool AOS, typename T, typename I, typename J, typename C>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void sddmm_coox_sample_kernel(J M,
                                  J N,
                                  I nnz,
                                  const C* __restrict__ dense_C,
                                  J lda,
                                  C* __restrict__ coo_val,
                                  const I* __restrict__ coo_row,
                                  const I* __restrict__ coo_col,
                                  rocsparse_index_base coo_base)
    {
        const auto NUM_THREADS = hipGridDim_x * BLOCKSIZE;

        const auto gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        for(auto idx = gid; idx < nnz; idx += NUM_THREADS)
        {
            const I row = coo_row[idx * ((AOS) ? 2 : 1)] - coo_base;
            const I col = coo_col[idx * ((AOS) ? 2 : 1)] - coo_base;

            coo_val[idx] = dense_C[col * lda + row];
        }
    }
}
