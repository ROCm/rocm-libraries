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

#include "rocsparse.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_sddmm.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t            BLOCKSIZE,
              uint32_t            WFSIZE,
              uint32_t            NTHREADS_PER_DOTPRODUCT,
              rocsparse_direction DIRECTION,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void sddmm_csx_kernel_wavefront_per_rowcol(rocsparse_operation transA,
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
                                               C* __restrict__ csx_val,
                                               const I* __restrict__ csx_ptr,
                                               const J* __restrict__ csx_ind,
                                               rocsparse_index_base csx_base,
                                               bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        // Each wavefront treats one row/column.
        static constexpr uint32_t NUM_SEQS = (WFSIZE / NTHREADS_PER_DOTPRODUCT);
        const uint32_t            bid      = hipBlockIdx_x;
        const uint32_t            tid      = hipThreadIdx_x;
        const uint32_t            wid      = tid / WFSIZE;
        const uint32_t            lid      = tid % WFSIZE;
        const uint32_t            swid     = lid / NTHREADS_PER_DOTPRODUCT;
        const uint32_t            slid     = lid % NTHREADS_PER_DOTPRODUCT;

        const uint32_t rowcol = (BLOCKSIZE / WFSIZE) * bid + wid;

        static constexpr bool ROW_ORIENTED = (DIRECTION == rocsparse_direction_row);

        if(rowcol >= ((ROW_ORIENTED) ? M : N))
        {
            return;
        }

        const int64_t incx = (orderA == rocsparse_order_column)
                                 ? ((transA == rocsparse_operation_none) ? lda : 1)
                                 : ((transA == rocsparse_operation_none) ? 1 : lda);

        const int64_t incy = (orderB == rocsparse_order_column)
                                 ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                                 : ((transB == rocsparse_operation_none) ? ldb : 1);

        const int64_t xinc = (ROW_ORIENTED) ? incx : incy;
        const int64_t yinc = (ROW_ORIENTED) ? incy : incx;

        const I start = csx_ptr[rowcol] - csx_base;
        const I end   = csx_ptr[rowcol + 1] - csx_base;

        if(ROW_ORIENTED)
        {
            const A* x = ((orderA == rocsparse_order_column)
                              ? ((transA == rocsparse_operation_none) ? (dense_A + rowcol)
                                                                      : (dense_A + lda * rowcol))
                              : ((transA == rocsparse_operation_none) ? (dense_A + lda * rowcol)
                                                                      : (dense_A + rowcol)));

            for(I at = start + swid; at < end; at += NUM_SEQS)
            {
                const I  ind = csx_ind[at] - csx_base;
                const B* y   = ((orderB == rocsparse_order_column)
                                    ? ((transB == rocsparse_operation_none) ? (dense_B + ldb * ind)
                                                                            : (dense_B + ind))
                                    : ((transB == rocsparse_operation_none) ? (dense_B + ind)
                                                                            : (dense_B + ldb * ind)));

                T sum = static_cast<T>(0);
                for(J k = slid; k < K; k += NTHREADS_PER_DOTPRODUCT)
                {
                    sum = rocsparse::fma<T>(x[k * xinc], y[k * yinc], sum);
                }

                sum = rocsparse::wfreduce_sum<NTHREADS_PER_DOTPRODUCT>(sum);

                if(slid == NTHREADS_PER_DOTPRODUCT - 1)
                {
                    csx_val[at] = rocsparse::fma<T>(beta, csx_val[at], alpha * sum);
                }
            }
        }
        else
        {
            const B* x = ((orderB == rocsparse_order_column)
                              ? ((transB == rocsparse_operation_none) ? (dense_B + ldb * rowcol)
                                                                      : (dense_B + rowcol))
                              : ((transB == rocsparse_operation_none) ? (dense_B + rowcol)
                                                                      : (dense_B + ldb * rowcol)));

            for(I at = start + swid; at < end; at += NUM_SEQS)
            {
                const I  ind = csx_ind[at] - csx_base;
                const A* y   = ((orderA == rocsparse_order_column)
                                    ? ((transA == rocsparse_operation_none) ? (dense_A + ind)
                                                                            : (dense_A + lda * ind))
                                    : ((transA == rocsparse_operation_none) ? (dense_A + lda * ind)
                                                                            : (dense_A + ind)));

                T sum = static_cast<T>(0);
                for(J k = slid; k < K; k += NTHREADS_PER_DOTPRODUCT)
                {
                    sum = rocsparse::fma<T>(x[k * xinc], y[k * yinc], sum);
                }

                sum = rocsparse::wfreduce_sum<NTHREADS_PER_DOTPRODUCT>(sum);

                if(slid == NTHREADS_PER_DOTPRODUCT - 1)
                {
                    csx_val[at] = rocsparse::fma<T>(beta, csx_val[at], alpha * sum);
                }
            }
        }
    }

    template <rocsparse_int       BLOCKSIZE,
              rocsparse_int       NTHREADS_PER_GROUP,
              rocsparse_direction DIRECTION,
              typename T,
              typename I,
              typename J,
              typename C>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void sddmm_csx_sample_kernel(J M,
                                 J N,
                                 I nnz,
                                 const C* __restrict__ dense_C,
                                 J lda,
                                 C* __restrict__ csx_val,
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

        const I start = csx_ptr[gwid] - csx_base;
        const I end   = csx_ptr[gwid + 1] - csx_base;

        for(I at = start + lid; at < end; at += NTHREADS_PER_GROUP)
        {
            const I ind = csx_ind[at] - csx_base;

            const J row = (row_oriented) ? gwid : ind;
            const J col = (row_oriented) ? ind : gwid;

            csx_val[at] = dense_C[col * lda + row];
        }
    }
}
