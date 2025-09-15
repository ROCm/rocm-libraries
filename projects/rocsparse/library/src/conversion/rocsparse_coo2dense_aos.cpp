/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_coo2dense_aos.hpp"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "coo2dense_device.h"

template <typename I, typename T>
rocsparse_status rocsparse::coo2dense_aos_template(rocsparse_handle          handle,
                                                   I                         m,
                                                   I                         n,
                                                   int64_t                   nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  coo_val,
                                                   const I*                  coo_ind,
                                                   T*                        A,
                                                   int64_t                   lda,
                                                   rocsparse_order           order)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Check matrix descriptor
    ROCSPARSE_CHECKARG_POINTER(4, descr);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcoo2dense_aos"),
                         m,
                         n,
                         nnz,
                         descr,
                         (const void*&)coo_val,
                         (const void*&)coo_ind,
                         (const void*&)A,
                         lda);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        4, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    // Check order
    ROCSPARSE_CHECKARG_ENUM(9, order);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(
        8, lda, lda < (order == rocsparse_order_column ? m : n), rocsparse_status_invalid_size);

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(7, A);

    ROCSPARSE_CHECKARG_POINTER(5, coo_val);
    ROCSPARSE_CHECKARG_POINTER(6, coo_ind);

    // Stream
    hipStream_t stream = handle->stream;

    // Note: hipMemset2DAsync does not seem to be supported by hipgraph but should be in the future.
    // Once hipgraph supports hipMemset2DAsync then the kernel memset2d_kernel can be replaced
    // with the hipMemset2DAsync call below.
    //
    // I mn = order == rocsparse_order_column ? m : n;
    // I nm = order == rocsparse_order_column ? n : m;
    // RETURN_IF_HIP_ERROR(hipMemset2DAsync(A, sizeof(T) * lda, 0, sizeof(T) * mn, nm, stream));

    // Set memory to zero.
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset_2d(handle,
                                                   static_cast<int64_t>(m),
                                                   static_cast<int64_t>(n),
                                                   lda,
                                                   static_cast<T>(0),
                                                   A,
                                                   order));

#define COO2DENSE_DIM 512
    const int64_t num_blocks_x = rocsparse::min(
        ((nnz - 1) / COO2DENSE_DIM + 1), static_cast<int64_t>(handle->properties.maxGridSize[0]));
    dim3 blocks(num_blocks_x);
    dim3 threads(COO2DENSE_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::coo2dense_aos_kernel<COO2DENSE_DIM>),
                                       blocks,
                                       threads,
                                       0,
                                       stream,
                                       m,
                                       n,
                                       nnz,
                                       lda,
                                       descr->base,
                                       coo_val,
                                       coo_ind,
                                       A,
                                       order);
#undef COO2DENSE_DIM

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                              \
    template rocsparse_status rocsparse::coo2dense_aos_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                      \
        ITYPE                     m,                                           \
        ITYPE                     n,                                           \
        int64_t                   nnz,                                         \
        const rocsparse_mat_descr descr,                                       \
        const TTYPE*              coo_val,                                     \
        const ITYPE*              coo_ind,                                     \
        TTYPE*                    A,                                           \
        int64_t                   lda,                                         \
        rocsparse_order           order)

INSTANTIATE(int32_t, _Float16);
INSTANTIATE(int32_t, rocsparse_bfloat16);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, _Float16);
INSTANTIATE(int64_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE
