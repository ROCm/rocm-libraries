/*! \file */
/* ************************************************************************
* Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "internal/conversion/rocsparse_coo2dense.h"
#include "rocsparse_common.h"
#include "rocsparse_coo2dense.hpp"

#include "coo2dense_device.h"
#include "rocsparse_common.hpp"

template <typename I, typename T>
rocsparse_status rocsparse::coo2dense_template(rocsparse_handle          handle, //0
                                               I                         m, //1
                                               I                         n, //2
                                               int64_t                   nnz, //3
                                               const rocsparse_mat_descr descr, //4
                                               const T*                  coo_val, //5
                                               const I*                  coo_row_ind, //6
                                               const I*                  coo_col_ind, //7
                                               T*                        A, //8
                                               int64_t                   lda, //9
                                               rocsparse_order           order) //10
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcoo2dense"),
                         m,
                         n,
                         nnz,
                         descr,
                         (const void*&)coo_val,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)A,
                         lda);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG(
        9, lda, (lda < (order == rocsparse_order_column ? m : n)), rocsparse_status_invalid_size);

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_val);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz, coo_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, (m * n), A);

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
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset_2d(handle, m, n, lda, static_cast<T>(0), A, order));

    if(nnz > 0)
    {
#define COO2DENSE_DIM 512
        dim3 blocks((nnz - 1) / COO2DENSE_DIM + 1);
        dim3 threads(COO2DENSE_DIM);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::coo2dense_kernel<COO2DENSE_DIM>),
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
                                           coo_row_ind,
                                           coo_col_ind,
                                           A,
                                           order);
#undef COO2DENSE_DIM
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                          \
    template rocsparse_status rocsparse::coo2dense_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                  \
        ITYPE                     m,                                       \
        ITYPE                     n,                                       \
        int64_t                   nnz,                                     \
        const rocsparse_mat_descr descr,                                   \
        const TTYPE*              coo_val,                                 \
        const ITYPE*              coo_row_ind,                             \
        const ITYPE*              coo_col_ind,                             \
        TTYPE*                    A,                                       \
        int64_t                   lda,                                     \
        rocsparse_order           order);

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

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

#define CIMPL(NAME, T)                                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                    \
                                     rocsparse_int             m,                         \
                                     rocsparse_int             n,                         \
                                     rocsparse_int             nnz,                       \
                                     const rocsparse_mat_descr descr,                     \
                                     const T*                  coo_val,                   \
                                     const rocsparse_int*      coo_row_ind,               \
                                     const rocsparse_int*      coo_col_ind,               \
                                     T*                        A,                         \
                                     rocsparse_int             ld)                        \
    try                                                                                   \
    {                                                                                     \
        ROCSPARSE_ROUTINE_TRACE;                                                          \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coo2dense_template(handle,                   \
                                                                m,                        \
                                                                n,                        \
                                                                nnz,                      \
                                                                descr,                    \
                                                                coo_val,                  \
                                                                coo_row_ind,              \
                                                                coo_col_ind,              \
                                                                A,                        \
                                                                ld,                       \
                                                                rocsparse_order_column)); \
        return rocsparse_status_success;                                                  \
    }                                                                                     \
    catch(...)                                                                            \
    {                                                                                     \
        RETURN_ROCSPARSE_EXCEPTION();                                                     \
    }

CIMPL(rocsparse_scoo2dense, float);
CIMPL(rocsparse_dcoo2dense, double);
CIMPL(rocsparse_ccoo2dense, rocsparse_float_complex);
CIMPL(rocsparse_zcoo2dense, rocsparse_double_complex);
#undef CIMPL
