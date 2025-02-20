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

#include "rocsparse_cscmv.hpp"
#include "rocsparse_csrmv.hpp"

#include "control.h"
#include "utility.h"

template <typename I, typename J, typename A>
rocsparse_status rocsparse::cscmv_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_operation       trans,
                                                    rocsparse::csrmv_alg      alg,
                                                    J                         m,
                                                    J                         n,
                                                    I                         nnz,
                                                    const rocsparse_mat_descr descr,
                                                    const A*                  csc_val,
                                                    const I*                  csc_col_ptr,
                                                    const J*                  csc_row_ind,
                                                    rocsparse_mat_info        info)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(trans)
    {
    case rocsparse_operation_none:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_analysis_template(handle,
                                                                     rocsparse_operation_transpose,
                                                                     alg,
                                                                     n,
                                                                     m,
                                                                     nnz,
                                                                     descr,
                                                                     csc_val,
                                                                     csc_col_ptr,
                                                                     csc_row_ind,
                                                                     info));
        return rocsparse_status_success;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_analysis_template(handle,
                                                                     rocsparse_operation_none,
                                                                     alg,
                                                                     n,
                                                                     m,
                                                                     nnz,
                                                                     descr,
                                                                     csc_val,
                                                                     csc_col_ptr,
                                                                     csc_row_ind,
                                                                     info));
        return rocsparse_status_success;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::cscmv_template(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse::csrmv_alg      alg,
                                           J                         m,
                                           J                         n,
                                           I                         nnz,
                                           const T*                  alpha,
                                           const rocsparse_mat_descr descr,
                                           const A*                  csc_val,
                                           const I*                  csc_col_ptr,
                                           const J*                  csc_row_ind,
                                           rocsparse_mat_info        info,
                                           const X*                  x,
                                           const T*                  beta,
                                           Y*                        y)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(trans)
    {
    case rocsparse_operation_none:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_template(handle,
                                                            rocsparse_operation_transpose,
                                                            alg,
                                                            n,
                                                            m,
                                                            nnz,
                                                            alpha,
                                                            descr,
                                                            csc_val,
                                                            csc_col_ptr,
                                                            csc_col_ptr + 1,
                                                            csc_row_ind,
                                                            info,
                                                            x,
                                                            beta,
                                                            y,
                                                            false));
        return rocsparse_status_success;
    }
    case rocsparse_operation_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_template(handle,
                                                            rocsparse_operation_none,
                                                            alg,
                                                            n,
                                                            m,
                                                            nnz,
                                                            alpha,
                                                            descr,
                                                            csc_val,
                                                            csc_col_ptr,
                                                            csc_col_ptr + 1,
                                                            csc_row_ind,
                                                            info,
                                                            x,
                                                            beta,
                                                            y,
                                                            false));
        return rocsparse_status_success;
    }
    case rocsparse_operation_conjugate_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_template(handle,
                                                            rocsparse_operation_none,
                                                            alg,
                                                            n,
                                                            m,
                                                            nnz,
                                                            alpha,
                                                            descr,
                                                            csc_val,
                                                            csc_col_ptr,
                                                            csc_col_ptr + 1,
                                                            csc_row_ind,
                                                            info,
                                                            x,
                                                            beta,
                                                            y,
                                                            true));
        return rocsparse_status_success;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}

#define INSTANTIATE(TTYPE, ITYPE, JTYPE)                                                            \
    template rocsparse_status rocsparse::cscmv_analysis_template(rocsparse_handle          handle,  \
                                                                 rocsparse_operation       trans,   \
                                                                 rocsparse::csrmv_alg      alg,     \
                                                                 JTYPE                     m,       \
                                                                 JTYPE                     n,       \
                                                                 ITYPE                     nnz,     \
                                                                 const rocsparse_mat_descr descr,   \
                                                                 const TTYPE*              csc_val, \
                                                                 const ITYPE*       csc_col_ptr,    \
                                                                 const JTYPE*       csc_row_ind,    \
                                                                 rocsparse_mat_info info);          \
    template rocsparse_status rocsparse::cscmv_template(rocsparse_handle          handle,           \
                                                        rocsparse_operation       trans,            \
                                                        rocsparse::csrmv_alg      alg,              \
                                                        JTYPE                     m,                \
                                                        JTYPE                     n,                \
                                                        ITYPE                     nnz,              \
                                                        const TTYPE*              alpha,            \
                                                        const rocsparse_mat_descr descr,            \
                                                        const TTYPE*              csc_val,          \
                                                        const ITYPE*              csc_col_ptr,      \
                                                        const JTYPE*              csc_row_ind,      \
                                                        rocsparse_mat_info        info,             \
                                                        const TTYPE*              x,                \
                                                        const TTYPE*              beta,             \
                                                        TTYPE*                    y);

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(float, int64_t, int64_t);
INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(double, int64_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE_MIXED_ANALYSIS(ITYPE, JTYPE, ATYPE)                                             \
    template rocsparse_status rocsparse::cscmv_analysis_template(rocsparse_handle          handle,  \
                                                                 rocsparse_operation       trans,   \
                                                                 rocsparse::csrmv_alg      alg,     \
                                                                 JTYPE                     m,       \
                                                                 JTYPE                     n,       \
                                                                 ITYPE                     nnz,     \
                                                                 const rocsparse_mat_descr descr,   \
                                                                 const ATYPE*              csc_val, \
                                                                 const ITYPE*       csc_col_ptr,    \
                                                                 const JTYPE*       csc_row_ind,    \
                                                                 rocsparse_mat_info info);

INSTANTIATE_MIXED_ANALYSIS(int32_t, int32_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int32_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int64_t, int8_t);
#undef INSTANTIATE_MIXED_ANALYSIS

#define INSTANTIATE_MIXED(TTYPE, ITYPE, JTYPE, ATYPE, XTYPE, YTYPE)                            \
    template rocsparse_status rocsparse::cscmv_template(rocsparse_handle          handle,      \
                                                        rocsparse_operation       trans,       \
                                                        rocsparse::csrmv_alg      alg,         \
                                                        JTYPE                     m,           \
                                                        JTYPE                     n,           \
                                                        ITYPE                     nnz,         \
                                                        const TTYPE*              alpha,       \
                                                        const rocsparse_mat_descr descr,       \
                                                        const ATYPE*              csc_val,     \
                                                        const ITYPE*              csc_col_ptr, \
                                                        const JTYPE*              csc_row_ind, \
                                                        rocsparse_mat_info        info,        \
                                                        const XTYPE*              x,           \
                                                        const TTYPE*              beta,        \
                                                        YTYPE*                    y);

INSTANTIATE_MIXED(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(rocsparse_float_complex,
                  int32_t,
                  int32_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_MIXED(rocsparse_float_complex,
                  int64_t,
                  int32_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_MIXED(rocsparse_float_complex,
                  int64_t,
                  int64_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex);

INSTANTIATE_MIXED(rocsparse_double_complex,
                  int32_t,
                  int32_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int32_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int64_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

INSTANTIATE_MIXED(double, int32_t, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, int64_t, float, double, double);

INSTANTIATE_MIXED(rocsparse_double_complex,
                  int32_t,
                  int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int64_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

#undef INSTANTIATE_MIXED
