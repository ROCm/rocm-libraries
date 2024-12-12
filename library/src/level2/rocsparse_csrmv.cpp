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

#include "internal/level2/rocsparse_csrmv.h"
#include "common.h"
#include "control.h"
#include "rocsparse_common.h"
#include "utility.h"

#include "rocsparse_csrmv.hpp"

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse::csrmv_alg value_)
{
    switch(value_)
    {
    case rocsparse::csrmv_alg_stream:
    case rocsparse::csrmv_alg_adaptive:
    case rocsparse::csrmv_alg_lrb:
    {
        return false;
    }
    }
    return true;
};

template <typename I, typename J, typename A>
rocsparse_status rocsparse::csrmv_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_operation       trans,
                                                    rocsparse::csrmv_alg      alg,
                                                    J                         m,
                                                    J                         n,
                                                    I                         nnz,
                                                    const rocsparse_mat_descr descr,
                                                    const A*                  csr_val,
                                                    const I*                  csr_row_ptr,
                                                    const J*                  csr_col_ind,
                                                    rocsparse_mat_info        info)
{
    if(m == 0 || n == 0 || nnz == 0)
    {
        // No matrix analysis required as matrix never accessed
        return rocsparse_status_success;
    }

    switch(alg)
    {
    case rocsparse::csrmv_alg_adaptive:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_analysis_adaptive_template_dispatch(
            handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info));
        return rocsparse_status_success;
    }

    case rocsparse::csrmv_alg_lrb:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_analysis_lrb_template_dispatch(
            handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info));
        return rocsparse_status_success;
    }

    case rocsparse::csrmv_alg_stream:
    {
        return rocsparse_status_success;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::csrmv_template(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse::csrmv_alg      alg,
                                           J                         m,
                                           J                         n,
                                           I                         nnz,
                                           const T*                  alpha_device_host,
                                           const rocsparse_mat_descr descr,
                                           const A*                  csr_val,
                                           const I*                  csr_row_ptr_begin,
                                           const I*                  csr_row_ptr_end,
                                           const J*                  csr_col_ind,
                                           rocsparse_mat_info        info,
                                           const X*                  x,
                                           const T*                  beta_device_host,
                                           Y*                        y,
                                           bool                      force_conj)
{

    const rocsparse_int ysize = (trans == rocsparse_operation_none) ? m : n;

    // Another quick return.
    if(m == 0 || n == 0 || nnz == 0)
    {
        // matrix never accessed however still need to update y vector
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, ysize, beta_device_host, y));
        return rocsparse_status_success;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    if(info == nullptr || info->csrmv_info == nullptr || trans != rocsparse_operation_none
       || (alg == rocsparse::csrmv_alg_lrb && descr->type == rocsparse_matrix_type_symmetric))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_stream_template_dispatch(handle,
                                                                            trans,
                                                                            m,
                                                                            n,
                                                                            nnz,
                                                                            alpha_device_host,
                                                                            descr,
                                                                            csr_val,
                                                                            csr_row_ptr_begin,
                                                                            csr_row_ptr_end,
                                                                            csr_col_ind,
                                                                            x,
                                                                            beta_device_host,
                                                                            y,
                                                                            force_conj));
        return rocsparse_status_success;
    }
    else
    {
        // If csrmv info is available, call csrmv adaptive or lrb

        //
        // Rows must be stored contiguously.
        //
        if((csr_row_ptr_begin + 1) != csr_row_ptr_end)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }

        switch(alg)
        {
        case rocsparse::csrmv_alg_stream:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_stream_template_dispatch(handle,
                                                                                trans,
                                                                                m,
                                                                                n,
                                                                                nnz,
                                                                                alpha_device_host,
                                                                                descr,
                                                                                csr_val,
                                                                                csr_row_ptr_begin,
                                                                                csr_row_ptr_end,
                                                                                csr_col_ind,
                                                                                x,
                                                                                beta_device_host,
                                                                                y,
                                                                                force_conj));
            return rocsparse_status_success;
        }
        case rocsparse::csrmv_alg_adaptive:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_adaptive_template_dispatch(handle,
                                                                                  trans,
                                                                                  m,
                                                                                  n,
                                                                                  nnz,
                                                                                  alpha_device_host,
                                                                                  descr,
                                                                                  csr_val,
                                                                                  csr_row_ptr_begin,
                                                                                  csr_col_ind,
                                                                                  info->csrmv_info,
                                                                                  x,
                                                                                  beta_device_host,
                                                                                  y,
                                                                                  force_conj));
            return rocsparse_status_success;
        }
        case rocsparse::csrmv_alg_lrb:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_lrb_template_dispatch(handle,
                                                                             trans,
                                                                             m,
                                                                             n,
                                                                             nnz,
                                                                             alpha_device_host,
                                                                             descr,
                                                                             csr_val,
                                                                             csr_row_ptr_begin,
                                                                             csr_col_ind,
                                                                             info->csrmv_info,
                                                                             x,
                                                                             beta_device_host,
                                                                             y,
                                                                             force_conj));
            return rocsparse_status_success;
        }
        }
    }
}

template <typename I, typename J, typename A>
rocsparse_status rocsparse_csrmv_analysis_impl(rocsparse_handle          handle,
                                               rocsparse_operation       trans,
                                               J                         m,
                                               J                         n,
                                               I                         nnz,
                                               const rocsparse_mat_descr descr,
                                               const A*                  csr_val,
                                               const I*                  csr_row_ptr,
                                               const J*                  csr_col_ind,
                                               rocsparse_mat_info        info)
{
    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(5, descr);
    ROCSPARSE_CHECKARG_POINTER(9, info);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csrmv_analysis",
                         trans,
                         m,
                         n,
                         nnz,
                         (const void*&)descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)info);

    ROCSPARSE_CHECKARG_ENUM(1, trans);

    // Check matrix type
    ROCSPARSE_CHECKARG(5,
                       descr,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular
                        && descr->type != rocsparse_matrix_type_symmetric),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG(5,
                       descr,
                       ((descr->type == rocsparse_matrix_type_symmetric
                         || descr->type == rocsparse_matrix_type_triangular)
                        && (m != n)),
                       rocsparse_status_invalid_size);

    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(5,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, n);
    ROCSPARSE_CHECKARG_SIZE(4, nnz);

    // Another quick return.
    if(m == 0 || n == 0 || nnz == 0)
    {
        // No matrix analysis required as matrix never accessed
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_ARRAY(7, m, csr_row_ptr);

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(8, nnz, csr_col_ind);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_analysis_template(
        handle,
        trans,
        info != nullptr ? rocsparse::csrmv_alg_adaptive : rocsparse::csrmv_alg_stream,
        m,
        n,
        nnz,
        descr,
        csr_val,
        csr_row_ptr,
        csr_col_ind,
        info));
    return rocsparse_status_success;
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse_csrmv_impl(rocsparse_handle          handle,
                                      rocsparse_operation       trans,
                                      J                         m,
                                      J                         n,
                                      I                         nnz,
                                      const T*                  alpha_device_host,
                                      const rocsparse_mat_descr descr,
                                      const A*                  csr_val,
                                      const I*                  csr_row_ptr,
                                      const J*                  csr_col_ind,
                                      rocsparse_mat_info        info,
                                      const X*                  x,
                                      const T*                  beta_device_host,
                                      Y*                        y,
                                      bool                      force_conj)
{
    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsrmv"),
                         trans,
                         m,
                         n,
                         nnz,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                         (const void*&)descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)x,
                         LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
                         (const void*&)y);

    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, n);
    ROCSPARSE_CHECKARG_SIZE(4, nnz);
    ROCSPARSE_CHECKARG_POINTER(5, alpha_device_host);
    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG(6,
                       descr,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular
                        && descr->type != rocsparse_matrix_type_symmetric),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(6,
                       descr,
                       (descr->type == rocsparse_matrix_type_symmetric
                        || descr->type == rocsparse_matrix_type_triangular)
                           && (m != n),
                       rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(6,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(9, nnz, csr_col_ind);

    const rocsparse_int xsize = (trans == rocsparse_operation_none) ? n : m;
    const rocsparse_int ysize = (trans == rocsparse_operation_none) ? m : n;

    ROCSPARSE_CHECKARG_ARRAY(11, xsize, x);
    ROCSPARSE_CHECKARG_POINTER(12, beta_device_host);
    ROCSPARSE_CHECKARG_ARRAY(13, ysize, y);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_template(
        handle,
        trans,
        info != nullptr ? rocsparse::csrmv_alg_adaptive : rocsparse::csrmv_alg_stream,
        m,
        n,
        nnz,
        alpha_device_host,
        descr,
        csr_val,
        csr_row_ptr,
        csr_row_ptr + 1,
        csr_col_ind,
        info,
        x,
        beta_device_host,
        y,
        false));
    return rocsparse_status_success;
}

#define INSTANTIATE(TTYPE, ITYPE, JTYPE)                                                             \
    template rocsparse_status rocsparse::csrmv_analysis_template(rocsparse_handle          handle,   \
                                                                 rocsparse_operation       trans,    \
                                                                 rocsparse::csrmv_alg      alg,      \
                                                                 JTYPE                     m,        \
                                                                 JTYPE                     n,        \
                                                                 ITYPE                     nnz,      \
                                                                 const rocsparse_mat_descr descr,    \
                                                                 const TTYPE*              csr_val,  \
                                                                 const ITYPE*       csr_row_ptr,     \
                                                                 const JTYPE*       csr_col_ind,     \
                                                                 rocsparse_mat_info info);           \
    template rocsparse_status rocsparse::csrmv_template(rocsparse_handle          handle,            \
                                                        rocsparse_operation       trans,             \
                                                        rocsparse::csrmv_alg      alg,               \
                                                        JTYPE                     m,                 \
                                                        JTYPE                     n,                 \
                                                        ITYPE                     nnz,               \
                                                        const TTYPE*              alpha_device_host, \
                                                        const rocsparse_mat_descr descr,             \
                                                        const TTYPE*              csr_val,           \
                                                        const ITYPE*              csr_row_ptr_begin, \
                                                        const ITYPE*              csr_row_ptr_end,   \
                                                        const JTYPE*              csr_col_ind,       \
                                                        rocsparse_mat_info        info,              \
                                                        const TTYPE*              x,                 \
                                                        const TTYPE*              beta_device_host,  \
                                                        TTYPE*                    y,                 \
                                                        bool                      force_conj);

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
    template rocsparse_status rocsparse::csrmv_analysis_template(rocsparse_handle          handle,  \
                                                                 rocsparse_operation       trans,   \
                                                                 rocsparse::csrmv_alg      alg,     \
                                                                 JTYPE                     m,       \
                                                                 JTYPE                     n,       \
                                                                 ITYPE                     nnz,     \
                                                                 const rocsparse_mat_descr descr,   \
                                                                 const ATYPE*              csr_val, \
                                                                 const ITYPE*       csr_row_ptr,    \
                                                                 const JTYPE*       csr_col_ind,    \
                                                                 rocsparse_mat_info info);

INSTANTIATE_MIXED_ANALYSIS(int32_t, int32_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int32_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int64_t, int8_t);
#undef INSTANTIATE_MIXED_ANALYSIS

#define INSTANTIATE_MIXED(TTYPE, ITYPE, JTYPE, ATYPE, XTYPE, YTYPE)                                  \
    template rocsparse_status rocsparse::csrmv_template(rocsparse_handle          handle,            \
                                                        rocsparse_operation       trans,             \
                                                        rocsparse::csrmv_alg      alg,               \
                                                        JTYPE                     m,                 \
                                                        JTYPE                     n,                 \
                                                        ITYPE                     nnz,               \
                                                        const TTYPE*              alpha_device_host, \
                                                        const rocsparse_mat_descr descr,             \
                                                        const ATYPE*              csr_val,           \
                                                        const ITYPE*              csr_row_ptr_begin, \
                                                        const ITYPE*              csr_row_ptr_end,   \
                                                        const JTYPE*              csr_col_ind,       \
                                                        rocsparse_mat_info        info,              \
                                                        const XTYPE*              x,                 \
                                                        const TTYPE*              beta_device_host,  \
                                                        YTYPE*                    y,                 \
                                                        bool                      force_conj);

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

INSTANTIATE_MIXED(double, int32_t, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, int64_t, float, double, double);

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

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

//
// rocsparse_xcsrmv_analysis
//
#define C_IMPL(NAME, TYPE)                                                              \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                  \
                                     rocsparse_operation       trans,                   \
                                     rocsparse_int             m,                       \
                                     rocsparse_int             n,                       \
                                     rocsparse_int             nnz,                     \
                                     const rocsparse_mat_descr descr,                   \
                                     const TYPE*               csr_val,                 \
                                     const rocsparse_int*      csr_row_ptr,             \
                                     const rocsparse_int*      csr_col_ind,             \
                                     rocsparse_mat_info        info)                    \
    try                                                                                 \
    {                                                                                   \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrmv_analysis_impl(                        \
            handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info)); \
        return rocsparse_status_success;                                                \
    }                                                                                   \
    catch(...)                                                                          \
    {                                                                                   \
        RETURN_ROCSPARSE_EXCEPTION();                                                   \
    }

C_IMPL(rocsparse_scsrmv_analysis, float);
C_IMPL(rocsparse_dcsrmv_analysis, double);
C_IMPL(rocsparse_ccsrmv_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrmv_analysis, rocsparse_double_complex);

#undef C_IMPL

//
// rocsparse_xcsrmv
//
#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             m,           \
                                     rocsparse_int             n,           \
                                     rocsparse_int             nnz,         \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     rocsparse_mat_info        info,        \
                                     const TYPE*               x,           \
                                     const TYPE*               beta,        \
                                     TYPE*                     y)           \
    try                                                                     \
    {                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrmv_impl(handle,              \
                                                       trans,               \
                                                       m,                   \
                                                       n,                   \
                                                       nnz,                 \
                                                       alpha,               \
                                                       descr,               \
                                                       csr_val,             \
                                                       csr_row_ptr,         \
                                                       csr_col_ind,         \
                                                       info,                \
                                                       x,                   \
                                                       beta,                \
                                                       y,                   \
                                                       false));             \
        return rocsparse_status_success;                                    \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

C_IMPL(rocsparse_scsrmv, float);
C_IMPL(rocsparse_dcsrmv, double);
C_IMPL(rocsparse_ccsrmv, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrmv, rocsparse_double_complex);
#undef C_IMPL

extern "C" rocsparse_status rocsparse_csrmv_clear(rocsparse_handle handle, rocsparse_mat_info info)
try
{
    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, info);

    // Logging
    rocsparse::log_trace(handle, "rocsparse_csrmv_clear", (const void*&)info);

    // Destroy csrmv info struct
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_csrmv_info(info->csrmv_info));
    info->csrmv_info = nullptr;

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
