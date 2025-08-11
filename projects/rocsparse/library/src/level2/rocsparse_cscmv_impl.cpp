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

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

template <typename I, typename J, typename A>
rocsparse_status rocsparse::cscmv_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_operation       trans,
                                                    rocsparse::csrmv_alg      alg,
                                                    int64_t                   m_,
                                                    int64_t                   n_,
                                                    int64_t                   nnz_,
                                                    const rocsparse_mat_descr descr,
                                                    const void*               csc_val_,
                                                    const void*               csc_col_ptr_,
                                                    const void*               csc_row_ind_,
                                                    rocsparse_csrmv_info*     p_csrmv_info)
{
    ROCSPARSE_ROUTINE_TRACE;

    const J  m           = static_cast<J>(m_);
    const J  n           = static_cast<J>(n_);
    const I  nnz         = static_cast<I>(nnz_);
    const A* csc_val     = reinterpret_cast<const A*>(csc_val_);
    const I* csc_col_ptr = reinterpret_cast<const I*>(csc_col_ptr_);
    const J* csc_row_ind = reinterpret_cast<const J*>(csc_row_ind_);

    p_csrmv_info[0] = nullptr;

    switch(trans)
    {
    case rocsparse_operation_none:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmv_analysis_template<I, J, A>(handle,
                                                         rocsparse_operation_transpose,
                                                         alg,
                                                         n,
                                                         m,
                                                         nnz,
                                                         descr,
                                                         csc_val,
                                                         csc_col_ptr,
                                                         csc_row_ind,
                                                         p_csrmv_info)));
        return rocsparse_status_success;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmv_analysis_template<I, J, A>(handle,
                                                         rocsparse_operation_none,
                                                         alg,
                                                         n,
                                                         m,
                                                         nnz,
                                                         descr,
                                                         csc_val,
                                                         csc_col_ptr,
                                                         csc_row_ind,
                                                         p_csrmv_info)));
        return rocsparse_status_success;
    }
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::cscmv_template(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse::csrmv_alg      alg,
                                           int64_t                   m_,
                                           int64_t                   n_,
                                           int64_t                   nnz_,
                                           const void*               alpha_,
                                           const rocsparse_mat_descr descr,
                                           const void*               csc_val_,
                                           const void*               csc_col_ptr_,
                                           const void*               csc_row_ind_,
                                           rocsparse_csrmv_info      csrmv_info,
                                           const void*               x_,
                                           const void*               beta_,
                                           void*                     y_,
                                           bool                      fallback_algorithm)
{
    ROCSPARSE_ROUTINE_TRACE;

    const J  m           = static_cast<J>(m_);
    const J  n           = static_cast<J>(n_);
    const I  nnz         = static_cast<I>(nnz_);
    const A* csc_val     = reinterpret_cast<const A*>(csc_val_);
    const I* csc_col_ptr = reinterpret_cast<const I*>(csc_col_ptr_);
    const J* csc_row_ind = reinterpret_cast<const J*>(csc_row_ind_);
    const T* alpha       = reinterpret_cast<const T*>(alpha_);
    const T* beta        = reinterpret_cast<const T*>(beta_);
    const X* x           = reinterpret_cast<const X*>(x_);
    Y*       y           = reinterpret_cast<Y*>(y_);

    switch(trans)
    {
    case rocsparse_operation_none:
    {
        static constexpr bool force_conj = false;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmv_template<T, I, J, A, X, Y>(handle,
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
                                                         csrmv_info,
                                                         x,
                                                         beta,
                                                         y,
                                                         force_conj,
                                                         fallback_algorithm)));
        return rocsparse_status_success;
    }
    case rocsparse_operation_transpose:
    {
        static constexpr bool force_conj = false;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmv_template<T, I, J, A, X, Y>(handle,
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
                                                         csrmv_info,
                                                         x,
                                                         beta,
                                                         y,
                                                         force_conj,
                                                         fallback_algorithm)));
        return rocsparse_status_success;
    }
    case rocsparse_operation_conjugate_transpose:
    {
        static constexpr bool force_conj = true;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmv_template<T, I, J, A, X, Y>(handle,
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
                                                         csrmv_info,
                                                         x,
                                                         beta,
                                                         y,
                                                         force_conj,
                                                         fallback_algorithm)));
        return rocsparse_status_success;
    }
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

#define INSTANTIATE(T, I, J)                                               \
    template rocsparse_status rocsparse::cscmv_analysis_template<I, J, T>( \
        rocsparse_handle          handle,                                  \
        rocsparse_operation       trans,                                   \
        rocsparse::csrmv_alg      alg,                                     \
        int64_t                   m,                                       \
        int64_t                   n,                                       \
        int64_t                   nnz,                                     \
        const rocsparse_mat_descr descr,                                   \
        const void*               csc_val,                                 \
        const void*               csc_col_ptr,                             \
        const void*               csc_row_ind,                             \
        rocsparse_csrmv_info*     p_csrmv_info);                               \
    template rocsparse_status rocsparse::cscmv_template<T, I, J, T, T, T>( \
        rocsparse_handle,                                                  \
        rocsparse_operation,                                               \
        rocsparse::csrmv_alg,                                              \
        int64_t,                                                           \
        int64_t,                                                           \
        int64_t,                                                           \
        const void*,                                                       \
        const rocsparse_mat_descr,                                         \
        const void*,                                                       \
        const void*,                                                       \
        const void*,                                                       \
        rocsparse_csrmv_info,                                              \
        const void*,                                                       \
        const void*,                                                       \
        void*,                                                             \
        bool)

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

#define INSTANTIATE_MIXED_ANALYSIS(I, J, A)                                \
    template rocsparse_status rocsparse::cscmv_analysis_template<I, J, A>( \
        rocsparse_handle,                                                  \
        rocsparse_operation,                                               \
        rocsparse::csrmv_alg,                                              \
        int64_t,                                                           \
        int64_t,                                                           \
        int64_t,                                                           \
        const rocsparse_mat_descr,                                         \
        const void*,                                                       \
        const void*,                                                       \
        const void*,                                                       \
        rocsparse_csrmv_info*)

INSTANTIATE_MIXED_ANALYSIS(int32_t, int32_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int32_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int64_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int32_t, int32_t, _Float16);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int32_t, _Float16);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int64_t, _Float16);
INSTANTIATE_MIXED_ANALYSIS(int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int64_t, rocsparse_bfloat16);
#undef INSTANTIATE_MIXED_ANALYSIS

#define INSTANTIATE_MIXED(T, I, J, A, X, Y)                                \
    template rocsparse_status rocsparse::cscmv_template<T, I, J, A, X, Y>( \
        rocsparse_handle,                                                  \
        rocsparse_operation,                                               \
        rocsparse::csrmv_alg,                                              \
        int64_t,                                                           \
        int64_t,                                                           \
        int64_t,                                                           \
        const void*,                                                       \
        const rocsparse_mat_descr,                                         \
        const void*,                                                       \
        const void*,                                                       \
        const void*,                                                       \
        rocsparse_csrmv_info,                                              \
        const void*,                                                       \
        const void*,                                                       \
        void*,                                                             \
        bool)

INSTANTIATE_MIXED(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
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
