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

#include <map>
#include <sstream>

#include "internal/generic/rocsparse_sddmm.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_determine_indextype.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_sddmm.hpp"

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_sddmm_alg value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_sddmm_alg_default);
        CASE(rocsparse_sddmm_alg_dense);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_sddmm_alg value_)
{
    switch(value_)
    {
    case rocsparse_sddmm_alg_default:
    {
        return false;
    }
    case rocsparse_sddmm_alg_dense:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{

    template <typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename... Ts>
    static rocsparse_status sddmm_buffer_size_dispatch_format(rocsparse_format format, Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(format)
        {
        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_coo, T, I, I, A, B, C>::
                     buffer_size_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_csr, T, I, J, A, B, C>::
                     buffer_size_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_coo_aos:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_coo_aos, T, I, I, A, B, C>::
                     buffer_size_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csc:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_csc, T, I, J, A, B, C>::
                     buffer_size_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_ell:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_ell, T, I, I, A, B, C>::
                     buffer_size_template(ts...)));
            return rocsparse_status_success;
        }
        case rocsparse_format_bell:
        case rocsparse_format_bsr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
            // LCOV_EXCL_START
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    static rocsparse_status sddmm_buffer_size_dispatch(rocsparse_format            format,
                                                       rocsparse_handle            handle,
                                                       rocsparse_operation         trans_A,
                                                       rocsparse_operation         trans_B,
                                                       const void*                 alpha,
                                                       rocsparse_const_dnmat_descr mat_A,
                                                       rocsparse_const_dnmat_descr mat_B,
                                                       const void*                 beta,
                                                       const rocsparse_spmat_descr mat_C,
                                                       rocsparse_datatype          compute_type,
                                                       rocsparse_sddmm_alg         alg,
                                                       size_t*                     buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return sddmm_buffer_size_dispatch_format<T, I, J, A, B, C>(format,
                                                                   handle,
                                                                   trans_A,
                                                                   trans_B,
                                                                   alpha,
                                                                   mat_A,
                                                                   mat_B,
                                                                   beta,
                                                                   mat_C,
                                                                   compute_type,
                                                                   alg,
                                                                   buffer_size);
    }

    typedef rocsparse_status (*sddmm_buffer_size_template_t)(rocsparse_format            format,
                                                             rocsparse_handle            handle,
                                                             rocsparse_operation         trans_A,
                                                             rocsparse_operation         trans_B,
                                                             const void*                 alpha,
                                                             rocsparse_const_dnmat_descr mat_A,
                                                             rocsparse_const_dnmat_descr mat_B,
                                                             const void*                 beta,
                                                             const rocsparse_spmat_descr mat_C,
                                                             rocsparse_datatype  compute_type,
                                                             rocsparse_sddmm_alg alg,
                                                             size_t*             buffer_size);

    using sddmm_buffer_size_template_tuple = std::tuple<rocsparse_datatype,
                                                        rocsparse_indextype,
                                                        rocsparse_indextype,
                                                        rocsparse_datatype,
                                                        rocsparse_datatype,
                                                        rocsparse_datatype>;

    // clang-format off
#define SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(T_, I_, J_, A_, B_, C_)                        \
    {                                                                                    \
        sddmm_buffer_size_template_tuple(T_, I_, J_, A_, B_, C_),                        \
            sddmm_buffer_size_dispatch<typename rocsparse::datatype_traits<T_>::type_t,  \
                                       typename rocsparse::indextype_traits<I_>::type_t, \
                                       typename rocsparse::indextype_traits<J_>::type_t, \
                                       typename rocsparse::datatype_traits<A_>::type_t,  \
                                       typename rocsparse::datatype_traits<B_>::type_t,  \
                                       typename rocsparse::datatype_traits<C_>::type_t>  \
    }
    // clang-format on

    static const std::map<sddmm_buffer_size_template_tuple, sddmm_buffer_size_template_t>
        s_sddmm_buffer_size_template_dispatch{{

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f16_r,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f16_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f16_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i64,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i64,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f32_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f32_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i64,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f16_r,
                                              rocsparse_datatype_f32_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_bf16_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_bf16_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i64,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_bf16_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_f32_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_f32_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i64,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_bf16_r,
                                              rocsparse_datatype_f32_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f32_r,
                                              rocsparse_datatype_f32_r,
                                              rocsparse_datatype_f32_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f32_r,
                                              rocsparse_datatype_f32_r,
                                              rocsparse_datatype_f32_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i64,
                                              rocsparse_datatype_f32_r,
                                              rocsparse_datatype_f32_r,
                                              rocsparse_datatype_f32_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f64_r,
                                              rocsparse_datatype_f64_r,
                                              rocsparse_datatype_f64_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f64_r,
                                              rocsparse_datatype_f64_r,
                                              rocsparse_datatype_f64_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i64,
                                              rocsparse_datatype_f64_r,
                                              rocsparse_datatype_f64_r,
                                              rocsparse_datatype_f64_r),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f64_c,
                                              rocsparse_datatype_f64_c,
                                              rocsparse_datatype_f64_c),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f64_c,
                                              rocsparse_datatype_f64_c,
                                              rocsparse_datatype_f64_c),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i64,
                                              rocsparse_datatype_f64_c,
                                              rocsparse_datatype_f64_c,
                                              rocsparse_datatype_f64_c),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f32_c,
                                              rocsparse_datatype_f32_c,
                                              rocsparse_datatype_f32_c),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i32,
                                              rocsparse_datatype_f32_c,
                                              rocsparse_datatype_f32_c,
                                              rocsparse_datatype_f32_c),

            SDDMM_BUFFER_SIZE_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                                              rocsparse_indextype_i64,
                                              rocsparse_indextype_i64,
                                              rocsparse_datatype_f32_c,
                                              rocsparse_datatype_f32_c,
                                              rocsparse_datatype_f32_c)}};

    static rocsparse_status
        sddmm_buffer_size_template_find(sddmm_buffer_size_template_t* sddmm_buffer_size_function_,
                                        rocsparse_datatype            compute_type_,
                                        rocsparse_indextype           i_type_,
                                        rocsparse_indextype           j_type_,
                                        rocsparse_datatype            a_type_,
                                        rocsparse_datatype            b_type_,
                                        rocsparse_datatype            c_type_)
    {
        const auto& it = rocsparse::s_sddmm_buffer_size_template_dispatch.find(
            rocsparse::sddmm_buffer_size_template_tuple(
                compute_type_, i_type_, j_type_, a_type_, b_type_, c_type_));

        if(it != rocsparse::s_sddmm_buffer_size_template_dispatch.end())
        {
            sddmm_buffer_size_function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "compute_type: " << rocsparse::enum_utils::to_string(compute_type_)
                 << ", i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", j_type: " << rocsparse::enum_utils::to_string(j_type_)
                 << ", a_type: " << rocsparse::enum_utils::to_string(a_type_)
                 << ", b_type: " << rocsparse::enum_utils::to_string(b_type_)
                 << ", c_type: " << rocsparse::enum_utils::to_string(c_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

}

extern "C" rocsparse_status rocsparse_sddmm_buffer_size(rocsparse_handle            handle,
                                                        rocsparse_operation         trans_A,
                                                        rocsparse_operation         trans_B,
                                                        const void*                 alpha,
                                                        rocsparse_const_dnmat_descr mat_A,
                                                        rocsparse_const_dnmat_descr mat_B,
                                                        const void*                 beta,
                                                        const rocsparse_spmat_descr mat_C,
                                                        rocsparse_datatype          compute_type,
                                                        rocsparse_sddmm_alg         alg,
                                                        size_t*                     buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_sddmm_buffer_size",
                         trans_A,
                         trans_B,
                         (const void*&)alpha,
                         (const void*&)mat_A,
                         (const void*&)mat_B,
                         (const void*&)beta,
                         (const void*&)mat_C,
                         compute_type,
                         alg,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, mat_A);
    ROCSPARSE_CHECKARG(4, mat_A, mat_A->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(5, mat_B);
    ROCSPARSE_CHECKARG(5, mat_B, mat_B->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(6, beta);
    ROCSPARSE_CHECKARG_POINTER(7, mat_C);
    ROCSPARSE_CHECKARG(7, mat_C, mat_C->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_ENUM(8, compute_type);

    ROCSPARSE_CHECKARG_ENUM(9, alg);
    ROCSPARSE_CHECKARG_POINTER(10, buffer_size);

    ROCSPARSE_CHECKARG(1,
                       trans_A,
                       (trans_A == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(2,
                       trans_B,
                       (trans_B == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);

    rocsparse::sddmm_buffer_size_template_t sddmm_buffer_size_function;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::sddmm_buffer_size_template_find(&sddmm_buffer_size_function,
                                                   compute_type,
                                                   rocsparse::determine_I_indextype(mat_C),
                                                   rocsparse::determine_J_indextype(mat_C),
                                                   mat_A->data_type,
                                                   mat_B->data_type,
                                                   mat_C->data_type));

    RETURN_IF_ROCSPARSE_ERROR(sddmm_buffer_size_function(mat_C->format,
                                                         handle,
                                                         trans_A,
                                                         trans_B,
                                                         alpha,
                                                         mat_A,
                                                         mat_B,
                                                         beta,
                                                         mat_C,
                                                         compute_type,
                                                         alg,
                                                         buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{

    template <typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename... Ts>
    static rocsparse_status sddmm_preprocess_dispatch_format(rocsparse_format format, Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(format)
        {
        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_coo, T, I, I, A, B, C>::
                     preprocess_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_csr, T, I, J, A, B, C>::
                     preprocess_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_coo_aos:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_coo_aos, T, I, I, A, B, C>::
                     preprocess_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csc:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_csc, T, I, J, A, B, C>::
                     preprocess_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_ell:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_ell, T, I, I, A, B, C>::
                     preprocess_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_bell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_format_bsr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
            // LCOV_EXCL_START
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    static rocsparse_status sddmm_preprocess_dispatch(rocsparse_format            format,
                                                      rocsparse_handle            handle,
                                                      rocsparse_operation         trans_A,
                                                      rocsparse_operation         trans_B,
                                                      const void*                 alpha,
                                                      rocsparse_const_dnmat_descr mat_A,
                                                      rocsparse_const_dnmat_descr mat_B,
                                                      const void*                 beta,
                                                      const rocsparse_spmat_descr mat_C,
                                                      rocsparse_datatype          compute_type,
                                                      rocsparse_sddmm_alg         alg,
                                                      void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return sddmm_preprocess_dispatch_format<T, I, J, A, B, C>(format,
                                                                  handle,
                                                                  trans_A,
                                                                  trans_B,
                                                                  alpha,
                                                                  mat_A,
                                                                  mat_B,
                                                                  beta,
                                                                  mat_C,
                                                                  compute_type,
                                                                  alg,
                                                                  buffer);
    }

    typedef rocsparse_status (*sddmm_preprocess_template_t)(rocsparse_format            format,
                                                            rocsparse_handle            handle,
                                                            rocsparse_operation         trans_A,
                                                            rocsparse_operation         trans_B,
                                                            const void*                 alpha,
                                                            rocsparse_const_dnmat_descr mat_A,
                                                            rocsparse_const_dnmat_descr mat_B,
                                                            const void*                 beta,
                                                            const rocsparse_spmat_descr mat_C,
                                                            rocsparse_datatype  compute_type,
                                                            rocsparse_sddmm_alg alg,
                                                            void*               buffer);

    using sddmm_preprocess_template_tuple = std::tuple<rocsparse_datatype,
                                                       rocsparse_indextype,
                                                       rocsparse_indextype,
                                                       rocsparse_datatype,
                                                       rocsparse_datatype,
                                                       rocsparse_datatype>;

    // clang-format off
#define SDDMM_PREPROCESS_TEMPLATE_CONFIG(T_, I_, J_, A_, B_, C_)                        \
    {                                                                                   \
        sddmm_preprocess_template_tuple(T_, I_, J_, A_, B_, C_),                        \
            sddmm_preprocess_dispatch<typename rocsparse::datatype_traits<T_>::type_t,  \
                                      typename rocsparse::indextype_traits<I_>::type_t, \
                                      typename rocsparse::indextype_traits<J_>::type_t, \
                                      typename rocsparse::datatype_traits<A_>::type_t,  \
                                      typename rocsparse::datatype_traits<B_>::type_t,  \
                                      typename rocsparse::datatype_traits<C_>::type_t>  \
    }
    // clang-format on

    static const std::map<sddmm_preprocess_template_tuple, sddmm_preprocess_template_t>
        s_sddmm_preprocess_template_dispatch{{

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f16_r,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f16_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f16_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f32_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f32_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f16_r,
                                             rocsparse_datatype_f32_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_bf16_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_bf16_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_bf16_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_f32_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_f32_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_bf16_r,
                                             rocsparse_datatype_f32_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f32_r,
                                             rocsparse_datatype_f32_r,
                                             rocsparse_datatype_f32_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f32_r,
                                             rocsparse_datatype_f32_r,
                                             rocsparse_datatype_f32_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_datatype_f32_r,
                                             rocsparse_datatype_f32_r,
                                             rocsparse_datatype_f32_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f64_r,
                                             rocsparse_datatype_f64_r,
                                             rocsparse_datatype_f64_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f64_r,
                                             rocsparse_datatype_f64_r,
                                             rocsparse_datatype_f64_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_datatype_f64_r,
                                             rocsparse_datatype_f64_r,
                                             rocsparse_datatype_f64_r),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f64_c,
                                             rocsparse_datatype_f64_c,
                                             rocsparse_datatype_f64_c),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f64_c,
                                             rocsparse_datatype_f64_c,
                                             rocsparse_datatype_f64_c),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_datatype_f64_c,
                                             rocsparse_datatype_f64_c,
                                             rocsparse_datatype_f64_c),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f32_c,
                                             rocsparse_datatype_f32_c,
                                             rocsparse_datatype_f32_c),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i32,
                                             rocsparse_datatype_f32_c,
                                             rocsparse_datatype_f32_c,
                                             rocsparse_datatype_f32_c),

            SDDMM_PREPROCESS_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                                             rocsparse_indextype_i64,
                                             rocsparse_indextype_i64,
                                             rocsparse_datatype_f32_c,
                                             rocsparse_datatype_f32_c,
                                             rocsparse_datatype_f32_c)}};

    static rocsparse_status
        sddmm_preprocess_template_find(sddmm_preprocess_template_t* sddmm_preprocess_function_,
                                       rocsparse_datatype           compute_type_,
                                       rocsparse_indextype          i_type_,
                                       rocsparse_indextype          j_type_,
                                       rocsparse_datatype           a_type_,
                                       rocsparse_datatype           b_type_,
                                       rocsparse_datatype           c_type_)
    {
        const auto& it = rocsparse::s_sddmm_preprocess_template_dispatch.find(
            rocsparse::sddmm_preprocess_template_tuple(
                compute_type_, i_type_, j_type_, a_type_, b_type_, c_type_));

        if(it != rocsparse::s_sddmm_preprocess_template_dispatch.end())
        {
            sddmm_preprocess_function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "compute_type: " << rocsparse::enum_utils::to_string(compute_type_)
                 << ", i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", j_type: " << rocsparse::enum_utils::to_string(j_type_)
                 << ", a_type: " << rocsparse::enum_utils::to_string(a_type_)
                 << ", b_type: " << rocsparse::enum_utils::to_string(b_type_)
                 << ", c_type: " << rocsparse::enum_utils::to_string(c_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_sddmm_preprocess(rocsparse_handle            handle, //0
                                                       rocsparse_operation         trans_A, //1
                                                       rocsparse_operation         trans_B, //2
                                                       const void*                 alpha, //3
                                                       rocsparse_const_dnmat_descr mat_A, //4
                                                       rocsparse_const_dnmat_descr mat_B, //5
                                                       const void*                 beta, //6
                                                       const rocsparse_spmat_descr mat_C, //7
                                                       rocsparse_datatype          compute_type, //8
                                                       rocsparse_sddmm_alg         alg, //9
                                                       void*                       temp_buffer) //10
try
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         "rocsparse_sddmm_preprocess",
                         trans_A,
                         trans_B,
                         (const void*&)alpha,
                         (const void*&)mat_A,
                         (const void*&)mat_B,
                         (const void*&)beta,
                         (const void*&)mat_C,
                         compute_type,
                         alg,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, mat_A);
    ROCSPARSE_CHECKARG(4, mat_A, mat_A->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(5, mat_B);
    ROCSPARSE_CHECKARG(5, mat_B, mat_B->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(6, beta);
    ROCSPARSE_CHECKARG_POINTER(7, mat_C);
    ROCSPARSE_CHECKARG(7, mat_C, mat_C->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_ENUM(8, compute_type);

    ROCSPARSE_CHECKARG_ENUM(9, alg);

    ROCSPARSE_CHECKARG(1,
                       trans_A,
                       (trans_A == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(2,
                       trans_B,
                       (trans_B == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);

    if(mat_C->nnz == 0)
    {
        return rocsparse_status_success;
    }

    rocsparse::sddmm_preprocess_template_t sddmm_preprocess_function;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::sddmm_preprocess_template_find(&sddmm_preprocess_function,
                                                  compute_type,
                                                  rocsparse::determine_I_indextype(mat_C),
                                                  rocsparse::determine_J_indextype(mat_C),
                                                  mat_A->data_type,
                                                  mat_B->data_type,
                                                  mat_C->data_type));

    RETURN_IF_ROCSPARSE_ERROR(sddmm_preprocess_function(mat_C->format,
                                                        handle,
                                                        trans_A,
                                                        trans_B,
                                                        alpha,
                                                        mat_A,
                                                        mat_B,
                                                        beta,
                                                        mat_C,
                                                        compute_type,
                                                        alg,
                                                        temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{

    template <typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename... Ts>
    static rocsparse_status sddmm_dispatch_format(rocsparse_format format, Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(format)
        {
        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_coo, T, I, I, A, B, C>::
                     compute_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_csr, T, I, J, A, B, C>::
                     compute_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_coo_aos:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_coo_aos, T, I, I, A, B, C>::
                     compute_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csc:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_csc, T, I, J, A, B, C>::
                     compute_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_ell:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<rocsparse_format_ell, T, I, I, A, B, C>::
                     compute_template(ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_bell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_format_bsr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
            // LCOV_EXCL_START
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    static rocsparse_status sddmm_dispatch(rocsparse_format            format,
                                           rocsparse_handle            handle,
                                           rocsparse_operation         trans_A,
                                           rocsparse_operation         trans_B,
                                           const void*                 alpha,
                                           rocsparse_const_dnmat_descr mat_A,
                                           rocsparse_const_dnmat_descr mat_B,
                                           const void*                 beta,
                                           const rocsparse_spmat_descr mat_C,
                                           rocsparse_datatype          compute_type,
                                           rocsparse_sddmm_alg         alg,
                                           void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return sddmm_dispatch_format<T, I, J, A, B, C>(format,
                                                       handle,
                                                       trans_A,
                                                       trans_B,
                                                       alpha,
                                                       mat_A,
                                                       mat_B,
                                                       beta,
                                                       mat_C,
                                                       compute_type,
                                                       alg,
                                                       buffer);
    }

    typedef rocsparse_status (*sddmm_template_t)(rocsparse_format            format,
                                                 rocsparse_handle            handle,
                                                 rocsparse_operation         trans_A,
                                                 rocsparse_operation         trans_B,
                                                 const void*                 alpha,
                                                 rocsparse_const_dnmat_descr mat_A,
                                                 rocsparse_const_dnmat_descr mat_B,
                                                 const void*                 beta,
                                                 const rocsparse_spmat_descr mat_C,
                                                 rocsparse_datatype          compute_type,
                                                 rocsparse_sddmm_alg         alg,
                                                 void*                       buffer);

    using sddmm_template_tuple = std::tuple<rocsparse_datatype,
                                            rocsparse_indextype,
                                            rocsparse_indextype,
                                            rocsparse_datatype,
                                            rocsparse_datatype,
                                            rocsparse_datatype>;

    // clang-format off
#define SDDMM_TEMPLATE_CONFIG(T_, I_, J_, A_, B_, C_)                        \
    {                                                                        \
        sddmm_template_tuple(T_, I_, J_, A_, B_, C_),                        \
            sddmm_dispatch<typename rocsparse::datatype_traits<T_>::type_t,  \
                           typename rocsparse::indextype_traits<I_>::type_t, \
                           typename rocsparse::indextype_traits<J_>::type_t, \
                           typename rocsparse::datatype_traits<A_>::type_t,  \
                           typename rocsparse::datatype_traits<B_>::type_t,  \
                           typename rocsparse::datatype_traits<C_>::type_t> \
    }
    // clang-format on

    static const std::map<sddmm_template_tuple, sddmm_template_t> s_sddmm_template_dispatch{{

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f16_r,
                              rocsparse_indextype_i32,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f16_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f16_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i64,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i32,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i64,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i32,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f32_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f32_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i64,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f16_r,
                              rocsparse_datatype_f32_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i32,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_bf16_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_bf16_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i64,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_bf16_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i32,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_f32_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_f32_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i64,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_bf16_r,
                              rocsparse_datatype_f32_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i32,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f32_r,
                              rocsparse_datatype_f32_r,
                              rocsparse_datatype_f32_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f32_r,
                              rocsparse_datatype_f32_r,
                              rocsparse_datatype_f32_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i64,
                              rocsparse_datatype_f32_r,
                              rocsparse_datatype_f32_r,
                              rocsparse_datatype_f32_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                              rocsparse_indextype_i32,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f64_r,
                              rocsparse_datatype_f64_r,
                              rocsparse_datatype_f64_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f64_r,
                              rocsparse_datatype_f64_r,
                              rocsparse_datatype_f64_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i64,
                              rocsparse_datatype_f64_r,
                              rocsparse_datatype_f64_r,
                              rocsparse_datatype_f64_r),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                              rocsparse_indextype_i32,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f64_c,
                              rocsparse_datatype_f64_c,
                              rocsparse_datatype_f64_c),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f64_c,
                              rocsparse_datatype_f64_c,
                              rocsparse_datatype_f64_c),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i64,
                              rocsparse_datatype_f64_c,
                              rocsparse_datatype_f64_c,
                              rocsparse_datatype_f64_c),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                              rocsparse_indextype_i32,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f32_c,
                              rocsparse_datatype_f32_c,
                              rocsparse_datatype_f32_c),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i32,
                              rocsparse_datatype_f32_c,
                              rocsparse_datatype_f32_c,
                              rocsparse_datatype_f32_c),

        SDDMM_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                              rocsparse_indextype_i64,
                              rocsparse_indextype_i64,
                              rocsparse_datatype_f32_c,
                              rocsparse_datatype_f32_c,
                              rocsparse_datatype_f32_c)}};

    static rocsparse_status sddmm_template_find(sddmm_template_t*   sddmm_function_,
                                                rocsparse_datatype  compute_type_,
                                                rocsparse_indextype i_type_,
                                                rocsparse_indextype j_type_,
                                                rocsparse_datatype  a_type_,
                                                rocsparse_datatype  b_type_,
                                                rocsparse_datatype  c_type_)
    {
        const auto& it = rocsparse::s_sddmm_template_dispatch.find(rocsparse::sddmm_template_tuple(
            compute_type_, i_type_, j_type_, a_type_, b_type_, c_type_));

        if(it != rocsparse::s_sddmm_template_dispatch.end())
        {
            sddmm_function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "compute_type: " << rocsparse::enum_utils::to_string(compute_type_)
                 << ", i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", j_type: " << rocsparse::enum_utils::to_string(j_type_)
                 << ", a_type: " << rocsparse::enum_utils::to_string(a_type_)
                 << ", b_type: " << rocsparse::enum_utils::to_string(b_type_)
                 << ", c_type: " << rocsparse::enum_utils::to_string(c_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_sddmm(rocsparse_handle            handle, //0
                                            rocsparse_operation         trans_A, //1
                                            rocsparse_operation         trans_B, //2
                                            const void*                 alpha, //3
                                            rocsparse_const_dnmat_descr mat_A, //4
                                            rocsparse_const_dnmat_descr mat_B, //5
                                            const void*                 beta, //6
                                            const rocsparse_spmat_descr mat_C, //7
                                            rocsparse_datatype          compute_type, //8
                                            rocsparse_sddmm_alg         alg, //9
                                            void*                       temp_buffer) //19
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_sddmm",
                         trans_A,
                         trans_B,
                         (const void*&)alpha,
                         (const void*&)mat_A,
                         (const void*&)mat_B,
                         (const void*&)beta,
                         (const void*&)mat_C,
                         compute_type,
                         alg,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, mat_A);
    ROCSPARSE_CHECKARG(4, mat_A, mat_A->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(5, mat_B);
    ROCSPARSE_CHECKARG(5, mat_B, mat_B->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(6, beta);
    ROCSPARSE_CHECKARG_POINTER(7, mat_C);
    ROCSPARSE_CHECKARG(7, mat_C, mat_C->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_ENUM(8, compute_type);

    ROCSPARSE_CHECKARG_ENUM(9, alg);

    ROCSPARSE_CHECKARG(1,
                       trans_A,
                       (trans_A == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(2,
                       trans_B,
                       (trans_B == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);

    if(mat_C->nnz == 0)
    {
        return rocsparse_status_success;
    }

    rocsparse::sddmm_template_t sddmm_function;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::sddmm_template_find(&sddmm_function,
                                       compute_type,
                                       rocsparse::determine_I_indextype(mat_C),
                                       rocsparse::determine_J_indextype(mat_C),
                                       mat_A->data_type,
                                       mat_B->data_type,
                                       mat_C->data_type));

    RETURN_IF_ROCSPARSE_ERROR(sddmm_function(mat_C->format,
                                             handle,
                                             trans_A,
                                             trans_B,
                                             alpha,
                                             mat_A,
                                             mat_B,
                                             beta,
                                             mat_C,
                                             compute_type,
                                             alg,
                                             temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
