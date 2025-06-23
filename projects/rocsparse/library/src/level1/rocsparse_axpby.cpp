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

#include "internal/generic/rocsparse_axpby.h"
#include "rocsparse_axpyi.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include <map>

namespace rocsparse
{
    template <typename T, typename I, typename X, typename Y>
    rocsparse_status axpby_template(rocsparse_handle            handle,
                                    const void*                 alpha,
                                    rocsparse_const_spvec_descr x,
                                    const void*                 beta,
                                    rocsparse_dnvec_descr       y)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return
        if(y->size == 0)
        {
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::scale_array(handle, (I)y->size, (const T*)beta, (Y*)y->values));

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::axpyi_template<T, I, X, Y>)(handle,
                                                    (I)x->nnz,
                                                    (const T*)alpha,
                                                    (const X*)x->const_val_data,
                                                    (const I*)x->const_idx_data,
                                                    (Y*)y->values,
                                                    x->idx_base));
        return rocsparse_status_success;
    }

    typedef rocsparse_status (*axpby_t)(rocsparse_handle            handle,
                                        const void*                 alpha,
                                        rocsparse_const_spvec_descr x,
                                        const void*                 beta,
                                        rocsparse_dnvec_descr       y);

    using axpby_tuple = std::
        tuple<rocsparse_datatype, rocsparse_indextype, rocsparse_datatype, rocsparse_datatype>;

    // clang-format off
#define AXPBY_CONFIG(T, I, X, Y)                                        \
    {axpby_tuple(T, I, X, Y),                                           \
     axpby_template<typename rocsparse::datatype_traits<T>::type_t,     \
                    typename rocsparse::indextype_traits<I>::type_t,    \
                    typename rocsparse::datatype_traits<X>::type_t,     \
                    typename rocsparse::datatype_traits<Y>::type_t>}
    // clang-format on

    static const std::map<axpby_tuple, axpby_t> s_axpby_dispatch{
        {AXPBY_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),
         AXPBY_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),
         AXPBY_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),
         AXPBY_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         AXPBY_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),
         AXPBY_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),
         AXPBY_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),
         AXPBY_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         AXPBY_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r),
         AXPBY_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r),
         AXPBY_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r),
         AXPBY_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r)}};

    static rocsparse_status axpby_find(axpby_t*            function_,
                                       rocsparse_datatype  t_type_,
                                       rocsparse_indextype i_type_,
                                       rocsparse_datatype  x_type_,
                                       rocsparse_datatype  y_type_)
    {
        const auto& it = rocsparse::s_axpby_dispatch.find(
            rocsparse::axpby_tuple(t_type_, i_type_, x_type_, y_type_));

        if(it != rocsparse::s_axpby_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "t_type: " << rocsparse::enum_utils::to_string(t_type_) << std::endl
                      << ", i_type: " << rocsparse::enum_utils::to_string(i_type_) << std::endl
                      << ", x_type: " << rocsparse::enum_utils::to_string(x_type_) << std::endl
                      << ", y_type: " << rocsparse::enum_utils::to_string(y_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_axpby_dispatch)
            {
                const auto& t      = p.first;
                const auto  t_type = std::get<0>(t);
                const auto  i_type = std::get<1>(t);
                const auto  x_type = std::get<2>(t);
                const auto  y_type = std::get<3>(t);
                std::cout << std::endl
                          << std::endl
                          << "t_type: " << rocsparse::enum_utils::to_string(t_type) << std::endl
                          << ", i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl
                          << ", x_type: " << rocsparse::enum_utils::to_string(x_type) << std::endl
                          << ", y_type: " << rocsparse::enum_utils::to_string(y_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "t_type: " << rocsparse::enum_utils::to_string(t_type_)
                 << ", i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", x_type: " << rocsparse::enum_utils::to_string(x_type_)
                 << ", y_type: " << rocsparse::enum_utils::to_string(y_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_axpby(rocsparse_handle            handle,
                                            const void*                 alpha,
                                            rocsparse_const_spvec_descr x,
                                            const void*                 beta,
                                            rocsparse_dnvec_descr       y)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for invalid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_axpby",
                         (const void*&)alpha,
                         (const void*&)x,
                         (const void*&)beta,
                         (const void*&)y);

    // Check for invalid descriptors
    ROCSPARSE_CHECKARG_POINTER(1, alpha);
    ROCSPARSE_CHECKARG_POINTER(2, x);
    ROCSPARSE_CHECKARG_POINTER(3, beta);
    ROCSPARSE_CHECKARG_POINTER(4, y);

    // Check if descriptors are initialized
    ROCSPARSE_CHECKARG(2, x, (x->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(4, y, (y->init == false), rocsparse_status_not_initialized);

    // Check for matching types while we do not support mixed precision computation
    ROCSPARSE_CHECKARG(4, y, (y->data_type != x->data_type), rocsparse_status_not_implemented);

    rocsparse::axpby_t f;
    if(x->data_type == rocsparse_datatype_f16_r || x->data_type == rocsparse_datatype_bf16_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::axpby_find(
            &f, rocsparse_datatype_f32_r, x->idx_type, x->data_type, y->data_type));
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::axpby_find(&f, x->data_type, x->idx_type, x->data_type, y->data_type));
    }

    RETURN_IF_ROCSPARSE_ERROR(f(handle, alpha, x, beta, y));

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
