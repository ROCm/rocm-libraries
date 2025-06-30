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

#include "internal/generic/rocsparse_spvv.h"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_dotci.hpp"
#include "rocsparse_doti.hpp"

#include <map>

namespace rocsparse
{
    template <typename T, typename I, typename X, typename Y>
    rocsparse_status spvv_template_real(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_const_spvec_descr x,
                                        rocsparse_const_dnvec_descr y,
                                        void*                       result,
                                        rocsparse_datatype          compute_type,
                                        size_t*                     buffer_size,
                                        void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // If temp_buffer is nullptr, return buffer_size
        if(temp_buffer == nullptr)
        {
            // We do not need a buffer
            *buffer_size = 4;

            return rocsparse_status_success;
        }

        // real precision
        if(compute_type == rocsparse_datatype_i32_r || compute_type == rocsparse_datatype_f32_r
           || compute_type == rocsparse_datatype_f64_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::doti_template(handle,
                                                               (I)x->nnz,
                                                               (const X*)x->val_data,
                                                               (const I*)x->idx_data,
                                                               (const Y*)y->values,
                                                               (T*)result,
                                                               x->idx_base));
            return rocsparse_status_success;
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }

    template <typename T, typename I, typename X, typename Y>
    rocsparse_status spvv_template_complex(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spvec_descr x,
                                           rocsparse_const_dnvec_descr y,
                                           void*                       result,
                                           rocsparse_datatype          compute_type,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // If temp_buffer is nullptr, return buffer_size
        if(temp_buffer == nullptr)
        {
            // We do not need a buffer
            *buffer_size = 4;
            return rocsparse_status_success;
        }

        // complex precision
        if(compute_type == rocsparse_datatype_f32_c || compute_type == rocsparse_datatype_f64_c)
        {
            // non transpose
            if(trans == rocsparse_operation_none)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::doti_template(handle,
                                                                   (I)x->nnz,
                                                                   (const X*)x->val_data,
                                                                   (const I*)x->idx_data,
                                                                   (const Y*)y->values,
                                                                   (T*)result,
                                                                   x->idx_base));
                return rocsparse_status_success;
            }

            // conjugate transpose
            if(trans == rocsparse_operation_conjugate_transpose)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dotci_template(handle,
                                                                    (I)x->nnz,
                                                                    (const X*)x->val_data,
                                                                    (const I*)x->idx_data,
                                                                    (const Y*)y->values,
                                                                    (T*)result,
                                                                    x->idx_base));
                return rocsparse_status_success;
            }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }

    typedef rocsparse_status (*spvv_t)(rocsparse_handle            handle,
                                       rocsparse_operation         trans,
                                       rocsparse_const_spvec_descr x,
                                       rocsparse_const_dnvec_descr y,
                                       void*                       result,
                                       rocsparse_datatype          compute_type,
                                       size_t*                     buffer_size,
                                       void*                       temp_buffer);

    using spvv_tuple = std::
        tuple<rocsparse_datatype, rocsparse_indextype, rocsparse_datatype, rocsparse_datatype>;

#define SPVV_REAL_CONFIG(T, I, X, Y)                                            \
    {                                                                           \
        spvv_tuple(T, I, X, Y),                                                 \
            spvv_template_real<typename rocsparse::datatype_traits<T>::type_t,  \
                               typename rocsparse::indextype_traits<I>::type_t, \
                               typename rocsparse::datatype_traits<X>::type_t,  \
                               typename rocsparse::datatype_traits<Y>::type_t>  \
    }

#define SPVV_COMPLEX_CONFIG(T, I, X, Y)                                            \
    {                                                                              \
        spvv_tuple(T, I, X, Y),                                                    \
            spvv_template_complex<typename rocsparse::datatype_traits<T>::type_t,  \
                                  typename rocsparse::indextype_traits<I>::type_t, \
                                  typename rocsparse::datatype_traits<X>::type_t,  \
                                  typename rocsparse::datatype_traits<Y>::type_t>  \
    }

    static const std::map<spvv_tuple, spvv_t> s_spvv_dispatch{
        {SPVV_REAL_CONFIG(rocsparse_datatype_f32_r,
                          rocsparse_indextype_i32,
                          rocsparse_datatype_f32_r,
                          rocsparse_datatype_f32_r),
         SPVV_REAL_CONFIG(rocsparse_datatype_f64_r,
                          rocsparse_indextype_i32,
                          rocsparse_datatype_f64_r,
                          rocsparse_datatype_f64_r),
         SPVV_COMPLEX_CONFIG(rocsparse_datatype_f32_c,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c),
         SPVV_COMPLEX_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

         SPVV_REAL_CONFIG(rocsparse_datatype_f32_r,
                          rocsparse_indextype_i64,
                          rocsparse_datatype_f32_r,
                          rocsparse_datatype_f32_r),
         SPVV_REAL_CONFIG(rocsparse_datatype_f64_r,
                          rocsparse_indextype_i64,
                          rocsparse_datatype_f64_r,
                          rocsparse_datatype_f64_r),
         SPVV_COMPLEX_CONFIG(rocsparse_datatype_f32_c,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c),
         SPVV_COMPLEX_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

         SPVV_REAL_CONFIG(rocsparse_datatype_f32_r,
                          rocsparse_indextype_i32,
                          rocsparse_datatype_i8_r,
                          rocsparse_datatype_i8_r),
         SPVV_REAL_CONFIG(rocsparse_datatype_i32_r,
                          rocsparse_indextype_i32,
                          rocsparse_datatype_i8_r,
                          rocsparse_datatype_i8_r),
         SPVV_REAL_CONFIG(rocsparse_datatype_f32_r,
                          rocsparse_indextype_i64,
                          rocsparse_datatype_i8_r,
                          rocsparse_datatype_i8_r),
         SPVV_REAL_CONFIG(rocsparse_datatype_i32_r,
                          rocsparse_indextype_i64,
                          rocsparse_datatype_i8_r,
                          rocsparse_datatype_i8_r),

         SPVV_REAL_CONFIG(rocsparse_datatype_f32_r,
                          rocsparse_indextype_i32,
                          rocsparse_datatype_f16_r,
                          rocsparse_datatype_f16_r),
         SPVV_REAL_CONFIG(rocsparse_datatype_f32_r,
                          rocsparse_indextype_i32,
                          rocsparse_datatype_bf16_r,
                          rocsparse_datatype_bf16_r),
         SPVV_REAL_CONFIG(rocsparse_datatype_f32_r,
                          rocsparse_indextype_i64,
                          rocsparse_datatype_f16_r,
                          rocsparse_datatype_f16_r),
         SPVV_REAL_CONFIG(rocsparse_datatype_f32_r,
                          rocsparse_indextype_i64,
                          rocsparse_datatype_bf16_r,
                          rocsparse_datatype_bf16_r)}};

    static rocsparse_status spvv_find(spvv_t*             function_,
                                      rocsparse_datatype  t_type_,
                                      rocsparse_indextype i_type_,
                                      rocsparse_datatype  x_type_,
                                      rocsparse_datatype  y_type_)
    {
        const auto& it = rocsparse::s_spvv_dispatch.find(
            rocsparse::spvv_tuple(t_type_, i_type_, x_type_, y_type_));

        if(it != rocsparse::s_spvv_dispatch.end())
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
            for(const auto& p : rocsparse::s_spvv_dispatch)
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

extern "C" rocsparse_status rocsparse_spvv(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spvec_descr x,
                                           rocsparse_const_dnvec_descr y,
                                           void*                       result,
                                           rocsparse_datatype          compute_type,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for invalid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_spvv",
                         trans,
                         (const void*&)x,
                         (const void*&)y,
                         (const void*&)result,
                         compute_type,
                         (const void*&)buffer_size,
                         (const void*&)temp_buffer);

    // Check operation
    ROCSPARSE_CHECKARG_ENUM(1, trans);

    // Check compute type
    ROCSPARSE_CHECKARG_ENUM(5, compute_type);

    // Check for invalid descriptors
    ROCSPARSE_CHECKARG_POINTER(2, x);
    ROCSPARSE_CHECKARG_POINTER(3, y);

    // Check for valid pointers
    ROCSPARSE_CHECKARG_POINTER(4, result);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        ROCSPARSE_CHECKARG_POINTER(6, buffer_size);
    }

    // Check if descriptors are initialized
    ROCSPARSE_CHECKARG(2, x, x->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(3, y, y->init == false, rocsparse_status_not_initialized);

    rocsparse::spvv_t f;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spvv_find(&f, compute_type, x->idx_type, x->data_type, y->data_type));
    RETURN_IF_ROCSPARSE_ERROR(
        f(handle, trans, x, y, result, compute_type, buffer_size, temp_buffer));

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
