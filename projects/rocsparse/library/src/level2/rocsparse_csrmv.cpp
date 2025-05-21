/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrmv.hpp"
#include "to_string.hpp"
#include "utility.h"

#include <map>
#include <sstream>

namespace rocsparse

{

    typedef rocsparse_status (*csrmv_t)(rocsparse_handle,
                                        rocsparse_operation,
                                        rocsparse::csrmv_alg,
                                        int64_t,
                                        int64_t,
                                        int64_t,
                                        const void*,
                                        const rocsparse_mat_descr,
                                        const void*,
                                        const void*,
                                        const void*,
                                        const void*,
                                        rocsparse_csrmv_info,
                                        const void*,
                                        const void*,
                                        void*,
                                        bool,
                                        bool);

    using csrmv_tuple = std::tuple<rocsparse_datatype,
                                   rocsparse_indextype,
                                   rocsparse_indextype,
                                   rocsparse_datatype,
                                   rocsparse_datatype,
                                   rocsparse_datatype>;

    // clang-format off
#define CSRMV_CONFIG(T, I, J, A, X, Y)                                      \
    {                                                                       \
        csrmv_tuple(T, I, J, A, X, Y),                                      \
            csrmv_template<typename rocsparse::datatype_traits<T>::type_t,  \
                           typename rocsparse::indextype_traits<I>::type_t, \
                           typename rocsparse::indextype_traits<J>::type_t, \
                           typename rocsparse::datatype_traits<A>::type_t,  \
                           typename rocsparse::datatype_traits<X>::type_t,  \
                           typename rocsparse::datatype_traits<Y>::type_t>  \
    }
    // clang-format on

    static const std::map<csrmv_tuple, csrmv_t> s_csrmv_dispatch{
        {CSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),

         CSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         CSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         CSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         CSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),

         CSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         CSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         CSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         CSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),

         CSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         CSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         CSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         CSRMV_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),

         CSRMV_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),

         CSRMV_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),

         CSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),

         CSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),

         CSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),

         CSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),

         CSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),

         CSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),

         CSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         CSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         CSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         CSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         CSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         CSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         CSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         CSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         CSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         CSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         CSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         CSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c)}};

    static rocsparse_status csrmv_find(csrmv_t*            function_,
                                       rocsparse_datatype  t_type_,
                                       rocsparse_indextype i_type_,
                                       rocsparse_indextype j_type_,
                                       rocsparse_datatype  a_type_,
                                       rocsparse_datatype  x_type_,
                                       rocsparse_datatype  y_type_)
    {
        const auto& it = rocsparse::s_csrmv_dispatch.find(
            rocsparse::csrmv_tuple(t_type_, i_type_, j_type_, a_type_, x_type_, y_type_));

        if(it != rocsparse::s_csrmv_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {

#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "t_type: " << rocsparse::to_string(t_type_) << std::endl
                      << ", i_type: " << rocsparse::to_string(i_type_) << std::endl
                      << ", j_type: " << rocsparse::to_string(j_type_) << std::endl
                      << ", a_type: " << rocsparse::to_string(a_type_) << std::endl
                      << ", x_type: " << rocsparse::to_string(x_type_) << std::endl
                      << ", y_type: " << rocsparse::to_string(y_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_csrmv_dispatch)
            {
                const auto& t      = p.first;
                const auto  t_type = std::get<0>(t);
                const auto  i_type = std::get<1>(t);
                const auto  j_type = std::get<2>(t);
                const auto  a_type = std::get<3>(t);
                const auto  x_type = std::get<4>(t);
                const auto  y_type = std::get<5>(t);
                std::cout << std::endl
                          << std::endl
                          << "t_type: " << rocsparse::to_string(t_type) << std::endl
                          << ", i_type: " << rocsparse::to_string(i_type) << std::endl
                          << ", j_type: " << rocsparse::to_string(j_type) << std::endl
                          << ", a_type: " << rocsparse::to_string(a_type) << std::endl
                          << ", x_type: " << rocsparse::to_string(x_type) << std::endl
                          << ", y_type: " << rocsparse::to_string(y_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "t_type: " << rocsparse::to_string(t_type_)
                 << ", i_type: " << rocsparse::to_string(i_type_)
                 << ", j_type: " << rocsparse::to_string(j_type_)
                 << ", a_type: " << rocsparse::to_string(a_type_)
                 << ", x_type: " << rocsparse::to_string(x_type_)
                 << ", y_type: " << rocsparse::to_string(y_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::csrmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse::csrmv_alg      alg,
                                  int64_t                   m,
                                  int64_t                   n,
                                  int64_t                   nnz,
                                  rocsparse_datatype        alpha_device_host_datatype,
                                  const void*               alpha_device_host,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_datatype        csr_val_datatype,
                                  const void*               csr_val,
                                  rocsparse_indextype       csr_row_ptr_indextype,
                                  const void*               csr_row_ptr,
                                  rocsparse_indextype       csr_row_ptr_end_indextype,
                                  const void*               csr_row_ptr_end,
                                  rocsparse_indextype       csr_col_ind_indextype,
                                  const void*               csr_col_ind,
                                  rocsparse_csrmv_info      csrmv_info,
                                  rocsparse_datatype        x_datatype,
                                  const void*               x,
                                  rocsparse_datatype        beta_device_host_datatype,
                                  const void*               beta_device_host,
                                  rocsparse_datatype        y_datatype,
                                  void*                     y,
                                  bool                      fallback_algorithm)
{

    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::csrmv_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_find(&f,
                                                    alpha_device_host_datatype,
                                                    csr_row_ptr_indextype,
                                                    csr_col_ind_indextype,
                                                    csr_val_datatype,
                                                    x_datatype,
                                                    y_datatype));

    static constexpr bool force_conj = false;
    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                trans,
                                alg,
                                m,
                                n,
                                nnz,
                                alpha_device_host,
                                descr,
                                csr_val,
                                csr_row_ptr,
                                csr_row_ptr_end,
                                csr_col_ind,
                                csrmv_info,
                                x,
                                beta_device_host,
                                y,
                                force_conj,
                                fallback_algorithm));

    return rocsparse_status_success;
}
