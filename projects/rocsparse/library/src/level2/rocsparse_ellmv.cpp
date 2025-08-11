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

#include "rocsparse_ellmv.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

#include <map>
#include <sstream>

namespace rocsparse

{
    typedef rocsparse_status (*ellmv_t)(rocsparse_handle          handle,
                                        rocsparse_operation       trans,
                                        int64_t                   m,
                                        int64_t                   n,
                                        const void*               alpha,
                                        const rocsparse_mat_descr descr,
                                        const void*               ell_val,
                                        const void*               ell_col_ind,
                                        int64_t                   ell_width,
                                        const void*               x,
                                        const void*               beta,
                                        void*                     y);

    using ellmv_tuple = std::tuple<rocsparse_datatype,
                                   rocsparse_indextype,
                                   rocsparse_datatype,
                                   rocsparse_datatype,
                                   rocsparse_datatype>;

    // clang-format off
#define ELLMV_CONFIG(T, I, A, X, Y)                                  \
    {ellmv_tuple(T, I, A, X, Y),                                     \
     ellmv_template<typename rocsparse::datatype_traits<T>::type_t,  \
                    typename rocsparse::indextype_traits<I>::type_t, \
                    typename rocsparse::datatype_traits<A>::type_t,  \
                    typename rocsparse::datatype_traits<X>::type_t,  \
                    typename rocsparse::datatype_traits<Y>::type_t>}
    // clang-format on

    static const std::map<ellmv_tuple, ellmv_t> s_ellmv_dispatch{
        {ELLMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),

         ELLMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         ELLMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         ELLMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         ELLMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),

         ELLMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         ELLMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         ELLMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         ELLMV_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),

         ELLMV_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),

         ELLMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),

         ELLMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),

         ELLMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),

         ELLMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),

         ELLMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_f32_r),

         ELLMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_f32_r),

         ELLMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         ELLMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         ELLMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         ELLMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         ELLMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         ELLMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         ELLMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         ELLMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c)

        }};

    static rocsparse_status ellmv_find(ellmv_t*            function_,
                                       rocsparse_datatype  t_type_,
                                       rocsparse_indextype i_type_,
                                       rocsparse_datatype  a_type_,
                                       rocsparse_datatype  x_type_,
                                       rocsparse_datatype  y_type_)
    {

        const auto& it = rocsparse::s_ellmv_dispatch.find(
            rocsparse::ellmv_tuple(t_type_, i_type_, a_type_, x_type_, y_type_));

        if(it != rocsparse::s_ellmv_dispatch.end())
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
                      << ", a_type: " << rocsparse::enum_utils::to_string(a_type_) << std::endl
                      << ", x_type: " << rocsparse::enum_utils::to_string(x_type_) << std::endl
                      << ", y_type: " << rocsparse::enum_utils::to_string(y_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_ellmv_dispatch)
            {
                const auto& t      = p.first;
                const auto  t_type = std::get<0>(t);
                const auto  i_type = std::get<1>(t);
                const auto  a_type = std::get<2>(t);
                const auto  x_type = std::get<3>(t);
                const auto  y_type = std::get<4>(t);
                std::cout << std::endl
                          << std::endl
                          << "t_type: " << rocsparse::enum_utils::to_string(t_type) << std::endl
                          << ", i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl
                          << ", a_type: " << rocsparse::enum_utils::to_string(a_type) << std::endl
                          << ", x_type: " << rocsparse::enum_utils::to_string(x_type) << std::endl
                          << ", y_type: " << rocsparse::enum_utils::to_string(y_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "t_type: " << rocsparse::enum_utils::to_string(t_type_)
                 << ", i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", a_type: " << rocsparse::enum_utils::to_string(a_type_)
                 << ", x_type: " << rocsparse::enum_utils::to_string(x_type_)
                 << ", y_type: " << rocsparse::enum_utils::to_string(y_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

}

rocsparse_status rocsparse::ellmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  int64_t                   m,
                                  int64_t                   n,
                                  rocsparse_datatype        alpha_device_host_datatype,
                                  const void*               alpha_device_host,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_datatype        ell_val_datatype,
                                  const void*               ell_val,
                                  rocsparse_indextype       ell_col_ind_indextype,
                                  const void*               ell_col_ind,
                                  int64_t                   ell_width,
                                  rocsparse_datatype        x_datatype,
                                  const void*               x,
                                  rocsparse_datatype        beta_device_host_datatype,
                                  const void*               beta_device_host,
                                  rocsparse_datatype        y_datatype,
                                  void*                     y)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::ellmv_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellmv_find(&f,
                                                    alpha_device_host_datatype,
                                                    ell_col_ind_indextype,
                                                    ell_val_datatype,
                                                    x_datatype,
                                                    y_datatype));

    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                trans,
                                m,
                                n,
                                alpha_device_host,
                                descr,
                                ell_val,
                                ell_col_ind,
                                ell_width,
                                x,
                                beta_device_host,
                                y));

    return rocsparse_status_success;
}
