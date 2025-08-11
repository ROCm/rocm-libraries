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

#include "rocsparse_bsrmv.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

#include <map>
#include <sstream>

namespace rocsparse

{

    typedef rocsparse_status (*bsrmv_t)(rocsparse_handle,
                                        rocsparse_direction,
                                        rocsparse_operation,
                                        int64_t,
                                        int64_t,
                                        int64_t,
                                        const void*,
                                        const rocsparse_mat_descr,
                                        const void*,
                                        const void*,
                                        const void*,
                                        int64_t,
                                        rocsparse_bsrmv_info,
                                        const void*,
                                        const void*,
                                        void*);

    using bsrmv_tuple = std::tuple<rocsparse_datatype,
                                   rocsparse_indextype,
                                   rocsparse_indextype,
                                   rocsparse_datatype,
                                   rocsparse_datatype,
                                   rocsparse_datatype>;

    // clang-format off
#define BSRMV_CONFIG(T, I, J, A, X, Y)                               \
    {bsrmv_tuple(T, I, J, A, X, Y),                                  \
     bsrmv_template<typename rocsparse::datatype_traits<T>::type_t,  \
                    typename rocsparse::indextype_traits<I>::type_t, \
                    typename rocsparse::indextype_traits<J>::type_t, \
                    typename rocsparse::datatype_traits<A>::type_t,  \
                    typename rocsparse::datatype_traits<X>::type_t,  \
                    typename rocsparse::datatype_traits<Y>::type_t>}
    // clang-format on

    static const std::map<bsrmv_tuple, bsrmv_t> s_bsrmv_dispatch{
        {BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         BSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         BSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         BSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         BSRMV_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),

         BSRMV_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),

         BSRMV_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_f32_r),

         BSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         BSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         BSRMV_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         BSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         BSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         BSRMV_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         BSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         BSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         BSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         BSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         BSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         BSRMV_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c)

        }};

    static rocsparse_status bsrmv_find(bsrmv_t*            function_,
                                       rocsparse_datatype  t_type_,
                                       rocsparse_indextype i_type_,
                                       rocsparse_indextype j_type_,
                                       rocsparse_datatype  a_type_,
                                       rocsparse_datatype  x_type_,
                                       rocsparse_datatype  y_type_)
    {
        const auto& it = rocsparse::s_bsrmv_dispatch.find(
            rocsparse::bsrmv_tuple(t_type_, i_type_, j_type_, a_type_, x_type_, y_type_));

        if(it != rocsparse::s_bsrmv_dispatch.end())
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
                      << ", j_type: " << rocsparse::enum_utils::to_string(j_type_) << std::endl
                      << ", a_type: " << rocsparse::enum_utils::to_string(a_type_) << std::endl
                      << ", x_type: " << rocsparse::enum_utils::to_string(x_type_) << std::endl
                      << ", y_type: " << rocsparse::enum_utils::to_string(y_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_bsrmv_dispatch)
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
                          << "t_type: " << rocsparse::enum_utils::to_string(t_type) << std::endl
                          << ", i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl
                          << ", j_type: " << rocsparse::enum_utils::to_string(j_type) << std::endl
                          << ", a_type: " << rocsparse::enum_utils::to_string(a_type) << std::endl
                          << ", x_type: " << rocsparse::enum_utils::to_string(x_type) << std::endl
                          << ", y_type: " << rocsparse::enum_utils::to_string(y_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "t_type: " << rocsparse::enum_utils::to_string(t_type_)
                 << ", i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", j_type: " << rocsparse::enum_utils::to_string(j_type_)
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

rocsparse_status rocsparse::bsrmv(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans,
                                  int64_t                   mb,
                                  int64_t                   nb,
                                  int64_t                   nnzb,
                                  rocsparse_datatype        alpha_device_host_datatype,
                                  const void*               alpha_device_host,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_datatype        bsr_val_datatype,
                                  const void*               bsr_val,
                                  rocsparse_indextype       bsr_row_ptr_indextype,
                                  const void*               bsr_row_ptr,
                                  rocsparse_indextype       bsr_col_ind_indextype,
                                  const void*               bsr_col_ind,
                                  int64_t                   block_dim,
                                  rocsparse_bsrmv_info      bsrmv_info,
                                  rocsparse_datatype        x_datatype,
                                  const void*               x,
                                  rocsparse_datatype        beta_device_host_datatype,
                                  const void*               beta_device_host,
                                  rocsparse_datatype        y_datatype,
                                  void*                     y)
{

    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::bsrmv_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_find(&f,
                                                    alpha_device_host_datatype,
                                                    bsr_row_ptr_indextype,
                                                    bsr_col_ind_indextype,
                                                    bsr_val_datatype,
                                                    x_datatype,
                                                    y_datatype));

    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                dir,
                                trans,
                                mb,
                                nb,
                                nnzb,
                                alpha_device_host,
                                descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                block_dim,
                                bsrmv_info,
                                x,
                                beta_device_host,
                                y));

    return rocsparse_status_success;
}
