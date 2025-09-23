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

#include "rocsparse_coomm.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

#include <map>
#include <sstream>

namespace rocsparse
{
    typedef rocsparse_status (*coomm_t)(rocsparse_handle          handle,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        rocsparse_coomm_alg       alg,
                                        int64_t                   m,
                                        int64_t                   n,
                                        int64_t                   k,
                                        int64_t                   nnz,
                                        int64_t                   batch_count_A,
                                        int64_t                   batch_stride_A,
                                        const void*               alpha,
                                        const rocsparse_mat_descr descr,
                                        const void*               coo_val,
                                        const void*               coo_row_ind,
                                        const void*               coo_col_ind,
                                        const void*               dense_B,
                                        int64_t                   ldb,
                                        int64_t                   batch_count_B,
                                        int64_t                   batch_stride_B,
                                        rocsparse_order           order_B,
                                        const void*               beta,
                                        void*                     dense_C,
                                        int64_t                   ldc,
                                        int64_t                   batch_count_C,
                                        int64_t                   batch_stride_C,
                                        rocsparse_order           order_C,
                                        void*                     temp_buffer);

    using coomm_tuple = std::tuple<rocsparse_datatype,
                                   rocsparse_indextype,
                                   rocsparse_datatype,
                                   rocsparse_datatype,
                                   rocsparse_datatype>;

    // clang-format off
#define COOMM_CONFIG(T, I, A, B, C)                                      \
    {                                                                       \
        coomm_tuple(T, I, A, B, C),                                      \
            coomm_template<typename rocsparse::datatype_traits<T>::type_t,  \
                           typename rocsparse::indextype_traits<I>::type_t, \
                           typename rocsparse::datatype_traits<A>::type_t,  \
                           typename rocsparse::datatype_traits<B>::type_t,  \
                           typename rocsparse::datatype_traits<C>::type_t>  \
    }
    // clang-format on

    static const std::map<coomm_tuple, coomm_t> s_coomm_dispatch{
        {// Uniform precision
         COOMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),
         COOMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),

         COOMM_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),
         COOMM_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         COOMM_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),
         COOMM_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         COOMM_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),
         COOMM_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         // Mixed precisions
         COOMM_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),
         COOMM_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),

         COOMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),
         COOMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),

         COOMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),
         COOMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),

         COOMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_f32_r),
         COOMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_f32_r)}};

    static rocsparse_status coomm_find(coomm_t*            function_,
                                       rocsparse_datatype  t_type_,
                                       rocsparse_indextype i_type_,
                                       rocsparse_datatype  a_type_,
                                       rocsparse_datatype  b_type_,
                                       rocsparse_datatype  c_type_)
    {
        const auto& it = rocsparse::s_coomm_dispatch.find(
            rocsparse::coomm_tuple(t_type_, i_type_, a_type_, b_type_, c_type_));

        if(it != rocsparse::s_coomm_dispatch.end())
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
                      << ", x_type: " << rocsparse::enum_utils::to_string(b_type_) << std::endl
                      << ", y_type: " << rocsparse::enum_utils::to_string(c_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_coomm_dispatch)
            {
                const auto& t      = p.first;
                const auto  t_type = std::get<0>(t);
                const auto  i_type = std::get<1>(t);
                const auto  a_type = std::get<2>(t);
                const auto  b_type = std::get<3>(t);
                const auto  c_type = std::get<4>(t);
                std::cout << std::endl
                          << std::endl
                          << "t_type: " << rocsparse::enum_utils::to_string(t_type) << std::endl
                          << ", i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl
                          << ", a_type: " << rocsparse::enum_utils::to_string(a_type) << std::endl
                          << ", b_type: " << rocsparse::enum_utils::to_string(b_type) << std::endl
                          << ", c_type: " << rocsparse::enum_utils::to_string(c_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "t_type: " << rocsparse::enum_utils::to_string(t_type_)
                 << ", i_type: " << rocsparse::enum_utils::to_string(i_type_)
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

rocsparse_status rocsparse::coomm(rocsparse_handle          handle,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_coomm_alg       alg,
                                  int64_t                   m,
                                  int64_t                   n,
                                  int64_t                   k,
                                  int64_t                   nnz,
                                  int64_t                   batch_count_A,
                                  int64_t                   batch_stride_A,
                                  rocsparse_datatype        alpha_datatype,
                                  const void*               alpha,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_datatype        coo_val_datatype,
                                  const void*               coo_val,
                                  rocsparse_indextype       coo_row_ind_indextype,
                                  const void*               coo_row_ind,
                                  rocsparse_indextype       coo_col_ind_indextype,
                                  const void*               coo_col_ind,
                                  rocsparse_datatype        dense_B_datatype,
                                  const void*               dense_B,
                                  int64_t                   ldb,
                                  int64_t                   batch_count_B,
                                  int64_t                   batch_stride_B,
                                  rocsparse_order           order_B,
                                  rocsparse_datatype        beta_datatype,
                                  const void*               beta,
                                  rocsparse_datatype        dense_C_datatype,
                                  void*                     dense_C,
                                  int64_t                   ldc,
                                  int64_t                   batch_count_C,
                                  int64_t                   batch_stride_C,
                                  rocsparse_order           order_C,
                                  void*                     temp_buffer)
{

    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::coomm_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_find(&f,
                                                    alpha_datatype,
                                                    coo_row_ind_indextype,
                                                    coo_val_datatype,
                                                    dense_B_datatype,
                                                    dense_C_datatype));

    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                trans_A,
                                trans_B,
                                alg,
                                m,
                                n,
                                k,
                                nnz,
                                batch_count_A,
                                batch_stride_A,
                                alpha,
                                descr,
                                coo_val,
                                coo_row_ind,
                                coo_col_ind,
                                dense_B,
                                ldb,
                                batch_count_B,
                                batch_stride_B,
                                order_B,
                                beta,
                                dense_C,
                                ldc,
                                batch_count_C,
                                batch_stride_C,
                                order_C,
                                temp_buffer));

    return rocsparse_status_success;
}
