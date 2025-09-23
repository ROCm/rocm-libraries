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

#include "rocsparse_bsrmm.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

#include <map>
#include <sstream>

namespace rocsparse
{
    typedef rocsparse_status (*bsrmm_t)(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        rocsparse_bsrmm_alg       alg,
                                        int64_t                   mb,
                                        int64_t                   n,
                                        int64_t                   kb,
                                        int64_t                   nnzb,
                                        int64_t                   batch_count_A,
                                        int64_t                   offsets_batch_stride_A,
                                        int64_t                   columns_values_batch_stride_A,
                                        const void*               alpha,
                                        const rocsparse_mat_descr descr,
                                        const void*               bsr_val,
                                        const void*               bsr_row_ptr,
                                        const void*               bsr_col_ind,
                                        int64_t                   block_dim,
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
                                        rocsparse_order           order_C);

    using bsrmm_tuple = std::tuple<rocsparse_datatype,
                                   rocsparse_indextype,
                                   rocsparse_indextype,
                                   rocsparse_datatype,
                                   rocsparse_datatype,
                                   rocsparse_datatype>;

    // clang-format off
#define BSRMM_CONFIG(T, I, J, A, B, C)                                      \
    {                                                                       \
        bsrmm_tuple(T, I, J, A, B, C),                                      \
            bsrmm_template<typename rocsparse::datatype_traits<T>::type_t,  \
                           typename rocsparse::indextype_traits<I>::type_t, \
                           typename rocsparse::indextype_traits<J>::type_t, \
                           typename rocsparse::datatype_traits<A>::type_t,  \
                           typename rocsparse::datatype_traits<B>::type_t,  \
                           typename rocsparse::datatype_traits<C>::type_t>  \
    }
    // clang-format on

    static const std::map<bsrmm_tuple, bsrmm_t> s_bsrmm_dispatch{
        {// Uniform precision
         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),
         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),
         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r,
                      rocsparse_datatype_f32_r),

         BSRMM_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),
         BSRMM_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),
         BSRMM_CONFIG(rocsparse_datatype_f64_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r,
                      rocsparse_datatype_f64_r),

         BSRMM_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),
         BSRMM_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),
         BSRMM_CONFIG(rocsparse_datatype_f32_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c,
                      rocsparse_datatype_f32_c),

         BSRMM_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),
         BSRMM_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),
         BSRMM_CONFIG(rocsparse_datatype_f64_c,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c,
                      rocsparse_datatype_f64_c),

         // Mixed precisions
         BSRMM_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),
         BSRMM_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),
         BSRMM_CONFIG(rocsparse_datatype_i32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i32_r),

         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),
         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),
         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_i8_r,
                      rocsparse_datatype_f32_r),

         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),
         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),
         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f16_r,
                      rocsparse_datatype_f32_r),

         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i32,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_f32_r),
         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i32,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_f32_r),
         BSRMM_CONFIG(rocsparse_datatype_f32_r,
                      rocsparse_indextype_i64,
                      rocsparse_indextype_i64,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_bf16_r,
                      rocsparse_datatype_f32_r)}};

    static rocsparse_status bsrmm_find(bsrmm_t*            function_,
                                       rocsparse_datatype  t_type_,
                                       rocsparse_indextype i_type_,
                                       rocsparse_indextype j_type_,
                                       rocsparse_datatype  a_type_,
                                       rocsparse_datatype  b_type_,
                                       rocsparse_datatype  c_type_)
    {
        const auto& it = rocsparse::s_bsrmm_dispatch.find(
            rocsparse::bsrmm_tuple(t_type_, i_type_, j_type_, a_type_, b_type_, c_type_));

        if(it != rocsparse::s_bsrmm_dispatch.end())
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
                      << ", x_type: " << rocsparse::enum_utils::to_string(b_type_) << std::endl
                      << ", y_type: " << rocsparse::enum_utils::to_string(c_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_bsrmm_dispatch)
            {
                const auto& t      = p.first;
                const auto  t_type = std::get<0>(t);
                const auto  i_type = std::get<1>(t);
                const auto  j_type = std::get<2>(t);
                const auto  a_type = std::get<3>(t);
                const auto  b_type = std::get<4>(t);
                const auto  c_type = std::get<5>(t);
                std::cout << std::endl
                          << std::endl
                          << "t_type: " << rocsparse::enum_utils::to_string(t_type) << std::endl
                          << ", i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl
                          << ", j_type: " << rocsparse::enum_utils::to_string(j_type) << std::endl
                          << ", a_type: " << rocsparse::enum_utils::to_string(a_type) << std::endl
                          << ", b_type: " << rocsparse::enum_utils::to_string(b_type) << std::endl
                          << ", c_type: " << rocsparse::enum_utils::to_string(c_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "t_type: " << rocsparse::enum_utils::to_string(t_type_)
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

rocsparse_status rocsparse::bsrmm(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_bsrmm_alg       alg,
                                  int64_t                   mb,
                                  int64_t                   n,
                                  int64_t                   kb,
                                  int64_t                   nnzb,
                                  int64_t                   batch_count_A,
                                  int64_t                   offsets_batch_stride_A,
                                  int64_t                   columns_values_batch_stride_A,
                                  rocsparse_datatype        alpha_datatype,
                                  const void*               alpha,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_datatype        bsr_val_datatype,
                                  const void*               bsr_val,
                                  rocsparse_indextype       bsr_row_ptr_indextype,
                                  const void*               bsr_row_ptr,
                                  rocsparse_indextype       bsr_col_ind_indextype,
                                  const void*               bsr_col_ind,
                                  int64_t                   block_dim,
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
                                  rocsparse_order           order_C)
{

    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::bsrmm_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_find(&f,
                                                    alpha_datatype,
                                                    bsr_row_ptr_indextype,
                                                    bsr_col_ind_indextype,
                                                    bsr_val_datatype,
                                                    dense_B_datatype,
                                                    dense_C_datatype));

    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                dir,
                                trans_A,
                                trans_B,
                                alg,
                                mb,
                                n,
                                kb,
                                nnzb,
                                batch_count_A,
                                offsets_batch_stride_A,
                                columns_values_batch_stride_A,
                                alpha,
                                descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                block_dim,
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
                                order_C));

    return rocsparse_status_success;
}
