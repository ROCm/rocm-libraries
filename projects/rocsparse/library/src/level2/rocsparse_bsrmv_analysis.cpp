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
    typedef rocsparse_status (*bsrmv_analysis_t)(rocsparse_handle,
                                                 rocsparse_direction,
                                                 rocsparse_operation,
                                                 int64_t,
                                                 int64_t,
                                                 int64_t,
                                                 const rocsparse_mat_descr,
                                                 const void*,
                                                 const void*,
                                                 const void*,
                                                 int64_t,
                                                 rocsparse_bsrmv_info*);

    using bsrmv_analysis_tuple
        = std::tuple<rocsparse_indextype, rocsparse_indextype, rocsparse_datatype>;

    // clang-format off
#define BSRMV_ANALYSIS_CONFIG(I_, J_, A_)                                             \
    {                                                                                 \
        bsrmv_analysis_tuple(I_, J_, A_),                                             \
            bsrmv_analysis_template<typename rocsparse::indextype_traits<I_>::type_t, \
                                    typename rocsparse::indextype_traits<J_>::type_t, \
                                    typename rocsparse::datatype_traits<A_>::type_t>  \
    }
    // clang-format on

    static const std::map<bsrmv_analysis_tuple, bsrmv_analysis_t> s_bsrmv_analysis_dispatch{
        {BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f32_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f64_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f32_c),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f64_c),

         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f32_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f64_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f32_c),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f64_c),

         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f32_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f64_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f32_c),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f64_c),

         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_i8_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_i8_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_i8_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f16_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f16_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f16_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_bf16_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_bf16_r),
         BSRMV_ANALYSIS_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_bf16_r)}};

    static rocsparse_status bsrmv_analysis_find(bsrmv_analysis_t*   function_,
                                                rocsparse_indextype i_type_,
                                                rocsparse_indextype j_type_,
                                                rocsparse_datatype  a_type_)
    {
        const auto& it = rocsparse::s_bsrmv_analysis_dispatch.find(
            rocsparse::bsrmv_analysis_tuple(i_type_, j_type_, a_type_));

        if(it != rocsparse::s_bsrmv_analysis_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "i_type: " << rocsparse::enum_utils::to_string(i_type_) << std::endl
                      << ", j_type: " << rocsparse::enum_utils::to_string(j_type_) << std::endl
                      << ", a_type: " << rocsparse::enum_utils::to_string(a_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_bsrmv_analysis_dispatch)
            {
                const auto& t      = p.first;
                const auto  i_type = std::get<0>(t);
                const auto  j_type = std::get<1>(t);
                const auto  a_type = std::get<2>(t);
                std::cout << std::endl
                          << std::endl
                          << "i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl
                          << ", j_type: " << rocsparse::enum_utils::to_string(j_type) << std::endl
                          << ", a_type: " << rocsparse::enum_utils::to_string(a_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", j_type: " << rocsparse::enum_utils::to_string(j_type_)
                 << ", a_type: " << rocsparse::enum_utils::to_string(a_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::bsrmv_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans,
                                           int64_t                   mb,
                                           int64_t                   nb,
                                           int64_t                   nnzb,
                                           const rocsparse_mat_descr descr,
                                           rocsparse_datatype        bsr_val_datatype,
                                           const void*               bsr_val,
                                           rocsparse_indextype       bsr_row_ptr_indextype,
                                           const void*               bsr_row_ptr,
                                           rocsparse_indextype       bsr_col_ind_indextype,
                                           const void*               bsr_col_ind,
                                           int64_t                   block_dim,
                                           rocsparse_bsrmv_info*     p_bsrmv_info)
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::bsrmv_analysis_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_analysis_find(
        &f, bsr_row_ptr_indextype, bsr_col_ind_indextype, bsr_val_datatype));
    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                dir,
                                trans,
                                mb,
                                nb,
                                nnzb,
                                descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                block_dim,
                                p_bsrmv_info));
    return rocsparse_status_success;
}
