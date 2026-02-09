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

#include "rocsparse_coomv.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"
#include <map>
#include <sstream>

namespace rocsparse
{
    typedef rocsparse_status (*coomv_analysis_t)(rocsparse_handle,
                                                 rocsparse_operation,
                                                 rocsparse_coomv_alg,
                                                 int64_t,
                                                 int64_t,
                                                 int64_t,
                                                 const rocsparse_mat_descr,
                                                 const void*,
                                                 const void*,
                                                 const void*);

    using coomv_analysis_tuple = std::tuple<rocsparse_indextype, rocsparse_datatype>;

    // clang-format off
#define COOMV_ANALYSIS_CONFIG(I_, A_)                                                 \
    {                                                                                 \
        coomv_analysis_tuple(I_, A_),                                                 \
            coomv_analysis_template<typename rocsparse::indextype_traits<I_>::type_t, \
                                    typename rocsparse::datatype_traits<A_>::type_t>  \
    }
    // clang-format on

    static const std::map<coomv_analysis_tuple, coomv_analysis_t> s_coomv_analysis_dispatch{
        {COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_i8_r),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f16_r),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_bf16_r),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f32_r),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f64_r),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f32_c),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f64_c),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_i8_r),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f16_r),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_bf16_r),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f32_r),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f64_r),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f32_c),
         COOMV_ANALYSIS_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f64_c)}};

    static rocsparse_status coomv_analysis_find(coomv_analysis_t*   function_,
                                                rocsparse_indextype i_type_,
                                                rocsparse_datatype  a_type_)
    {
        const auto& it = rocsparse::s_coomv_analysis_dispatch.find(
            rocsparse::coomv_analysis_tuple(i_type_, a_type_));

        if(it != rocsparse::s_coomv_analysis_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "i_type: " << rocsparse::enum_utils::to_string(i_type_) << std::endl
                      << ", a_type: " << rocsparse::enum_utils::to_string(a_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_coomv_analysis_dispatch)
            {
                const auto& t      = p.first;
                const auto  i_type = std::get<0>(t);
                const auto  a_type = std::get<1>(t);
                std::cout << std::endl
                          << std::endl
                          << "i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl
                          << ", a_type: " << rocsparse::enum_utils::to_string(a_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", a_type: " << rocsparse::enum_utils::to_string(a_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::coomv_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse_coomv_alg       alg,
                                           int64_t                   m,
                                           int64_t                   n,
                                           int64_t                   nnz,
                                           const rocsparse_mat_descr descr,
                                           rocsparse_datatype        coo_val_datatype,
                                           const void*               coo_val,
                                           rocsparse_indextype       coo_row_indextype,
                                           const void*               coo_row_ind,
                                           rocsparse_indextype       coo_col_indextype,
                                           const void*               coo_col_ind)
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::coomv_analysis_t f;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::coomv_analysis_find(&f, coo_row_indextype, coo_val_datatype));
    RETURN_IF_ROCSPARSE_ERROR(
        f(handle, trans, alg, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind));
    return rocsparse_status_success;
}
