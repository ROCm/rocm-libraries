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

#include "rocsparse_csrsv.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"
#include <map>
#include <sstream>

namespace rocsparse
{
    typedef rocsparse_status (*csrsv_solve_t)(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int64_t                   m,
                                              int64_t                   nnz,
                                              const void*               alpha,
                                              const rocsparse_mat_descr descr,
                                              const void*               csr_val,
                                              const void*               csr_row_ptr,
                                              const void*               csr_col_ind,
                                              rocsparse_mat_info        info,
                                              const void*               x,
                                              int64_t                   x_inc,
                                              void*                     y,
                                              rocsparse_solve_policy    policy,
                                              rocsparse_csrsv_info      csrsv_info,
                                              void*                     temp_buffer);

    using csrsv_solve_tuple
        = std::tuple<rocsparse_indextype, rocsparse_indextype, rocsparse_datatype>;

#define CSRSV_SOLVE_CONFIG(I_, J_, T_)                                             \
    {                                                                              \
        csrsv_solve_tuple(I_, J_, T_),                                             \
            csrsv_solve_template<typename rocsparse::indextype_traits<I_>::type_t, \
                                 typename rocsparse::indextype_traits<J_>::type_t, \
                                 typename rocsparse::datatype_traits<T_>::type_t>  \
    }

    static const std::map<csrsv_solve_tuple, csrsv_solve_t> s_csrsv_solve_dispatch{
        {CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f32_r),
         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f64_r),
         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f32_c),
         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f64_c),

         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f32_r),
         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f64_r),
         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f32_c),
         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f64_c),

         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f32_r),
         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f64_r),
         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f32_c),
         CSRSV_SOLVE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f64_c)}};

    static rocsparse_status csrsv_solve_find(csrsv_solve_t*      function_,
                                             rocsparse_indextype i_type_,
                                             rocsparse_indextype j_type_,
                                             rocsparse_datatype  t_type_)
    {
        const auto& it = rocsparse::s_csrsv_solve_dispatch.find(
            rocsparse::csrsv_solve_tuple(i_type_, j_type_, t_type_));

        if(it != rocsparse::s_csrsv_solve_dispatch.end())
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
                      << ", t_type: " << rocsparse::enum_utils::to_string(t_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_csrsv_solve_dispatch)
            {
                const auto& t      = p.first;
                const auto  i_type = std::get<0>(t);
                const auto  j_type = std::get<1>(t);
                const auto  t_type = std::get<2>(t);
                std::cout << std::endl
                          << std::endl
                          << "i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl
                          << ", j_type: " << rocsparse::enum_utils::to_string(j_type) << std::endl
                          << ", t_type: " << rocsparse::enum_utils::to_string(t_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", j_type: " << rocsparse::enum_utils::to_string(j_type_)
                 << ", t_type: " << rocsparse::enum_utils::to_string(t_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::csrsv_solve(rocsparse_handle          handle,
                                        rocsparse_operation       trans,
                                        int64_t                   m,
                                        int64_t                   nnz,
                                        rocsparse_datatype        alpha_datatype,
                                        const void*               alpha,
                                        const rocsparse_mat_descr descr,
                                        rocsparse_datatype        csr_val_datatype,
                                        const void*               csr_val,
                                        rocsparse_indextype       csr_row_ptr_indextype,
                                        const void*               csr_row_ptr,
                                        rocsparse_indextype       csr_col_ind_indextype,
                                        const void*               csr_col_ind,
                                        rocsparse_mat_info        info,
                                        rocsparse_datatype        x_val_datatype,
                                        const void*               x,
                                        int64_t                   x_inc,
                                        rocsparse_datatype        y_val_datatype,
                                        void*                     y,
                                        rocsparse_solve_policy    policy,
                                        rocsparse_csrsv_info      csrsv_info,
                                        void*                     temp_buffer)
{
    rocsparse::csrsv_solve_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve_find(
        &f, csr_row_ptr_indextype, csr_col_ind_indextype, csr_val_datatype));
    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                trans,
                                m,
                                nnz,
                                alpha,
                                descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                info,
                                x,
                                x_inc,
                                y,
                                policy,
                                csrsv_info,
                                temp_buffer));
    return rocsparse_status_success;
}
