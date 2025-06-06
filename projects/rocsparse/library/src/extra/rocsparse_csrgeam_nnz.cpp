/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <map>
#include <sstream>

#include "rocsparse_csrgeam.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    typedef rocsparse_status (*csrgeam_nnz_t)(rocsparse_handle             handle,
                                              const rocsparse_spgeam_descr descr,
                                              rocsparse_operation          trans_A,
                                              rocsparse_operation          trans_B,
                                              int64_t                      m,
                                              int64_t                      n,
                                              const rocsparse_mat_descr    descr_A,
                                              int64_t                      nnz_A,
                                              const void*                  csr_row_ptr_A,
                                              const void*                  csr_col_ind_A,
                                              const rocsparse_mat_descr    descr_B,
                                              int64_t                      nnz_B,
                                              const void*                  csr_row_ptr_B,
                                              const void*                  csr_col_ind_B,
                                              const rocsparse_mat_descr    descr_C,
                                              void*                        csr_row_ptr_C,
                                              void*                        nnz_C,
                                              void*                        temp_buffer,
                                              bool                         called_from_spgeam);

    using csrgeam_nnz_tuple = std::tuple<rocsparse_indextype, rocsparse_indextype>;

    // clang-format off
#define CSRGEAM_NNZ_CONFIG(I, J)                                           \
    {csrgeam_nnz_tuple(I, J),                                              \
     csrgeam_nnz_template<typename rocsparse::indextype_traits<I>::type_t, \
                          typename rocsparse::indextype_traits<J>::type_t>}
    // clang-format on

    static const std::map<csrgeam_nnz_tuple, csrgeam_nnz_t> s_csrgeam_nnz_dispatch{
        {CSRGEAM_NNZ_CONFIG(rocsparse_indextype_i32, rocsparse_indextype_i32),
         CSRGEAM_NNZ_CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i32),
         CSRGEAM_NNZ_CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i64)}};

    static rocsparse_status csrgeam_nnz_find(csrgeam_nnz_t*      function_,
                                             rocsparse_indextype i_type_,
                                             rocsparse_indextype j_type_)
    {
        const auto& it = rocsparse::s_csrgeam_nnz_dispatch.find(
            rocsparse::csrgeam_nnz_tuple(i_type_, j_type_));

        if(it != rocsparse::s_csrgeam_nnz_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << ", i_type: " << rocsparse::enum_utils::to_string(i_type_) << std::endl
                      << ", j_type: " << rocsparse::enum_utils::to_string(j_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_csrgeam_nnz_dispatch)
            {
                const auto& t      = p.first;
                const auto  i_type = std::get<0>(t);
                const auto  j_type = std::get<1>(t);
                std::cout << std::endl
                          << std::endl
                          << ", i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl
                          << ", j_type: " << rocsparse::enum_utils::to_string(j_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << ", i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", j_type: " << rocsparse::enum_utils::to_string(j_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::csrgeam_nnz(rocsparse_handle             handle,
                                        const rocsparse_spgeam_descr descr,
                                        rocsparse_operation          trans_A,
                                        rocsparse_operation          trans_B,
                                        int64_t                      m,
                                        int64_t                      n,
                                        const rocsparse_mat_descr    descr_A,
                                        int64_t                      nnz_A,
                                        rocsparse_indextype          csr_row_ptr_A_indextype,
                                        const void*                  csr_row_ptr_A,
                                        rocsparse_indextype          csr_col_ind_A_indextype,
                                        const void*                  csr_col_ind_A,
                                        const rocsparse_mat_descr    descr_B,
                                        int64_t                      nnz_B,
                                        rocsparse_indextype          csr_row_ptr_B_indextype,
                                        const void*                  csr_row_ptr_B,
                                        rocsparse_indextype          csr_col_ind_B_indextype,
                                        const void*                  csr_col_ind_B,
                                        const rocsparse_mat_descr    descr_C,
                                        rocsparse_indextype          csr_row_ptr_C_indextype,
                                        void*                        csr_row_ptr_C,
                                        void*                        nnz_C,
                                        void*                        temp_buffer,
                                        bool                         called_from_spgeam)
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::csrgeam_nnz_t f;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csrgeam_nnz_find(&f, csr_row_ptr_A_indextype, csr_col_ind_A_indextype));

    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                descr,
                                trans_A,
                                trans_B,
                                m,
                                n,
                                descr_A,
                                nnz_A,
                                csr_row_ptr_A,
                                csr_col_ind_A,
                                descr_B,
                                nnz_B,
                                csr_row_ptr_B,
                                csr_col_ind_B,
                                descr_C,
                                csr_row_ptr_C,
                                nnz_C,
                                temp_buffer,
                                called_from_spgeam));

    return rocsparse_status_success;
}
