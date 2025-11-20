/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "rocsparse_handle.hpp"

namespace rocsparse
{
    rocsparse_status csrsv_zero_pivot(rocsparse_handle     handle,
                                      rocsparse_csrsv_info info,
                                      rocsparse_indextype  indextype,
                                      void*                position);

    template <typename I, typename J, typename T>
    rocsparse_status csrsv_buffer_size_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans,
                                                int64_t                   m,
                                                int64_t                   nnz,
                                                const rocsparse_mat_descr descr,
                                                const void*               csr_val,
                                                const void*               csr_row_ptr,
                                                const void*               csr_col_ind,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

    template <typename I, typename J, typename T>
    rocsparse_status csrsv_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             int64_t                   m,
                                             int64_t                   nnz,
                                             const rocsparse_mat_descr descr,
                                             const void*               csr_val,
                                             const void*               csr_row_ptr,
                                             const void*               csr_col_ind,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             rocsparse_csrsv_info*     p_csrsv_info,
                                             void*                     temp_buffer);

    template <typename I, typename J, typename T>
    rocsparse_status csrsv_solve_template(rocsparse_handle          handle,
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

    rocsparse_status csrsv_buffer_size(rocsparse_handle          handle,
                                       rocsparse_operation       trans,
                                       int64_t                   m,
                                       int64_t                   nnz,
                                       const rocsparse_mat_descr descr,
                                       rocsparse_datatype        csr_val_datatype,
                                       const void*               csr_val,
                                       rocsparse_indextype       csr_row_ptr_indextype,
                                       const void*               csr_row_ptr,
                                       rocsparse_indextype       csr_col_ind_indextype,
                                       const void*               csr_col_ind,
                                       rocsparse_mat_info        info,
                                       size_t*                   buffer_size);

    rocsparse_status csrsv_analysis(rocsparse_handle          handle,
                                    rocsparse_operation       trans,
                                    int64_t                   m,
                                    int64_t                   nnz,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_datatype        csr_val_datatype,
                                    const void*               csr_val,
                                    rocsparse_indextype       csr_row_ptr_indextype,
                                    const void*               csr_row_ptr,
                                    rocsparse_indextype       csr_col_ind_indextype,
                                    const void*               csr_col_ind,
                                    rocsparse_mat_info        info,
                                    rocsparse_analysis_policy analysis,
                                    rocsparse_solve_policy    solve,
                                    rocsparse_csrsv_info*     p_csrsv_info,
                                    void*                     temp_buffer);

    rocsparse_status csrsv_solve(rocsparse_handle          handle,
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
                                 void*                     temp_buffer);

}
