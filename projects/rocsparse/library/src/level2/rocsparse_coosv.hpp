/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <typename I, typename T>
    rocsparse_status coosv_buffer_size_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans,
                                                int64_t                   m,
                                                int64_t                   nnz,
                                                const rocsparse_mat_descr descr,
                                                const void*               coo_val,
                                                const void*               coo_row_ind,
                                                const void*               coo_col_ind,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

    template <typename I, typename T>
    rocsparse_status coosv_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             int64_t                   m,
                                             int64_t                   nnz,
                                             const rocsparse_mat_descr descr,
                                             const void*               coo_val,
                                             const void*               coo_row_ind,
                                             const void*               coo_col_ind,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             rocsparse_csrsv_info*     p_csrsv_info,
                                             void*                     temp_buffer);

    template <typename I, typename T>
    rocsparse_status coosv_solve_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          int64_t                   m,
                                          int64_t                   nnz,
                                          const void*               alpha,
                                          const rocsparse_mat_descr descr,
                                          const void*               coo_val,
                                          const void*               coo_row_ind,
                                          const void*               coo_col_ind,
                                          rocsparse_mat_info        info,
                                          const void*               x,
                                          void*                     y,
                                          rocsparse_solve_policy    policy,
                                          rocsparse_csrsv_info      csrsv_info,
                                          void*                     temp_buffer);

    rocsparse_status coosv_buffer_size(rocsparse_handle          handle,
                                       rocsparse_operation       trans,
                                       int64_t                   m,
                                       int64_t                   nnz,
                                       const rocsparse_mat_descr descr,
                                       rocsparse_datatype        coo_val_datatype,
                                       const void*               coo_val,
                                       rocsparse_indextype       coo_row_ind_indextype,
                                       const void*               coo_row_ind,
                                       rocsparse_indextype       coo_col_ind_indextype,
                                       const void*               coo_col_ind,
                                       rocsparse_mat_info        info,
                                       size_t*                   buffer_size);

    rocsparse_status coosv_analysis(rocsparse_handle          handle,
                                    rocsparse_operation       trans,
                                    int64_t                   m,
                                    int64_t                   nnz,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_datatype        coo_val_datatype,
                                    const void*               coo_val,
                                    rocsparse_indextype       coo_row_ind_indextype,
                                    const void*               coo_row_ind,
                                    rocsparse_indextype       coo_col_ind_indextype,
                                    const void*               coo_col_ind,
                                    rocsparse_mat_info        info,
                                    rocsparse_analysis_policy analysis,
                                    rocsparse_solve_policy    solve,
                                    rocsparse_csrsv_info*     p_csrsv_info,
                                    void*                     temp_buffer);

    rocsparse_status coosv_solve(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 int64_t                   m,
                                 int64_t                   nnz,
                                 rocsparse_datatype        alpha_datatype,
                                 const void*               alpha,
                                 const rocsparse_mat_descr descr,
                                 rocsparse_datatype        coo_val_datatype,
                                 const void*               coo_val,
                                 rocsparse_indextype       coo_row_ind_indextype,
                                 const void*               coo_row_ind,
                                 rocsparse_indextype       coo_col_ind_indextype,
                                 const void*               coo_col_ind,
                                 rocsparse_mat_info        info,
                                 rocsparse_datatype        x_datatype,
                                 const void*               x,
                                 rocsparse_datatype        y_datatype,
                                 void*                     y,
                                 rocsparse_solve_policy    policy,
                                 rocsparse_csrsv_info      csrsv_info,
                                 void*                     temp_buffer);

}
