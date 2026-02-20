/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status csritsv_buffer_size_template(rocsparse_handle          handle,
                                                  rocsparse_operation       trans,
                                                  J                         m,
                                                  I                         nnz,
                                                  const rocsparse_mat_descr descr,
                                                  const T*                  csr_val,
                                                  const I*                  csr_row_ptr,
                                                  const J*                  csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size);

    template <typename I, typename J, typename T>
    rocsparse_status csritsv_analysis_template(rocsparse_handle          handle,
                                               rocsparse_operation       trans,
                                               J                         m,
                                               I                         nnz,
                                               const rocsparse_mat_descr descr,
                                               const T*                  csr_val,
                                               const I*                  csr_row_ptr,
                                               const J*                  csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer);

    template <typename I, typename J, typename T>
    rocsparse_status csritsv_solve_ex_template(rocsparse_handle          handle,
                                               rocsparse_int*            host_nmaxiter,
                                               rocsparse_int             host_nfreeiter,
                                               const floating_data_t<T>* host_tol,
                                               floating_data_t<T>*       host_history,
                                               rocsparse_operation       trans,
                                               J                         m,
                                               I                         nnz,
                                               const T*                  alpha,
                                               const rocsparse_mat_descr descr,
                                               const T*                  csr_val,
                                               const I*                  csr_row_ptr,
                                               const J*                  csr_col_ind,
                                               rocsparse_mat_info        info,
                                               const T*                  x,
                                               T*                        y,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer);
}
