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

#include "rocsparse_csrmv.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename A>
    rocsparse_status cscmv_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse::csrmv_alg      alg,
                                             int64_t                   m,
                                             int64_t                   n,
                                             int64_t                   nnz,
                                             const rocsparse_mat_descr descr,
                                             const void*               csc_val,
                                             const void*               csc_col_ptr,
                                             const void*               csc_row_ind,
                                             rocsparse_csrmv_info*     p_csrmv_info);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status cscmv_template(rocsparse_handle          handle,
                                    rocsparse_operation       trans,
                                    rocsparse::csrmv_alg      alg,
                                    int64_t                   m,
                                    int64_t                   n,
                                    int64_t                   nnz,
                                    const void*               alpha,
                                    const rocsparse_mat_descr descr,
                                    const void*               csc_val,
                                    const void*               csc_col_ptr,
                                    const void*               csc_row_ind,
                                    rocsparse_csrmv_info      csrmv_info,
                                    const void*               x,
                                    const void*               beta,
                                    void*                     y,
                                    bool                      fallback_algorithm);

    rocsparse_status cscmv_analysis(rocsparse_handle          handle,
                                    rocsparse_operation       trans,
                                    rocsparse::csrmv_alg      alg,
                                    int64_t                   m,
                                    int64_t                   n,
                                    int64_t                   nnz,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_datatype        csc_val_datatype,
                                    const void*               csc_val,
                                    rocsparse_indextype       csc_row_ptr_indextype,
                                    const void*               csc_row_ptr,
                                    rocsparse_indextype       csc_col_ind_indextype,
                                    const void*               csc_col_ind,
                                    rocsparse_csrmv_info*     p_csrmv_info);

    rocsparse_status cscmv(rocsparse_handle          handle,
                           rocsparse_operation       trans,
                           rocsparse::csrmv_alg      alg,
                           int64_t                   m,
                           int64_t                   n,
                           int64_t                   nnz,
                           rocsparse_datatype        alpha_device_host_datatype,
                           const void*               alpha_device_host,
                           const rocsparse_mat_descr descr,
                           rocsparse_datatype        csc_val_datatype,
                           const void*               csc_val,
                           rocsparse_indextype       csc_col_ptr_indextype,
                           const void*               csc_col_ptr,
                           rocsparse_indextype       csc_row_ind_indextype,
                           const void*               csc_row_ind,
                           rocsparse_csrmv_info      csrmv_info,
                           rocsparse_datatype        x_datatype,
                           const void*               x,
                           rocsparse_datatype        beta_device_host_datatype,
                           const void*               beta_device_host,
                           rocsparse_datatype        y_datatype,
                           void*                     y,
                           bool                      fallback_algorithm);

}
