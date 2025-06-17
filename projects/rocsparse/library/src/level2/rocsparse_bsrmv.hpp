/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <typename I, typename J, typename A>
    rocsparse_status bsrmv_analysis_template(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_operation       trans,
                                             int64_t                   mb,
                                             int64_t                   nb,
                                             int64_t                   nnzb,
                                             const rocsparse_mat_descr descr,
                                             const void*               bsr_val,
                                             const void*               bsr_row_ptr,
                                             const void*               bsr_col_ind,
                                             int64_t                   block_dim,
                                             rocsparse_bsrmv_info*     p_bsrmv_info);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status bsrmv_template_dispatch(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_operation       trans,
                                             J                         mb,
                                             J                         nb,
                                             I                         nnzb,
                                             const T*                  alpha_device_host,
                                             const rocsparse_mat_descr descr,
                                             const A*                  bsr_val,
                                             const I*                  bsr_row_ptr,
                                             const J*                  bsr_col_ind,
                                             J                         block_dim,
                                             const X*                  x,
                                             const T*                  beta_device_host,
                                             Y*                        y);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status bsrmv_adaptive_template_dispatch(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans,
                                                      J                         mb,
                                                      J                         nb,
                                                      I                         nnzb,
                                                      const T*                  alpha_device_host,
                                                      const rocsparse_mat_descr descr,
                                                      const A*                  bsr_val,
                                                      const I*                  bsr_row_ptr,
                                                      const J*                  bsr_col_ind,
                                                      J                         block_dim,
                                                      rocsparse_bsrmv_info      bsrmv_info,
                                                      const X*                  x,
                                                      const T*                  beta_device_host,
                                                      Y*                        y);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status bsrmv_template(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_operation       trans,
                                    int64_t                   mb,
                                    int64_t                   nb,
                                    int64_t                   nnzb,
                                    const void*               alpha_device_host,
                                    const rocsparse_mat_descr descr,
                                    const void*               bsr_val,
                                    const void*               bsr_row_ptr,
                                    const void*               bsr_col_ind,
                                    int64_t                   block_dim,
                                    rocsparse_bsrmv_info      bsrmv_info,
                                    const void*               x,
                                    const void*               beta_device_host,
                                    void*                     y);

    rocsparse_status bsrmv_analysis(rocsparse_handle          handle,
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
                                    rocsparse_bsrmv_info*     p_bsrmv_info);

    rocsparse_status bsrmv(rocsparse_handle          handle,
                           rocsparse_direction       dir,
                           rocsparse_operation       trans,
                           int64_t                   mb,
                           int64_t                   nb,
                           int64_t                   nnzb,
                           rocsparse_datatype        alpha_device_host_datatype,
                           const void*               alpha_device_host,
                           const rocsparse_mat_descr descr,
                           rocsparse_datatype        bsr_val_datatype,
                           const void*               bsr_val,
                           rocsparse_indextype       bsr_row_ptr_indextype,
                           const void*               bsr_row_ptr,
                           rocsparse_indextype       bsr_col_ind_indextype,
                           const void*               bsr_col_ind,
                           int64_t                   block_dim,
                           rocsparse_bsrmv_info      bsrmv_info,
                           rocsparse_datatype        x_datatype,
                           const void*               x,
                           rocsparse_datatype        beta_device_host_datatype,
                           const void*               beta_device_host,
                           rocsparse_datatype        y_datatype,
                           void*                     y);

}
