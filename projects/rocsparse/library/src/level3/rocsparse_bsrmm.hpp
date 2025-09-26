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

#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"

typedef enum rocsparse_bsrmm_alg_
{
    rocsparse_bsrmm_alg_default = 0,
    rocsparse_bsrmm_alg_bsr
} rocsparse_bsrmm_alg;

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmm_template_dispatch(rocsparse_handle    handle,
                                             rocsparse_direction dir,
                                             rocsparse_operation trans_A,
                                             rocsparse_operation trans_B,
                                             rocsparse_bsrmm_alg alg,
                                             J                   mb,
                                             J                   n,
                                             J                   kb,
                                             I                   nnzb,
                                             J                   batch_count_A,
                                             int64_t             offsets_batch_stride_A,
                                             int64_t             columns_values_batch_stride_A,
                                             const T*            alpha,
                                             const rocsparse_mat_descr descr,
                                             const A*                  bsr_val,
                                             const I*                  bsr_row_ptr,
                                             const J*                  bsr_col_ind,
                                             J                         block_dim,
                                             const B*                  dense_B,
                                             int64_t                   ldb,
                                             J                         batch_count_B,
                                             int64_t                   batch_stride_B,
                                             rocsparse_order           order_B,
                                             const T*                  beta,
                                             C*                        dense_C,
                                             int64_t                   ldc,
                                             J                         batch_count_C,
                                             int64_t                   batch_stride_C,
                                             rocsparse_order           order_C);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmm_template(rocsparse_handle          handle,
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

    rocsparse_status bsrmm_buffer_size(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_bsrmm_alg       alg,
                                       int64_t                   mb,
                                       int64_t                   n,
                                       int64_t                   kb,
                                       int64_t                   nnzb,
                                       const rocsparse_mat_descr descr,
                                       rocsparse_datatype        bsr_val_datatype,
                                       const void*               bsr_val,
                                       rocsparse_indextype       bsr_row_ptr_indextype,
                                       const void*               bsr_row_ptr,
                                       rocsparse_indextype       bsr_col_ind_indextype,
                                       const void*               bsr_col_ind,
                                       int64_t                   block_dim,
                                       size_t*                   buffer_size);

    rocsparse_status bsrmm_analysis(rocsparse_handle          handle,
                                    rocsparse_operation       trans_A,
                                    rocsparse_bsrmm_alg       alg,
                                    int64_t                   mb,
                                    int64_t                   n,
                                    int64_t                   kb,
                                    int64_t                   nnzb,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_datatype        bsr_val_datatype,
                                    const void*               bsr_val,
                                    rocsparse_indextype       bsr_row_ptr_indextype,
                                    const void*               bsr_row_ptr,
                                    rocsparse_indextype       bsr_col_ind_indextype,
                                    const void*               bsr_col_ind,
                                    int64_t                   block_dim,
                                    void*                     temp_buffer);

    rocsparse_status bsrmm(rocsparse_handle          handle,
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
                           rocsparse_order           order_C);
}
