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

typedef enum rocsparse_csrmm_alg_
{
    rocsparse_csrmm_alg_default = 0,
    rocsparse_csrmm_alg_row_split,
    rocsparse_csrmm_alg_nnz_split,
    rocsparse_csrmm_alg_merge_path
} rocsparse_csrmm_alg;

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A>
    rocsparse_status csrmm_buffer_size_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_csrmm_alg       alg,
                                                int64_t                   m,
                                                int64_t                   n,
                                                int64_t                   k,
                                                int64_t                   nnz,
                                                const rocsparse_mat_descr descr,
                                                const void*               csr_val,
                                                const void*               csr_row_ptr,
                                                const void*               csr_col_ind,
                                                size_t*                   buffer_size);

    template <typename I, typename J, typename A>
    rocsparse_status csrmm_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_csrmm_alg       alg,
                                             int64_t                   m,
                                             int64_t                   n,
                                             int64_t                   k,
                                             int64_t                   nnz,
                                             const rocsparse_mat_descr descr,
                                             const void*               csr_val,
                                             const void*               csr_row_ptr,
                                             const void*               csr_col_ind,
                                             void*                     temp_buffer);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_template_dispatch(rocsparse_handle    handle,
                                             rocsparse_operation trans_A,
                                             rocsparse_operation trans_B,
                                             rocsparse_csrmm_alg alg,
                                             J                   m,
                                             J                   n,
                                             J                   k,
                                             I                   nnz,
                                             J                   batch_count_A,
                                             int64_t             offsets_batch_stride_A,
                                             int64_t             columns_values_batch_stride_A,
                                             const T*            alpha_device_host,
                                             const rocsparse_mat_descr descr,
                                             const A*                  csr_val,
                                             const I*                  csr_row_ptr,
                                             const J*                  csr_col_ind,
                                             const B*                  dense_B,
                                             int64_t                   ldb,
                                             J                         batch_count_B,
                                             int64_t                   batch_stride_B,
                                             rocsparse_order           order_B,
                                             const T*                  beta_device_host,
                                             C*                        dense_C,
                                             int64_t                   ldc,
                                             J                         batch_count_C,
                                             int64_t                   batch_stride_C,
                                             rocsparse_order           order_C,
                                             void*                     temp_buffer,
                                             bool                      force_conj_A);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_template(rocsparse_handle          handle,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_csrmm_alg       alg,
                                    int64_t                   m,
                                    int64_t                   n,
                                    int64_t                   k,
                                    int64_t                   nnz,
                                    int64_t                   batch_count_A,
                                    int64_t                   offsets_batch_stride_A,
                                    int64_t                   columns_values_batch_stride_A,
                                    const void*               alpha,
                                    const rocsparse_mat_descr descr,
                                    const void*               csr_val,
                                    const void*               csr_row_ptr,
                                    const void*               csr_col_ind,
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
                                    rocsparse_order           order_C,
                                    void*                     temp_buffer,
                                    bool                      force_conj_A);

    rocsparse_status csrmm_buffer_size(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_csrmm_alg       alg,
                                       int64_t                   m,
                                       int64_t                   n,
                                       int64_t                   k,
                                       int64_t                   nnz,
                                       const rocsparse_mat_descr descr,
                                       rocsparse_datatype        compute_datatype,
                                       rocsparse_datatype        csr_val_datatype,
                                       const void*               csr_val,
                                       rocsparse_indextype       csr_row_ptr_indextype,
                                       const void*               csr_row_ptr,
                                       rocsparse_indextype       csr_col_ind_indextype,
                                       const void*               csr_col_ind,
                                       size_t*                   buffer_size);

    rocsparse_status csrmm_analysis(rocsparse_handle          handle,
                                    rocsparse_operation       trans_A,
                                    rocsparse_csrmm_alg       alg,
                                    int64_t                   m,
                                    int64_t                   n,
                                    int64_t                   k,
                                    int64_t                   nnz,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_datatype        csr_val_datatype,
                                    const void*               csr_val,
                                    rocsparse_indextype       csr_row_ptr_indextype,
                                    const void*               csr_row_ptr,
                                    rocsparse_indextype       csr_col_ind_indextype,
                                    const void*               csr_col_ind,
                                    void*                     temp_buffer);

    rocsparse_status csrmm(rocsparse_handle          handle,
                           rocsparse_operation       trans_A,
                           rocsparse_operation       trans_B,
                           rocsparse_csrmm_alg       alg,
                           int64_t                   m,
                           int64_t                   n,
                           int64_t                   k,
                           int64_t                   nnz,
                           int64_t                   batch_count_A,
                           int64_t                   offsets_batch_stride_A,
                           int64_t                   columns_values_batch_stride_A,
                           rocsparse_datatype        alpha_datatype,
                           const void*               alpha,
                           const rocsparse_mat_descr descr,
                           rocsparse_datatype        csr_val_datatype,
                           const void*               csr_val,
                           rocsparse_indextype       csr_row_ptr_indextype,
                           const void*               csr_row_ptr,
                           rocsparse_indextype       csr_col_ind_indextype,
                           const void*               csr_col_ind,
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
                           rocsparse_order           order_C,
                           void*                     temp_buffer,
                           bool                      force_conj_A);

}
