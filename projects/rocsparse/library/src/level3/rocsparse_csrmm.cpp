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

#include "rocsparse_csrmm.hpp"
#include "rocsparse_csrmm_transpose.hpp"
#include "rocsparse_csrmm_merge_path.hpp"
#include "rocsparse_csrmm_nnz_split.hpp"
#include "rocsparse_csrmm_row_split.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrmm_buffer_size(rocsparse_handle          handle,
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
                                              size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(trans_A == rocsparse_operation_transpose
       || trans_A == rocsparse_operation_conjugate_transpose)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size_transpose(handle,
                                                                         trans_A,
                                                                         m,
                                                                         n,
                                                                         k,
                                                                         nnz,
                                                                         descr,
                                                                         compute_datatype,
                                                                         csr_val_datatype,
                                                                         csr_val,
                                                                         csr_row_ptr_indextype,
                                                                         csr_row_ptr,
                                                                         csr_col_ind_indextype,
                                                                         csr_col_ind,
                                                                         buffer_size));
        return rocsparse_status_success;
    }

    switch(alg)
    {
    case rocsparse_csrmm_alg_default:
    case rocsparse_csrmm_alg_row_split:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size_row_split(handle,
                                                                         trans_A,
                                                                         m,
                                                                         n,
                                                                         k,
                                                                         nnz,
                                                                         descr,
                                                                         compute_datatype,
                                                                         csr_val_datatype,
                                                                         csr_val,
                                                                         csr_row_ptr_indextype,
                                                                         csr_row_ptr,
                                                                         csr_col_ind_indextype,
                                                                         csr_col_ind,
                                                                         buffer_size));
        return rocsparse_status_success;
    }
    case rocsparse_csrmm_alg_nnz_split:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size_nnz_split(handle,
                                                                         trans_A,
                                                                         m,
                                                                         n,
                                                                         k,
                                                                         nnz,
                                                                         descr,
                                                                         compute_datatype,
                                                                         csr_val_datatype,
                                                                         csr_val,
                                                                         csr_row_ptr_indextype,
                                                                         csr_row_ptr,
                                                                         csr_col_ind_indextype,
                                                                         csr_col_ind,
                                                                         buffer_size));
        return rocsparse_status_success;
    }
    case rocsparse_csrmm_alg_merge_path:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size_merge_path(handle,
                                                                     trans_A,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     nnz,
                                                                     descr,
                                                                     compute_datatype,
                                                                     csr_val_datatype,
                                                                     csr_val,
                                                                     csr_row_ptr_indextype,
                                                                     csr_row_ptr,
                                                                     csr_col_ind_indextype,
                                                                     csr_col_ind,
                                                                     buffer_size));
        return rocsparse_status_success;
    }
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrmm_analysis(rocsparse_handle          handle,
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
                                           void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(trans_A == rocsparse_operation_transpose
       || trans_A == rocsparse_operation_conjugate_transpose)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_analysis_transpose(handle,
                                                                      trans_A,
                                                                      m,
                                                                      n,
                                                                      k,
                                                                      nnz,
                                                                      descr,
                                                                      csr_val_datatype,
                                                                      csr_val,
                                                                      csr_row_ptr_indextype,
                                                                      csr_row_ptr,
                                                                      csr_col_ind_indextype,
                                                                      csr_col_ind,
                                                                      temp_buffer));
        return rocsparse_status_success;
    }

    switch(alg)
    {
    case rocsparse_csrmm_alg_default:
    case rocsparse_csrmm_alg_row_split:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_analysis_row_split(handle,
                                                                      trans_A,
                                                                      m,
                                                                      n,
                                                                      k,
                                                                      nnz,
                                                                      descr,
                                                                      csr_val_datatype,
                                                                      csr_val,
                                                                      csr_row_ptr_indextype,
                                                                      csr_row_ptr,
                                                                      csr_col_ind_indextype,
                                                                      csr_col_ind,
                                                                      temp_buffer));
        return rocsparse_status_success;
    }
    case rocsparse_csrmm_alg_nnz_split:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_analysis_nnz_split(handle,
                                                                      trans_A,
                                                                      m,
                                                                      n,
                                                                      k,
                                                                      nnz,
                                                                      descr,
                                                                      csr_val_datatype,
                                                                      csr_val,
                                                                      csr_row_ptr_indextype,
                                                                      csr_row_ptr,
                                                                      csr_col_ind_indextype,
                                                                      csr_col_ind,
                                                                      temp_buffer));
        return rocsparse_status_success;
    }
    case rocsparse_csrmm_alg_merge_path:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_analysis_merge_path(handle,
                                                                  trans_A,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  nnz,
                                                                  descr,
                                                                  csr_val_datatype,
                                                                  csr_val,
                                                                  csr_row_ptr_indextype,
                                                                  csr_row_ptr,
                                                                  csr_col_ind_indextype,
                                                                  csr_col_ind,
                                                                  temp_buffer));
        return rocsparse_status_success;
    }
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrmm(rocsparse_handle          handle,
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
                                  bool                      force_conj_A)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(trans_A == rocsparse_operation_transpose
       || trans_A == rocsparse_operation_conjugate_transpose)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_transpose(handle,
                                                             trans_A,
                                                             trans_B,
                                                             m,
                                                             n,
                                                             k,
                                                             nnz,
                                                             batch_count_A,
                                                             offsets_batch_stride_A,
                                                             columns_values_batch_stride_A,
                                                             alpha_datatype,
                                                             alpha,
                                                             descr,
                                                             csr_val_datatype,
                                                             csr_val,
                                                             csr_row_ptr_indextype,
                                                             csr_row_ptr,
                                                             csr_col_ind_indextype,
                                                             csr_col_ind,
                                                             dense_B_datatype,
                                                             dense_B,
                                                             ldb,
                                                             batch_count_B,
                                                             batch_stride_B,
                                                             order_B,
                                                             beta_datatype,
                                                             beta,
                                                             dense_C_datatype,
                                                             dense_C,
                                                             ldc,
                                                             batch_count_C,
                                                             batch_stride_C,
                                                             order_C,
                                                             temp_buffer,
                                                             force_conj_A));
        return rocsparse_status_success;
    }

    switch(alg)
    {
    case rocsparse_csrmm_alg_default:
    case rocsparse_csrmm_alg_row_split:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_row_split(handle,
                                                             trans_A,
                                                             trans_B,
                                                             m,
                                                             n,
                                                             k,
                                                             nnz,
                                                             batch_count_A,
                                                             offsets_batch_stride_A,
                                                             columns_values_batch_stride_A,
                                                             alpha_datatype,
                                                             alpha,
                                                             descr,
                                                             csr_val_datatype,
                                                             csr_val,
                                                             csr_row_ptr_indextype,
                                                             csr_row_ptr,
                                                             csr_col_ind_indextype,
                                                             csr_col_ind,
                                                             dense_B_datatype,
                                                             dense_B,
                                                             ldb,
                                                             batch_count_B,
                                                             batch_stride_B,
                                                             order_B,
                                                             beta_datatype,
                                                             beta,
                                                             dense_C_datatype,
                                                             dense_C,
                                                             ldc,
                                                             batch_count_C,
                                                             batch_stride_C,
                                                             order_C,
                                                             temp_buffer,
                                                             force_conj_A));
        return rocsparse_status_success;
    }
    case rocsparse_csrmm_alg_nnz_split:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_nnz_split(handle,
                                                             trans_A,
                                                             trans_B,
                                                             m,
                                                             n,
                                                             k,
                                                             nnz,
                                                             batch_count_A,
                                                             offsets_batch_stride_A,
                                                             columns_values_batch_stride_A,
                                                             alpha_datatype,
                                                             alpha,
                                                             descr,
                                                             csr_val_datatype,
                                                             csr_val,
                                                             csr_row_ptr_indextype,
                                                             csr_row_ptr,
                                                             csr_col_ind_indextype,
                                                             csr_col_ind,
                                                             dense_B_datatype,
                                                             dense_B,
                                                             ldb,
                                                             batch_count_B,
                                                             batch_stride_B,
                                                             order_B,
                                                             beta_datatype,
                                                             beta,
                                                             dense_C_datatype,
                                                             dense_C,
                                                             ldc,
                                                             batch_count_C,
                                                             batch_stride_C,
                                                             order_C,
                                                             temp_buffer,
                                                             force_conj_A));
        return rocsparse_status_success;
    }
    case rocsparse_csrmm_alg_merge_path:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_merge_path(handle,
                                                         trans_A,
                                                         trans_B,
                                                         m,
                                                         n,
                                                         k,
                                                         nnz,
                                                         batch_count_A,
                                                         offsets_batch_stride_A,
                                                         columns_values_batch_stride_A,
                                                         alpha_datatype,
                                                         alpha,
                                                         descr,
                                                         csr_val_datatype,
                                                         csr_val,
                                                         csr_row_ptr_indextype,
                                                         csr_row_ptr,
                                                         csr_col_ind_indextype,
                                                         csr_col_ind,
                                                         dense_B_datatype,
                                                         dense_B,
                                                         ldb,
                                                         batch_count_B,
                                                         batch_stride_B,
                                                         order_B,
                                                         beta_datatype,
                                                         beta,
                                                         dense_C_datatype,
                                                         dense_C,
                                                         ldc,
                                                         batch_count_C,
                                                         batch_stride_C,
                                                         order_C,
                                                         temp_buffer,
                                                         force_conj_A));
        return rocsparse_status_success;
    }
    }

    return rocsparse_status_success;
}
