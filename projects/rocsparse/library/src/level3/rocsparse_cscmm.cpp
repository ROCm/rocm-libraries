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

#include "rocsparse_cscmm.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::cscmm_buffer_size(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_csrmm_alg       alg,
                                              int64_t                   m,
                                              int64_t                   n,
                                              int64_t                   k,
                                              int64_t                   nnz,
                                              const rocsparse_mat_descr descr,
                                              rocsparse_datatype        compute_datatype,
                                              rocsparse_datatype        csc_val_datatype,
                                              const void*               csc_val,
                                              rocsparse_indextype       csc_col_ptr_indextype,
                                              const void*               csc_col_ptr,
                                              rocsparse_indextype       csc_row_ind_indextype,
                                              const void*               csc_row_ind,
                                              size_t*                   buffer_size)
{
    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size(handle,
                                                               rocsparse_operation_transpose,
                                                               alg,
                                                               k,
                                                               n,
                                                               m,
                                                               nnz,
                                                               descr,
                                                               compute_datatype,
                                                               csc_val_datatype,
                                                               csc_val,
                                                               csc_col_ptr_indextype,
                                                               csc_col_ptr,
                                                               csc_row_ind_indextype,
                                                               csc_row_ind,
                                                               buffer_size));
        return rocsparse_status_success;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size(handle,
                                                               rocsparse_operation_none,
                                                               alg,
                                                               k,
                                                               n,
                                                               m,
                                                               nnz,
                                                               descr,
                                                               compute_datatype,
                                                               csc_val_datatype,
                                                               csc_val,
                                                               csc_col_ptr_indextype,
                                                               csc_col_ptr,
                                                               csc_row_ind_indextype,
                                                               csc_row_ind,
                                                               buffer_size));
        return rocsparse_status_success;
    }
    }
}

rocsparse_status rocsparse::cscmm_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_csrmm_alg       alg,
                                           int64_t                   m,
                                           int64_t                   n,
                                           int64_t                   k,
                                           int64_t                   nnz,
                                           const rocsparse_mat_descr descr,
                                           rocsparse_datatype        csc_val_datatype,
                                           const void*               csc_val,
                                           rocsparse_indextype       csc_col_ptr_indextype,
                                           const void*               csc_col_ptr,
                                           rocsparse_indextype       csc_row_ind_indextype,
                                           const void*               csc_row_ind,
                                           void*                     temp_buffer)
{
    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_analysis(handle,
                                                            rocsparse_operation_transpose,
                                                            alg,
                                                            k,
                                                            n,
                                                            m,
                                                            nnz,
                                                            descr,
                                                            csc_val_datatype,
                                                            csc_val,
                                                            csc_col_ptr_indextype,
                                                            csc_col_ptr,
                                                            csc_row_ind_indextype,
                                                            csc_row_ind,
                                                            temp_buffer));
        return rocsparse_status_success;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_analysis(handle,
                                                            rocsparse_operation_none,
                                                            alg,
                                                            k,
                                                            n,
                                                            m,
                                                            nnz,
                                                            descr,
                                                            csc_val_datatype,
                                                            csc_val,
                                                            csc_col_ptr_indextype,
                                                            csc_col_ptr,
                                                            csc_row_ind_indextype,
                                                            csc_row_ind,
                                                            temp_buffer));
        return rocsparse_status_success;
    }
    }
}

rocsparse_status rocsparse::cscmm(rocsparse_handle          handle,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_csrmm_alg       alg,
                                  int64_t                   m,
                                  int64_t                   n,
                                  int64_t                   k,
                                  int64_t                   nnz,
                                  int64_t                   batch_count_A,
                                  int64_t                   offsets_batch_stride_A,
                                  int64_t                   rows_values_batch_stride_A,
                                  rocsparse_datatype        alpha_datatype,
                                  const void*               alpha,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_datatype        csc_val_datatype,
                                  const void*               csc_val,
                                  rocsparse_indextype       csc_col_ptr_indextype,
                                  const void*               csc_col_ptr,
                                  rocsparse_indextype       csc_row_ind_indextype,
                                  const void*               csc_row_ind,
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
                                  void*                     temp_buffer)
{
    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm(handle,
                                                   rocsparse_operation_transpose,
                                                   trans_B,
                                                   alg,
                                                   k,
                                                   n,
                                                   m,
                                                   nnz,
                                                   batch_count_A,
                                                   offsets_batch_stride_A,
                                                   rows_values_batch_stride_A,
                                                   alpha_datatype,
                                                   alpha,
                                                   descr,
                                                   csc_val_datatype,
                                                   csc_val,
                                                   csc_col_ptr_indextype,
                                                   csc_col_ptr,
                                                   csc_row_ind_indextype,
                                                   csc_row_ind,
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
                                                   false));
        return rocsparse_status_success;
    }
    case rocsparse_operation_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm(handle,
                                                   rocsparse_operation_none,
                                                   trans_B,
                                                   alg,
                                                   k,
                                                   n,
                                                   m,
                                                   nnz,
                                                   batch_count_A,
                                                   offsets_batch_stride_A,
                                                   rows_values_batch_stride_A,
                                                   alpha_datatype,
                                                   alpha,
                                                   descr,
                                                   csc_val_datatype,
                                                   csc_val,
                                                   csc_col_ptr_indextype,
                                                   csc_col_ptr,
                                                   csc_row_ind_indextype,
                                                   csc_row_ind,
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
                                                   false));
        return rocsparse_status_success;
    }
    case rocsparse_operation_conjugate_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm(handle,
                                                   rocsparse_operation_none,
                                                   trans_B,
                                                   alg,
                                                   k,
                                                   n,
                                                   m,
                                                   nnz,
                                                   batch_count_A,
                                                   offsets_batch_stride_A,
                                                   rows_values_batch_stride_A,
                                                   alpha_datatype,
                                                   alpha,
                                                   descr,
                                                   csc_val_datatype,
                                                   csc_val,
                                                   csc_col_ptr_indextype,
                                                   csc_col_ptr,
                                                   csc_row_ind_indextype,
                                                   csc_row_ind,
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
                                                   true));
        return rocsparse_status_success;
    }
    }
}