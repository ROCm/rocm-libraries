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

namespace rocsparse
{
    rocsparse_status csrgeam_buffer_size(rocsparse_handle          handle,
                                         rocsparse_operation       trans_A,
                                         rocsparse_operation       trans_B,
                                         int64_t                   m,
                                         int64_t                   n,
                                         const rocsparse_mat_descr descr_A,
                                         int64_t                   nnz_A,
                                         const void*               csr_row_ptr_A,
                                         const void*               csr_col_ind_A,
                                         const rocsparse_mat_descr descr_B,
                                         int64_t                   nnz_B,
                                         const void*               csr_row_ptr_B,
                                         const void*               csr_col_ind_B,
                                         const void*               csr_row_ptr_C,
                                         size_t*                   buffer_size);

    template <typename I, typename J>
    rocsparse_status csrgeam_nnz_template(rocsparse_handle             handle,
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

    template <typename T, typename I, typename J>
    rocsparse_status csrgeam_template(rocsparse_handle          handle,
                                      rocsparse_operation       trans_A,
                                      rocsparse_operation       trans_B,
                                      int64_t                   m,
                                      int64_t                   n,
                                      const void*               alpha_device_host,
                                      const rocsparse_mat_descr descr_A,
                                      int64_t                   nnz_A,
                                      const void*               csr_val_A,
                                      const void*               csr_row_ptr_A,
                                      const void*               csr_col_ind_A,
                                      const void*               beta_device_host,
                                      const rocsparse_mat_descr descr_B,
                                      int64_t                   nnz_B,
                                      const void*               csr_val_B,
                                      const void*               csr_row_ptr_B,
                                      const void*               csr_col_ind_B,
                                      const rocsparse_mat_descr descr_C,
                                      void*                     csr_val_C,
                                      const void*               csr_row_ptr_C,
                                      void*                     csr_col_ind_C,
                                      void*                     temp_buffer);

    rocsparse_status csrgeam_nnz(rocsparse_handle             handle,
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
                                 bool                         called_from_spgeam);

    rocsparse_status csrgeam(rocsparse_handle          handle,
                             rocsparse_operation       trans_A,
                             rocsparse_operation       trans_B,
                             int64_t                   m,
                             int64_t                   n,
                             rocsparse_datatype        alpha_device_host_datatype,
                             const void*               alpha,
                             const rocsparse_mat_descr descr_A,
                             int64_t                   nnz_A,
                             rocsparse_datatype        csr_val_A_datatype,
                             const void*               csr_val_A,
                             rocsparse_indextype       csr_row_ptr_A_indextype,
                             const void*               csr_row_ptr_A,
                             rocsparse_indextype       csr_col_ind_A_indextype,
                             const void*               csr_col_ind_A,
                             rocsparse_datatype        beta_device_host_datatype,
                             const void*               beta,
                             const rocsparse_mat_descr descr_B,
                             int64_t                   nnz_B,
                             rocsparse_datatype        csr_val_B_datatype,
                             const void*               csr_val_B,
                             rocsparse_indextype       csr_row_ptr_B_indextype,
                             const void*               csr_row_ptr_B,
                             rocsparse_indextype       csr_col_ind_B_indextype,
                             const void*               csr_col_ind_B,
                             const rocsparse_mat_descr descr_C,
                             rocsparse_datatype        csr_val_C_datatype,
                             void*                     csr_val_C,
                             rocsparse_indextype       csr_row_ptr_C_indextype,
                             const void*               csr_row_ptr_C,
                             rocsparse_indextype       csr_col_ind_C_indextype,
                             void*                     csr_col_ind_C,
                             void*                     temp_buffer);

}
