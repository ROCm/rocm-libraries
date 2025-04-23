/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "control.h"

namespace rocsparse
{
    rocsparse_status csrgemm_numeric_scal_quickreturn(rocsparse_handle          handle,
                                                      int64_t                   m,
                                                      int64_t                   n,
                                                      const void*               beta_device_host,
                                                      const rocsparse_mat_descr descr_D,
                                                      int64_t                   nnz_D,
                                                      const void*               csr_val_D,
                                                      const void*               csr_row_ptr_D,
                                                      const void*               csr_col_ind_D,
                                                      const rocsparse_mat_descr descr_C,
                                                      int64_t                   nnz_C,
                                                      void*                     csr_val_C,
                                                      const void*               csr_row_ptr_C,
                                                      const void*               csr_col_ind_C,
                                                      const rocsparse_mat_info  info_C,
                                                      void*                     temp_buffer);

    template <typename I, typename J, typename T>
    rocsparse_status csrgemm_numeric_scal_core(rocsparse_handle          handle,
                                               J                         m,
                                               J                         n,
                                               const T*                  beta_device_host,
                                               const rocsparse_mat_descr descr_D,
                                               I                         nnz_D,
                                               const T*                  csr_val_D,
                                               const I*                  csr_row_ptr_D,
                                               const J*                  csr_col_ind_D,
                                               const rocsparse_mat_descr descr_C,
                                               I                         nnz_C,
                                               T*                        csr_val_C,
                                               const I*                  csr_row_ptr_C,
                                               const J*                  csr_col_ind_C,
                                               const rocsparse_mat_info  info_C,
                                               void*                     temp_buffer);
}
