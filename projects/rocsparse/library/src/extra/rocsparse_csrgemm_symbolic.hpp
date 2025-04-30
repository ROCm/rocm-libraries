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
    rocsparse_status csrgemm_symbolic_quickreturn(rocsparse_handle          handle,
                                                  rocsparse_operation       trans_A,
                                                  rocsparse_operation       trans_B,
                                                  int64_t                   m,
                                                  int64_t                   n,
                                                  int64_t                   k,
                                                  const rocsparse_mat_descr descr_A,
                                                  int64_t                   nnz_A,
                                                  const void*               csr_row_ptr_A,
                                                  const void*               csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  int64_t                   nnz_B,
                                                  const void*               csr_row_ptr_B,
                                                  const void*               csr_col_ind_B,
                                                  const rocsparse_mat_descr descr_D,
                                                  int64_t                   nnz_D,
                                                  const void*               csr_row_ptr_D,
                                                  const void*               csr_col_ind_D,
                                                  const rocsparse_mat_descr descr_C,
                                                  int64_t                   nnz_C,
                                                  const void*               csr_row_ptr_C,
                                                  void*                     csr_col_ind_C,
                                                  const rocsparse_mat_info  info_C,
                                                  void*                     temp_buffer);

    template <typename I, typename J>
    rocsparse_status csrgemm_symbolic_core(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           J                         m,
                                           J                         n,
                                           J                         k,
                                           const rocsparse_mat_descr descr_A,
                                           I                         nnz_A,
                                           const I*                  csr_row_ptr_A,
                                           const J*                  csr_col_ind_A,
                                           const rocsparse_mat_descr descr_B,
                                           I                         nnz_B,
                                           const I*                  csr_row_ptr_B,
                                           const J*                  csr_col_ind_B,
                                           const rocsparse_mat_descr descr_D,
                                           I                         nnz_D,
                                           const I*                  csr_row_ptr_D,
                                           const J*                  csr_col_ind_D,
                                           const rocsparse_mat_descr descr_C,
                                           I                         nnz_C,
                                           const I*                  csr_row_ptr_C,
                                           J*                        csr_col_ind_C,
                                           const rocsparse_mat_info  info_C,
                                           void*                     temp_buffer);

    template <typename... P>
    rocsparse_status csrgemm_symbolic_template(P&&... p)
    {
        const rocsparse_status status = rocsparse::csrgemm_symbolic_quickreturn(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_symbolic_core(p...));
        return rocsparse_status_success;
    }
}
