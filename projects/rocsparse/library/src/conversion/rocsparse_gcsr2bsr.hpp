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

#include "rocsparse-types.h"

namespace rocsparse
{
    rocsparse_status gcsr2bsr_nnz(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  int64_t                   m,
                                  int64_t                   n,
                                  const rocsparse_mat_descr csr_descr,
                                  rocsparse_indextype       csr_row_ptr_indextype,
                                  const void*               csr_row_ptr,
                                  rocsparse_indextype       csr_col_ind_indextype,
                                  const void*               csr_col_ind,
                                  int64_t                   block_dim,
                                  const rocsparse_mat_descr bsr_descr,
                                  rocsparse_indextype       bsr_row_ptr_indextype,
                                  void*                     bsr_row_ptr,
                                  int64_t*                  bsr_nnz);

    rocsparse_status gcsr2bsr(rocsparse_handle          handle,
                              rocsparse_direction       direction,
                              int64_t                   m,
                              int64_t                   n,
                              const rocsparse_mat_descr csr_descr,
                              rocsparse_datatype        csr_val_datatype,
                              const void*               csr_val,
                              rocsparse_indextype       csr_row_ptr_indextype,
                              const void*               csr_row_ptr,
                              rocsparse_indextype       csr_col_ind_indextype,
                              const void*               csr_col_ind,
                              int64_t                   block_dim,
                              const rocsparse_mat_descr bsr_descr,
                              rocsparse_datatype        bsr_val_datatype,
                              void*                     bsr_val,
                              rocsparse_indextype       bsr_row_ptr_indextype,
                              void*                     bsr_row_ptr,
                              rocsparse_indextype       bsr_col_ind_indextype,
                              void*                     bsr_col_ind);

    rocsparse_status spmat_csr2bsr_nnz(rocsparse_handle            handle,
                                       rocsparse_const_spmat_descr source,
                                       rocsparse_spmat_descr       target,
                                       int64_t*                    bsr_nnz);

    rocsparse_status spmat_csr2bsr(rocsparse_handle            handle,
                                   rocsparse_const_spmat_descr source,
                                   rocsparse_spmat_descr       target,
                                   size_t                      buffer_size,
                                   void*                       buffer);

    rocsparse_status spmat_csr2bsr_buffer_size(rocsparse_handle            handle,
                                               rocsparse_const_spmat_descr source,
                                               rocsparse_const_spmat_descr target,
                                               size_t*                     buffer_size);
}
