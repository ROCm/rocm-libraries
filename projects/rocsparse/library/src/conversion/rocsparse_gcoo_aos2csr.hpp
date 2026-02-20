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
    rocsparse_status gcoo_aos2csr_buffer_size(rocsparse_handle     handle,
                                              int64_t              m,
                                              int64_t              nnz,
                                              rocsparse_indextype  source_ind_type,
                                              const void*          source_ind_data,
                                              rocsparse_index_base source_idx_base,
                                              rocsparse_indextype  target_row_type,
                                              rocsparse_indextype  target_col_type,
                                              size_t*              buffer_size);

    rocsparse_status gcoo_aos2csr(rocsparse_handle     handle,
                                  int64_t              m,
                                  int64_t              nnz,
                                  rocsparse_indextype  source_ind_type,
                                  const void*          source_ind_data,
                                  rocsparse_datatype   source_data_type,
                                  const void*          source_val_data,
                                  rocsparse_index_base source_idx_base,
                                  rocsparse_indextype  target_row_type,
                                  void*                target_row_data,
                                  rocsparse_indextype  target_col_type,
                                  void*                target_col_data,
                                  rocsparse_datatype   target_data_type,
                                  void*                target_val_data,
                                  rocsparse_index_base target_idx_base,
                                  size_t               buffer_size,
                                  void*                buffer_);

    rocsparse_status spmat_coo_aos2csr_buffer_size(rocsparse_handle            handle,
                                                   rocsparse_const_spmat_descr source_,
                                                   rocsparse_spmat_descr       target_,
                                                   size_t*                     buffer_size_);

    rocsparse_status spmat_coo_aos2csr(rocsparse_handle            handle,
                                       rocsparse_const_spmat_descr source_,
                                       rocsparse_spmat_descr       target_,
                                       size_t                      buffer_size_,
                                       void*                       buffer_);
}
