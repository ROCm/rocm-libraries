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
    rocsparse_status gcsr2csc_buffer_size(rocsparse_handle    handle,
                                          int64_t             m,
                                          int64_t             n,
                                          int64_t             nnz,
                                          rocsparse_indextype indextype_ptr,
                                          rocsparse_indextype indextype_ind,
                                          const void*         csr_row_ptr,
                                          const void*         csr_col_ind,
                                          rocsparse_action    copy_values,
                                          size_t*             buffer_size);

    rocsparse_status gcsr2csc(rocsparse_handle     handle,
                              int64_t              m,
                              int64_t              n,
                              int64_t              nnz,
                              rocsparse_datatype   datatype,
                              rocsparse_indextype  indextype_ptr,
                              rocsparse_indextype  indextype_ind,
                              const void*          csr_val,
                              const void*          csr_row_ptr,
                              const void*          csr_col_ind,
                              void*                csc_val,
                              void*                csc_row_ind,
                              void*                csc_col_ptr,
                              rocsparse_action     copy_values,
                              rocsparse_index_base idx_base,
                              void*                temp_buffer);

    rocsparse_status spmat_csr2csc_buffer_size(rocsparse_handle            handle,
                                               rocsparse_const_spmat_descr source_,
                                               rocsparse_spmat_descr       target_,
                                               size_t*                     buffer_size_);

    rocsparse_status spmat_csr2csc(rocsparse_handle            handle,
                                   rocsparse_const_spmat_descr source_,
                                   rocsparse_spmat_descr       target_,
                                   size_t                      buffer_size_,
                                   void*                       buffer_);
}
