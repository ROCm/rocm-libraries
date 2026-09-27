/*! \file */
/* ************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

namespace rocsparse
{
    template <typename T>
    rocsparse_status dense2bell_nnz_template(rocsparse_handle          handle,
                                             rocsparse_order           order,
                                             int64_t                   m,
                                             int64_t                   n,
                                             const rocsparse_mat_descr descr,
                                             const T*                  A,
                                             int64_t                   ld,
                                             int64_t                   ell_block_size,
                                             int64_t*                  nnzb_per_row,
                                             int64_t*                  ell_cols);

    template <typename I, typename T>
    rocsparse_status dense2bell_template(rocsparse_handle          handle,
                                         rocsparse_order           order,
                                         int64_t                   m,
                                         int64_t                   n,
                                         const rocsparse_mat_descr descr,
                                         const T*                  A,
                                         int64_t                   ld,
                                         int64_t                   ell_block_size,
                                         int64_t                   ell_cols,
                                         T*                        bell_val,
                                         I*                        bell_col_ind);
}
