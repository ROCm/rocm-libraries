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

#include "rocsparse-types.h"

#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)

namespace rocsparse
{
    rocsparse_status diagonal_solve(rocsparse_handle            handle,
                                    rocsparse_operation         trans,
                                    rocsparse_diagonal_modifier modifier,
                                    const void*                 alpha,
                                    rocsparse_const_spmat_descr A,
                                    rocsparse_indextype         diag_ind_type,
                                    const void*                 diag_ind,
                                    const void*                 transposed_perm,
                                    int64_t                     nrhs,
                                    const void*                 x,
                                    int64_t                     x_row_stride,
                                    int64_t                     x_col_stride,
                                    int64_t                     x_batch_stride,
                                    void*                       y,
                                    int64_t                     y_row_stride,
                                    int64_t                     y_col_stride,
                                    int64_t                     y_batch_stride,
                                    int64_t                     batch_count,
                                    bool                        conj_x,
                                    void*                       zero_pivot,
                                    int64_t                     zero_pivot_stride,
                                    bool                        is_host_mode);
}

#endif
