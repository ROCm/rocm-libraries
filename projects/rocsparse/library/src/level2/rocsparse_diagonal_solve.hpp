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
    // Reusable, format-agnostic diagonal backsolve for one or many right-hand sides:
    // Y = alpha * op(X) ./ D, where D is the diagonal of A seen as a sparse vector,
    // located through \p diag_ind (the per-row diagonal offset collected during the
    // triangular-solve analysis, i.e. rocsparse::trm_info_t::diag_ind). When the
    // analysis ran on a transposed structure (CSR transpose / conjugate-transpose,
    // or any CSC), \p diag_ind indexes that transposed layout; \p transposed_perm
    // (rocsparse::trm_info_t::transposed_perm) then maps it back into A's shared val
    // array. Pass \p transposed_perm as nullptr when \p diag_ind already indexes val
    // (non-transposed CSR). All of the solve semantics are folded in:
    // \p alpha scaling, the conjugation implied by a conjugate transpose \p trans
    // (conjugates d), the \p diagonal_mode (signed → /D, absolute → /|D|) and
    // numeric/structural zero-pivot reporting into \p zero_pivot.
    //
    // The RHS is a dense block of \p nrhs columns given by raw pointers and generic
    // strides: a plain vector is nrhs == 1 with a column stride of 0; a dense matrix
    // uses its leading dimension / order. \p conj_x conjugates the x entries on the
    // fly (conjugate-transpose RHS). One single launch covers all columns and batches.
    //
    // \p alpha must already be expressed in A's data type and point to host or device
    // memory according to the handle's pointer mode.
    rocsparse_status diagonal_solve(rocsparse_handle            handle,
                                    rocsparse_operation         trans,
                                    rocsparse_diagonal_mode     diagonal_mode,
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
