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

struct _rocsparse_csrsv_info;
typedef _rocsparse_csrsv_info* rocsparse_csrsv_info;

namespace rocsparse
{
    struct spdiag_view
    {
        rocsparse_indextype offset_type{}; // index type of diag_ind / transposed_perm
        const void*         diag_ind{nullptr}; // per-row diagonal position
        const void*         transposed_perm{nullptr}; // remap into val, or nullptr
    };

    rocsparse_status build_spdiag_view(rocsparse_const_spmat_descr A,
                                       rocsparse_operation         trans,
                                       rocsparse_csrsv_info        info,
                                       rocsparse::spdiag_view*     view);

    rocsparse_status diagonal_solve(rocsparse_handle              handle,
                                    rocsparse_operation           trans,
                                    rocsparse_diagonal_modifier   modifier,
                                    const void*                   alpha,
                                    rocsparse_const_spmat_descr   A,
                                    const rocsparse::spdiag_view& diag,
                                    int64_t                       nrhs,
                                    const void*                   x,
                                    int64_t                       x_row_stride,
                                    int64_t                       x_col_stride,
                                    int64_t                       x_batch_stride,
                                    void*                         y,
                                    int64_t                       y_row_stride,
                                    int64_t                       y_col_stride,
                                    int64_t                       y_batch_stride,
                                    int64_t                       batch_count,
                                    bool                          conj_x,
                                    void*                         zero_pivot,
                                    int64_t                       zero_pivot_stride,
                                    bool                          is_host_mode);

    // Format-specific entry points that perform a complete diagonal solve: they
    // build the diagonal view from the analysis info, seed the numeric zero-pivot
    // buffer, and launch the solve. CSR and CSC differ only in how the analysis
    // pivot is typed, so callers dispatch on the matrix format rather than sharing
    // a single CSR-centric path.
    rocsparse_status diagonal_solve_csr(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_diagonal_modifier modifier,
                                        const void*                 alpha,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_csrsv_info        info,
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
                                        bool                        conj_x);

    rocsparse_status diagonal_solve_csc(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_diagonal_modifier modifier,
                                        const void*                 alpha,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_csrsv_info        info,
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
                                        bool                        conj_x);
}

#endif
