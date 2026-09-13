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
    // Complete diagonal solve: builds the diagonal view from the analysis info,
    // seeds the numeric zero-pivot buffer, and launches the solve. Only CSR and CSC
    // matrices are supported; callers switch on the matrix format so that a format
    // added later fetches its own analysis info rather than a CSR-named handle.
    //
    // The dense operands are taken as descriptors: rocsparse_sptrsv solves a single
    // right-hand side held in dense vectors, whereas rocsparse_sptrsm solves the
    // columns of a dense matrix and may transpose or conjugate its right-hand side.

    rocsparse_status diagonal_solve_csr(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_diagonal_modifier modifier,
                                        const void*                 alpha,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_csrsv_info        info,
                                        rocsparse_const_dnvec_descr x,
                                        rocsparse_dnvec_descr       y);

    rocsparse_status diagonal_solve_csc(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_diagonal_modifier modifier,
                                        const void*                 alpha,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_csrsv_info        info,
                                        rocsparse_const_dnvec_descr x,
                                        rocsparse_dnvec_descr       y);

    rocsparse_status diagonal_solve_csr(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_diagonal_modifier modifier,
                                        const void*                 alpha,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_csrsv_info        info,
                                        rocsparse_operation         x_operation,
                                        rocsparse_const_dnmat_descr X,
                                        rocsparse_dnmat_descr       Y);

    rocsparse_status diagonal_solve_csc(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_diagonal_modifier modifier,
                                        const void*                 alpha,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_csrsv_info        info,
                                        rocsparse_operation         x_operation,
                                        rocsparse_const_dnmat_descr X,
                                        rocsparse_dnmat_descr       Y);
}

#endif
