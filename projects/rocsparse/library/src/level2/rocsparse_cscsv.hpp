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

#include "rocsparse_csc_to_csr_descr.hpp"
#include "rocsparse_cscsv_info.hpp"
#include "rocsparse_csrsv.hpp"

namespace rocsparse
{
    // A CSC matrix A is the transpose of the CSR matrix that shares its arrays, so op(A) maps to
    // op'(A^T) with the operation and the fill mode flipped (see build_csr_from_csc). A
    // conjugate-transpose CSC solve is handled as a non-transposed CSR solve with on-the-fly
    // conjugation (see force_conj in csrsv_solve), so it maps to none.
    static inline rocsparse_operation cscsv_operation_to_csr(rocsparse_operation trans)
    {
        return (trans == rocsparse_operation_none) ? rocsparse_operation_transpose
                                                   : rocsparse_operation_none;
    }

    // CSC triangular solve. Without transpose, the system is solved on the columns of the matrix
    // (when rocSPARSE is built with BUILD_WITH_SPTRSV_CSC); the other operations solve the
    // transposed CSR system.
    rocsparse_status cscsv_analysis_buffer_size(rocsparse_handle            handle,
                                                rocsparse_operation         trans,
                                                rocsparse_const_spmat_descr A,
                                                size_t*                     buffer_size);

    bool cscsv_is_analyzed(rocsparse_cscsv_info        info,
                           rocsparse_operation         trans,
                           rocsparse_const_spmat_descr A);

    rocsparse_status cscsv_solve_buffer_size(rocsparse_handle            handle,
                                             rocsparse_operation         trans,
                                             rocsparse_const_spmat_descr A,
                                             rocsparse_const_dnvec_descr x,
                                             rocsparse_const_dnvec_descr y,
                                             size_t*                     buffer_size);

    rocsparse_status cscsv_analysis(rocsparse_handle            handle,
                                    rocsparse_operation         trans,
                                    rocsparse_const_spmat_descr A,
                                    rocsparse_analysis_policy   analysis_policy,
                                    rocsparse_solve_policy      solve_policy,
                                    rocsparse_cscsv_info        info,
                                    size_t                      buffer_size_in_bytes,
                                    void*                       buffer);

    rocsparse_status cscsv_solve(rocsparse_handle            handle,
                                 rocsparse_operation         trans,
                                 rocsparse_datatype          alpha_datatype,
                                 const void*                 alpha,
                                 int64_t                     alpha_stride,
                                 rocsparse_const_spmat_descr A,
                                 rocsparse_const_dnvec_descr x,
                                 rocsparse_dnvec_descr       y,
                                 rocsparse_solve_policy      solve_policy,
                                 rocsparse_cscsv_info        info,
                                 size_t                      buffer_size_in_bytes,
                                 void*                       buffer);

    rocsparse_status cscsv_diagonal_solve(rocsparse_handle            handle,
                                          rocsparse_operation         trans,
                                          rocsparse_diagonal_modifier modifier,
                                          const void*                 alpha,
                                          rocsparse_const_spmat_descr A,
                                          rocsparse_const_dnvec_descr x,
                                          rocsparse_dnvec_descr       y,
                                          rocsparse_cscsv_info        info);

    rocsparse_status cscsv_zero_pivot(rocsparse_handle     handle,
                                      rocsparse_cscsv_info info,
                                      rocsparse_indextype  indextype,
                                      void*                position);
}
