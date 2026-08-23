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
#include "rocsparse_csrsv.hpp"

namespace rocsparse
{
    //
    // CSC triangular solve, expressed as a CSR triangular solve on the transpose.
    //
    // A CSC matrix A is the transpose of the CSR matrix that shares its arrays, so
    // op(A) maps to op'(A^T) with both the operation and the triangular fill mode
    // flipped (see build_csr_from_csc). These wrappers mirror the csrsv_* interface
    // one-to-one so callers can dispatch CSC exactly like CSR, without building any
    // descriptor (and therefore without leaking one).
    //

    // Map the requested CSC operation to the equivalent CSR operation. A
    // conjugate-transpose CSC solve is handled as a non-transposed CSR solve with
    // on-the-fly conjugation (see force_conj in csrsv_solve), so it maps to none.
    static inline rocsparse_operation cscsv_operation_to_csr(rocsparse_operation trans)
    {
        return (trans == rocsparse_operation_none) ? rocsparse_operation_transpose
                                                   : rocsparse_operation_none;
    }

    // Report whether the analysis for the requested CSC operation is already
    // cached in \p info. Because the CSC solve is expressed as a transposed CSR
    // solve, the analysis is stored under the CSR-mapped operation and the flipped
    // fill mode (see build_csr_from_csc), so the lookup key must be mapped the same
    // way. This lets callers gate the analysis per (operation, fill_mode) instead
    // of using the descriptor-wide mat->analysed flag.
    inline bool cscsv_is_analyzed(rocsparse_csrsv_info        info,
                                  rocsparse_operation         trans,
                                  rocsparse_const_spmat_descr A)
    {
        _rocsparse_mat_descr   descr_csr;
        _rocsparse_spmat_descr mat_csr;
        rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);

        return info->get(rocsparse::cscsv_operation_to_csr(trans), descr_csr.fill_mode) != nullptr;
    }

    inline rocsparse_status cscsv_analysis_buffer_size(rocsparse_handle            handle,
                                                       rocsparse_operation         trans,
                                                       rocsparse_const_spmat_descr A,
                                                       size_t*                     buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        _rocsparse_mat_descr   descr_csr;
        _rocsparse_spmat_descr mat_csr;
        rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_analysis_buffer_size(
            handle, rocsparse::cscsv_operation_to_csr(trans), &mat_csr, buffer_size));
        return rocsparse_status_success;
    }

    inline rocsparse_status cscsv_solve_buffer_size(rocsparse_handle            handle,
                                                    rocsparse_operation         trans,
                                                    rocsparse_const_spmat_descr A,
                                                    rocsparse_const_dnvec_descr x,
                                                    rocsparse_const_dnvec_descr y,
                                                    size_t*                     buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        _rocsparse_mat_descr   descr_csr;
        _rocsparse_spmat_descr mat_csr;
        rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);

        // conjugate_transpose runs as a non-transposed CSR solve with on-the-fly
        // conjugation, which needs the same extra buffer space as transposition.
        const rocsparse_operation trans_csr = (trans == rocsparse_operation_conjugate_transpose)
                                                  ? rocsparse_operation_transpose
                                                  : rocsparse::cscsv_operation_to_csr(trans);

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrsv_solve_buffer_size(handle, trans_csr, &mat_csr, x, y, buffer_size));
        return rocsparse_status_success;
    }

    inline rocsparse_status cscsv_analysis(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_analysis_policy   analysis_policy,
                                           rocsparse_solve_policy      solve_policy,
                                           rocsparse_csrsv_info*       p_csrsv_info,
                                           void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        _rocsparse_mat_descr   descr_csr;
        _rocsparse_spmat_descr mat_csr;
        rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrsv_analysis(handle,
                                      rocsparse::cscsv_operation_to_csr(trans),
                                      &mat_csr,
                                      analysis_policy,
                                      solve_policy,
                                      p_csrsv_info,
                                      temp_buffer));
        return rocsparse_status_success;
    }

    inline rocsparse_status cscsv_solve(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_datatype          alpha_datatype,
                                        const void*                 alpha,
                                        int64_t                     alpha_stride,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_const_dnvec_descr x,
                                        rocsparse_dnvec_descr       y,
                                        rocsparse_solve_policy      policy,
                                        rocsparse_csrsv_info        csrsv_info,
                                        void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        _rocsparse_mat_descr   descr_csr;
        _rocsparse_spmat_descr mat_csr;
        rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);

        // A conjugate-transpose CSC solve becomes a non-transposed CSR solve with
        // on-the-fly conjugation of the matrix values.
        const bool force_conj = (trans == rocsparse_operation_conjugate_transpose);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve(handle,
                                                         rocsparse::cscsv_operation_to_csr(trans),
                                                         alpha_datatype,
                                                         alpha,
                                                         alpha_stride,
                                                         &mat_csr,
                                                         x,
                                                         y,
                                                         policy,
                                                         csrsv_info,
                                                         temp_buffer,
                                                         force_conj));
        return rocsparse_status_success;
    }
}
