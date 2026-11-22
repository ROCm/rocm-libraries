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

    // Whether the limited-memory version is used for a CSC matrix without transpose. It does not
    // build the transposed matrix: once an unknown is computed, its column of the matrix updates
    // the unknowns that depend on it. Otherwise, the transposed CSR system is solved.
    static inline bool cscsv_use_limited_memory(rocsparse_operation trans, bool limited_memory)
    {
#if defined(ROCSPARSE_WITH_SPTRSV_CSC)
        return limited_memory && trans == rocsparse_operation_none;
#else
        return false;
#endif
    }

    // A CSR matrix with transpose is the CSC matrix without transpose that shares its arrays (see
    // build_csc_from_csr), and with conjugate transpose the same with conjugated values, so it can
    // use the limited-memory version too.
    static inline bool csrsv_use_limited_memory(rocsparse_operation trans, bool limited_memory)
    {
#if defined(ROCSPARSE_WITH_SPTRSV_CSC)
        return limited_memory && trans != rocsparse_operation_none;
#else
        return false;
#endif
    }

    static inline bool is_limited_memory(rocsparse_sptrsv_alg alg)
    {
#if defined(ROCSPARSE_WITH_SPTRSV_CSC)
        return alg == rocsparse_sptrsv_alg_limited_memory;
#else
        return false;
#endif
    }

    static inline bool is_limited_memory(rocsparse_spsv_alg alg)
    {
#if defined(ROCSPARSE_WITH_SPTRSV_CSC)
        return alg == rocsparse_spsv_alg_limited_memory;
#else
        return false;
#endif
    }

    // Square, general or triangular, sorted: the CSC solve without transpose does not go
    // through the csrsv checks.
    rocsparse_status cscsv_checkarg_matrix_properties(rocsparse_const_spmat_descr A);

    // Lanes per task: smallest power of two >= min_lanes, at least 2, at most the wavefront size.
    uint32_t cscsv_num_lanes(int64_t min_lanes, uint32_t wavefront_size);

    // Buffer of the solve without transpose: ticket, remaining dependencies and partial sums of
    // each batch instance.
    struct cscsv_solve_buffer_layout_t
    {
        cscsv_solve_buffer_layout_t(int64_t             m,
                                    int64_t             batch_count,
                                    rocsparse_indextype j_type,
                                    rocsparse_datatype  t_type);

        size_t   size() const;
        size_t   ticket_size() const;
        int32_t* ticket(void* buffer) const;
        void*    rem(void* buffer) const;
        void*    sum(void* buffer) const;

    private:
        size_t m_ticket_size{};
        size_t m_rem_size{};
        size_t m_sum_size{};
    };

    // CSC triangular solve, with the limited-memory version if
    // cscsv_use_limited_memory(trans, limited_memory). conj conjugates the matrix values in the
    // limited-memory version (CSR matrix with conjugate transpose, see build_csc_from_csr).
    rocsparse_status cscsv_analysis_buffer_size(rocsparse_handle            handle,
                                                rocsparse_operation         trans,
                                                rocsparse_const_spmat_descr A,
                                                size_t*                     buffer_size_in_bytes,
                                                bool                        limited_memory);

    bool cscsv_is_analyzed(rocsparse_cscsv_info        info,
                           rocsparse_operation         trans,
                           rocsparse_const_spmat_descr A,
                           bool                        limited_memory);

    rocsparse_status cscsv_solve_buffer_size(rocsparse_handle            handle,
                                             rocsparse_operation         trans,
                                             rocsparse_const_spmat_descr A,
                                             rocsparse_const_dnvec_descr x,
                                             rocsparse_const_dnvec_descr y,
                                             size_t*                     buffer_size_in_bytes,
                                             bool                        limited_memory);

    rocsparse_status cscsv_analysis(rocsparse_handle            handle,
                                    rocsparse_operation         trans,
                                    rocsparse_const_spmat_descr A,
                                    rocsparse_analysis_policy   analysis_policy,
                                    rocsparse_solve_policy      solve_policy,
                                    rocsparse_cscsv_info        info,
                                    size_t                      buffer_size_in_bytes,
                                    void*                       buffer,
                                    bool                        limited_memory);

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
                                 void*                       buffer,
                                 bool                        limited_memory,
                                 bool                        conj);

    rocsparse_status cscsv_diagonal_solve(rocsparse_handle            handle,
                                          rocsparse_operation         trans,
                                          rocsparse_diagonal_modifier modifier,
                                          const void*                 alpha,
                                          rocsparse_const_spmat_descr A,
                                          rocsparse_const_dnvec_descr x,
                                          rocsparse_dnvec_descr       y,
                                          rocsparse_cscsv_info        info,
                                          bool                        limited_memory,
                                          bool                        conj);

    // Each batch instance starts from the structural zero pivot of the analysis, or from
    // no pivot if the diagonal is unit.
    rocsparse_status cscsv_seed_zero_pivot(rocsparse_handle     handle,
                                           rocsparse_indextype  indextype,
                                           rocsparse_diag_type  diag_type,
                                           int64_t              batch_count,
                                           rocsparse_cscsv_info info);

    rocsparse_status cscsv_zero_pivot(rocsparse_handle     handle,
                                      rocsparse_cscsv_info info,
                                      rocsparse_indextype  indextype,
                                      void*                position);
}
