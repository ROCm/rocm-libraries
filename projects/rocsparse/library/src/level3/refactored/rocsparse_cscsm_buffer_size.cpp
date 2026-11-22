/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_control.hpp"
#include "rocsparse_cscsm.hpp"
#include "rocsparse_csrsm.hpp"
#include "rocsparse_utility.hpp"

#include "../../level2/rocsparse_cscsv.hpp"
#include "../../level2/rocsparse_cscsv_build_tasks_kernel.hpp"

bool rocsparse::cscsm_use_limited_memory(rocsparse_operation         op_A,
                                         rocsparse_operation         op_B,
                                         rocsparse_const_dnvec_descr alpha,
                                         rocsparse_const_spmat_descr A,
                                         rocsparse_const_dnmat_descr B,
                                         bool                        limited_memory)
{
#if defined(ROCSPARSE_WITH_SPTRSM_CSC)
    return limited_memory && (op_A == rocsparse_operation_none)
           && (op_B == rocsparse_operation_none || op_B == rocsparse_operation_transpose)
           && (A->rows == A->cols) && (A->descr != nullptr)
           && (A->descr->storage_mode == rocsparse_storage_mode_sorted)
           && (A->descr->type == rocsparse_matrix_type_general
               || A->descr->type == rocsparse_matrix_type_triangular)
           && (B->data_type == A->data_type)
           && (alpha == nullptr || alpha->data_type == A->data_type);
#else
    return false;
#endif
}

size_t rocsparse::cscsm_push_analysis_buffer_size(rocsparse_handle            handle,
                                                  rocsparse_const_spmat_descr A)
{
    size_t size{};
#if defined(ROCSPARSE_WITH_SPTRSM_CSC)
    if(A->cols > 0)
    {
        rocsparse::cscsv_build_tasks_buffer_size_t launch;
        if(rocsparse::cscsv_build_tasks_buffer_size_find(&launch, A->col_type)
               == rocsparse_status_success
           && launch(handle, A->cols, &size) == rocsparse_status_success)
        {
            return size;
        }
    }
#endif
    return size;
}

size_t rocsparse::cscsm_push_solve_buffer_size(int64_t                     nrhs,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_const_dnmat_descr B)
{
    const int64_t m       = A->cols;
    const int64_t panel   = rocsparse::cscsm_panel_width(nrhs);
    const int64_t npanels = (nrhs + panel - 1) / panel;
    const int64_t nslots  = npanels * B->batch_count;
    // Tickets of the batch instances, dependency counters of each (batch, panel) and, with one
    // right-hand side, the partial sums of the CSC sptrsv kernels. With more right-hand sides, the
    // partial sums are accumulated in B.
    return rocsparse::align_size(sizeof(int32_t) * B->batch_count)
           + rocsparse::align_size(rocsparse::indextype_sizeof(A->row_type) * m * nslots)
           + ((nrhs == 1) ? rocsparse::align_size(rocsparse::datatype_sizeof(A->data_type) * m
                                                  * B->batch_count)
                          : 0);
}

rocsparse_status rocsparse::cscsm_analysis_buffer_size(rocsparse_handle            handle,
                                                       const int64_t               nrhs,
                                                       rocsparse_operation         op_A,
                                                       rocsparse_operation         op_B,
                                                       rocsparse_const_dnvec_descr alpha,
                                                       rocsparse_const_spmat_descr A,
                                                       rocsparse_const_dnmat_descr X,
                                                       size_t*          p_buffer_size_in_bytes,
                                                       rocsparse_error* p_error,
                                                       bool             limited_memory)
{

    ROCSPARSE_ROUTINE_TRACE;

    if(rocsparse::cscsm_use_limited_memory(op_A, op_B, alpha, A, X, limited_memory))
    {
        p_buffer_size_in_bytes[0] = rocsparse::cscsm_push_analysis_buffer_size(handle, A);
        return rocsparse_status_success;
    }

    //
    // To avoid runtime allocation of the descriptor.
    //
    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr A_csr(A, rocsparse_format_csr, &descr_csr, A->info);

#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
    //
    // A points now to the local A_csr.
    //
    A = &A_csr;
#endif
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_analysis_buffer_size(
        handle,
        nrhs,

        (op_A == rocsparse_operation_none) ? rocsparse_operation_transpose
                                           : rocsparse_operation_none,

        op_B,
        alpha,
#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
        A,
#else
        &A_csr,
#endif
        X,
        p_buffer_size_in_bytes,
        p_error));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsm_solve_buffer_size(rocsparse_handle            handle,
                                                    const int64_t               nrhs,
                                                    rocsparse_operation         op_A,
                                                    rocsparse_operation         op_B,
                                                    rocsparse_const_dnvec_descr alpha,
                                                    rocsparse_const_spmat_descr A,
                                                    rocsparse_const_dnmat_descr X,
                                                    size_t*          p_buffer_size_in_bytes,
                                                    rocsparse_error* p_error,
                                                    bool             limited_memory)
{

    ROCSPARSE_ROUTINE_TRACE;

    if(rocsparse::cscsm_use_limited_memory(op_A, op_B, alpha, A, X, limited_memory))
    {
        p_buffer_size_in_bytes[0] = rocsparse::cscsm_push_solve_buffer_size(nrhs, A, X);
        return rocsparse_status_success;
    }

    //
    // To avoid runtime allocation of the descriptor.
    //
    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr A_csr(A, rocsparse_format_csr, &descr_csr, A->info);

#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
    //
    // A points now to the local A_csr.
    //
    A = &A_csr;
#endif

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve_buffer_size(handle,
                                                                 nrhs,

                                                                 (op_A == rocsparse_operation_none)
                                                                     ? rocsparse_operation_transpose
                                                                     : rocsparse_operation_none,

                                                                 op_B,
                                                                 alpha,
#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
                                                                 A,
#else
                                                                 &A_csr,
#endif
                                                                 X,
                                                                 p_buffer_size_in_bytes,
                                                                 p_error));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsm_buffer_size(rocsparse_handle            handle,
                                              const int64_t               nrhs,
                                              rocsparse_operation         op_A,
                                              rocsparse_operation         op_B,
                                              rocsparse_const_dnvec_descr alpha,
                                              rocsparse_const_spmat_descr A,
                                              rocsparse_const_dnmat_descr X,
                                              size_t*                     p_buffer_size_in_bytes,
                                              rocsparse_error*            p_error,
                                              bool                        limited_memory)
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsm_analysis_buffer_size(
        handle, nrhs, op_A, op_B, alpha, A, X, p_buffer_size_in_bytes, p_error, limited_memory));

    size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsm_solve_buffer_size(
        handle, nrhs, op_A, op_B, alpha, A, X, &buffer_size_in_bytes, p_error, limited_memory));

    p_buffer_size_in_bytes[0] = std::max(p_buffer_size_in_bytes[0], buffer_size_in_bytes);

    return rocsparse_status_success;
}
