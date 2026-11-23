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
#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::coosm_analysis_buffer_size(rocsparse_handle            handle,
                                                       const int64_t               nrhs,
                                                       rocsparse_operation         op_A,
                                                       rocsparse_operation         op_B,
                                                       rocsparse_const_dnvec_descr alpha,
                                                       rocsparse_const_spmat_descr A,
                                                       rocsparse_const_dnmat_descr X,
                                                       size_t*          p_buffer_size_in_bytes,
                                                       rocsparse_error* p_error)
{

    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_indextype indextype = (A->nnz <= std::numeric_limits<int32_t>::max())
                                              ? rocsparse_indextype_i32
                                              : rocsparse_indextype_i64;

    //
    // Trick since it is not used in csrsm_buffer_size, otherwise we need to create a proper ptr array for nothing.
    //
    const void* const_ptr = (const void*)0x4;
    void*       ptr       = (void*)0x4;

    _rocsparse_spmat_descr A_csr(rocsparse_format_csr,
                                 A->batch_count,
                                 A->rows,
                                 A->cols,
                                 A->nnz,
                                 A->data_type,
                                 A->const_val_data,
                                 A->val_data,
                                 A->batch_stride,
                                 indextype,
                                 const_ptr,
                                 ptr,
                                 0,
                                 A->col_type,
                                 A->const_col_data,
                                 A->col_data,
                                 0,
                                 A->idx_base,
                                 A->descr,
                                 A->info);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_analysis_buffer_size(
        handle, nrhs, op_A, op_B, alpha, &A_csr, X, p_buffer_size_in_bytes, p_error));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::coosm_solve_buffer_size(rocsparse_handle            handle,
                                                    const int64_t               nrhs,
                                                    rocsparse_operation         op_A,
                                                    rocsparse_operation         op_B,
                                                    rocsparse_const_dnvec_descr alpha,
                                                    rocsparse_const_spmat_descr A,
                                                    rocsparse_const_dnmat_descr X,
                                                    size_t*          p_buffer_size_in_bytes,
                                                    rocsparse_error* p_error)
{

    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_indextype indextype = (A->nnz <= std::numeric_limits<int32_t>::max())
                                              ? rocsparse_indextype_i32
                                              : rocsparse_indextype_i64;

    //
    // Trick since it is not used in csrsm_buffer_size, otherwise we need to create a proper ptr array for nothing.
    //
    const void* const_ptr = (const void*)0x4;
    void*       ptr       = (void*)0x4;

    _rocsparse_spmat_descr A_csr(rocsparse_format_csr,
                                 A->batch_count,
                                 A->rows,
                                 A->cols,
                                 A->nnz,
                                 A->data_type,
                                 A->const_val_data,
                                 A->val_data,
                                 A->batch_stride,
                                 indextype,
                                 const_ptr,
                                 ptr,
                                 0,
                                 A->col_type,
                                 A->const_col_data,
                                 A->col_data,
                                 0,
                                 A->idx_base,
                                 A->descr,
                                 A->info);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve_buffer_size(
        handle, nrhs, op_A, op_B, alpha, &A_csr, X, p_buffer_size_in_bytes, p_error));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::coosm_buffer_size(rocsparse_handle            handle,
                                              const int64_t               nrhs,
                                              rocsparse_operation         op_A,
                                              rocsparse_operation         op_B,
                                              rocsparse_const_dnvec_descr alpha,
                                              rocsparse_const_spmat_descr A,
                                              rocsparse_const_dnmat_descr X,
                                              size_t*                     p_buffer_size_in_bytes,
                                              rocsparse_error*            p_error)
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_analysis_buffer_size(
        handle, nrhs, op_A, op_B, alpha, A, X, p_buffer_size_in_bytes, p_error));

    size_t buffer_size_in_bytes;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve_buffer_size(
        handle, nrhs, op_A, op_B, alpha, A, X, &buffer_size_in_bytes, p_error));

    p_buffer_size_in_bytes[0] = std::max(p_buffer_size_in_bytes[0], buffer_size_in_bytes);

    return rocsparse_status_success;
}
