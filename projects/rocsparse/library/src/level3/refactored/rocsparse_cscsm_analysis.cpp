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

#include "rocsparse_control.hpp"
#include "rocsparse_cscsm.hpp"
#include "rocsparse_csrsm.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::cscsm_analysis(rocsparse_handle            handle,
                                           int64_t                     nrhs,
                                           rocsparse_operation         op_A,
                                           rocsparse_operation         op_B,
                                           rocsparse_const_dnvec_descr alpha,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_const_dnmat_descr B,
                                           rocsparse_analysis_policy   analysis,
                                           rocsparse_csrsm_info*       p_csrsm_info,
                                           size_t                      buffer_size_in_bytes,
                                           void*                       buffer,
                                           rocsparse_error*            p_error)
{

    ROCSPARSE_ROUTINE_TRACE;

    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr A_csr(A, rocsparse_format_csr, &descr_csr, A->info);

    op_A = (op_A == rocsparse_operation_none) ? rocsparse_operation_transpose
                                              : rocsparse_operation_none;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_analysis(handle,
                                                        nrhs,
                                                        op_A,
                                                        op_B,
                                                        alpha,
                                                        &A_csr,
                                                        B,
                                                        analysis,
                                                        p_csrsm_info,
                                                        buffer_size_in_bytes,
                                                        buffer,
                                                        p_error));

    return rocsparse_status_success;
}
