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

#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::coosm_compute(rocsparse_handle            handle,
                                          const int64_t               nrhs,
                                          rocsparse_operation         op_A,
                                          rocsparse_operation         op_B,
                                          rocsparse_const_dnvec_descr alpha,
                                          rocsparse_const_spmat_descr A,
                                          rocsparse_dnmat_descr       B,
                                          rocsparse_csrsm_info        csrsm_info,
                                          size_t                      buffer_size_in_bytes,
                                          void*                       buffer,
                                          rocsparse_error*            p_error)
{

    if(nrhs == 0)
    {
        return rocsparse_status_success;
    }
    rocsparse::sorted_coo2csr_info_t* sorted_coo2csr_info = A->info->get_sorted_coo2csr_info();
    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
        (sorted_coo2csr_info) ? rocsparse_status_success : rocsparse_status_internal_error,
        "sorted_coo2csr_info is not available, it looks like the analysis phase of this "
        "algorithm was not previously executed.");

    _rocsparse_spmat_descr A_csr(rocsparse_format_csr,
                                 A->batch_count,
                                 A->rows,
                                 A->cols,
                                 A->nnz,
                                 A->data_type,
                                 A->const_val_data,
                                 A->val_data,
                                 A->batch_stride,
                                 sorted_coo2csr_info->get_row_ptr_indextype(),
                                 sorted_coo2csr_info->get_row_ptr(),
                                 nullptr,
                                 0,
                                 A->col_type,
                                 A->const_col_data,
                                 A->col_data,
                                 0,
                                 A->idx_base,
                                 A->descr,
                                 A->info);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_compute(handle,
                                                       nrhs,
                                                       op_A,
                                                       op_B,
                                                       alpha,
                                                       &A_csr,
                                                       B,
                                                       csrsm_info,
                                                       buffer_size_in_bytes,
                                                       buffer,
                                                       p_error));

    return rocsparse_status_success;
}
