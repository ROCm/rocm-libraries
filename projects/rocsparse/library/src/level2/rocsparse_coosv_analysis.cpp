/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_coosv.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::coosv_analysis(rocsparse_handle            handle, // 0
                                           rocsparse_operation         trans, // 1
                                           rocsparse_const_spmat_descr A, // 2
                                           rocsparse_analysis_policy   analysis, // 3
                                           rocsparse_solve_policy      solve, // 4
                                           rocsparse_csrsv_info*       p_csrsv_info, // 5
                                           void*                       buffer) // 6
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_ENUM(3, analysis);
    ROCSPARSE_CHECKARG_ENUM(4, solve);
    ROCSPARSE_CHECKARG_POINTER(5, p_csrsv_info);
    ROCSPARSE_CHECKARG_ARRAY(6, (A->rows > 0 && A->batch_count > 0), buffer);

    // Quick return if possible
    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }
    rocsparse_mat_descr descr = A->descr;
    rocsparse_mat_info  info  = A->info;
    // Check matrix type
    ROCSPARSE_CHECKARG(5,
                       descr,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);
    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(5,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // All must be null (zero matrix) or none null
    const bool                use_32 = A->nnz < std::numeric_limits<int32_t>::max();
    const rocsparse_indextype indextype
        = use_32 ? rocsparse_indextype_i32 : rocsparse_indextype_i64;

    // Buffer
    rocsparse::sorted_coo2csr_info_t* sorted_coo2csr_info = info->get_sorted_coo2csr_info();
    if(sorted_coo2csr_info == nullptr)
    {
        sorted_coo2csr_info
            = new rocsparse::sorted_coo2csr_info_t(A->rows, indextype, handle->stream);

        //
        // Assign it first, because if an error occurs in calculate below, then we won't have a memory leak.
        //
        info->set_sorted_coo2csr_info(sorted_coo2csr_info);
        RETURN_IF_ROCSPARSE_ERROR(
            sorted_coo2csr_info->calculate(handle, A->nnz, A->row_data, A->row_type, descr->base));
    }

    const rocsparse_indextype csr_row_ptr_indextype
        = (use_32) ? rocsparse_indextype_i32 : rocsparse_indextype_i64;
    const void*   csr_row_ptr        = sorted_coo2csr_info->get_row_ptr();
    const int64_t csr_row_ptr_stride = (A->offsets_batch_stride == 0) ? 0 : (A->rows + 1);

    _rocsparse_spmat_descr csr(rocsparse_format_csr,
                               A->analysed,
                               A->batch_count,
                               A->rows,
                               A->cols,
                               A->nnz,

                               A->data_type,
                               A->const_val_data,
                               A->val_data,
                               A->batch_stride,

                               //
                               csr_row_ptr_indextype,
                               csr_row_ptr,
                               nullptr,
                               csr_row_ptr_stride,

                               A->col_type,
                               A->const_col_data,
                               A->col_data,
                               A->columns_values_batch_stride,

                               A->idx_base,
                               A->descr,
                               A->info);

    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::csrsv_analysis(handle, trans, &csr, analysis, solve, p_csrsv_info, buffer)));

    return rocsparse_status_success;
}
