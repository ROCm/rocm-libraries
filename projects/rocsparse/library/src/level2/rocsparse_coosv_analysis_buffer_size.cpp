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

rocsparse_status rocsparse::coosv_analysis_buffer_size(rocsparse_handle            handle,
                                                       rocsparse_operation         trans,
                                                       rocsparse_const_spmat_descr A,
                                                       size_t* buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_POINTER(3, buffer_size_in_bytes);

    // Quick return if possible
    if(A->rows == 0 || A->batch_count == 0)
    {
        *buffer_size_in_bytes = 0;
        return rocsparse_status_success;
    }

    rocsparse_mat_descr descr = A->descr;
    ROCSPARSE_CHECKARG(2,
                       descr,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(2,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    *buffer_size_in_bytes            = 0;
    const bool                use_32 = (A->nnz < std::numeric_limits<int32_t>::max());
    const rocsparse_indextype indextype
        = (use_32) ? rocsparse_indextype_i32 : rocsparse_indextype_i64;

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

                               indextype,
                               (const void*)0x4,
                               (void*)0x4,
                               0,

                               A->col_type,
                               A->const_col_data,
                               A->col_data,
                               A->columns_values_batch_stride,

                               A->idx_base,
                               A->descr,
                               A->info);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csrsv_analysis_buffer_size(handle, trans, &csr, buffer_size_in_bytes));

    return rocsparse_status_success;
}
