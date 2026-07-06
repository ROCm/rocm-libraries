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

rocsparse_status rocsparse::coosv_solve(rocsparse_handle            handle, // 0
                                        rocsparse_operation         trans, // 1
                                        rocsparse_datatype          alpha_datatype, //2
                                        const void*                 alpha, // 3
                                        int64_t                     alpha_stride, // 4
                                        rocsparse_const_spmat_descr A, // 5
                                        rocsparse_const_dnvec_descr x, // 6
                                        rocsparse_dnvec_descr       y, // 7
                                        rocsparse_solve_policy      policy, // 8
                                        rocsparse_csrsv_info        csrsv_info, // 9
                                        void*                       temp_buffer) // 10
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_ENUM(2, alpha_datatype);
    ROCSPARSE_CHECKARG_POINTER(5, A);
    ROCSPARSE_CHECKARG_ARRAY(3, A->batch_count, alpha);
    ROCSPARSE_CHECKARG_POINTER(6, x);
    ROCSPARSE_CHECKARG_POINTER(7, y);
    ROCSPARSE_CHECKARG_ENUM(8, policy);

    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(9, csrsv_info);
    rocsparse_mat_descr descr = A->descr;
    rocsparse_mat_info  info  = A->info;
    // Check matrix type
    ROCSPARSE_CHECKARG(5,
                       A,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);

    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(5,
                       A,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    rocsparse::sorted_coo2csr_info_t* sorted_coo2csr_info = info->get_sorted_coo2csr_info();
    if(sorted_coo2csr_info == nullptr)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_internal_error,
            "sorted_coo2csr_info is not available, it looks like the analysis phase of this "
            "algorithm was not previously executed.");
    }

    const bool                use_32 = A->nnz < std::numeric_limits<int32_t>::max();
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

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_solve(handle,
                                                      trans,
                                                      alpha_datatype,
                                                      alpha,
                                                      alpha_stride,
                                                      &csr,
                                                      x,
                                                      y,
                                                      policy,
                                                      csrsv_info,
                                                      temp_buffer)));

    return rocsparse_status_success;
}
