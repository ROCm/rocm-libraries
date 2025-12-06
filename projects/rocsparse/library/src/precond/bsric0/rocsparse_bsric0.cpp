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

#include "rocsparse_bsric0.hpp"
#include "rocsparse_bsric0_strided_batched_kernel_launch.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::bsric0(rocsparse_handle       handle,
                                   rocsparse_spmat_descr  A,
                                   rocsparse_solve_policy policy,
                                   rocsparse_bsric0_info  bsric0_info,
                                   void*                  buffer)
{
    const bool quick_return = (A->rows == 0 || A->batch_count == 0);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, A);
    ROCSPARSE_CHECKARG_ENUM(2, policy);
    ROCSPARSE_CHECKARG_POINTER(3, bsric0_info);
    ROCSPARSE_CHECKARG(4,
                       buffer,
                       ((buffer == nullptr) && (quick_return == false)),
                       rocsparse_status_invalid_pointer);

    if(quick_return)
    {
        return rocsparse_status_success;
    }

    const rocsparse_mat_descr descr = A->descr;
    ROCSPARSE_CHECKARG(
        1, A, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG(1,
                       A,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    rocsparse::trm_info_t* trm_info
        = bsric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
    ROCSPARSE_CHECKARG(4, bsric0_info, (trm_info == nullptr), rocsparse_status_invalid_pointer);

    // Buffer
    char* ptr = reinterpret_cast<char*>(buffer);
    ptr += 256;

    // done array
    int32_t*      done_array        = reinterpret_cast<int32_t*>(ptr);
    const int64_t done_array_stride = A->rows;

    // Initialize buffers
    RETURN_IF_HIP_ERROR(
        hipMemsetAsync(done_array, 0, sizeof(int32_t) * A->rows * A->batch_count, handle->stream));
    void*         zero_pivot        = bsric0_info->get_zero_pivot();
    const int64_t zero_pivot_stride = bsric0_info->get_zero_pivot_stride();
    const void*   bsr_diag_ind      = trm_info->get_diag_ind();
    const void*   row_map           = trm_info->get_row_map();
    const int64_t max_nnzb          = trm_info->get_max_nnz();

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsric0_strided_batched_kernel_launch(handle,
                                                                              A->block_dir,
                                                                              A->batch_count,
                                                                              A->rows,
                                                                              A->data_type,
                                                                              A->val_data,
                                                                              A->batch_stride,
                                                                              A->row_type,
                                                                              A->const_row_data,
                                                                              A->col_type,
                                                                              A->const_col_data,
                                                                              bsr_diag_ind,
                                                                              A->block_dim,
                                                                              done_array,
                                                                              done_array_stride,
                                                                              row_map,
                                                                              zero_pivot,
                                                                              zero_pivot_stride,
                                                                              descr->base,
                                                                              max_nnzb));

    return rocsparse_status_success;
}
