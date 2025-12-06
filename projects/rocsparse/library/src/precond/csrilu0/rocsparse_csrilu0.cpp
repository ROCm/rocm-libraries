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

#include "rocsparse_csrilu0.hpp"
#include "rocsparse_csrilu0_kernel_launch.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrilu0_solve(rocsparse_handle       handle,
                                          rocsparse_spmat_descr  A,
                                          rocsparse_solve_policy policy,
                                          rocsparse_csrilu0_info csrilu0_info,
                                          int                    boost_enable,
                                          size_t                 boost_tol_size,
                                          const void*            boost_tol,
                                          const void*            boost_val,
                                          void*                  temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, A);
    ROCSPARSE_CHECKARG_ENUM(2, policy);
    ROCSPARSE_CHECKARG_POINTER(3, csrilu0_info);

    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(8, temp_buffer);

    ROCSPARSE_CHECKARG(1, A, (A->descr == nullptr), rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(
        1, A, (A->descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(1,
                       A,
                       (A->descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    auto        trm_info = csrilu0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
    hipStream_t stream   = handle->stream;

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    ptr += 256;

    // done array
    int32_t* d_done_array = reinterpret_cast<int32_t*>(ptr);

    // Initialize buffers
    RETURN_IF_HIP_ERROR(
        hipMemsetAsync(d_done_array, 0, sizeof(int32_t) * A->rows * A->batch_count, stream));
    const int64_t d_done_array_stride = A->rows;

    // Max nnz per row
    const int64_t max_nnz = trm_info->get_max_nnz();

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csrilu0_strided_batched_kernel_launch(handle,
                                                         A->batch_count,
                                                         A->rows,
                                                         A->row_type,
                                                         A->const_row_data,
                                                         A->col_type,
                                                         A->const_col_data,
                                                         A->data_type,
                                                         A->val_data,
                                                         A->batch_stride,
                                                         trm_info->get_diag_ind(),
                                                         d_done_array,
                                                         d_done_array_stride,
                                                         trm_info->get_row_map(),
                                                         csrilu0_info->get_zero_pivot(),
                                                         csrilu0_info->get_zero_pivot_stride(),
                                                         csrilu0_info->get_singular_pivot(),
                                                         csrilu0_info->get_singular_pivot_stride(),
                                                         csrilu0_info->get_singular_tol(),
                                                         A->descr->base,
                                                         boost_enable,
                                                         boost_tol_size,
                                                         boost_tol,
                                                         boost_val,
                                                         max_nnz));
    return rocsparse_status_success;
}
