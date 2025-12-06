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

#include "rocsparse_bsrilu0.hpp"
#include "rocsparse_bsrilu0_strided_batched_kernel_launch.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::bsrilu0(rocsparse_handle       handle,
                                    rocsparse_spmat_descr  A,
                                    rocsparse_solve_policy policy,
                                    rocsparse_bsrilu0_info bsrilu0_info,
                                    void*                  temp_buffer)
{
    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }

    rocsparse::trm_info_t* trm_info
        = bsrilu0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
    ROCSPARSE_CHECKARG(3,
                       bsrilu0_info,
                       ((A->rows > 0) && (trm_info == nullptr)),
                       rocsparse_status_invalid_pointer);

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    ptr += 256;

    // done array
    int32_t*      done_array        = reinterpret_cast<int32_t*>(ptr);
    const int64_t done_array_stride = A->rows;

    // Initialize buffers
    RETURN_IF_HIP_ERROR(
        hipMemsetAsync(done_array, 0, sizeof(int32_t) * A->rows * A->batch_count, handle->stream));
    void*         zero_pivot        = bsrilu0_info->get_zero_pivot();
    const int64_t zero_pivot_stride = bsrilu0_info->get_zero_pivot_stride();
    const void*   bsr_diag_ind      = trm_info->get_diag_ind();
    const void*   row_map           = trm_info->get_row_map();

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::bsrilu0_strided_batched_kernel_launch(handle,
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
                                                         A->descr->base,
                                                         A->info->boost_enable,
                                                         A->info->boost_tol_size,
                                                         A->info->boost_tol,
                                                         A->info->boost_val));

    return rocsparse_status_success;
}
