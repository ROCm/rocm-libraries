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

#include "rocsparse_csric0.hpp"
#include "rocsparse_csric0_launch_kernel.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csric0(rocsparse_handle       handle,
                                   rocsparse_spmat_descr  A,
                                   rocsparse_solve_policy policy,
                                   rocsparse_csric0_info  csric0_info,
                                   void*                  buffer)
{
    if(A->rows == 0)
    {
        //
        // Quick return
        //
        return rocsparse_status_success;
    }

    auto info          = A->info;
    auto descr         = A->descr;
    auto trm_info      = info->get_csric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
    hipStream_t stream = handle->stream;

    char* ptr = reinterpret_cast<char*>(buffer);
    ptr += 256;

    int32_t*      d_done_array        = reinterpret_cast<int32_t*>(ptr);
    const int64_t d_done_array_stride = A->rows;

    RETURN_IF_HIP_ERROR(
        hipMemsetAsync(d_done_array, 0, sizeof(int32_t) * A->rows * A->batch_count, stream));

    const int64_t max_nnz                = trm_info->get_max_nnz();
    void*         zero_pivots            = csric0_info->get_zero_pivot();
    const int64_t zero_pivots_stride     = 1;
    void*         singular_pivots        = csric0_info->get_singular_pivot();
    const int64_t singular_pivots_stride = 1;
    const void*   row_map                = trm_info->get_row_map();
    const void*   csr_diag_ind           = trm_info->get_diag_ind();
    const double  singular_tol           = csric0_info->get_singular_tol();
    const auto    base                   = descr->base;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csric0_launch_kernel(handle,
                                                              A->batch_count,
                                                              A->rows,
                                                              A->row_type,
                                                              A->const_row_data,
                                                              A->col_type,
                                                              A->const_col_data,
                                                              A->data_type,
                                                              A->val_data,
                                                              A->batch_stride,
                                                              csr_diag_ind,
                                                              d_done_array,
                                                              d_done_array_stride,
                                                              row_map,
                                                              zero_pivots,
                                                              zero_pivots_stride,
                                                              singular_pivots,
                                                              singular_pivots_stride,
                                                              singular_tol,
                                                              base,
                                                              max_nnz));

    return rocsparse_status_success;
}
