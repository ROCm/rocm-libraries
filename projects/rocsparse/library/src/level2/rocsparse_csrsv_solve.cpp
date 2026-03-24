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

#include "../level1/rocsparse_gthr.hpp"
#include "csrsv_device.h"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_csrsv_solve_kernel.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrsv_solve(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_datatype          alpha_datatype,
                                        const void*                 alpha,
                                        int64_t                     alpha_stride,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_const_dnvec_descr x,
                                        rocsparse_dnvec_descr       y,
                                        rocsparse_solve_policy      policy,
                                        rocsparse_csrsv_info        csrsv_info,
                                        void*                       temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const int64_t batch_count = y->batch_count;

    // Quick return if possible
    if(A->rows == 0 || batch_count == 0)
    {
        return rocsparse_status_success;
    }

    rocsparse_mat_descr descr = A->descr;
    // Check matrix type
    ROCSPARSE_CHECKARG(8,
                       descr,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);

    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(8,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Stream
    hipStream_t stream = handle->stream;

    const rocsparse_diag_type diag_type = descr->diag_type;

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    ptr += 256;
    // done array
    int32_t*     done_array = reinterpret_cast<int32_t*>(ptr);
    const size_t done_array_size_in_bytes
        = ((sizeof(int32_t) * A->rows * batch_count - 1) / 256 + 1) * 256;
    ptr += done_array_size_in_bytes;

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemsetAsync(done_array, 0, done_array_size_in_bytes, stream));

    const rocsparse::trm_info_t* csrsv = csrsv_info->get(trans, descr->fill_mode);
    if(csrsv == nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    csrsv_info->create_singularity_numeric_exact(batch_count, A->col_type, handle->stream);

    // If diag type is unit, re-initialize zero pivot to remove structural zeros

    switch(diag_type)
    {
    case rocsparse_diag_type_unit:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::assign_max_async(1, A->col_type, csrsv_info->get_position(), stream));
        if(A->col_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
                batch_count,
                (int32_t*)csrsv_info->get_singularity_numeric_exact()->get_position(),
                (const int32_t*)csrsv_info->get_position(),
                handle->stream));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
                batch_count,
                (int64_t*)csrsv_info->get_singularity_numeric_exact()->get_position(),
                (const int64_t*)csrsv_info->get_position(),
                handle->stream));
        }

        break;
    }
    case rocsparse_diag_type_non_unit:
    {
        if(A->col_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
                batch_count,
                (int32_t*)csrsv_info->get_singularity_numeric_exact()->get_position(),
                (const int32_t*)csrsv_info->get_position(),
                handle->stream));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
                batch_count,
                (int64_t*)csrsv_info->get_singularity_numeric_exact()->get_position(),
                (const int64_t*)csrsv_info->get_position(),
                handle->stream));
        }
        break;
    }
    }

    // Pointers to differentiate between transpose mode
    const void* local_row_data = A->const_row_data;
    const void* local_col_data = A->const_col_data;
    const void* local_val_data = A->const_val_data;

    int64_t             local_val_data_inc    = 1;
    int64_t             local_val_data_stride = A->batch_stride;
    rocsparse_fill_mode fill_mode             = descr->fill_mode;

    // When computing transposed triangular solve, we first need to update the
    // transposed matrix values
    if(trans == rocsparse_operation_transpose || trans == rocsparse_operation_conjugate_transpose)
    {
        void*                    csrt_val          = ptr;
        const int64_t            csrt_val_stride   = A->nnz;
        const rocsparse_datatype csrt_val_datatype = A->data_type;

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::gthr_strided_batched(handle,
                                                                   A->batch_count,
                                                                   A->nnz,
                                                                   A->data_type,
                                                                   A->const_val_data,
                                                                   A->batch_stride,
                                                                   csrt_val_datatype,
                                                                   csrt_val,
                                                                   csrt_val_stride,
                                                                   A->row_type,
                                                                   csrsv->get_transposed_perm(),
                                                                   rocsparse_index_base_zero)));

        if(trans == rocsparse_operation_conjugate_transpose)
        {
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::conjugate_strided_batched(
                handle, A->batch_count, A->nnz, A->data_type, csrt_val, csrt_val_stride)));
        }

        local_row_data        = csrsv->get_transposed_row_ptr();
        local_col_data        = csrsv->get_transposed_col_ind();
        local_val_data        = csrt_val;
        local_val_data_stride = (A->batch_count > 1) ? A->nnz : 0;
        fill_mode             = (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                                         : rocsparse_fill_mode_lower;
    }

    const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
    const int         asicRev       = handle->asic_rev;
    const bool        sleep_  = (gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2);
    const uint32_t    wfsize_ = sleep_ ? 64 : handle->wavefront_size;
    rocsparse::csrsv_launch_kernel_t csrsv_launch_kernel{};
    RETURN_IF_ROCSPARSE_ERROR(csrsv_launch_kernel_find(
        &csrsv_launch_kernel, 1024, wfsize_, sleep_, A->row_type, A->col_type, A->data_type));

#undef CSRSV_DIM
    auto numeric_exact_position = csrsv_info->get_singularity_numeric_exact();
    csrsv_launch_kernel(handle,
                        batch_count,
                        A->rows,
                        alpha,
                        alpha_stride,
                        local_row_data,
                        local_col_data,
                        local_val_data,
                        local_val_data_inc,
                        local_val_data_stride,
                        x->const_values,
                        x->inc,
                        x->batch_stride,
                        y->values,
                        y->inc,
                        y->batch_stride,
                        done_array,
                        csrsv->get_row_map(),
                        0,
                        numeric_exact_position->get_position(),
                        1,
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
    return rocsparse_status_success;
}
