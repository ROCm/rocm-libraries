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

#include "rocsparse_csrilu0.hpp"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_csrilu0_kernel_launch.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrilu0(rocsparse_handle          handle,
                                    rocsparse_spmat_descr     A,
                                    rocsparse_solve_policy    policy,
                                    rocsparse_csrilu0_info    csrilu0_info,
                                    rocsparse::numeric_boost* boost,
                                    size_t                    buffer_size,
                                    void*                     buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, A);
    ROCSPARSE_CHECKARG_ENUM(2, policy);

    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(3, csrilu0_info);

    ROCSPARSE_CHECKARG_POINTER(8, buffer);

    ROCSPARSE_CHECKARG(1, A, (A->descr == nullptr), rocsparse_status_invalid_pointer);

    ROCSPARSE_CHECKARG(
        1, A, (A->descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG(1,
                       A,
                       (A->descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    csrilu0_info->create_singularity_numeric_exact(A->batch_count, A->col_type, handle->stream);
    if(A->col_type == rocsparse_indextype_i32)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
            A->batch_count,
            (int32_t*)csrilu0_info->get_singularity_numeric_exact()->get_position(),
            (const int32_t*)csrilu0_info->get_position(),
            handle->stream));
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
            A->batch_count,
            (int64_t*)csrilu0_info->get_singularity_numeric_exact()->get_position(),
            (const int64_t*)csrilu0_info->get_position(),
            handle->stream));
    }

    csrilu0_info->create_singularity_numeric_near(A->batch_count, A->col_type, handle->stream);

    if(A->col_type == rocsparse_indextype_i32)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
            A->batch_count,
            (int32_t*)csrilu0_info->get_singularity_numeric_near()->get_position(),
            (const int32_t*)csrilu0_info->get_position(),
            handle->stream));
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
            A->batch_count,
            (int64_t*)csrilu0_info->get_singularity_numeric_near()->get_position(),
            (const int64_t*)csrilu0_info->get_position(),
            handle->stream));
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csrilu0_kernel_launch(handle, csrilu0_info, A, boost, buffer_size, buffer));
    return rocsparse_status_success;
}
