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

#include "rocsparse_csric0.hpp"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_csric0_kernel_launch.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csric0(rocsparse_handle      handle,
                                   rocsparse_csric0_info csric0_info,
                                   rocsparse_spmat_descr A,
                                   size_t                buffer_size,
                                   void*                 buffer)
{
    if(A->rows == 0)
    {
        //
        // Quick return
        //
        return rocsparse_status_success;
    }

    csric0_info->create_singularity_numeric_near(A->batch_count, A->col_type, handle->stream);
    csric0_info->create_singularity_numeric_exact(A->batch_count, A->col_type, handle->stream);
    if(A->col_type == rocsparse_indextype_i32)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
            A->batch_count,
            reinterpret_cast<int32_t*>(csric0_info->get_singularity_numeric_near()->get_position()),
            reinterpret_cast<const int32_t*>(csric0_info->get_position()),
            handle->stream));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
            A->batch_count,
            reinterpret_cast<int32_t*>(
                csric0_info->get_singularity_numeric_exact()->get_position()),
            reinterpret_cast<const int32_t*>(csric0_info->get_position()),
            handle->stream));
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
            A->batch_count,
            reinterpret_cast<int64_t*>(csric0_info->get_singularity_numeric_near()->get_position()),
            reinterpret_cast<const int64_t*>(csric0_info->get_position()),
            handle->stream));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
            A->batch_count,
            reinterpret_cast<int64_t*>(
                csric0_info->get_singularity_numeric_exact()->get_position()),
            reinterpret_cast<const int64_t*>(csric0_info->get_position()),
            handle->stream));
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csric0_kernel_launch(handle, csric0_info, A, buffer_size, buffer));

    return rocsparse_status_success;
}
