/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_csritsv.h"
#include "control.h"
#include "rocsparse_csritsv.hpp"
#include "utility.h"

extern "C" rocsparse_status rocsparse_csritsv_zero_pivot(rocsparse_handle          handle,
                                                         const rocsparse_mat_descr descr,
                                                         rocsparse_mat_info        info,
                                                         rocsparse_int*            position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(2, info);

    // Logging
    rocsparse::log_trace(
        handle, "rocsparse_csritsv_zero_pivot", (const void*&)info, (const void*&)position);

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(3, position);

    // Stream
    hipStream_t stream = handle->stream;

    // If m == 0 || nnz == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->zero_pivot == nullptr)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 0xFF, sizeof(rocsparse_int), stream));
        }
        else
        {
            *position = -1;
        }

        return rocsparse_status_success;
    }

    // Differentiate between pointer modes
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // rocsparse_pointer_mode_device
        rocsparse_int zero_pivot;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &zero_pivot, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        if(zero_pivot == std::numeric_limits<rocsparse_int>::max())
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 0xFF, sizeof(rocsparse_int), stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(position,
                                               info->zero_pivot,
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToDevice,
                                               stream));

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
        }
    }
    else
    {
        // rocsparse_pointer_mode_host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(position,
                                           info->zero_pivot,
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        // If no zero pivot is found, set -1
        if(*position == std::numeric_limits<rocsparse_int>::max())
        {

            *position = -1;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
        }
    }

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_csritsv_clear(rocsparse_handle          handle,
                                                    const rocsparse_mat_descr descr,
                                                    rocsparse_mat_info        info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, info);

    // Logging
    rocsparse::log_trace(
        handle, "rocsparse_csritsv_clear", (const void*&)descr, (const void*&)info);

    // Clear csritsv meta data (this includes lower, upper and their transposed equivalents
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_csritsv_info(info->csritsv_info));
    info->csritsv_info = nullptr;
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
