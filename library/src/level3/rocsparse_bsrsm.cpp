/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level3/rocsparse_bsrsm.h"
#include "control.h"
#include "utility.h"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_bsrsm_zero_pivot(rocsparse_handle   handle,
                                                       rocsparse_mat_info info,
                                                       rocsparse_int*     position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(
        handle, "rocsparse_bsrsm_zero_pivot", (const void*&)info, (const void*&)position);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    // Stream
    hipStream_t stream = handle->stream;

    // If mb == 0 || nrhs == 0 || nnzb == 0 it can happen, that info structure is not created.
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
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            position, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

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

extern "C" rocsparse_status rocsparse_bsrsm_clear(rocsparse_handle handle, rocsparse_mat_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle, "rocsparse_bsrsm_clear", (const void*&)info);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, info);

    // Clear bsrsm meta data (this includes lower, upper and their transposed equivalents
    if(!rocsparse::check_trm_shared(info, info->bsrsm_lower_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_trm_info(info->bsrsm_lower_info));
    }
    if(!rocsparse::check_trm_shared(info, info->bsrsm_upper_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_trm_info(info->bsrsm_upper_info));
    }

    info->bsrsm_lower_info = nullptr;
    info->bsrsm_upper_info = nullptr;

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
