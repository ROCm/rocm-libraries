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
#include "rocsparse_utility.hpp"

#include "internal/precond/rocsparse_csric0.h"

extern "C" rocsparse_status rocsparse_csric0_singular_pivot(rocsparse_handle   handle,
                                                            rocsparse_mat_info info,
                                                            rocsparse_int*     position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(
        handle, "rocsparse_csric0_singular_pivot", (const void*&)info, (const void*&)position);

    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    // Stream
    hipStream_t stream = handle->stream;

    auto    csric0_info = info->get_csric0_info();
    int64_t zero_pivot_value;
    auto    status = csric0_info->copy_zero_pivot_async(rocsparse_pointer_mode_host,
                                                     rocsparse::get_indextype<int64_t>(),
                                                     &zero_pivot_value,
                                                     handle->stream);
    if(status != rocsparse_status_zero_pivot)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
    }

    int64_t singular_pivot_value;
    status = csric0_info->copy_singular_pivot_async(rocsparse_pointer_mode_host,
                                                    rocsparse::get_indextype<int64_t>(),
                                                    &singular_pivot_value,
                                                    handle->stream);

    if(status != rocsparse_status_zero_pivot)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
    }

    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    int64_t singular_pivot;
    if((singular_pivot_value == -1) || (zero_pivot_value == -1))
    {
        singular_pivot = ((singular_pivot_value == -1) ? zero_pivot_value : singular_pivot_value);
    }
    else
    {
        singular_pivot = std::min(zero_pivot_value, singular_pivot_value);
    }

    // Differentiate between pointer modes
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // rocsparse_pointer_mode_device
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            position, &singular_pivot, sizeof(rocsparse_int), hipMemcpyHostToDevice, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }
    else
    {
        // rocsparse_pointer_mode_host
        *position = singular_pivot;
    }

    return (rocsparse_status_success);
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
