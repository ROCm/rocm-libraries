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
#include "rocsparse_control.hpp"
#include "rocsparse_csritsv.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_utility.hpp"

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
    auto csritsv_info = info->csritsv_info;
    if(csritsv_info == nullptr)
    {
        rocsparse::set_minus_one_async(handle->stream,
                                       handle->pointer_mode,
                                       rocsparse::get_indextype<rocsparse_int>(),
                                       position);
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
        return rocsparse_status_success;
    }
    auto status = csritsv_info->copy_zero_pivot_async(
        handle->pointer_mode, rocsparse::get_indextype<rocsparse_int>(), position, handle->stream);
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    if(status == rocsparse_status_zero_pivot)
    {
        return status;
    }
    RETURN_IF_ROCSPARSE_ERROR(status);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

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
    if(info->csritsv_info != nullptr)
    {
        delete info->csritsv_info;
        info->csritsv_info = nullptr;
    }
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
