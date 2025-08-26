/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_csrsv.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_utility.hpp"
namespace rocsparse
{
    rocsparse_status csrsv_zero_pivot(rocsparse_handle     handle,
                                      rocsparse_csrsv_info info,
                                      rocsparse_indextype  indextype,
                                      void*                position)

    {
        ROCSPARSE_ROUTINE_TRACE;
        if(info == nullptr)
        {
            rocsparse::set_minus_one_async(
                handle->stream, handle->pointer_mode, indextype, position);
            return rocsparse_status_success;
        }
        auto status = info->copy_zero_pivot_async(
            handle->pointer_mode, indextype, position, handle->stream);
        if(status == rocsparse_status_zero_pivot)
        {
            return status;
        }
        RETURN_IF_ROCSPARSE_ERROR(status);

        return rocsparse_status_success;
    }
}
extern "C" rocsparse_status rocsparse_csrsv_zero_pivot(rocsparse_handle          handle,
                                                       const rocsparse_mat_descr descr,
                                                       rocsparse_mat_info        info,
                                                       rocsparse_int*            position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, info);

    // Logging
    rocsparse::log_trace(
        handle, "rocsparse_csrsv_zero_pivot", (const void*&)info, (const void*&)position);

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(2, position);
    auto csrsv_info = info->get_csrsv_info();

    auto status = rocsparse::csrsv_zero_pivot(
        handle, csrsv_info, rocsparse::get_indextype<rocsparse_int>(), position);
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

extern "C" rocsparse_status rocsparse_csrsv_clear(rocsparse_handle          handle,
                                                  const rocsparse_mat_descr descr,
                                                  rocsparse_mat_info        info)
try
{
    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, info);

    // Logging
    rocsparse::log_trace(handle, "rocsparse_csrsv_clear", (const void*&)descr, (const void*&)info);

    // Clear csrsv meta data (this includes lower, upper and their transposed equivalents
    info->clear_csrsv_info();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
