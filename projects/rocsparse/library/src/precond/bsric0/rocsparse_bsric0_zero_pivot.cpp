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

#include "internal/precond/rocsparse_bsric0.h"
#include "rocsparse_utility.hpp"

extern "C" rocsparse_status rocsparse_bsric0_zero_pivot(rocsparse_handle   handle,
                                                        rocsparse_mat_info info,
                                                        rocsparse_int*     position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    ROCSPARSE_CHECKARG_POINTER(1, info);

    ROCSPARSE_CHECKARG_POINTER(2, position);

    auto bsric0_info = info->get_bsric0_info();
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::singularity_get_position_async(
        handle,
        1,
        bsric0_info,
        (bsric0_info != nullptr) ? bsric0_info->get_singularity_numeric_exact() : nullptr,
        nullptr,
        handle->pointer_mode,
        rocsparse::get_indextype<rocsparse_int>(),
        position));

    rocsparse_int pos;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&pos, position, sizeof(rocsparse_int), hipMemcpyDefault, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    if(pos != -1)
    {
        return rocsparse_status_zero_pivot;
    }

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
