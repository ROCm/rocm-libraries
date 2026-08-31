/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_csrsm.hpp"

#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_utility.hpp"

#include "../level1/rocsparse_gthr.hpp"
#include "../level2/rocsparse_csrsv.hpp"
#include "csrsm_device.h"

namespace rocsparse
{
    rocsparse_status csrsm_zero_pivot(rocsparse_handle     handle,
                                      rocsparse_csrsm_info info,
                                      rocsparse_indextype  indextype,
                                      void*                position)

    {
        ROCSPARSE_ROUTINE_TRACE;
        auto numeric_exact_position = (info) ? info->get_singularity_numeric_exact() : nullptr;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::singularity_get_position_async(handle,
                                                                            1,
                                                                            info,
                                                                            numeric_exact_position,
                                                                            nullptr,
                                                                            handle->pointer_mode,
                                                                            indextype,
                                                                            position));
        switch(indextype)
        {
        case rocsparse_indextype_i32:
        {
            int32_t p;
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                &p, position, sizeof(int32_t), hipMemcpyDefault, handle->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
            if(p != -1)
            {
                return rocsparse_status_zero_pivot;
            }
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            int64_t p;
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                &p, position, sizeof(int64_t), hipMemcpyDefault, handle->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
            if(p != -1)
            {
                return rocsparse_status_zero_pivot;
            }
            return rocsparse_status_success;
        }

        case deprecated_rocsparse_indextype_u16:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   "rocsparse_indextype_u16 not supported");
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
}

extern "C" rocsparse_status rocsparse_csrsm_zero_pivot(rocsparse_handle   handle,
                                                       rocsparse_mat_info info,
                                                       rocsparse_int*     position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(
        handle, "rocsparse_csrsm_zero_pivot", (const void*&)info, (const void*&)position);
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    auto csrsm_info = info->get_csrsm_info();
    auto status     = rocsparse::csrsm_zero_pivot(
        handle, csrsm_info, rocsparse::get_indextype<rocsparse_int>(), position);
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

extern "C" rocsparse_status rocsparse_csrsm_clear(rocsparse_handle handle, rocsparse_mat_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle, "rocsparse_csrsm_clear", (const void*&)info);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, info);
    info->clear_csrsm_info();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
