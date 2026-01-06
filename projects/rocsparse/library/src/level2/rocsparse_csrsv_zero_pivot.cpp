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

#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    rocsparse_status csrsv_strided_batched_zero_pivot(rocsparse_handle     handle,
                                                      rocsparse_csrsv_info info,
                                                      rocsparse_indextype  indextype,
                                                      rocsparse_int        batch_count,
                                                      void*                position)

    {
        ROCSPARSE_ROUTINE_TRACE;

        if(batch_count == 0)
        {
            return rocsparse_status_success;
        }
        // Stream
        hipStream_t stream = handle->stream;

        if(info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::set_minus_one_async(
                handle->pointer_mode, indextype, batch_count, position, stream));
            return rocsparse_status_success;
        }
        auto status = info->copy_zero_pivot_async(
            batch_count, handle->pointer_mode, indextype, position, handle->stream);

        if(status == rocsparse_status_zero_pivot)
        {
            return status;
        }
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status
    rocsparse_csrsv_strided_batched_zero_pivot(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_mat_info        info,
                                               rocsparse_int             batch_count,
                                               rocsparse_int*            position)
try
{

    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, info);
    ROCSPARSE_CHECKARG_SIZE(3, batch_count);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csrsv_strided_batched_zero_pivot(handle,
                                                    info->get_csrsv_info(),
                                                    rocsparse::get_indextype<rocsparse_int>(),
                                                    batch_count,
                                                    position));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
