/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_cscsv.hpp"
#include "rocsparse_singularity.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::cscsv_zero_pivot(rocsparse_handle     handle,
                                             rocsparse_cscsv_info info,
                                             rocsparse_indextype  indextype,
                                             void*                position)
{
    ROCSPARSE_ROUTINE_TRACE;
    if(info == nullptr)
    {
        return rocsparse::csrsv_zero_pivot(handle, nullptr, indextype, position);
    }
    if(!info->uses_csrsv_info())
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::singularity_get_position_async(handle,
                                                      1,
                                                      info,
                                                      info->get_singularity_numeric_exact(),
                                                      nullptr,
                                                      handle->pointer_mode,
                                                      indextype,
                                                      position));

        // position lives where the handle's pointer mode says it does.
        const hipMemcpyKind kind = (handle->pointer_mode == rocsparse_pointer_mode_host)
                                       ? hipMemcpyHostToHost
                                       : hipMemcpyDeviceToHost;

        int64_t p = -1;
        if(indextype == rocsparse_indextype_i32)
        {
            int32_t p32;
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMemcpyAsync(&p32, position, sizeof(int32_t), kind, handle->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
            p = p32;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMemcpyAsync(&p, position, sizeof(int64_t), kind, handle->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
        }
        return (p != -1) ? rocsparse_status_zero_pivot : rocsparse_status_success;
    }
    return rocsparse::csrsv_zero_pivot(handle, info->get_csrsv_info(), indextype, position);
}
