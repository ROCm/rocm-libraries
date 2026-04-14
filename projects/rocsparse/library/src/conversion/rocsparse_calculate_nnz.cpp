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

#include "rocsparse_calculate_nnz.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_indextype_utils.hpp"

rocsparse_status rocsparse::calculate_nnz(
    int64_t m, rocsparse_indextype indextype, const void* ptr, int64_t* nnz, hipStream_t stream)
{
    if(m == 0)
    {
        nnz[0] = 0;
        return rocsparse_status_success;
    }
    const char* p = reinterpret_cast<const char*>(ptr) + rocsparse::indextype_sizeof(indextype) * m;
    int64_t     end, start;
    switch(indextype)
    {
    case rocsparse_indextype_i32:
    {
        int32_t u, v;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &u, ptr, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &v, p, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
        start = u;
        end   = v;
        break;
    }
    case rocsparse_indextype_i64:
    {
        int64_t u, v;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &u, ptr, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &v, p, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
        start = u;
        end   = v;
        break;
    }
    case rocsparse_indextype_u16:
    {
        uint16_t u, v;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &u, ptr, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &v, p, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
        start = u;
        end   = v;
        break;
    }
    }
    nnz[0] = end - start;
    return rocsparse_status_success;
}
