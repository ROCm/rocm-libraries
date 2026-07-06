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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_indextype_utils.hpp"

namespace rocsparse
{
    template <typename T>
    ROCSPARSE_KERNEL(32)
    void assign_kernel(T* dest, T value)
    {
        const uint32_t batch_index = blockIdx.y;
        if(hipThreadIdx_x == 0)
        {
            dest[batch_index] = value;
        }
    }

    template <typename T>
    ROCSPARSE_KERNEL(32)
    void assign_device_kernel(T* dest, const T* value)
    {
        const uint32_t batch_index = blockIdx.y;
        if(hipThreadIdx_x == 0)
        {
            dest[batch_index] = value[0];
        }
    }
}

template <typename T>
rocsparse_status
    rocsparse::assign_device_async(int64_t n, T* dest, const T* value, hipStream_t stream)
{
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        rocsparse::assign_device_kernel, dim3(1, n), dim3(32), 0, stream, dest, value);
    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse::assign_async(int64_t n, T* dest, T value, hipStream_t stream)
{
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        rocsparse::assign_kernel, dim3(1, n), dim3(32), 0, stream, dest, value);
    return rocsparse_status_success;
}

template rocsparse_status
    rocsparse::assign_async(int64_t n, int32_t* dest, int32_t value, hipStream_t stream);

template rocsparse_status
    rocsparse::assign_async(int64_t n, int64_t* dest, int64_t value, hipStream_t stream);

template rocsparse_status rocsparse::assign_device_async(int64_t        n,
                                                         int32_t*       dest,
                                                         const int32_t* value,
                                                         hipStream_t    stream);

template rocsparse_status rocsparse::assign_device_async(int64_t        n,
                                                         int64_t*       dest,
                                                         const int64_t* value,
                                                         hipStream_t    stream);

rocsparse_status rocsparse::assign_max_async(int64_t             n,
                                             rocsparse_indextype indextype,
                                             void*               dest,
                                             hipStream_t         stream)
{
    switch(indextype)
    {
    case rocsparse_indextype_i32:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_async(
            n, reinterpret_cast<int32_t*>(dest), std::numeric_limits<int32_t>::max(), stream));

        return rocsparse_status_success;
    }
    case rocsparse_indextype_i64:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_async(
            n, reinterpret_cast<int64_t*>(dest), std::numeric_limits<int64_t>::max(), stream));
        return rocsparse_status_success;
    }
    // LCOV_EXCL_START
    case deprecated_rocsparse_indextype_u16:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented,
                                               "unsupported indextype: rocsparse_indextype_u16");
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}
