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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_control.hpp"

namespace rocsparse
{

    template <typename T>
    ROCSPARSE_KERNEL(1)
    void assign_kernel(T* dest, T value)
    {
        *dest = value;
    }

}

template <typename T>
rocsparse_status rocsparse::assign_async(T* dest, T value, hipStream_t stream)
{
    // Use a kernel instead of memcpy, because memcpy is synchronous if the source is not in
    // pinned memory.
    // Memset lacks a 64bit option, but would involve a similar implicit kernel anyways.
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        rocsparse::assign_kernel, dim3(1), dim3(1), 0, stream, dest, value);
    return rocsparse_status_success;
}

template rocsparse_status rocsparse::assign_async(int32_t* dest, int32_t value, hipStream_t stream);

template rocsparse_status rocsparse::assign_async(int64_t* dest, int64_t value, hipStream_t stream);
