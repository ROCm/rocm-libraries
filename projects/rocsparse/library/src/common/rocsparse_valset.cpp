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

#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void valset_device(I length, int64_t value, T* __restrict__ array)
    {
        I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;
        if(idx >= length)
        {
            return;
        }

        array[idx] = value;
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void valset_kernel(I length, int64_t value, T* array)
    {
        rocsparse::valset_device<BLOCKSIZE>(length, value, array);
    }

    template <typename T>
    static rocsparse_status
        launch_valset(rocsparse_handle handle, int64_t length, int64_t value, void* array)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::valset_kernel<256>),
                                           dim3((length - 1) / 256 + 1),
                                           dim3(256),
                                           0,
                                           handle->stream,
                                           length,
                                           value,
                                           reinterpret_cast<T*>(array));

        return rocsparse_status_success;
    }

}

rocsparse_status rocsparse::valset(rocsparse_handle    handle,
                                   int64_t             length,
                                   int64_t             value,
                                   rocsparse_indextype array_indextype,
                                   void*               array)
{

    auto f = launch_valset<int32_t>;
    switch(array_indextype)
    {
    case rocsparse_indextype_i32:
        break;
    case rocsparse_indextype_i64:
    {
        f = launch_valset<int64_t>;
        break;
    }
    case deprecated_rocsparse_indextype_u16:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented,
                                               "rocsparse_indextype_u16 case not implemented");
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(f(handle, length, value, array));
    return rocsparse_status_success;
}
