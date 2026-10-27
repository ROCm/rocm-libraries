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
#include "rocsparse_grid.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    //
    // AISPARSE-700. launch_valset sizes grid.x from an int64_t length, correctly, but
    // the kernel took the length back as a template index type I and formed its
    // element index as
    //     I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;
    // BLOCKSIZE * hipBlockIdx_x is unsigned-int arithmetic whatever I is, so the
    // index wrapped at 2^32 elements while the grid still covered the whole range,
    // and there was no grid-stride loop to recover the tail. valset is the most
    // widely reached site of this ticket: 15+ conversion routines,
    // csrgemm/csrgeam/bsrgeam/bsrgemm and csrsv_analysis.
    //
    // The length is now carried at full width -- launch_valset always passed an
    // int64_t, so there never was a narrower instantiation to truncate it, and
    // dropping the index template parameter makes that structural -- the index is
    // 64-bit, and grid.x is clamped with a grid-stride loop behind it. The loop bound
    // depends only on hipBlockIdx_x, hipGridDim_x, BLOCKSIZE and the length, all
    // block uniform, so every thread of a block runs the same number of iterations
    // (AISPARSE-666 idiom).
    //
    template <typename T>
    ROCSPARSE_DEVICE_ILF void valset_device(int64_t idx, int64_t value, T* __restrict__ array)
    {
        array[idx] = value;
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void valset_kernel(int64_t length, int64_t value, T* array)
    {
        for(int64_t base = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE; base < length;
            base += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            const int64_t idx = base + hipThreadIdx_x;
            if(idx < length)
            {
                rocsparse::valset_device(idx, value, array);
            }
        }
    }

    template <typename T>
    static rocsparse_status
        launch_valset(rocsparse_handle handle, int64_t length, int64_t value, void* array)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::valset_kernel<256>),
            dim3(rocsparse::get_grid_size_x(handle, (length - 1) / 256 + 1, 256)),
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
    switch(static_cast<int>(array_indextype))
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
