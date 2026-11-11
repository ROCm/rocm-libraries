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

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void
        conjugate_device(int64_t length, T* __restrict__ array, int64_t array_inc)
    {
        auto idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;
        if(idx >= length)
        {
            return;
        }
        array += idx * array_inc;
        array[0] = rocsparse::conj(array[0]);
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void conjugate_kernel(int64_t batch_count,
                          int64_t length,
                          T* __restrict__ array,
                          int64_t array_inc,
                          int64_t array_dist)
    {

        for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
        {
            auto p = rocsparse::load_pointer(array, batch, array_dist);
            rocsparse::conjugate_device<BLOCKSIZE>(length, p, array_inc);
        }
    }

    template <typename T>
    static rocsparse_status conjugate_strided_batched_kernel_launch(rocsparse_handle handle,
                                                                    int64_t          batch_count,
                                                                    int64_t          length,
                                                                    void*            array,
                                                                    int64_t          array_inc,
                                                                    int64_t          array_stride)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::conjugate_kernel<256, T>),
            dim3(rocsparse::get_grid_size_x(handle, (length - 1) / 256 + 1, 256),
                 rocsparse::get_grid_size_y(handle, batch_count)),
            dim3(256),
            0,
            handle->stream,
            batch_count,
            length,
            reinterpret_cast<T*>(array),
            array_inc,
            array_stride);
        return rocsparse_status_success;
    }

    typedef rocsparse_status (*conjugate_strided_batched_kernel_launch_t)(rocsparse_handle handle,
                                                                          int64_t batch_count,
                                                                          int64_t length,
                                                                          void*   array,
                                                                          int64_t array_inc,
                                                                          int64_t array_stride);

    static rocsparse::conjugate_strided_batched_kernel_launch_t
        find_conjugate_strided_batched_kernel_launch(rocsparse_datatype datatype)
    {
        switch(datatype)
        {
        case rocsparse_datatype_f32_c:
        {
            return conjugate_strided_batched_kernel_launch<rocsparse_float_complex>;
        }
        case rocsparse_datatype_f64_c:
        {
            return conjugate_strided_batched_kernel_launch<rocsparse_double_complex>;
        }
        default:
        {
            return nullptr;
        }
        }
    }
}

rocsparse_status rocsparse::conjugate_strided_batched(rocsparse_handle   handle,
                                                      int64_t            batch_count,
                                                      int64_t            length,
                                                      rocsparse_datatype datatype,
                                                      void*              array,
                                                      int64_t            array_inc,
                                                      int64_t            array_stride)
{
    if(rocsparse::datatype_is_complex(datatype))
    {
        auto launch_kernel = rocsparse::find_conjugate_strided_batched_kernel_launch(datatype);
        if(launch_kernel == nullptr)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   "find_conjugate_launch failed");
        }
        RETURN_IF_ROCSPARSE_ERROR(
            launch_kernel(handle, batch_count, length, array, array_inc, array_stride));
    }
    return rocsparse_status_success;
}

rocsparse_status rocsparse::conjugate_strided_batched(rocsparse_handle   handle,
                                                      int64_t            batch_count,
                                                      int64_t            length,
                                                      rocsparse_datatype datatype,
                                                      void*              array,
                                                      int64_t            array_stride)
{
    if(rocsparse::datatype_is_complex(datatype))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::conjugate_strided_batched(
            handle, batch_count, length, datatype, array, static_cast<int64_t>(1), array_stride));
    }
    return rocsparse_status_success;
}

rocsparse_status rocsparse::conjugate(rocsparse_handle   handle,
                                      int64_t            length,
                                      rocsparse_datatype datatype,
                                      void*              array)
{
    if(rocsparse::datatype_is_complex(datatype))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::conjugate_strided_batched(handle,
                                                                       static_cast<int64_t>(1),
                                                                       length,
                                                                       datatype,
                                                                       array,
                                                                       static_cast<int64_t>(1),
                                                                       static_cast<int64_t>(0)));
    }
    return rocsparse_status_success;
}
