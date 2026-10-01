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

    // GRID_STRIDE is set when grid.x was clamped below the block count that
    // length needs. Otherwise grid.x * BLOCKSIZE <= 2^32 - 1, so the 32-bit
    // index of the straight-line path is exact.
    template <uint32_t BLOCKSIZE, bool GRID_STRIDE, typename T>
    ROCSPARSE_DEVICE_ILF void conjugate_device(int64_t length, T* __restrict__ array)
    {
        if constexpr(GRID_STRIDE)
        {
            const int64_t gid    = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
            const int64_t stride = static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE;
            for(int64_t idx = gid; idx < length; idx += stride)
            {
                array[idx] = rocsparse::conj(array[idx]);
            }
        }
        else
        {
            auto idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;
            if(idx >= length)
            {
                return;
            }

            array[idx] = rocsparse::conj(array[idx]);
        }
    }

    template <uint32_t BLOCKSIZE, bool GRID_STRIDE, typename T, typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void conjugate_kernel(int64_t length, int64_t batch_count, U array, int64_t array_dist)
    {
        // Grid-stride over the batch dimension so batch counts above the grid-y
        // limit (65535) are handled correctly.
        for(int64_t batch_index = hipBlockIdx_y; batch_index < batch_count;
            batch_index += hipGridDim_y)
        {
            auto p = batched_pointer(batch_index, array, array_dist);
            rocsparse::conjugate_device<BLOCKSIZE, GRID_STRIDE>(length, p);
        }
    }

    template <typename T>
    static rocsparse_status conjugate_strided_batched_kernel_launch(rocsparse_handle handle,
                                                                    int64_t          batch_count,
                                                                    int64_t          length,
                                                                    void*            array,
                                                                    int64_t          array_stride)
    {
        const int64_t  blocks_x = (length - 1) / 256 + 1;
        const uint32_t grid_x   = rocsparse::get_grid_size_x(handle, blocks_x, 256);
        const dim3     blocks(grid_x, rocsparse::get_grid_size_y(handle, batch_count));

        if(grid_x < blocks_x)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::conjugate_kernel<256, true, T, T*>),
                                               blocks,
                                               dim3(256),
                                               0,
                                               handle->stream,
                                               length,
                                               batch_count,
                                               reinterpret_cast<T*>(array),
                                               array_stride);
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::conjugate_kernel<256, false, T, T*>),
                                               blocks,
                                               dim3(256),
                                               0,
                                               handle->stream,
                                               length,
                                               batch_count,
                                               reinterpret_cast<T*>(array),
                                               array_stride);
        }
        return rocsparse_status_success;
    }

    typedef rocsparse_status (*conjugate_strided_batched_kernel_launch_t)(rocsparse_handle handle,
                                                                          int64_t batch_count,
                                                                          int64_t length,
                                                                          void*   array,
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
                                                      int64_t            array_stride)
{
    // Conjugation is the identity for real types.
    if(datatype == rocsparse_datatype_f32_r || datatype == rocsparse_datatype_f64_r)
    {
        return rocsparse_status_success;
    }

    auto launch_kernel = rocsparse::find_conjugate_strided_batched_kernel_launch(datatype);
    if(launch_kernel == nullptr)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "find_conjugate_launch failed");
    }
    RETURN_IF_ROCSPARSE_ERROR(launch_kernel(handle, batch_count, length, array, array_stride));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::conjugate(rocsparse_handle   handle,
                                      int64_t            length,
                                      rocsparse_datatype datatype,
                                      void*              array)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::conjugate_strided_batched(
        handle, static_cast<int64_t>(1), length, datatype, array, static_cast<int64_t>(0)));
    return rocsparse_status_success;
}
