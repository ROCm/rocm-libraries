/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include <hip/hip_runtime.h>

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void
        gthr_device(I nnz, const T* y, T* x_val, const I* x_ind, rocsparse_index_base idx_base)
    {
        // Cast to I before the multiply so the index arithmetic does not wrap in
        // 32-bit when nnz exceeds the range of unsigned int, and grid-stride so
        // every element is gathered even when grid.x is clamped below the ideal
        // block count.
        const I stride = static_cast<I>(hipGridDim_x) * BLOCKSIZE;
        for(I idx = static_cast<I>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x; idx < nnz;
            idx += stride)
        {
            x_val[idx] = y[x_ind[idx] - idx_base];
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gthr_kernel(I                    nnz,
                     int64_t              batch_count,
                     const T*             y,
                     int64_t              y_stride,
                     T*                   x_val,
                     int64_t              x_val_stride,
                     const I*             x_ind,
                     rocsparse_index_base idx_base)
    {
        // Grid-stride over the batch axis so batch_count is not limited by the
        // 65,535 grid.y hardware cap.
        for(int64_t batch_index = hipBlockIdx_y; batch_index < batch_count;
            batch_index += hipGridDim_y)
        {
            gthr_device<BLOCKSIZE, I, T>(nnz,
                                         y + batch_index * y_stride,
                                         x_val + batch_index * x_val_stride,
                                         x_ind,
                                         idx_base);
        }
    }
}
