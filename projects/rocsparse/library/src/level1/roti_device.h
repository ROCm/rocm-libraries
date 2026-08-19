/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
        roti_device(I nnz, T* x_val, const I* x_ind, T* y, T c, T s, rocsparse_index_base idx_base)
    {
        // Cast to the (possibly 64-bit) index type I before the multiply so the
        // element index does not overflow 32-bit arithmetic for large nnz.
        const I stride = static_cast<I>(hipGridDim_x) * BLOCKSIZE;

        for(I idx = static_cast<I>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x; idx < nnz;
            idx += stride)
        {
            I i = x_ind[idx] - idx_base;

            T xr = x_val[idx];
            T yr = y[i];

            x_val[idx] = rocsparse::fma<T>(c, xr, s * yr);
            y[i]       = rocsparse::fma<T>(c, yr, -s * xr);
        }
    }
}
