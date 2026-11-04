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
    // Create identity permutation
    // GRID_STRIDE is set when grid.x was clamped below the number of blocks n
    // needs. Otherwise grid.x * BLOCKSIZE fits the 32-bit dispatch limit and the
    // unsigned-int global id of the straight-line path is exact.
    template <uint32_t BLOCKSIZE, bool GRID_STRIDE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void identity_kernel(I n, I* p)
    {
        if constexpr(GRID_STRIDE)
        {
            const int64_t stride = static_cast<int64_t>(BLOCKSIZE) * hipGridDim_x;

            for(int64_t gid = static_cast<int64_t>(BLOCKSIZE) * hipBlockIdx_x + hipThreadIdx_x;
                gid < n;
                gid += stride)
            {
                p[gid] = static_cast<I>(gid);
            }
        }
        else
        {
            const int64_t gid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;

            if(gid >= n)
            {
                return;
            }

            p[gid] = static_cast<I>(gid);
        }
    }
}
