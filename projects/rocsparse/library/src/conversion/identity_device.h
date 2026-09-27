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
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void identity_kernel(I n, I* p)
    {
        // AISPARSE-686. `hipBlockIdx_x * BLOCKSIZE` is unsigned-int arithmetic --
        // both operands are unsigned int -- so it wrapped at 2^32 before it was ever
        // assigned to I, and there was no grid-stride loop to cover a grid.x that the
        // caller clamped against the device limit. The id is formed in int64_t now --
        // not in I, because the caller may clamp grid.x, so BLOCKSIZE * hipGridDim_x
        // is no longer bounded by n.
        //
        // Block-uniform stride bound: every term is hipBlockIdx_x, hipGridDim_x, a
        // kernel argument or a compile-time constant. This kernel has no
        // __syncthreads(), and the early `return` it used to take is now the loop
        // condition itself.
        const int64_t stride = static_cast<int64_t>(BLOCKSIZE) * hipGridDim_x;

        for(int64_t gid = static_cast<int64_t>(BLOCKSIZE) * hipBlockIdx_x + hipThreadIdx_x; gid < n;
            gid += stride)
        {
            p[gid] = static_cast<I>(gid);
        }
    }
}
