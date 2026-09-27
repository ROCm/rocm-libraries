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
    template <uint32_t BLOCKSIZE, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2csc_permute_kernel(I nnz,
                                const J* __restrict__ in1,
                                const T* __restrict__ in2,
                                const I* __restrict__ map,
                                J* __restrict__ out1,
                                T* __restrict__ out2)
    {
        // AISPARSE-685. The global id used to be a rocsparse_int computed from
        // `hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x`. Both operands are unsigned
        // int, so that product is evaluated in 32-bit arithmetic and wraps at 2^32 no
        // matter what it is assigned to, and the rocsparse_int it was assigned to
        // truncated at INT_MAX on top of that -- while nnz is the index type I, which
        // is int64_t on the 64-bit instantiations reached from rocsparse_Xcsr2csc and
        // rocsparse_sparse_to_sparse. The id is formed in int64_t now -- not in I,
        // because the caller may clamp grid.x, so BLOCKSIZE * hipGridDim_x is no
        // longer bounded by nnz -- and the stride loop covers what the clamp dropped.
        //
        // Block-uniform stride bound: every term of the loop is hipBlockIdx_x,
        // hipGridDim_x, a kernel argument or a compile-time constant. This kernel has
        // no __syncthreads(), and the early `return` it used to take is now the loop
        // condition itself.
        const int64_t stride = static_cast<int64_t>(BLOCKSIZE) * hipGridDim_x;

        for(int64_t gid = static_cast<int64_t>(BLOCKSIZE) * hipBlockIdx_x + hipThreadIdx_x;
            gid < nnz;
            gid += stride)
        {
            const I idx = map[gid];
            out1[gid]   = in1[idx];
            out2[gid]   = in2[idx];
        }
    }
}
