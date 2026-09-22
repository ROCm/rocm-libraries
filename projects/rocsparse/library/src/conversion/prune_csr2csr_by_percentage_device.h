/*! \file */
/* ************************************************************************
* Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.hpp"
#include "rocsparse_handle.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void abs_kernel(int64_t nnz_A, const T* __restrict__ csr_val_A, T* __restrict__ output)
    {
        // Cast before each multiply: hipBlockDim_x, hipBlockIdx_x and
        // hipGridDim_x are unsigned int, so the products wrap at 2^32 and the
        // int64_t destination cannot recover the lost high bits. The launch
        // already caps grid_x at 2147483647 and spills the remainder onto
        // grid_y, so a 256-thread block reaches that wrap at 2^32 elements.
        const int64_t gid_x = static_cast<int64_t>(hipBlockDim_x) * hipBlockIdx_x + hipThreadIdx_x;
        const int64_t gid_y = static_cast<int64_t>(hipBlockDim_y) * hipBlockIdx_y + hipThreadIdx_y;

        const int64_t grid_dim_x = static_cast<int64_t>(hipGridDim_x) * hipBlockDim_x;

        // Map a 2D HIP grid to a 1D index
        const int64_t gid = grid_dim_x * gid_y + gid_x;

        if(gid >= nnz_A)
        {
            return;
        }

        output[gid] = rocsparse::abs(csr_val_A[gid]);
    }
}
