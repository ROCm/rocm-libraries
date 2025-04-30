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
    // Compute lower bound by binary search
    template <typename I, typename J>
    ROCSPARSE_DEVICE_ILF I lower_bound(const J* arr, J key, I low, I high)
    {
        while(low < high)
        {
            I mid = low + ((high - low) >> 1);

            if(arr[mid] < key)
            {
                low = mid + 1;
            }
            else
            {
                high = mid;
            }
        }

        return low;
    }

    // COO to CSR matrix conversion kernel
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void coo2csr_kernel(
        J m, I nnz, const J* coo_row_ind, I* csr_row_ptr, rocsparse_index_base idx_base)
    {
        J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= m)
        {
            return;
        }

        if(gid == 0)
        {
            csr_row_ptr[0] = idx_base;
            csr_row_ptr[m] = nnz + idx_base;
            return;
        }

        // Binary search
        csr_row_ptr[gid]
            = lower_bound<I, J>(coo_row_ind, gid + idx_base, static_cast<I>(0), nnz) + idx_base;
    }
}
