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

#include "rocsparse_common.hpp"

namespace rocsparse
{
    // y = a * x + y kernel for sparse x and dense y
    template <uint32_t BLOCKSIZE, typename T, typename I, typename X, typename Y>
    ROCSPARSE_DEVICE_ILF void axpyi_device(
        I nnz, T alpha, const X* x_val, const I* x_ind, Y* y, rocsparse_index_base idx_base)
    {
        // Keep the grid-stride arithmetic wide even when I is 32-bit.
        const int64_t gid    = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        const int64_t stride = static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE;

        // Grid-stride loop so the full vector is processed even when the number
        // of required blocks exceeds the grid size.
        for(int64_t idx = gid; idx < nnz; idx += stride)
        {
            I i  = x_ind[idx] - idx_base;
            y[i] = rocsparse::fma<T>(alpha, x_val[idx], y[i]);
        }
    }
}
