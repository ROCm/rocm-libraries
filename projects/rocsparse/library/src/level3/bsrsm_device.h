/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_scalar.hpp"

namespace rocsparse
{
    // Body of ONE row of the bsrsm right hand side setup. `row` is a parameter
    // rather than a read of blockIdx.x/threadIdx.x so the grid-stride loop lives in
    // the bsrsm_copy_scale __global__ wrapper (the AISPARSE-666 idiom the reviewer
    // required). `m` is the total scalar row count mb * block_dim, which is 64 bit:
    // the caller used to form that product in signed 32 bit both for this argument
    // and for the launch grid (AISPARSE-670).
    template <typename T>
    ROCSPARSE_DEVICE_ILF void bsrsm_copy_scale_device(int64_t       row,
                                                      int64_t       m,
                                                      rocsparse_int n,
                                                      T             alpha,
                                                      const T*      B,
                                                      int64_t       ldb,
                                                      T*            X,
                                                      int64_t       ldx)
    {
        // Redundant with the wrapper's loop bound, kept so the device function is
        // still safe to call with an arbitrary row. There is no barrier and no
        // shared memory in this function, so a per-thread bound is fine here.
        if(row >= m)
        {
            return;
        }

        for(rocsparse_int i = 0; i < n; ++i)
        {
            const int64_t idx_B = row * ldb + i;
            const int64_t idx_X = row * ldx + i;

            X[idx_X] = alpha * B[idx_B];
        }
    }

    // The bsrsm right hand side setup. Lives in this header rather than in
    // rocsparse_bsrsm_template_large.cpp so clients/unittests can launch it with a
    // deliberately undersized grid, which is what the clamp on its launch produces.
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsm_copy_scale(int64_t       m,
                          rocsparse_int n,
                          ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                          const T* B,
                          int64_t  ldb,
                          T*       X,
                          int64_t  ldx,
                          bool     is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        // Grid-stride loop so a grid clamped against maxGridSize[0] still covers
        // all m = mb * block_dim rows. blockIdx.x * BLOCKSIZE is formed in 64 bit;
        // it used to be a 32 bit product assigned to a rocsparse_int. This kernel
        // contains no __syncthreads() and no shared memory, so the per-thread loop
        // bound cannot diverge at a barrier (AISPARSE-670).
        const int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCKSIZE;

        for(int64_t row = static_cast<int64_t>(blockIdx.x) * BLOCKSIZE + threadIdx.x; row < m;
            row += stride)
        {
            rocsparse::bsrsm_copy_scale_device(row, m, n, alpha, B, ldb, X, ldx);
        }
    }
}
