/*! \file */
/* ************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    // Large size problems

    // Parallel cyclic reduction algorithm using global memory for partitioning large matrices into
    // multiple small ones that can be solved in parallel in stage 2
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_pow2_stage1_kernel(rocsparse_int stride,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int ldb,
                                             const T* __restrict__ a0,
                                             const T* __restrict__ b0,
                                             const T* __restrict__ c0,
                                             const T* __restrict__ rhs0,
                                             T* __restrict__ a1,
                                             T* __restrict__ b1,
                                             T* __restrict__ c1,
                                             T* __restrict__ rhs1)
    {
        rocsparse_int gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        rocsparse_int right = gid + stride;
        if(right >= m)
            right = m - 1;

        rocsparse_int left = gid - stride;
        if(left < 0)
            left = 0;

        T k1 = a0[gid] / b0[left];
        T k2 = c0[gid] / b0[right];
        T k3 = -a0[left];
        T k4 = -c0[right];

        // The first entry of the lower diagonal and the last entry of the upper
        // diagonal should be treated as zero
        if(gid == 0)
        {
            k1 = static_cast<T>(0);
        }
        if(left == 0)
        {
            k3 = static_cast<T>(0);
        }
        if(gid == (m - 1))
        {
            k2 = static_cast<T>(0);
        }
        if(right == (m - 1))
        {
            k4 = static_cast<T>(0);
        }

        b1[gid] = b0[gid] - c0[left] * k1 - a0[right] * k2;
        a1[gid] = k3 * k1;
        c1[gid] = k4 * k2;

        for(rocsparse_int i = 0; i < n; i++)
        {
            const T* rhs0_col = rhs0 + ldb * i;
            T*       rhs1_col = rhs1 + m * i;

            k3            = rhs0_col[right];
            k4            = rhs0_col[left];
            rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
        }
    }

    // Parallel cyclic reduction algorithm using global memory for partitioning large matrices into
    // multiple small ones that can be solved in parallel in stage 2
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_stage1_kernel(rocsparse_int stride,
                                        rocsparse_int m,
                                        rocsparse_int n,
                                        rocsparse_int ldb,
                                        const T* __restrict__ a0,
                                        const T* __restrict__ b0,
                                        const T* __restrict__ c0,
                                        const T* __restrict__ rhs0,
                                        T* __restrict__ a1,
                                        T* __restrict__ b1,
                                        T* __restrict__ c1,
                                        T* __restrict__ rhs1)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int gid = tid + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= m)
        {
            return;
        }

        rocsparse_int right = gid + stride;
        if(right >= m)
            right = m - 1;

        rocsparse_int left = gid - stride;
        if(left < 0)
            left = 0;

        T k1 = a0[gid] / b0[left];
        T k2 = c0[gid] / b0[right];
        T k3 = -a0[left];
        T k4 = -c0[right];

        // The first entry of the lower diagonal and the last entry of the upper
        // diagonal should be treated as zero
        if(gid == 0)
        {
            k1 = static_cast<T>(0);
        }
        if(left == 0)
        {
            k3 = static_cast<T>(0);
        }
        if(gid == (m - 1))
        {
            k2 = static_cast<T>(0);
        }
        if(right == (m - 1))
        {
            k4 = static_cast<T>(0);
        }

        b1[gid] = b0[gid] - c0[left] * k1 - a0[right] * k2;
        a1[gid] = k3 * k1;
        c1[gid] = k4 * k2;

        for(rocsparse_int i = 0; i < n; i++)
        {
            const T* rhs0_col = rhs0 + ldb * i;
            T*       rhs1_col = rhs1 + m * i;

            k3            = rhs0_col[right];
            k4            = rhs0_col[left];
            rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
        }
    }

    // See Nikolai Sakharnykh. Efficient tridiagonal solvers for adi methods and fluid simulation.
    // In NVIDIA GPU Technology Conference 2010, September 2010.
    template <uint32_t BLOCKSIZE, uint32_t SYSTEM_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_thomas_pow2_stage2_kernel(rocsparse_int stride,
                                                rocsparse_int m,
                                                rocsparse_int n,
                                                rocsparse_int ldb,
                                                const T* __restrict__ a0,
                                                const T* __restrict__ b0,
                                                const T* __restrict__ c0,
                                                const T* __restrict__ rhs0,
                                                T* __restrict__ a1,
                                                T* __restrict__ b1,
                                                T* __restrict__ c1,
                                                T* __restrict__ rhs1,
                                                T* __restrict__ B)
    {
        rocsparse_int gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= stride)
        {
            return;
        }

        // Forward elimination
        c1[gid]                       = c0[gid] / b0[gid];
        rhs1[gid + m * hipBlockIdx_y] = rhs0[gid + m * hipBlockIdx_y] / b0[gid];

        for(rocsparse_int i = 1; i < SYSTEM_SIZE; i++)
        {
            rocsparse_int index = stride * i + gid;
            rocsparse_int minus = stride * (i - 1) + gid;

            T k = static_cast<T>(1) / (b0[index] - c1[minus] * a0[index]);

            c1[index] = c0[index] * k;
            rhs1[index + m * hipBlockIdx_y]
                = (rhs0[index + m * hipBlockIdx_y] - rhs1[minus + m * hipBlockIdx_y] * a0[index])
                  * k;
        }

        // backward substitution
        B[stride * (SYSTEM_SIZE - 1) + gid + ldb * hipBlockIdx_y]
            = rhs1[stride * (SYSTEM_SIZE - 1) + gid + m * hipBlockIdx_y];

        for(rocsparse_int i = SYSTEM_SIZE - 2; i >= 0; i--)
        {
            rocsparse_int index = stride * i + gid;
            rocsparse_int plus  = stride * (i + 1) + gid;

            B[index + ldb * hipBlockIdx_y]
                = rhs1[index + m * hipBlockIdx_y] - c1[index] * B[plus + ldb * hipBlockIdx_y];
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_thomas_stage2_kernel(rocsparse_int stride,
                                           rocsparse_int m,
                                           rocsparse_int n,
                                           rocsparse_int ldb,
                                           const T* __restrict__ a0,
                                           const T* __restrict__ b0,
                                           const T* __restrict__ c0,
                                           const T* __restrict__ rhs0,
                                           T* __restrict__ a1,
                                           T* __restrict__ b1,
                                           T* __restrict__ c1,
                                           T* __restrict__ rhs1,
                                           T* __restrict__ B)
    {
        rocsparse_int gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= stride)
        {
            return;
        }

        rocsparse_int system_size = (m - gid - 1) / stride + 1;

        // Forward elimination
        c1[gid]                       = c0[gid] / b0[gid];
        rhs1[gid + m * hipBlockIdx_y] = rhs0[gid + m * hipBlockIdx_y] / b0[gid];

        for(rocsparse_int i = 1; i < system_size; i++)
        {
            rocsparse_int index = stride * i + gid;
            rocsparse_int minus = stride * (i - 1) + gid;

            T k = static_cast<T>(1) / (b0[index] - c1[minus] * a0[index]);

            c1[index] = c0[index] * k;
            rhs1[index + m * hipBlockIdx_y]
                = (rhs0[index + m * hipBlockIdx_y] - rhs1[minus + m * hipBlockIdx_y] * a0[index])
                  * k;
        }

        // backward substitution
        B[stride * (system_size - 1) + gid + ldb * hipBlockIdx_y]
            = rhs1[stride * (system_size - 1) + gid + m * hipBlockIdx_y];

        for(rocsparse_int i = system_size - 2; i >= 0; i--)
        {
            rocsparse_int index = stride * i + gid;
            rocsparse_int plus  = stride * (i + 1) + gid;

            B[index + ldb * hipBlockIdx_y]
                = rhs1[index + m * hipBlockIdx_y] - c1[index] * B[plus + ldb * hipBlockIdx_y];
        }
    }
}