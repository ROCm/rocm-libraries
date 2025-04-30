/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"

namespace rocsparse
{
    // Matrix has form:
    //
    // [ b0 c0 0  0  0  0  0  0 ]
    // [ a1 b1 c1 0  0  0  0  0 ]
    // [ 0  a2 b2 c2 0  0  0  0 ]
    // [ 0  0  a3 b3 c3 0  0  0 ]
    // [ 0  0  0  a4 b4 c4 0  0 ]
    // [ 0  0  0  0  a5 b5 c5 0 ]
    // [ 0  0  0  0  0  a6 b6 c6]
    // [ 0  0  0  0  0  0  a7 b7]

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_interleaved_batch_thomas_kernel(rocsparse_int m,
                                              rocsparse_int batch_count,
                                              rocsparse_int batch_stride,
                                              const T* __restrict__ a0,
                                              const T* __restrict__ b0,
                                              const T* __restrict__ c0,
                                              T* __restrict__ c1,
                                              T* __restrict__ x1,
                                              T* __restrict__ x)
    {
        rocsparse_int gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= batch_count)
        {
            return;
        }

        // Forward elimination
        c1[gid] = c0[gid] / b0[gid];
        x1[gid] = x[gid] / b0[gid];

        for(rocsparse_int k = 1; k < m; k++)
        {
            rocsparse_int index = batch_count * k + gid;
            rocsparse_int minus = batch_count * (k - 1) + gid;

            T tc0 = c0[batch_stride * k + gid];
            T tb0 = b0[batch_stride * k + gid];
            T ta0 = a0[batch_stride * k + gid];
            T tx  = x[batch_stride * k + gid];

            c1[index] = tc0 / (tb0 - c1[minus] * ta0);
            x1[index] = (tx - x1[minus] * ta0) / (tb0 - c1[minus] * ta0);
        }

        // backward substitution
        x[batch_stride * (m - 1) + gid] = x1[batch_count * (m - 1) + gid];

        for(rocsparse_int k = m - 2; k >= 0; k--)
        {
            rocsparse_int index = batch_count * k + gid;

            x[batch_stride * k + gid] = x1[index] - c1[index] * x[batch_stride * (k + 1) + gid];
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_interleaved_batch_lu_kernel(rocsparse_int m,
                                          rocsparse_int batch_count,
                                          rocsparse_int batch_stride,
                                          T* __restrict__ dl,
                                          T* __restrict__ d,
                                          T* __restrict__ du,
                                          T* __restrict__ u2,
                                          rocsparse_int* __restrict__ p,
                                          T* __restrict__ x)
    {
        rocsparse_int gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= batch_count)
        {
            return;
        }

        p[gid] = 0;

        // LU decomposition
        for(rocsparse_int k = 0; k < m - 1; k++)
        {
            rocsparse_int ind_k   = batch_stride * k + gid;
            rocsparse_int ind_k_1 = batch_stride * (k + 1) + gid;

            T ak_1 = dl[ind_k_1];
            T bk   = d[ind_k];

            if(rocsparse::abs(bk) < rocsparse::abs(ak_1))
            {
                T bk_1 = d[ind_k_1];
                T ck   = du[ind_k];
                T ck_1 = du[ind_k_1];
                T dk   = u2[batch_count * k + gid];

                d[ind_k]                  = ak_1;
                du[ind_k]                 = bk_1;
                u2[batch_count * k + gid] = ck_1;

                d[ind_k_1]  = ck;
                du[ind_k_1] = dk;

                rocsparse_int pk               = p[batch_count * k + gid];
                p[batch_count * k + gid]       = k + 1;
                p[batch_count * (k + 1) + gid] = pk;

                T xk       = x[ind_k];
                x[ind_k]   = x[ind_k_1];
                x[ind_k_1] = xk;

                T lk_1      = bk / ak_1;
                dl[ind_k_1] = lk_1;

                d[ind_k_1]  = d[ind_k_1] - lk_1 * du[ind_k];
                du[ind_k_1] = du[ind_k_1] - lk_1 * u2[batch_count * k + gid];
            }
            else
            {
                p[batch_count * (k + 1) + gid] = k + 1;

                T lk_1      = ak_1 / bk;
                dl[ind_k_1] = lk_1;

                d[ind_k_1]  = d[ind_k_1] - lk_1 * du[ind_k];
                du[ind_k_1] = du[ind_k_1] - lk_1 * u2[batch_count * k + gid];
            }
        }

        // Forward elimination (L * x_new = x_old)
        rocsparse_int start = 0;
        for(rocsparse_int k = 1; k < m; k++)
        {
            rocsparse_int ind_k = batch_stride * k + gid;
            if(p[batch_count * k + gid] <= k) // no pivoting occurred, sum up result
            {
                T temp = static_cast<T>(0);
                for(rocsparse_int s = start; s < k; s++)
                {
                    temp = temp - dl[batch_stride * (s + 1) + gid] * x[batch_stride * s + gid];
                }
                x[ind_k] = x[ind_k] + temp;
                start += k - start;
            }
        }

        // backward substitution (U * x_newest = x_new)
        x[batch_stride * (m - 1) + gid]
            = x[batch_stride * (m - 1) + gid] / d[batch_stride * (m - 1) + gid];
        x[batch_stride * (m - 2) + gid]
            = (x[batch_stride * (m - 2) + gid]
               - du[batch_stride * (m - 2) + gid] * x[batch_stride * (m - 1) + gid])
              / d[batch_stride * (m - 2) + gid];
        for(rocsparse_int k = m - 3; k >= 0; k--)
        {
            rocsparse_int ind_k   = batch_stride * k + gid;
            rocsparse_int ind_k_1 = batch_stride * (k + 1) + gid;
            rocsparse_int ind_k_2 = batch_stride * (k + 2) + gid;

            x[ind_k] = (x[ind_k] - du[ind_k] * x[ind_k_1] - u2[batch_count * k + gid] * x[ind_k_2])
                       / d[ind_k];
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_interleaved_batch_qr_kernel(rocsparse_int m,
                                          rocsparse_int batch_count,
                                          rocsparse_int batch_stride,
                                          const T* __restrict__ dl,
                                          T* __restrict__ d,
                                          T* __restrict__ du,
                                          T* __restrict__ r2,
                                          T* __restrict__ x)
    {
        rocsparse_int gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= batch_count)
        {
            return;
        }

        for(rocsparse_int i = 0; i < m - 1; i++)
        {
            rocsparse_int ind_k   = batch_stride * i + gid;
            rocsparse_int ind_k_1 = batch_stride * (i + 1) + gid;

            T ak_1 = dl[ind_k_1];
            T bk   = d[ind_k];
            T bk_1 = d[ind_k_1];
            T ck   = du[ind_k];
            T ck_1 = du[ind_k_1];

            T radius = rocsparse::sqrt(
                rocsparse::abs(bk * rocsparse::conj(bk) + ak_1 * rocsparse::conj(ak_1)));

            // Apply Givens rotation
            // | cos  sin | |bk    ck   0   |
            // |-sin  cos | |ak_1  bk_1 ck_1|
            T cos_theta = rocsparse::conj(bk) / radius;
            T sin_theta = rocsparse::conj(ak_1) / radius;

            d[ind_k]   = rocsparse::fma(bk, cos_theta, ak_1 * sin_theta);
            d[ind_k_1] = rocsparse::fma(
                -ck, rocsparse::conj(sin_theta), bk_1 * rocsparse::conj(cos_theta));
            du[ind_k]                 = rocsparse::fma(ck, cos_theta, bk_1 * sin_theta);
            du[ind_k_1]               = ck_1 * rocsparse::conj(cos_theta);
            r2[batch_count * i + gid] = ck_1 * sin_theta;

            // Apply Givens rotation to rhs vector
            // | cos  sin | |xk  |
            // |-sin  cos | |xk_1|
            T xk       = x[ind_k];
            T xk_1     = x[ind_k_1];
            x[ind_k]   = rocsparse::fma(xk, cos_theta, xk_1 * sin_theta);
            x[ind_k_1] = rocsparse::fma(
                -xk, rocsparse::conj(sin_theta), xk_1 * rocsparse::conj(cos_theta));
        }

        x[batch_stride * (m - 1) + gid]
            = x[batch_stride * (m - 1) + gid] / d[batch_stride * (m - 1) + gid];
        x[batch_stride * (m - 2) + gid]
            = (x[batch_stride * (m - 2) + gid]
               - du[batch_stride * (m - 2) + gid] * x[batch_stride * (m - 1) + gid])
              / d[batch_stride * (m - 2) + gid];

        for(rocsparse_int i = m - 3; i >= 0; i--)
        {
            rocsparse_int ind_k   = batch_stride * i + gid;
            rocsparse_int ind_k_1 = batch_stride * (i + 1) + gid;
            rocsparse_int ind_k_2 = batch_stride * (i + 2) + gid;

            x[ind_k] = (x[ind_k] - du[ind_k] * x[ind_k_1] - r2[batch_count * i + gid] * x[ind_k_2])
                       / d[ind_k];
        }
    }
}
