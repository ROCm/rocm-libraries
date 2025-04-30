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

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gpsv_strided_gather(rocsparse_int m,
                             rocsparse_int batch_count,
                             rocsparse_int batch_stride,
                             const T* __restrict__ in,
                             T* __restrict__ out)
    {
        // Current batch this thread works on
        rocsparse_int b = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(b >= batch_count)
        {
            return;
        }

        // Process all entries of the batch
        for(rocsparse_int i = 0; i < m; ++i)
        {
            out[batch_count * i + b] = in[batch_stride * i + b];
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gpsv_interleaved_batch_householder_qr_kernel(rocsparse_int m,
                                                      rocsparse_int batch_count,
                                                      rocsparse_int batch_stride,
                                                      T* __restrict__ ds,
                                                      T* __restrict__ dl,
                                                      T* __restrict__ d,
                                                      T* __restrict__ du,
                                                      T* __restrict__ dw,
                                                      T* __restrict__ X,
                                                      T* __restrict__ t1,
                                                      T* __restrict__ t2,
                                                      T* __restrict__ B)
    {
        // Current batch this thread works on
        rocsparse_int b = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(b >= batch_count)
        {
            return;
        }

        // Process all rows but the last four
        for(rocsparse_int row = 0; row < m - 1; ++row)
        {
            bool second_last_row = (row == m - 2);

            // Some indices
            rocsparse_int idp0  = batch_stride * (row + 0) + b;
            rocsparse_int idp1  = batch_stride * (row + 1) + b;
            rocsparse_int idp2  = batch_stride * (row + 2) + b;
            rocsparse_int idp0c = batch_count * (row + 0) + b;
            rocsparse_int idp1c = batch_count * (row + 1) + b;
            rocsparse_int idp2c = batch_count * (row + 2) + b;

            // Some values
            T dlp1 = dl[idp1];
            T dp1  = d[idp1];
            T dup1 = du[idp1];
            T dwp1 = dw[idp1];
            T Bp1  = B[idp1c];

            // Prefetch
            T dsp2 = static_cast<T>(0);
            T dlp2 = static_cast<T>(0);
            T dp2  = static_cast<T>(0);
            T dup2 = static_cast<T>(0);
            T dwp2 = static_cast<T>(0);
            T Bp2  = static_cast<T>(0);

            if(!second_last_row)
            {
                dsp2 = ds[idp2];
                dlp2 = dl[idp2];
                dp2  = d[idp2];
                dup2 = du[idp2];
                dwp2 = dw[idp2];
                Bp2  = B[idp2c];
            }

            T v1 = dlp1;
            T v2 = dsp2;

            T v1v2sq = rocsparse::fma(v1, v1, v2 * v2);

            if(v1v2sq != static_cast<T>(0))
            {
                T diag = d[idp0];
                T val  = rocsparse::sqrt(rocsparse::fma(diag, diag, v1v2sq));
                T a_ii = (rocsparse::gt(diag, static_cast<T>(0))) ? diag + val : diag - val;

                v1 = v1 / a_ii;
                v2 = v2 / a_ii;

                T sq   = a_ii * a_ii;
                T beta = static_cast<T>(2) * sq / (v1v2sq + sq);
                T tau  = static_cast<T>(2)
                        / rocsparse::fma(v2, v2, (rocsparse::fma(v1, v1, static_cast<T>(1))));

                // Process the five diagonals
                T d1 = rocsparse::fma(v2, dsp2, rocsparse::fma(v1, dlp1, diag)) * beta;
                T d2 = rocsparse::fma(v2, dlp2, rocsparse::fma(v1, dp1, du[idp0])) * beta;
                T d3 = rocsparse::fma(v2, dp2, rocsparse::fma(v1, dup1, dw[idp0])) * beta;
                T d4 = rocsparse::fma(v2, dup2, rocsparse::fma(v1, dwp1, t1[idp0c])) * beta;
                T d5 = rocsparse::fma(v2, dwp2, rocsparse::fma(v1, t1[idp1c], t2[idp0c])) * beta;
                T fs = rocsparse::fma(v2, Bp2, rocsparse::fma(v1, Bp1, B[idp0c])) * tau;

                // Update
                d[idp0] -= d1;
                du[idp0] -= d2;
                dw[idp0] -= d3;
                t1[idp0c] -= d4;
                t2[idp0c] -= d5;
                B[idp0c] -= fs;

                dl[idp1]  = v1;
                d[idp1]   = rocsparse::fma(-d2, v1, dp1);
                du[idp1]  = rocsparse::fma(-d3, v1, dup1);
                dw[idp1]  = rocsparse::fma(-d4, v1, dwp1);
                t1[idp1c] = rocsparse::fma(-d5, v1, t1[idp1c]);
                B[idp1c]  = rocsparse::fma(-v1, fs, Bp1);

                if(!second_last_row)
                {
                    ds[idp2] = v2;
                    dl[idp2] = rocsparse::fma(-d2, v2, dlp2);
                    d[idp2]  = rocsparse::fma(-d3, v2, dp2);
                    du[idp2] = rocsparse::fma(-d4, v2, dup2);
                    dw[idp2] = rocsparse::fma(-d5, v2, dwp2);
                    B[idp2c] = rocsparse::fma(-v2, fs, Bp2);
                }
            }
        }

        // Backsolve
        for(rocsparse_int row = m - 1; row >= 0; --row)
        {
            rocsparse_int idp0  = batch_stride * row + b;
            rocsparse_int idp0c = batch_count * row + b;

            T sum = static_cast<T>(0);

            if(row + 1 < m)
                sum += du[idp0] * X[batch_stride * (row + 1) + b];
            if(row + 2 < m)
                sum += dw[idp0] * X[batch_stride * (row + 2) + b];
            if(row + 3 < m)
                sum += t1[idp0c] * X[batch_stride * (row + 3) + b];
            if(row + 4 < m)
                sum += t2[idp0c] * X[batch_stride * (row + 4) + b];

            X[idp0] = (B[idp0c] - sum) / d[idp0];
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gpsv_interleaved_batch_givens_qr_kernel(rocsparse_int m,
                                                 rocsparse_int batch_count,
                                                 rocsparse_int batch_stride,
                                                 T* __restrict__ ds,
                                                 T* __restrict__ dl,
                                                 T* __restrict__ d,
                                                 T* __restrict__ du,
                                                 T* __restrict__ dw,
                                                 T* __restrict__ r3,
                                                 T* __restrict__ r4,
                                                 T* __restrict__ x)
    {
        rocsparse_int gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= batch_count)
        {
            return;
        }

        for(rocsparse_int i = 0; i < m - 2; i++)
        {
            rocsparse_int ind_k   = batch_stride * i + gid;
            rocsparse_int ind_k_1 = batch_stride * (i + 1) + gid;
            rocsparse_int ind_k_2 = batch_stride * (i + 2) + gid;

            // For penta diagonal matrices, need to apply two givens rotations to remove lower and lower - 1 entries
            T radius    = static_cast<T>(0);
            T cos_theta = static_cast<T>(0);
            T sin_theta = static_cast<T>(0);

            // Apply first Givens rotation
            // | cos  sin | |lk_1 dk_1 uk_1 wk_1 0   |
            // |-sin  cos | |sk_2 lk_2 dk_2 uk_2 wk_2|
            T sk_2 = ds[ind_k_2];
            T lk_1 = dl[ind_k_1];
            T lk_2 = dl[ind_k_2];
            T dk_1 = d[ind_k_1];
            T dk_2 = d[ind_k_2];
            T uk_1 = du[ind_k_1];
            T uk_2 = du[ind_k_2];
            T wk_1 = dw[ind_k_1];
            T wk_2 = dw[ind_k_2];

            radius = rocsparse::sqrt(rocsparse::abs(
                rocsparse::fma(lk_1, rocsparse::conj(lk_1), sk_2 * rocsparse::conj(sk_2))));

            cos_theta = rocsparse::conj(lk_1) / radius;
            sin_theta = rocsparse::conj(sk_2) / radius;

            T dlk_1_new = rocsparse::fma(lk_1, cos_theta, sk_2 * sin_theta);
            T dk_1_new  = rocsparse::fma(dk_1, cos_theta, lk_2 * sin_theta);
            T duk_1_new = rocsparse::fma(uk_1, cos_theta, dk_2 * sin_theta);
            T dwk_1_new = rocsparse::fma(wk_1, cos_theta, uk_2 * sin_theta);

            dl[ind_k_1] = dlk_1_new;
            dl[ind_k_2] = rocsparse::fma(
                -dk_1, rocsparse::conj(sin_theta), lk_2 * rocsparse::conj(cos_theta));
            d[ind_k_1] = dk_1_new;
            d[ind_k_2] = rocsparse::fma(
                -uk_1, rocsparse::conj(sin_theta), dk_2 * rocsparse::conj(cos_theta));
            du[ind_k_1] = duk_1_new;
            du[ind_k_2] = rocsparse::fma(
                -wk_1, rocsparse::conj(sin_theta), uk_2 * rocsparse::conj(cos_theta));
            dw[ind_k_1]                     = dwk_1_new;
            dw[ind_k_2]                     = wk_2 * rocsparse::conj(cos_theta);
            r3[batch_count * (i + 1) + gid] = wk_2 * sin_theta;

            // Apply first Givens rotation to rhs vector
            // | cos  sin | |xk_1|
            // |-sin  cos | |xk_2|
            T xk_1     = x[ind_k_1];
            T xk_2     = x[ind_k_2];
            x[ind_k_1] = rocsparse::fma(xk_1, cos_theta, xk_2 * sin_theta);
            x[ind_k_2] = rocsparse::fma(
                -xk_1, rocsparse::conj(sin_theta), xk_2 * rocsparse::conj(cos_theta));

            // Apply second Givens rotation
            // | cos  sin | |dk   uk   wk   rk   0   |
            // |-sin  cos | |lk_1 dk_1 uk_1 wk_1 rk_1|
            lk_1   = dlk_1_new;
            T dk   = d[ind_k];
            dk_1   = dk_1_new;
            T uk   = du[ind_k];
            uk_1   = duk_1_new;
            T wk   = dw[ind_k];
            wk_1   = dwk_1_new;
            T rk   = r3[batch_count * i + gid];
            T rk_1 = r3[batch_count * (i + 1) + gid];

            radius    = rocsparse::sqrt(rocsparse::abs(
                rocsparse::fma(dk, rocsparse::conj(dk), lk_1 * rocsparse::conj(lk_1))));
            cos_theta = rocsparse::conj(dk) / radius;
            sin_theta = rocsparse::conj(lk_1) / radius;

            d[ind_k]   = rocsparse::fma(dk, cos_theta, lk_1 * sin_theta);
            d[ind_k_1] = rocsparse::fma(
                -uk, rocsparse::conj(sin_theta), dk_1 * rocsparse::conj(cos_theta));
            du[ind_k]   = rocsparse::fma(uk, cos_theta, dk_1 * sin_theta);
            du[ind_k_1] = rocsparse::fma(
                -wk, rocsparse::conj(sin_theta), uk_1 * rocsparse::conj(cos_theta));
            dw[ind_k]   = rocsparse::fma(wk, cos_theta, uk_1 * sin_theta);
            dw[ind_k_1] = rocsparse::fma(
                -rk, rocsparse::conj(sin_theta), wk_1 * rocsparse::conj(cos_theta));
            r3[batch_count * i + gid]       = rocsparse::fma(rk, cos_theta, wk_1 * sin_theta);
            r3[batch_count * (i + 1) + gid] = rk_1 * rocsparse::conj(cos_theta);
            r4[batch_count * i + gid]       = rk_1 * sin_theta;

            // Apply second Givens rotation to rhs vector
            // | cos  sin | |xk  |
            // |-sin  cos | |xk_1|
            T xk       = x[ind_k];
            xk_1       = x[ind_k_1];
            x[ind_k]   = rocsparse::fma(xk, cos_theta, xk_1 * sin_theta);
            x[ind_k_1] = rocsparse::fma(
                -xk, rocsparse::conj(sin_theta), xk_1 * rocsparse::conj(cos_theta));
        }

        // Apply last Givens rotation
        // | cos  sin | |dk   uk   wk   rk   0   |
        // |-sin  cos | |lk_1 dk_1 uk_1 wk_1 rk_1|
        T lk_1 = dl[batch_stride * (m - 1) + gid];
        T dk   = d[batch_stride * (m - 2) + gid];
        T dk_1 = d[batch_stride * (m - 1) + gid];
        T uk   = du[batch_stride * (m - 2) + gid];
        T uk_1 = du[batch_stride * (m - 1) + gid];
        T wk   = dw[batch_stride * (m - 2) + gid];
        T wk_1 = dw[batch_stride * (m - 1) + gid];
        T rk   = r3[batch_count * (m - 2) + gid];
        T rk_1 = r3[batch_count * (m - 1) + gid];

        T radius = rocsparse::sqrt(
            rocsparse::abs(rocsparse::fma(dk, rocsparse::conj(dk), lk_1 * rocsparse::conj(lk_1))));
        T cos_theta = rocsparse::conj(dk) / radius;
        T sin_theta = rocsparse::conj(lk_1) / radius;

        d[batch_stride * (m - 2) + gid] = rocsparse::fma(dk, cos_theta, lk_1 * sin_theta);
        d[batch_stride * (m - 1) + gid]
            = rocsparse::fma(-uk, rocsparse::conj(sin_theta), dk_1 * rocsparse::conj(cos_theta));
        du[batch_stride * (m - 2) + gid] = rocsparse::fma(uk, cos_theta, dk_1 * sin_theta);
        du[batch_stride * (m - 1) + gid]
            = rocsparse::fma(-wk, rocsparse::conj(sin_theta), uk_1 * rocsparse::conj(cos_theta));
        dw[batch_stride * (m - 2) + gid] = rocsparse::fma(wk, cos_theta, uk_1 * sin_theta);
        dw[batch_stride * (m - 1) + gid]
            = rocsparse::fma(-rk, rocsparse::conj(sin_theta), wk_1 * rocsparse::conj(cos_theta));
        r3[batch_count * (m - 2) + gid] = rocsparse::fma(rk, cos_theta, wk_1 * sin_theta);
        r3[batch_count * (m - 1) + gid] = rk_1 * rocsparse::conj(cos_theta);
        r4[batch_count * (m - 2) + gid] = rk_1 * sin_theta;

        // Apply last Givens rotation to rhs vector
        // | cos  sin | |xk  |
        // |-sin  cos | |xk_1|
        T xk                            = x[batch_stride * (m - 2) + gid];
        T xk_1                          = x[batch_stride * (m - 1) + gid];
        x[batch_stride * (m - 2) + gid] = rocsparse::fma(xk, cos_theta, xk_1 * sin_theta);
        x[batch_stride * (m - 1) + gid]
            = rocsparse::fma(-xk, rocsparse::conj(sin_theta), xk_1 * rocsparse::conj(cos_theta));

        // Backward substitution on upper triangular R * x = x
        x[batch_stride * (m - 1) + gid]
            = x[batch_stride * (m - 1) + gid] / d[batch_stride * (m - 1) + gid];
        x[batch_stride * (m - 2) + gid]
            = (x[batch_stride * (m - 2) + gid]
               - du[batch_stride * (m - 2) + gid] * x[batch_stride * (m - 1) + gid])
              / d[batch_stride * (m - 2) + gid];

        x[batch_stride * (m - 3) + gid]
            = (x[batch_stride * (m - 3) + gid]
               - du[batch_stride * (m - 3) + gid] * x[batch_stride * (m - 2) + gid]
               - dw[batch_stride * (m - 3) + gid] * x[batch_stride * (m - 1) + gid])
              / d[batch_stride * (m - 3) + gid];

        x[batch_stride * (m - 4) + gid]
            = (x[batch_stride * (m - 4) + gid]
               - du[batch_stride * (m - 4) + gid] * x[batch_stride * (m - 3) + gid]
               - dw[batch_stride * (m - 4) + gid] * x[batch_stride * (m - 2) + gid]
               - r3[batch_count * (m - 4) + gid] * x[batch_stride * (m - 1) + gid])
              / d[batch_stride * (m - 4) + gid];

        for(rocsparse_int i = m - 5; i >= 0; i--)
        {
            x[batch_stride * i + gid]
                = (x[batch_stride * i + gid]
                   - du[batch_stride * i + gid] * x[batch_stride * (i + 1) + gid]
                   - dw[batch_stride * i + gid] * x[batch_stride * (i + 2) + gid]
                   - r3[batch_count * i + gid] * x[batch_stride * (i + 3) + gid]
                   - r4[batch_count * i + gid] * x[batch_stride * (i + 4) + gid])
                  / d[batch_stride * i + gid];
        }
    }
}
