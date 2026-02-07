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

#include <vector>

namespace rocsparse
{
    // Small sized problems

    // Cyclic reduction + parallel cyclic reduction algorithms based on paper
    // "Fast Tridiagonal Solvers on the GPU" by Yao Zhang, Jonathan Cohen, and John Owens
    //
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

    // Cyclic reduction algorithm using shared memory
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_cr_pow2_shared_kernel(rocsparse_int m,
                                            rocsparse_int n,
                                            rocsparse_int ldb,
                                            const T* __restrict__ dl,
                                            const T* __restrict__ d,
                                            const T* __restrict__ du,
                                            T* __restrict__ B)
    {
        rocsparse_int tid = hipThreadIdx_x;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE));
        rocsparse_int stride = 1;
        rocsparse_int i      = BLOCKSIZE;

        // Cyclic reduction shared memory
        __shared__ T sa[2 * BLOCKSIZE];
        __shared__ T sb[2 * BLOCKSIZE];
        __shared__ T sc[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE];
        __shared__ T sx[2 * BLOCKSIZE];

        sa[tid]               = dl[tid];
        sa[tid + BLOCKSIZE]   = dl[tid + BLOCKSIZE];
        sb[tid]               = d[tid];
        sb[tid + BLOCKSIZE]   = d[tid + BLOCKSIZE];
        sc[tid]               = du[tid];
        sc[tid + BLOCKSIZE]   = du[tid + BLOCKSIZE];
        srhs[tid]             = B[tid + ldb * hipBlockIdx_x];
        srhs[tid + BLOCKSIZE] = B[tid + BLOCKSIZE + ldb * hipBlockIdx_x];

        __syncthreads();

        // Forward reduction using cyclic reduction
        for(rocsparse_int j = 0; j < iter; j++)
        {
            stride <<= 1; //stride *= 2;

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(right >= 2 * BLOCKSIZE)
                {
                    right = 2 * BLOCKSIZE - 1;
                }

                T k1 = sa[index] / sb[left];
                T k2 = sc[index] / sb[right];

                sb[index]   = sb[index] - sc[left] * k1 - sa[right] * k2;
                srhs[index] = srhs[index] - srhs[left] * k1 - srhs[right] * k2;
                sa[index]   = -sa[left] * k1;
                sc[index]   = -sc[right] * k2;
            }

            i >>= 1; //i /= 2;

            __syncthreads();
        }

        if(tid == 0)
        {
            // Solve 2x2 system
            rocsparse_int i   = stride - 1;
            rocsparse_int j   = 2 * stride - 1;
            T             det = sb[j] * sb[i] - sc[i] * sa[j];
            det               = static_cast<T>(1) / det;

            sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
            sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
        }

        // Backward substitution using cyclic reduction
        i = 2;
        for(rocsparse_int j = 0; j < iter; j++)
        {
            __syncthreads();

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride / 2 - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(left < 0)
                {
                    sx[index] = (srhs[index] - sc[index] * sx[right]) / sb[index];
                }
                else
                {
                    sx[index]
                        = (srhs[index] - sa[index] * sx[left] - sc[index] * sx[right]) / sb[index];
                }
            }

            stride >>= 1; //stride /= 2;
            i <<= 1; //i *= 2;
        }

        __syncthreads();

        B[tid + ldb * hipBlockIdx_x]             = sx[tid];
        B[tid + BLOCKSIZE + ldb * hipBlockIdx_x] = sx[tid + BLOCKSIZE];
    }

    // Parallel cyclic reduction algorithm using shared memory
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_pow2_shared_kernel(rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int ldb,
                                             const T* __restrict__ dl,
                                             const T* __restrict__ d,
                                             const T* __restrict__ du,
                                             T* __restrict__ B)
    {
        rocsparse_int tid = hipThreadIdx_x;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE / 2));
        rocsparse_int stride = 1;

        // Parallel cyclic reduction shared memory
        __shared__ T sa[BLOCKSIZE + 1];
        __shared__ T sb[BLOCKSIZE + 1];
        __shared__ T sc[BLOCKSIZE + 1];
        __shared__ T srhs[BLOCKSIZE + 1];
        __shared__ T sx[BLOCKSIZE + 1];

        // Fill parallel cyclic reduction shared memory
        sa[tid]   = dl[tid];
        sb[tid]   = d[tid];
        sc[tid]   = du[tid];
        srhs[tid] = B[tid + ldb * hipBlockIdx_x];

        __syncthreads();

        for(rocsparse_int j = 0; j < iter; j++)
        {
            rocsparse_int right = tid + stride;
            if(right >= BLOCKSIZE)
                right = BLOCKSIZE - 1;

            rocsparse_int left = tid - stride;
            if(left < 0)
                left = 0;

            T k1 = sa[tid] / sb[left];
            T k2 = sc[tid] / sb[right];

            T tb   = sb[tid] - sc[left] * k1 - sa[right] * k2;
            T trhs = srhs[tid] - srhs[left] * k1 - srhs[right] * k2;
            T ta   = -sa[left] * k1;
            T tc   = -sc[right] * k2;

            __syncthreads();

            sb[tid]   = tb;
            srhs[tid] = trhs;
            sa[tid]   = ta;
            sc[tid]   = tc;

            stride <<= 1; //stride *= 2;

            __syncthreads();
        }

        if(tid < BLOCKSIZE / 2)
        {
            // Solve 2x2 systems
            rocsparse_int i   = tid;
            rocsparse_int j   = tid + stride;
            T             det = sb[j] * sb[i] - sc[i] * sa[j];
            det               = static_cast<T>(1) / det;

            sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
            sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
        }

        __syncthreads();

        B[tid + ldb * hipBlockIdx_x] = sx[tid];
    }

























    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_2x2_kernel(rocsparse_int n,
                                 rocsparse_int ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // a0 | b0 c0 |
            //    | a1 b1 | c1
            const T a1 = dl[1];
            const T b0 = d[0];
            const T b1 = d[1];
            const T c0 = du[0];

            const T rhs0 = B[ldb * gid + 0];
            const T rhs1 = B[ldb * gid + 1];

            // det = b0 * b1 - a1 * c0
            const T det = static_cast<T>(1) / rocsparse::fma(b0, b1, -a1 * c0);

            B[ldb * gid + 0] = (rocsparse::fma(b1, rhs0, -c0 * rhs1)) * det;
            B[ldb * gid + 1] = (rocsparse::fma(rhs1, b0, -rhs0 * a1)) * det;
        }
    }

    // Kernel to solve 3x3 tridiagonal systems using Thomas algorithm
    // Each thread solves one system independently
    //
    // Matrix form:
    // | b0  c0   0 |   | x0 |   | rhs0 |
    // | a1  b1  c1 |   | x1 | = | rhs1 |
    // |  0  a2  b2 |   | x2 |   | rhs2 |
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_3x3_kernel(rocsparse_int n,
                                 rocsparse_int ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // Load matrix coefficients
            const T a1 = dl[1];
            const T a2 = dl[2];
            const T b0 = d[0];
            const T b1 = d[1];
            const T b2 = d[2];
            const T c0 = du[0];
            const T c1 = du[1];

            // Load RHS
            const T rhs0 = B[ldb * gid + 0];
            const T rhs1 = B[ldb * gid + 1];
            const T rhs2 = B[ldb * gid + 2];

            // Forward elimination with FMA
            // First row
            const T c0_prime   = c0 / b0;
            const T rhs0_prime = rhs0 / b0;

            // Second row: denom1 = b1 - a1 * c0_prime
            const T denom1     = rocsparse::fma(-a1, c0_prime, b1);
            const T inv_denom1 = static_cast<T>(1) / denom1;
            const T c1_prime   = c1 * inv_denom1;
            // rhs1_prime = (rhs1 - a1 * rhs0_prime) / denom1
            const T rhs1_prime = rocsparse::fma(-a1, rhs0_prime, rhs1) * inv_denom1;

            // Third row: denom2 = b2 - a2 * c1_prime
            const T denom2     = rocsparse::fma(-a2, c1_prime, b2);
            const T inv_denom2 = static_cast<T>(1) / denom2;
            // rhs2_prime = (rhs2 - a2 * rhs1_prime) / denom2
            const T rhs2_prime = rocsparse::fma(-a2, rhs1_prime, rhs2) * inv_denom2;

            // Back substitution
            const T x2 = rhs2_prime;
            // x1 = rhs1_prime - c1_prime * x2
            const T x1 = rocsparse::fma(-c1_prime, x2, rhs1_prime);
            // x0 = rhs0_prime - c0_prime * x1
            const T x0 = rocsparse::fma(-c0_prime, x1, rhs0_prime);

            // Write solution
            B[ldb * gid + 0] = x0;
            B[ldb * gid + 1] = x1;
            B[ldb * gid + 2] = x2;
        }
    }

    // Kernel to solve 4x4 tridiagonal systems using Thomas algorithm
    // Each thread solves one system independently
    //
    // Matrix form:
    // | b0  c0   0   0 |   | x0 |   | rhs0 |
    // | a1  b1  c1   0 |   | x1 | = | rhs1 |
    // |  0  a2  b2  c2 |   | x2 |   | rhs2 |
    // |  0   0  a3  b3 |   | x3 |   | rhs3 |
    //
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_4x4_kernel(rocsparse_int n,
                                 rocsparse_int ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[1];
            const T a2 = dl[2];
            const T a3 = dl[3];
            const T b0 = d[0];
            const T b1 = d[1];
            const T b2 = d[2];
            const T b3 = d[3];
            const T c0 = du[0];
            const T c1 = du[1];
            const T c2 = du[2];

            // Load RHS for this thread
            const T rhs0 = B[ldb * gid + 0];
            const T rhs1 = B[ldb * gid + 1];
            const T rhs2 = B[ldb * gid + 2];
            const T rhs3 = B[ldb * gid + 3];

            // Forward elimination (Thomas algorithm)

            // First row: normalize by b0
            const T inv_b0     = static_cast<T>(1) / b0;
            const T c0_prime   = c0 * inv_b0;
            const T rhs0_prime = rhs0 * inv_b0;

            // Second row: eliminate a1
            const T denom1     = rocsparse::fma(-a1, c0_prime, b1);
            const T inv_denom1 = static_cast<T>(1) / denom1;
            const T c1_prime   = c1 * inv_denom1;
            const T rhs1_prime = rocsparse::fma(-a1, rhs0_prime, rhs1) * inv_denom1;

            // Third row: eliminate a2
            const T denom2     = rocsparse::fma(-a2, c1_prime, b2);
            const T inv_denom2 = static_cast<T>(1) / denom2;
            const T c2_prime   = c2 * inv_denom2;
            const T rhs2_prime = rocsparse::fma(-a2, rhs1_prime, rhs2) * inv_denom2;

            // Fourth row: eliminate a3
            const T denom3     = rocsparse::fma(-a3, c2_prime, b3);
            const T inv_denom3 = static_cast<T>(1) / denom3;
            const T rhs3_prime = rocsparse::fma(-a3, rhs2_prime, rhs3) * inv_denom3;

            // Back substitution
            const T x3 = rhs3_prime;
            const T x2 = rocsparse::fma(-c2_prime, x3, rhs2_prime);
            const T x1 = rocsparse::fma(-c1_prime, x2, rhs1_prime);
            const T x0 = rocsparse::fma(-c0_prime, x1, rhs0_prime);

            // Write solution
            B[ldb * gid + 0] = x0;
            B[ldb * gid + 1] = x1;
            B[ldb * gid + 2] = x2;
            B[ldb * gid + 3] = x3;
        }
    }

    // Kernel to solve 5x5 tridiagonal systems using Thomas algorithm
    // Each thread solves one system independently
    //
    // Matrix form:
    // | b0  c0   0   0   0 |   | x0 |   | rhs0 |
    // | a1  b1  c1   0   0 |   | x1 |   | rhs1 |
    // |  0  a2  b2  c2   0 |   | x2 | = | rhs2 |
    // |  0   0  a3  b3  c3 |   | x3 |   | rhs3 |
    // |  0   0   0  a4  b4 |   | x4 |   | rhs4 |
    //
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_5x5_kernel(rocsparse_int n,
                                 rocsparse_int ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[1];
            const T a2 = dl[2];
            const T a3 = dl[3];
            const T a4 = dl[4];
            const T b0 = d[0];
            const T b1 = d[1];
            const T b2 = d[2];
            const T b3 = d[3];
            const T b4 = d[4];
            const T c0 = du[0];
            const T c1 = du[1];
            const T c2 = du[2];
            const T c3 = du[3];

            // Load RHS for this thread
            const T rhs0 = B[ldb * gid + 0];
            const T rhs1 = B[ldb * gid + 1];
            const T rhs2 = B[ldb * gid + 2];
            const T rhs3 = B[ldb * gid + 3];
            const T rhs4 = B[ldb * gid + 4];

            // Forward elimination (Thomas algorithm)

            // First row: normalize by b0
            const T inv_b0     = static_cast<T>(1) / b0;
            const T c0_prime   = c0 * inv_b0;
            const T rhs0_prime = rhs0 * inv_b0;

            // Second row: eliminate a1
            const T denom1     = rocsparse::fma(-a1, c0_prime, b1);
            const T inv_denom1 = static_cast<T>(1) / denom1;
            const T c1_prime   = c1 * inv_denom1;
            const T rhs1_prime = rocsparse::fma(-a1, rhs0_prime, rhs1) * inv_denom1;

            // Third row: eliminate a2
            const T denom2     = rocsparse::fma(-a2, c1_prime, b2);
            const T inv_denom2 = static_cast<T>(1) / denom2;
            const T c2_prime   = c2 * inv_denom2;
            const T rhs2_prime = rocsparse::fma(-a2, rhs1_prime, rhs2) * inv_denom2;

            // Fourth row: eliminate a3
            const T denom3     = rocsparse::fma(-a3, c2_prime, b3);
            const T inv_denom3 = static_cast<T>(1) / denom3;
            const T c3_prime   = c3 * inv_denom3;
            const T rhs3_prime = rocsparse::fma(-a3, rhs2_prime, rhs3) * inv_denom3;

            // Fifth row: eliminate a4
            const T denom4     = rocsparse::fma(-a4, c3_prime, b4);
            const T inv_denom4 = static_cast<T>(1) / denom4;
            const T rhs4_prime = rocsparse::fma(-a4, rhs3_prime, rhs4) * inv_denom4;

            // Back substitution
            const T x4 = rhs4_prime;
            const T x3 = rocsparse::fma(-c3_prime, x4, rhs3_prime);
            const T x2 = rocsparse::fma(-c2_prime, x3, rhs2_prime);
            const T x1 = rocsparse::fma(-c1_prime, x2, rhs1_prime);
            const T x0 = rocsparse::fma(-c0_prime, x1, rhs0_prime);

            // Write solution
            B[ldb * gid + 0] = x0;
            B[ldb * gid + 1] = x1;
            B[ldb * gid + 2] = x2;
            B[ldb * gid + 3] = x3;
            B[ldb * gid + 4] = x4;
        }
    }

    // Kernel to solve 6x6 tridiagonal systems using Thomas algorithm
    // Each thread solves one system independently
    //
    // Matrix form:
    // | b0  c0   0   0   0   0 |   | x0 |   | rhs0 |
    // | a1  b1  c1   0   0   0 |   | x1 |   | rhs1 |
    // |  0  a2  b2  c2   0   0 |   | x2 |   | rhs2 |
    // |  0   0  a3  b3  c3   0 |   | x3 | = | rhs3 |
    // |  0   0   0  a4  b4  c4 |   | x4 |   | rhs4 |
    // |  0   0   0   0  a5  b5 |   | x5 |   | rhs5 |
    //
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_6x6_kernel(rocsparse_int n,
                                 rocsparse_int ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[1];
            const T a2 = dl[2];
            const T a3 = dl[3];
            const T a4 = dl[4];
            const T a5 = dl[5];
            const T b0 = d[0];
            const T b1 = d[1];
            const T b2 = d[2];
            const T b3 = d[3];
            const T b4 = d[4];
            const T b5 = d[5];
            const T c0 = du[0];
            const T c1 = du[1];
            const T c2 = du[2];
            const T c3 = du[3];
            const T c4 = du[4];

            // Load RHS for this thread
            const T rhs0 = B[ldb * gid + 0];
            const T rhs1 = B[ldb * gid + 1];
            const T rhs2 = B[ldb * gid + 2];
            const T rhs3 = B[ldb * gid + 3];
            const T rhs4 = B[ldb * gid + 4];
            const T rhs5 = B[ldb * gid + 5];

            // Forward elimination (Thomas algorithm)

            // First row: normalize by b0
            const T inv_b0     = static_cast<T>(1) / b0;
            const T c0_prime   = c0 * inv_b0;
            const T rhs0_prime = rhs0 * inv_b0;

            // Second row: eliminate a1
            const T denom1     = rocsparse::fma(-a1, c0_prime, b1);
            const T inv_denom1 = static_cast<T>(1) / denom1;
            const T c1_prime   = c1 * inv_denom1;
            const T rhs1_prime = rocsparse::fma(-a1, rhs0_prime, rhs1) * inv_denom1;

            // Third row: eliminate a2
            const T denom2     = rocsparse::fma(-a2, c1_prime, b2);
            const T inv_denom2 = static_cast<T>(1) / denom2;
            const T c2_prime   = c2 * inv_denom2;
            const T rhs2_prime = rocsparse::fma(-a2, rhs1_prime, rhs2) * inv_denom2;

            // Fourth row: eliminate a3
            const T denom3     = rocsparse::fma(-a3, c2_prime, b3);
            const T inv_denom3 = static_cast<T>(1) / denom3;
            const T c3_prime   = c3 * inv_denom3;
            const T rhs3_prime = rocsparse::fma(-a3, rhs2_prime, rhs3) * inv_denom3;

            // Fifth row: eliminate a4
            const T denom4     = rocsparse::fma(-a4, c3_prime, b4);
            const T inv_denom4 = static_cast<T>(1) / denom4;
            const T c4_prime   = c4 * inv_denom4;
            const T rhs4_prime = rocsparse::fma(-a4, rhs3_prime, rhs4) * inv_denom4;

            // Sixth row: eliminate a5
            const T denom5     = rocsparse::fma(-a5, c4_prime, b5);
            const T inv_denom5 = static_cast<T>(1) / denom5;
            const T rhs5_prime = rocsparse::fma(-a5, rhs4_prime, rhs5) * inv_denom5;

            // Back substitution
            const T x5 = rhs5_prime;
            const T x4 = rocsparse::fma(-c4_prime, x5, rhs4_prime);
            const T x3 = rocsparse::fma(-c3_prime, x4, rhs3_prime);
            const T x2 = rocsparse::fma(-c2_prime, x3, rhs2_prime);
            const T x1 = rocsparse::fma(-c1_prime, x2, rhs1_prime);
            const T x0 = rocsparse::fma(-c0_prime, x1, rhs0_prime);

            // Write solution
            B[ldb * gid + 0] = x0;
            B[ldb * gid + 1] = x1;
            B[ldb * gid + 2] = x2;
            B[ldb * gid + 3] = x3;
            B[ldb * gid + 4] = x4;
            B[ldb * gid + 5] = x5;
        }
    }

    // Kernel to solve 7x7 tridiagonal systems using Thomas algorithm
    // Each thread solves one system independently
    //
    // Matrix form:
    // | b0  c0   0   0   0   0   0 |   | x0 |   | rhs0 |
    // | a1  b1  c1   0   0   0   0 |   | x1 |   | rhs1 |
    // |  0  a2  b2  c2   0   0   0 |   | x2 |   | rhs2 |
    // |  0   0  a3  b3  c3   0   0 |   | x3 |   | rhs3 |
    // |  0   0   0  a4  b4  c4   0 |   | x4 | = | rhs4 |
    // |  0   0   0   0  a5  b5  c5 |   | x5 |   | rhs5 |
    // |  0   0   0   0   0  a6  b6 |   | x6 |   | rhs6 |
    //
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_7x7_kernel(rocsparse_int n,
                                 rocsparse_int ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[1];
            const T a2 = dl[2];
            const T a3 = dl[3];
            const T a4 = dl[4];
            const T a5 = dl[5];
            const T a6 = dl[6];
            const T b0 = d[0];
            const T b1 = d[1];
            const T b2 = d[2];
            const T b3 = d[3];
            const T b4 = d[4];
            const T b5 = d[5];
            const T b6 = d[6];
            const T c0 = du[0];
            const T c1 = du[1];
            const T c2 = du[2];
            const T c3 = du[3];
            const T c4 = du[4];
            const T c5 = du[5];

            // Load RHS for this thread
            const T rhs0 = B[ldb * gid + 0];
            const T rhs1 = B[ldb * gid + 1];
            const T rhs2 = B[ldb * gid + 2];
            const T rhs3 = B[ldb * gid + 3];
            const T rhs4 = B[ldb * gid + 4];
            const T rhs5 = B[ldb * gid + 5];
            const T rhs6 = B[ldb * gid + 6];

            // Forward elimination (Thomas algorithm)

            // First row: normalize by b0
            const T inv_b0     = static_cast<T>(1) / b0;
            const T c0_prime   = c0 * inv_b0;
            const T rhs0_prime = rhs0 * inv_b0;

            // Second row: eliminate a1
            const T denom1     = rocsparse::fma(-a1, c0_prime, b1);
            const T inv_denom1 = static_cast<T>(1) / denom1;
            const T c1_prime   = c1 * inv_denom1;
            const T rhs1_prime = rocsparse::fma(-a1, rhs0_prime, rhs1) * inv_denom1;

            // Third row: eliminate a2
            const T denom2     = rocsparse::fma(-a2, c1_prime, b2);
            const T inv_denom2 = static_cast<T>(1) / denom2;
            const T c2_prime   = c2 * inv_denom2;
            const T rhs2_prime = rocsparse::fma(-a2, rhs1_prime, rhs2) * inv_denom2;

            // Fourth row: eliminate a3
            const T denom3     = rocsparse::fma(-a3, c2_prime, b3);
            const T inv_denom3 = static_cast<T>(1) / denom3;
            const T c3_prime   = c3 * inv_denom3;
            const T rhs3_prime = rocsparse::fma(-a3, rhs2_prime, rhs3) * inv_denom3;

            // Fifth row: eliminate a4
            const T denom4     = rocsparse::fma(-a4, c3_prime, b4);
            const T inv_denom4 = static_cast<T>(1) / denom4;
            const T c4_prime   = c4 * inv_denom4;
            const T rhs4_prime = rocsparse::fma(-a4, rhs3_prime, rhs4) * inv_denom4;

            // Sixth row: eliminate a5
            const T denom5     = rocsparse::fma(-a5, c4_prime, b5);
            const T inv_denom5 = static_cast<T>(1) / denom5;
            const T c5_prime   = c5 * inv_denom5;
            const T rhs5_prime = rocsparse::fma(-a5, rhs4_prime, rhs5) * inv_denom5;

            // Seventh row: eliminate a6
            const T denom6     = rocsparse::fma(-a6, c5_prime, b6);
            const T inv_denom6 = static_cast<T>(1) / denom6;
            const T rhs6_prime = rocsparse::fma(-a6, rhs5_prime, rhs6) * inv_denom6;

            // Back substitution
            const T x6 = rhs6_prime;
            const T x5 = rocsparse::fma(-c5_prime, x6, rhs5_prime);
            const T x4 = rocsparse::fma(-c4_prime, x5, rhs4_prime);
            const T x3 = rocsparse::fma(-c3_prime, x4, rhs3_prime);
            const T x2 = rocsparse::fma(-c2_prime, x3, rhs2_prime);
            const T x1 = rocsparse::fma(-c1_prime, x2, rhs1_prime);
            const T x0 = rocsparse::fma(-c0_prime, x1, rhs0_prime);

            // Write solution
            B[ldb * gid + 0] = x0;
            B[ldb * gid + 1] = x1;
            B[ldb * gid + 2] = x2;
            B[ldb * gid + 3] = x3;
            B[ldb * gid + 4] = x4;
            B[ldb * gid + 5] = x5;
            B[ldb * gid + 6] = x6;
        }
    }

    // Thomas algorithm kernel for solving multiple tridiagonal systems in parallel
    //
    // This kernel implements the Thomas algorithm (a specialized form of Gaussian elimination
    // for tridiagonal systems) where each thread independently solves one tridiagonal system.
    //
    // Algorithm Overview:
    // ------------------
    // Solves Ax = b where A is an M×M tridiagonal matrix of the form:
    //
    //   [ d[0]  du[0]   0      0    ...    0   ]   [ x[0]   ]   [ b[0]   ]
    //   [ dl[1] d[1]   du[1]   0    ...    0   ]   [ x[1]   ]   [ b[1]   ]
    //   [ 0     dl[2]  d[2]   du[2] ...    0   ] × [ x[2]   ] = [ b[2]   ]
    //   [ ...   ...    ...    ...   ...   ...  ]   [ ...    ]   [ ...    ]
    //   [ 0     0      0      0    dl[M-1] d[M-1]] [ x[M-1] ]   [ b[M-1] ]
    //
    // where:
    //   - dl: lower diagonal (M elements, dl[0] unused)
    //   - d:  main diagonal (M elements)
    //   - du: upper diagonal (M elements, du[M-1] unused)
    //   - B:  right-hand side vectors (n systems of size M, stored column-major with stride ldb)
    //
    // Two-Phase Approach:
    // -------------------
    // 1. Forward Sweep (Forward Elimination):
    //    Eliminates the lower diagonal by computing modified upper diagonal and RHS:
    //      du'[0] = du[0] / d[0]
    //      du'[i] = du[i] / (d[i] - dl[i] * du'[i-1])  for i = 1..M-2
    //
    //      B'[0] = B[0] / d[0]
    //      B'[i] = (B[i] - dl[i] * B'[i-1]) / (d[i] - dl[i] * du'[i-1])  for i = 1..M-1
    //
    // 2. Backward Sweep (Back Substitution):
    //    Solves for x using the modified system:
    //      x[M-1] = B'[M-1]
    //      x[i] = B'[i] - du'[i] * x[i+1]  for i = M-2..0
    //
    // Parallelization:
    // ----------------
    // - Each thread processes one independent tridiagonal system (one column of B)
    // - Thread gid solves the system using B[ldb*gid : ldb*gid+M-1] as RHS
    // - No inter-thread communication required since systems are independent
    // - Optimal for n >> M where n is the number of systems
    //
    // Template Parameters:
    // --------------------
    // - BLOCKSIZE: Number of threads per block
    // - M: Size of each tridiagonal system (must be known at compile time)
    // - T: Data type (float, double, or complex types)
    //
    // Note: This is a "no pivot" algorithm assuming the matrix is diagonally dominant
    //       or otherwise numerically stable without pivoting
    template <uint32_t BLOCKSIZE, uint32_t M, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_thomas_kernel(rocsparse_int n,
                                    rocsparse_int ldb,
                                    const T* __restrict__ dl,
                                    const T* __restrict__ d,
                                    const T* __restrict__ du,
                                    T* __restrict__ B)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            T du_prime[M];
            T B_prime[M];

            // Forward sweep
            du_prime[0] = du[0] / d[0];
            for(int i = 1; i < M - 1; i++)
            {
                T num       = du[i];
                T denom     = d[i] - dl[i] * du_prime[i - 1];
                du_prime[i] = num / denom;
            }

            B_prime[0] = B[ldb * gid + 0] / d[0];
            for(int i = 1; i < M; i++)
            {
                T num      = B[ldb * gid + i] - dl[i] * B_prime[i - 1];
                T denom    = d[i] - dl[i] * du_prime[i - 1];
                B_prime[i] = num / denom;
            }

            // Backward sweep
            B[ldb * gid + M - 1] = B_prime[M - 1];
            for(int i = M - 2; i >= 0; i--)
            {
                B[ldb * gid + i] = B_prime[i] - du_prime[i] * B[ldb * gid + i + 1];
            }
        }
    }

    template <uint32_t WF_SIZE, int NUM_ELEMENTS = 8, int SUB_GROUP_SIZE = 8>
    __device__ void wavefront_transpose(float* row)
    {
        static constexpr int NUM_STAGES = rocsparse::log2_pow2<SUB_GROUP_SIZE>::value;

        // laneId within the WF_SIZE-thread warp (0-WF_SIZE-1)
        int laneId = threadIdx.x % WF_SIZE;
        
        // Each thread only swaps with partners within its own 8-lane segment
        // Group 0: 0-7, Group 1: 8-15, etc.        
        for(int i = 0; i < NUM_STAGES; ++i)
        {
            int mask = 1 << i;

            for(int j = 0; j < NUM_ELEMENTS; ++j)
            {
                // The logic here mimics your square transpose but localized
                if((j & mask) == 0)
                {
                    int  regA   = j;
                    int  regB   = j | mask;
                    
                    // Determine if this specific thread is the 'high' or 'low' partner
                    bool bitSet = (laneId & mask) != 0;

                    float val        = bitSet ? row[regA] : row[regB];
                    
                    // __shfl_xor restricted to the 8-thread sub-group
                    float partnerVal = __shfl_xor(val, mask, SUB_GROUP_SIZE);

                    if(bitSet)
                        row[regA] = partnerVal;
                    else
                        row[regB] = partnerVal;
                }
            }
        }
    }

#if ROCSPARSE_USE_MOVE_DPP
    // DPP-based wavefront transpose (same semantics as shuffle version).
    // Uses __hip_move_dpp for XOR lane exchange: mask 1 -> 0x142, mask 2 -> 0x143, mask 4 -> 0x130.
    template <uint32_t WF_SIZE, int NUM_ELEMENTS = 8, int SUB_GROUP_SIZE = 8>
    __device__ void wavefront_transpose_dpp(float* row)
    {
        static constexpr int NUM_STAGES = rocsparse::log2_pow2<SUB_GROUP_SIZE>::value;
        int                  laneId   = threadIdx.x % WF_SIZE;

        typedef union
        {
            float    f;
            uint32_t u32;
        } flt_u32;

        for(int i = 0; i < NUM_STAGES; ++i)
        {
            int mask = 1 << i;
            // DPP control: 0x142 = quad swap (XOR 1), 0x143 = quad swap (XOR 2), 0x130 = row_ror:4 (XOR 4)
            int dpp_ctrl = (mask == 1) ? 0x142 : (mask == 2) ? 0x143 : 0x130;

            for(int j = 0; j < NUM_ELEMENTS; ++j)
            {
                if((j & mask) == 0)
                {
                    int  regA   = j;
                    int  regB   = j | mask;
                    bool bitSet = (laneId & mask) != 0;

                    flt_u32 val;
                    val.f = bitSet ? row[regA] : row[regB];

                    flt_u32 partner;
                    partner.u32 = __hip_move_dpp(val.u32, dpp_ctrl, 0xf, 0xf, false);

                    if(bitSet)
                        row[regA] = partner.f;
                    else
                        row[regB] = partner.f;
                }
            }
        }
    }

    template <uint32_t WF_SIZE, int NUM_ELEMENTS = 8, int SUB_GROUP_SIZE = 8>
    __device__ void wavefront_transpose_dpp(double* row)
    {
        static constexpr int NUM_STAGES = rocsparse::log2_pow2<SUB_GROUP_SIZE>::value;
        int                  laneId   = threadIdx.x % WF_SIZE;

        typedef union
        {
            double   d;
            uint32_t u32[2];
        } dbl_u32;

        for(int i = 0; i < NUM_STAGES; ++i)
        {
            int mask     = 1 << i;
            int dpp_ctrl = (mask == 1) ? 0x142 : (mask == 2) ? 0x143 : 0x130;

            for(int j = 0; j < NUM_ELEMENTS; ++j)
            {
                if((j & mask) == 0)
                {
                    int  regA   = j;
                    int  regB   = j | mask;
                    bool bitSet = (laneId & mask) != 0;

                    dbl_u32 val;
                    val.d = bitSet ? row[regA] : row[regB];

                    dbl_u32 partner;
                    partner.u32[0] = __hip_move_dpp(val.u32[0], dpp_ctrl, 0xf, 0xf, false);
                    partner.u32[1] = __hip_move_dpp(val.u32[1], dpp_ctrl, 0xf, 0xf, false);

                    if(bitSet)
                        row[regA] = partner.d;
                    else
                        row[regB] = partner.d;
                }
            }
        }
    }

    template <uint32_t WF_SIZE, int NUM_ELEMENTS = 8, int SUB_GROUP_SIZE = 8>
    __device__ void wavefront_transpose_dpp(rocsparse_float_complex* row)
    {
        static constexpr int NUM_STAGES = rocsparse::log2_pow2<SUB_GROUP_SIZE>::value;
        int                  laneId   = threadIdx.x % WF_SIZE;

        typedef union
        {
            float    f;
            uint32_t u32;
        } flt_u32;

        for(int i = 0; i < NUM_STAGES; ++i)
        {
            int mask     = 1 << i;
            int dpp_ctrl = (mask == 1) ? 0x142 : (mask == 2) ? 0x143 : 0x130;

            for(int j = 0; j < NUM_ELEMENTS; ++j)
            {
                if((j & mask) == 0)
                {
                    int  regA   = j;
                    int  regB   = j | mask;
                    bool bitSet = (laneId & mask) != 0;

                    flt_u32 valRe, valIm;
                    valRe.f = bitSet ? std::real(row[regA]) : std::real(row[regB]);
                    valIm.f = bitSet ? std::imag(row[regA]) : std::imag(row[regB]);

                    flt_u32 partnerRe, partnerIm;
                    partnerRe.u32 = __hip_move_dpp(valRe.u32, dpp_ctrl, 0xf, 0xf, false);
                    partnerIm.u32 = __hip_move_dpp(valIm.u32, dpp_ctrl, 0xf, 0xf, false);

                    if(bitSet)
                        row[regA] = rocsparse_float_complex(partnerRe.f, partnerIm.f);
                    else
                        row[regB] = rocsparse_float_complex(partnerRe.f, partnerIm.f);
                }
            }
        }
    }

    template <uint32_t WF_SIZE, int NUM_ELEMENTS = 8, int SUB_GROUP_SIZE = 8>
    __device__ void wavefront_transpose_dpp(rocsparse_double_complex* row)
    {
        static constexpr int NUM_STAGES = rocsparse::log2_pow2<SUB_GROUP_SIZE>::value;
        int                  laneId   = threadIdx.x % WF_SIZE;

        typedef union
        {
            double   d;
            uint32_t u32[2];
        } dbl_u32;

        for(int i = 0; i < NUM_STAGES; ++i)
        {
            int mask     = 1 << i;
            int dpp_ctrl = (mask == 1) ? 0x142 : (mask == 2) ? 0x143 : 0x130;

            for(int j = 0; j < NUM_ELEMENTS; ++j)
            {
                if((j & mask) == 0)
                {
                    int  regA   = j;
                    int  regB   = j | mask;
                    bool bitSet = (laneId & mask) != 0;

                    dbl_u32 valRe, valIm;
                    valRe.d = bitSet ? std::real(row[regA]) : std::real(row[regB]);
                    valIm.d = bitSet ? std::imag(row[regA]) : std::imag(row[regB]);

                    dbl_u32 partnerRe, partnerIm;
                    partnerRe.u32[0] = __hip_move_dpp(valRe.u32[0], dpp_ctrl, 0xf, 0xf, false);
                    partnerRe.u32[1] = __hip_move_dpp(valRe.u32[1], dpp_ctrl, 0xf, 0xf, false);
                    partnerIm.u32[0] = __hip_move_dpp(valIm.u32[0], dpp_ctrl, 0xf, 0xf, false);
                    partnerIm.u32[1] = __hip_move_dpp(valIm.u32[1], dpp_ctrl, 0xf, 0xf, false);

                    if(bitSet)
                        row[regA] = rocsparse_double_complex(partnerRe.d, partnerIm.d);
                    else
                        row[regB] = rocsparse_double_complex(partnerRe.d, partnerIm.d);
                }
            }
        }
    }
#endif /* ROCSPARSE_USE_MOVE_DPP */

    template <uint32_t WF_SIZE, int NUM_ELEMENTS = 8, int SUB_GROUP_SIZE = 8>
    __device__ void wavefront_transpose(double* row)
    {
        static constexpr int NUM_STAGES = rocsparse::log2_pow2<SUB_GROUP_SIZE>::value;

        // laneId within the WF_SIZE-thread warp (0-WF_SIZE-1)
        int laneId = threadIdx.x % WF_SIZE;
        
        // Each thread only swaps with partners within its own 8-lane segment
        // Group 0: 0-7, Group 1: 8-15, etc.        
        for(int i = 0; i < NUM_STAGES; ++i)
        {
            int mask = 1 << i;

            for(int j = 0; j < NUM_ELEMENTS; ++j)
            {
                // The logic here mimics your square transpose but localized
                if((j & mask) == 0)
                {
                    int  regA   = j;
                    int  regB   = j | mask;
                    
                    // Determine if this specific thread is the 'high' or 'low' partner
                    bool bitSet = (laneId & mask) != 0;

                    double val        = bitSet ? row[regA] : row[regB];
                    
                    // __shfl_xor restricted to the 8-thread sub-group
                    double partnerVal = __shfl_xor(val, mask, SUB_GROUP_SIZE);

                    if(bitSet)
                        row[regA] = partnerVal;
                    else
                        row[regB] = partnerVal;
                }
            }
        }
    }

    template <uint32_t WF_SIZE, int NUM_ELEMENTS = 8, int SUB_GROUP_SIZE = 8>
    __device__ void wavefront_transpose(rocsparse_float_complex* row)
    {
        static constexpr int NUM_STAGES = rocsparse::log2_pow2<SUB_GROUP_SIZE>::value;

        // laneId within the WF_SIZE-thread warp (0-WF_SIZE-1)
        int laneId = threadIdx.x % WF_SIZE;
        
        // Each thread only swaps with partners within its own 8-lane segment
        // Group 0: 0-7, Group 1: 8-15, etc.        
        for(int i = 0; i < NUM_STAGES; ++i)
        {
            int mask = 1 << i;

            for(int j = 0; j < NUM_ELEMENTS; ++j)
            {
                // The logic here mimics your square transpose but localized
                if((j & mask) == 0)
                {
                    int  regA   = j;
                    int  regB   = j | mask;
                    
                    // Determine if this specific thread is the 'high' or 'low' partner
                    bool bitSet = (laneId & mask) != 0;

                    float valRe     = bitSet ? std::real(row[regA]) : std::real(row[regB]);
                    float valIm     = bitSet ? std::imag(row[regA]) : std::imag(row[regB]);
                    float partnerRe = __shfl_xor(valRe, mask, SUB_GROUP_SIZE);
                    float partnerIm = __shfl_xor(valIm, mask, SUB_GROUP_SIZE);

                    if(bitSet)
                        row[regA] = rocsparse_float_complex(partnerRe, partnerIm);
                    else
                        row[regB] = rocsparse_float_complex(partnerRe, partnerIm);
                }
            }
        }
    }

    template <uint32_t WF_SIZE, int NUM_ELEMENTS = 8, int SUB_GROUP_SIZE = 8>
    __device__ void wavefront_transpose(rocsparse_double_complex* row)
    {
        static constexpr int NUM_STAGES = rocsparse::log2_pow2<SUB_GROUP_SIZE>::value;

        // laneId within the WF_SIZE-thread warp (0-WF_SIZE-1)
        int laneId = threadIdx.x & (WF_SIZE - 1);
        
        // Each thread only swaps with partners within its own 8-lane segment
        // Group 0: 0-7, Group 1: 8-15, etc.        
        for(int i = 0; i < NUM_STAGES; ++i)
        {
            int mask = 1 << i;

            for(int j = 0; j < NUM_ELEMENTS; ++j)
            {
                // The logic here mimics your square transpose but localized
                if((j & mask) == 0)
                {
                    int  regA   = j;
                    int  regB   = j | mask;
                    
                    // Determine if this specific thread is the 'high' or 'low' partner
                    bool bitSet = (laneId & mask) != 0;

                    double valRe     = bitSet ? std::real(row[regA]) : std::real(row[regB]);
                    double valIm     = bitSet ? std::imag(row[regA]) : std::imag(row[regB]);
                    double partnerRe = __shfl_xor(valRe, mask, SUB_GROUP_SIZE);
                    double partnerIm = __shfl_xor(valIm, mask, SUB_GROUP_SIZE);

                    if(bitSet)
                        row[regA] = rocsparse_double_complex(partnerRe, partnerIm);
                    else
                        row[regB] = rocsparse_double_complex(partnerRe, partnerIm);
                }
            }
        }
    }



    // // WF_SIZE must be power of 2 (2, 4, 8, 16, 32). Uses rocsparse::log2_pow2<WF_SIZE> for stages.
    // template <uint32_t WF_SIZE>
    // ROCSPARSE_DEVICE_ILF void wavefront_transpose(float* row)
    // {
    //     static constexpr int NUM_STAGES = rocsparse::log2_pow2<WF_SIZE>::value;
    //     int                  laneId    = threadIdx.x % WF_SIZE;

    //     for(int i = 0; i < NUM_STAGES; ++i)
    //     {
    //         int mask = 1 << i;

    //         for(int j = 0; j < WF_SIZE; ++j)
    //         {
    //             if((j & mask) == 0)
    //             {
    //                 int  regA   = j;
    //                 int  regB   = j | mask;
    //                 bool bitSet = (laneId & mask) != 0;

    //                 float val         = bitSet ? row[regA] : row[regB];
    //                 float partnerVal  = __shfl_xor(val, mask, WF_SIZE);

    //                 if(bitSet)
    //                     row[regA] = partnerVal;
    //                 else
    //                     row[regB] = partnerVal;
    //             }
    //         }
    //     }
    // }

    // template <uint32_t WF_SIZE>
    // ROCSPARSE_DEVICE_ILF void wavefront_transpose(double* row)
    // {
    //     static constexpr int NUM_STAGES = rocsparse::log2_pow2<WF_SIZE>::value;
    //     int                  laneId    = threadIdx.x % WF_SIZE;

    //     for(int i = 0; i < NUM_STAGES; ++i)
    //     {
    //         int mask = 1 << i;

    //         for(int j = 0; j < WF_SIZE; ++j)
    //         {
    //             if((j & mask) == 0)
    //             {
    //                 int  regA   = j;
    //                 int  regB   = j | mask;
    //                 bool bitSet = (laneId & mask) != 0;

    //                 double val         = bitSet ? row[regA] : row[regB];
    //                 double partnerVal  = __shfl_xor(val, mask, WF_SIZE);

    //                 if(bitSet)
    //                     row[regA] = partnerVal;
    //                 else
    //                     row[regB] = partnerVal;
    //             }
    //         }
    //     }
    // }

    // template <uint32_t WF_SIZE>
    // ROCSPARSE_DEVICE_ILF void wavefront_transpose(
    //     rocsparse_float_complex* row)
    // {
    //     static constexpr int NUM_STAGES = rocsparse::log2_pow2<WF_SIZE>::value;
    //     int                  laneId    = threadIdx.x % WF_SIZE;

    //     for(int i = 0; i < NUM_STAGES; ++i)
    //     {
    //         int mask = 1 << i;

    //         for(int j = 0; j < WF_SIZE; ++j)
    //         {
    //             if((j & mask) == 0)
    //             {
    //                 int  regA   = j;
    //                 int  regB   = j | mask;
    //                 bool bitSet = (laneId & mask) != 0;

    //                 float valRe     = bitSet ? std::real(row[regA]) : std::real(row[regB]);
    //                 float valIm     = bitSet ? std::imag(row[regA]) : std::imag(row[regB]);
    //                 float partnerRe = __shfl_xor(valRe, mask, WF_SIZE);
    //                 float partnerIm = __shfl_xor(valIm, mask, WF_SIZE);

    //                 if(bitSet)
    //                     row[regA] = rocsparse_float_complex(partnerRe, partnerIm);
    //                 else
    //                     row[regB] = rocsparse_float_complex(partnerRe, partnerIm);
    //             }
    //         }
    //     }
    // }

    // template <uint32_t WF_SIZE>
    // ROCSPARSE_DEVICE_ILF void wavefront_transpose(
    //     rocsparse_double_complex* row)
    // {
    //     static constexpr int NUM_STAGES = rocsparse::log2_pow2<WF_SIZE>::value;
    //     int                  laneId    = threadIdx.x % WF_SIZE;

    //     for(int i = 0; i < NUM_STAGES; ++i)
    //     {
    //         int mask = 1 << i;

    //         for(int j = 0; j < WF_SIZE; ++j)
    //         {
    //             if((j & mask) == 0)
    //             {
    //                 int   regA   = j;
    //                 int   regB   = j | mask;
    //                 bool  bitSet = (laneId & mask) != 0;

    //                 double valRe     = bitSet ? std::real(row[regA]) : std::real(row[regB]);
    //                 double valIm     = bitSet ? std::imag(row[regA]) : std::imag(row[regB]);
    //                 double partnerRe = __shfl_xor(valRe, mask, WF_SIZE);
    //                 double partnerIm = __shfl_xor(valIm, mask, WF_SIZE);

    //                 if(bitSet)
    //                     row[regA] = rocsparse_double_complex(partnerRe, partnerIm);
    //                 else
    //                     row[regB] = rocsparse_double_complex(partnerRe, partnerIm);
    //             }
    //         }
    //     }
    // }


    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t TILE_X, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE) 
    void thomas_shared_transpose_kernel(rocsparse_int m,
                                        rocsparse_int n,
                                        rocsparse_int ldb,
                                        const T* __restrict__ dl,
                                        const T* __restrict__ d,
                                        const T* __restrict__ du,
                                        T* __restrict__ B,
                                        T* __restrict__ temp) 
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;

        const int lid = tid & (WF_SIZE - 1); // 0..WF_SIZE-1
        const int wid = tid / WF_SIZE;  // 0..7

        const int lid_x = lid & (TILE_X - 1); // 0..TILE_X-1
        const int wid_x = lid / TILE_X; // 0..3

        __shared__ T total_shared[(BLOCKSIZE / WF_SIZE) * TILE_X * WF_SIZE];
        T* shared = &total_shared[TILE_X * WF_SIZE * wid];

        int B_offset = ldb * WF_SIZE * ((BLOCKSIZE / WF_SIZE) * bid + wid);

        // Loop over TILE_X and load 8x4=32 elements from B each iteration. 
        // In total this will load 8x(4*TILE_X) elements from B
        for(int i = 0; i < TILE_X; i++)
        {
            shared[WF_SIZE * i + TILE_X * wid_x + lid_x] = B[B_offset + WF_SIZE * i + TILE_X * wid_x + lid_x];
        }
        __syncthreads();

        T B_local[TILE_X];

        for(int i = 0; i < TILE_X; i++)
        {
            B_local[i] = shared[TILE_X * lid + i];
        }

        T du_prime[TILE_X];
        T B_prime[TILE_X];

        // Forward sweep
        du_prime[0] = du[0] / d[0];
        for(int i = 1; i < TILE_X - 1; i++)
        {
            T num       = du[i];
            T denom     = d[i] - dl[i] * du_prime[i - 1];
            du_prime[i] = num / denom;
        }

        B_prime[0] = B_local[0] / d[0];
        for(int i = 1; i < TILE_X; i++)
        {
            T num      = B_local[i] - dl[i] * B_prime[i - 1];
            T denom    = d[i] - dl[i] * du_prime[i - 1];
            B_prime[i] = num / denom;
        }

        // Backward sweep
        B_local[TILE_X - 1] = B_prime[TILE_X - 1];
        for(int i = TILE_X - 2; i >= 0; i--)
        {
            B_local[i] = B_prime[i] - du_prime[i] * B_local[i + 1];
        }

        __syncthreads();
        for(int i = 0; i < TILE_X; i++)
        {
           shared[TILE_X * lid + i] = B_local[i];
        }
        __syncthreads();
        
        for(int i = 0; i < TILE_X; i++)
        {
           B[B_offset + WF_SIZE * i + TILE_X * wid_x + lid_x] = shared[WF_SIZE * i + TILE_X * wid_x + lid_x];
        }
        // for(int i = 0; i < TILE_X; i++)
        // {
        //     B[B_offset + TILE_X * lid + i] = B_local[i];
        // }
    }

    /*template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t TILE_X, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE) 
    void thomas_shared_transpose_kernel2(rocsparse_int m,
                                        rocsparse_int n,
                                        rocsparse_int ldb,
                                        const T* __restrict__ dl,
                                        const T* __restrict__ d,
                                        const T* __restrict__ du,
                                        T* __restrict__ B,
                                        T* __restrict__ temp) 
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;

        const int lid = tid & (WF_SIZE - 1); // 0..WF_SIZE-1
        const int wid = tid / WF_SIZE;  // 0..7

        const int lid_x = lid & (TILE_X - 1); // 0..TILE_X-1
        const int wid_x = lid / TILE_X; // 0..3

        __shared__ T total_shared[(BLOCKSIZE / WF_SIZE) * TILE_X * WF_SIZE];
        T* shared = &total_shared[TILE_X * WF_SIZE * wid];

        int B_offset = ldb * WF_SIZE * ((BLOCKSIZE / WF_SIZE) * bid + wid);

        T du_prime[512];
        T B_prime[512];

        for()
        {
            // Loop over TILE_X and load 8x4=32 elements from B each iteration. 
            // In total this will load 8x(4*TILE_X) elements from B
            for(int i = 0; i < TILE_X; i++)
            {
                shared[WF_SIZE * i + TILE_X * wid_x + lid_x] = B[B_offset + WF_SIZE * i + TILE_X * wid_x + lid_x];
            }

            __syncthreads();

            T B_local[TILE_X];

            for(int i = 0; i < TILE_X; i++)
            {
                B_local[i] = shared[TILE_X * lid + i];
            }

            T du_prime_local[TILE_X];
            T B_prime_local[TILE_X];

            // Forward sweep
            du_prime_local[0] = du[0] / d[0];
            for(int i = 1; i < TILE_X - 1; i++)
            {
                T num       = du[i];
                T denom     = d[i] - dl[i] * du_prime_local[i - 1];
                du_prime_local[i] = num / denom;
            }

            B_prime_local[0] = B_local[0] / d[0];
            for(int i = 1; i < TILE_X; i++)
            {
                T num      = B_local[i] - dl[i] * B_prime_local[i - 1];
                T denom    = d[i] - dl[i] * du_prime_local[i - 1];
                B_prime_local[i] = num / denom;
            }

            // Write du_prime_local and B_prime_local to local memory arrays
            for()
            {
                du_prime[] = du_prime_local[];
                B_prime[] = B_prime[];
            }

            __syncthreads();

        }

        // Backward sweep
        for()
        {
            // Read du_prime and B_prime


            T B_local[TILE_X];

            B_local[TILE_X - 1] = B_prime[TILE_X - 1];
            for(int i = TILE_X - 2; i >= 0; i--)
            {
                B_local[i] = B_prime[i] - du_prime[i] * B_local[i + 1];
            }
        }

        // Backward sweep
        // B_local[TILE_X - 1] = B_prime[TILE_X - 1];
        // for(int i = TILE_X - 2; i >= 0; i--)
        // {
        //     B_local[i] = B_prime[i] - du_prime[i] * B_local[i + 1];
        // }
        // __syncthreads();

        for(int i = 0; i < TILE_X; i++)
        {
            shared[TILE_X * lid + i] = B_local[i];
        }

        __syncthreads();
        
        for(int i = 0; i < TILE_X; i++)
        {
            B[B_offset + WF_SIZE * i + TILE_X * wid_x + lid_x] = shared[WF_SIZE * i + TILE_X * wid_x + lid_x];
        }
    }*/


    
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t NUM_ELEMENTS, uint32_t SUB_GROUP_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_thomas_transpose_kernel(rocsparse_int m,
                                              rocsparse_int n,
                                              rocsparse_int ldb,
                                              const T* __restrict__ dl,
                                              const T* __restrict__ d,
                                              const T* __restrict__ du,
                                              T* __restrict__ B,
                                              T* __restrict__ temp)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;

        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        T B_local[NUM_ELEMENTS];

        // Each thread loads one column of the WF_SIZExNUM_ELEMENTS block (thread lid has column lid)
        for(int i = 0; i < NUM_ELEMENTS; i++)
        {
            //B_local[i]
            //    = temp[WF_SIZE * NUM_ELEMENTS * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
            B_local[i]
                = B[WF_SIZE * NUM_ELEMENTS * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
            // B_local[i]
                // = B[WF_SIZE * NUM_ELEMENTS * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * lid + i];
        }

        wavefront_transpose<WF_SIZE, NUM_ELEMENTS, SUB_GROUP_SIZE>(B_local);

        T du_prime[WF_SIZE];
        T B_prime[WF_SIZE];

        // Forward sweep
        du_prime[0] = du[0] / d[0];
        for(int i = 1; i < WF_SIZE - 1; i++)
        {
            T num       = du[i];
            T denom     = d[i] - dl[i] * du_prime[i - 1];
            du_prime[i] = num / denom;
        }

        B_prime[0] = B_local[0] / d[0];
        for(int i = 1; i < WF_SIZE; i++)
        {
            T num      = B_local[i] - dl[i] * B_prime[i - 1];
            T denom    = d[i] - dl[i] * du_prime[i - 1];
            B_prime[i] = num / denom;
        }

        // Backward sweep
        B_local[WF_SIZE - 1] = B_prime[WF_SIZE - 1];
        for(int i = WF_SIZE - 2; i >= 0; i--)
        {
            B_local[i] = B_prime[i] - du_prime[i] * B_local[i + 1];
        }

        wavefront_transpose<WF_SIZE, NUM_ELEMENTS, SUB_GROUP_SIZE>(B_local);

        for(int i = 0; i < NUM_ELEMENTS; i++)
        {
            // temp[WF_SIZE * NUM_ELEMENTS * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid] = B_local[i];
            B[WF_SIZE * NUM_ELEMENTS * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid] = B_local[i];
            // B[WF_SIZE * NUM_ELEMENTS * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * lid + i] = B_local[i];
        }
    }





























    // Combined Parallel cyclic reduction and cyclic reduction algorithm using shared memory
    template <uint32_t BLOCKSIZE, uint32_t PCR_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_crpcr_pow2_shared_kernel(rocsparse_int m,
                                               rocsparse_int n,
                                               rocsparse_int ldb,
                                               const T* __restrict__ dl,
                                               const T* __restrict__ d,
                                               const T* __restrict__ du,
                                               T* __restrict__ B)
    {
        rocsparse_int tid = hipThreadIdx_x;

        rocsparse_int tot_iter = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE));
        rocsparse_int pcr_iter = static_cast<rocsparse_int>(rocsparse::log2(PCR_SIZE / 2));
        rocsparse_int cr_iter  = tot_iter - pcr_iter;
        rocsparse_int stride   = 1;
        rocsparse_int i        = BLOCKSIZE;

        // Cyclic reduction shared memory
        __shared__ T sa[2 * BLOCKSIZE];
        __shared__ T sb[2 * BLOCKSIZE];
        __shared__ T sc[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE];
        __shared__ T sx[2 * BLOCKSIZE];

        // Parallel cyclic reduction shared memory
        __shared__ T spa[PCR_SIZE];
        __shared__ T spb[PCR_SIZE];
        __shared__ T spc[PCR_SIZE];
        __shared__ T sprhs[PCR_SIZE];
        __shared__ T spx[PCR_SIZE];

        // Fill cyclic reduction shared memory
        sa[tid]               = dl[tid];
        sa[tid + BLOCKSIZE]   = dl[tid + BLOCKSIZE];
        sb[tid]               = d[tid];
        sb[tid + BLOCKSIZE]   = d[tid + BLOCKSIZE];
        sc[tid]               = du[tid];
        sc[tid + BLOCKSIZE]   = du[tid + BLOCKSIZE];
        srhs[tid]             = B[tid + ldb * hipBlockIdx_x];
        srhs[tid + BLOCKSIZE] = B[tid + BLOCKSIZE + ldb * hipBlockIdx_x];

        __syncthreads();

        // Forward reduction using cyclic reduction
        for(rocsparse_int j = 0; j < cr_iter; j++)
        {
            stride <<= 1; //stride *= 2;

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(right >= 2 * BLOCKSIZE)
                {
                    right = 2 * BLOCKSIZE - 1;
                }

                T k1 = sa[index] / sb[left];
                T k2 = sc[index] / sb[right];

                sb[index]   = sb[index] - sc[left] * k1 - sa[right] * k2;
                srhs[index] = srhs[index] - srhs[left] * k1 - srhs[right] * k2;
                sa[index]   = -sa[left] * k1;
                sc[index]   = -sc[right] * k2;
            }

            i >>= 1; //i /= 2;

            __syncthreads();
        }

        // Parallel cyclic reduction
        if(tid < PCR_SIZE)
        {
            spa[tid]   = sa[tid * stride + stride - 1];
            spb[tid]   = sb[tid * stride + stride - 1];
            spc[tid]   = sc[tid * stride + stride - 1];
            sprhs[tid] = srhs[tid * stride + stride - 1];
        }

        __syncthreads();

        rocsparse_int pcr_stride = 1;
        for(rocsparse_int j = 0; j < pcr_iter; j++)
        {
            T ta;
            T tb;
            T tc;
            T trhs;

            if(tid < PCR_SIZE)
            {
                rocsparse_int right = tid + pcr_stride;
                if(right >= PCR_SIZE)
                    right = PCR_SIZE - 1;

                rocsparse_int left = tid - pcr_stride;
                if(left < 0)
                    left = 0;

                T k1 = spa[tid] / spb[left];
                T k2 = spc[tid] / spb[right];

                tb   = spb[tid] - spc[left] * k1 - spa[right] * k2;
                trhs = sprhs[tid] - sprhs[left] * k1 - sprhs[right] * k2;
                ta   = -spa[left] * k1;
                tc   = -spc[right] * k2;
            }

            __syncthreads();

            if(tid < PCR_SIZE)
            {
                spb[tid]   = tb;
                sprhs[tid] = trhs;
                spa[tid]   = ta;
                spc[tid]   = tc;
            }

            pcr_stride <<= 1; //pcr_stride *= 2;

            __syncthreads();
        }

        if(tid < pcr_stride) // same as PCR_SIZE / 2
        {
            // Solve 2x2 systems
            rocsparse_int i   = tid;
            rocsparse_int j   = tid + pcr_stride;
            T             det = spb[j] * spb[i] - spc[i] * spa[j];
            det               = static_cast<T>(1) / det;

            spx[i] = (spb[j] * sprhs[i] - spc[i] * sprhs[j]) * det;
            spx[j] = (sprhs[j] * spb[i] - sprhs[i] * spa[j]) * det;
        }

        __syncthreads();

        if(tid < PCR_SIZE)
        {
            sx[tid * stride + stride - 1] = spx[tid];
        }

        // Backward substitution using cyclic reduction
        i = PCR_SIZE;
        for(rocsparse_int j = 0; j < cr_iter; j++)
        {
            __syncthreads();

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride / 2 - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(left < 0)
                {
                    sx[index] = (srhs[index] - sc[index] * sx[right]) / sb[index];
                }
                else
                {
                    sx[index]
                        = (srhs[index] - sa[index] * sx[left] - sc[index] * sx[right]) / sb[index];
                }
            }

            (stride >>= 1); //stride /= 2;
            i <<= 1; //i *= 2;
        }

        __syncthreads();

        B[tid + ldb * hipBlockIdx_x]             = sx[tid];
        B[tid + BLOCKSIZE + ldb * hipBlockIdx_x] = sx[tid + BLOCKSIZE];
    }

    // Parallel cyclic reduction algorithm
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_shared_kernel(rocsparse_int m,
                                        rocsparse_int n,
                                        rocsparse_int ldb,
                                        const T* __restrict__ dl,
                                        const T* __restrict__ d,
                                        const T* __restrict__ du,
                                        T* __restrict__ B)
    {
        rocsparse_int tid = hipThreadIdx_x;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE / 2));
        rocsparse_int stride = 1;

        // Parallel cyclic reduction shared memory
        __shared__ T sa[BLOCKSIZE];
        __shared__ T sb[BLOCKSIZE];
        __shared__ T sc[BLOCKSIZE];
        __shared__ T srhs[BLOCKSIZE];
        __shared__ T sx[BLOCKSIZE];

        // Fill parallel cyclic reduction shared memory
        sa[tid]   = (tid < m) ? dl[tid] : static_cast<T>(0);
        sb[tid]   = (tid < m) ? d[tid] : static_cast<T>(0);
        sc[tid]   = (tid < m) ? du[tid] : static_cast<T>(0);
        srhs[tid] = (tid < m) ? B[tid + ldb * hipBlockIdx_x] : static_cast<T>(0);

        __syncthreads();

        for(rocsparse_int j = 0; j < iter; j++)
        {
            rocsparse_int right = tid + stride;
            if(right >= m)
                right = m - 1;

            rocsparse_int left = tid - stride;
            if(left < 0)
                left = 0;

            T k1 = sa[tid] / sb[left];
            T k2 = sc[tid] / sb[right];

            T tb   = sb[tid] - sc[left] * k1 - sa[right] * k2;
            T trhs = srhs[tid] - srhs[left] * k1 - srhs[right] * k2;
            T ta   = -sa[left] * k1;
            T tc   = -sc[right] * k2;

            __syncthreads();

            sb[tid]   = tb;
            srhs[tid] = trhs;
            sa[tid]   = ta;
            sc[tid]   = tc;

            stride <<= 1; //stride *= 2;

            __syncthreads();
        }

        if(tid < BLOCKSIZE / 2)
        {
            rocsparse_int i = tid;
            rocsparse_int j = tid + stride;

            if(j < m)
            {
                // Solve 2x2 systems
                T det = sb[j] * sb[i] - sc[i] * sa[j];
                det   = static_cast<T>(1) / det;

                sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
                sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
            }
            else
            {
                // Solve 1x1 systems
                sx[i] = srhs[i] / sb[i];
            }
        }

        __syncthreads();

        if(tid < m)
        {
            B[tid + ldb * hipBlockIdx_x] = sx[tid];
        }
    }

    // Medium sized problems

    // Parallel cyclic reduction algorithm using global memory for partitioning large matrices into
    // multiple small ones that can be solved in parallel in stage 2
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_pow2_stage1_n_kernel(rocsparse_int stride,
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

        const T* rhs0_col = rhs0 + ldb * hipBlockIdx_y;
        T*       rhs1_col = rhs1 + m * hipBlockIdx_y;

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

        b1[gid] = b0[gid] - c0[left] * k1 - a0[right] * k2;
        a1[gid] = k3 * k1;
        c1[gid] = k4 * k2;

        k3            = rhs0_col[right];
        k4            = rhs0_col[left];
        rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_stage1_n_kernel(rocsparse_int stride,
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

        b1[gid] = b0[gid] - c0[left] * k1 - a0[right] * k2;
        a1[gid] = k3 * k1;
        c1[gid] = k4 * k2;

        const T* rhs0_col = rhs0 + ldb * hipBlockIdx_y;
        T*       rhs1_col = rhs1 + m * hipBlockIdx_y;

        k3            = rhs0_col[right];
        k4            = rhs0_col[left];
        rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
    }

    // Cyclic reduction algorithm using shared memory to solve multiple small matrices produced from
    // stage 1 above in parallel
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_cr_pow2_stage2_kernel(rocsparse_int m,
                                            rocsparse_int n,
                                            rocsparse_int ldb,
                                            const T* __restrict__ a,
                                            const T* __restrict__ b,
                                            const T* __restrict__ c,
                                            const T* __restrict__ rhs,
                                            T* __restrict__ B)
    {
        rocsparse_int tid = hipThreadIdx_x;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE));
        rocsparse_int stride = 1;
        rocsparse_int i      = BLOCKSIZE;

        // Cyclic reduction shared memory
        __shared__ T sa[2 * BLOCKSIZE];
        __shared__ T sb[2 * BLOCKSIZE];
        __shared__ T sc[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE];
        __shared__ T sx[2 * BLOCKSIZE];

        sa[tid]   = a[hipGridDim_x * tid + hipBlockIdx_x];
        sb[tid]   = b[hipGridDim_x * tid + hipBlockIdx_x];
        sc[tid]   = c[hipGridDim_x * tid + hipBlockIdx_x];
        srhs[tid] = rhs[hipGridDim_x * tid + hipBlockIdx_x + m * hipBlockIdx_y];
        sx[tid]   = static_cast<T>(0);

        sa[tid + BLOCKSIZE] = a[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x];
        sb[tid + BLOCKSIZE] = b[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x];
        sc[tid + BLOCKSIZE] = c[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x];
        srhs[tid + BLOCKSIZE]
            = rhs[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x + m * hipBlockIdx_y];
        sx[tid + BLOCKSIZE] = static_cast<T>(0);

        __syncthreads();

        // Forward reduction using cyclic reduction
        for(rocsparse_int j = 0; j < iter; j++)
        {
            stride <<= 1; //stride *= 2;

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(right >= 2 * BLOCKSIZE)
                {
                    right = 2 * BLOCKSIZE - 1;
                }

                T k1 = sa[index] / sb[left];
                T k2 = sc[index] / sb[right];

                sb[index]   = sb[index] - sc[left] * k1 - sa[right] * k2;
                srhs[index] = srhs[index] - srhs[left] * k1 - srhs[right] * k2;
                sa[index]   = -sa[left] * k1;
                sc[index]   = -sc[right] * k2;
            }

            i >>= 1; //i /= 2;

            __syncthreads();
        }

        if(tid == 0)
        {
            // Solve 2x2 system
            rocsparse_int i   = stride - 1;
            rocsparse_int j   = 2 * stride - 1;
            T             det = sb[j] * sb[i] - sc[i] * sa[j];
            det               = static_cast<T>(1) / det;

            sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
            sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
        }

        // Backward substitution using cyclic reduction
        i = 2;
        for(rocsparse_int j = 0; j < iter; j++)
        {
            __syncthreads();

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride / 2 - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(left < 0)
                {
                    sx[index] = (srhs[index] - sc[index] * sx[right]) / sb[index];
                }
                else
                {
                    sx[index]
                        = (srhs[index] - sa[index] * sx[left] - sc[index] * sx[right]) / sb[index];
                }
            }

            (stride >>= 1); //stride /= 2;
            i <<= 1; //i *= 2;
        }

        __syncthreads();

        B[hipGridDim_x * tid + hipBlockIdx_x + ldb * hipBlockIdx_y] = sx[tid];
        B[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x + ldb * hipBlockIdx_y]
            = sx[tid + BLOCKSIZE];
    }

    // Parallel cyclic reduction algorithm using shared memory to solve multiple small matrices produced from
    // stage 1 above in parallel
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_stage2_kernel(rocsparse_int m,
                                        rocsparse_int n,
                                        rocsparse_int ldb,
                                        const T* __restrict__ a,
                                        const T* __restrict__ b,
                                        const T* __restrict__ c,
                                        const T* __restrict__ rhs,
                                        T* __restrict__ B)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int gid = hipGridDim_x * tid + hipBlockIdx_x;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE / 2));
        rocsparse_int stride = 1;

        // Parallel cyclic reduction shared memory
        __shared__ T sa[BLOCKSIZE];
        __shared__ T sb[BLOCKSIZE];
        __shared__ T sc[BLOCKSIZE];
        __shared__ T srhs[BLOCKSIZE];
        __shared__ T sx[BLOCKSIZE];

        // Fill parallel cyclic reduction shared memory
        sa[tid]   = (gid < m) ? a[gid] : a[m - hipGridDim_x + hipBlockIdx_x];
        sb[tid]   = (gid < m) ? b[gid] : b[m - hipGridDim_x + hipBlockIdx_x];
        sc[tid]   = (gid < m) ? c[gid] : c[m - hipGridDim_x + hipBlockIdx_x];
        srhs[tid] = (gid < m) ? rhs[gid + m * hipBlockIdx_y]
                              : rhs[m - hipGridDim_x + hipBlockIdx_x + m * hipBlockIdx_y];

        __syncthreads();

        for(rocsparse_int j = 0; j < iter; j++)
        {
            rocsparse_int right = tid + stride;
            if(right >= BLOCKSIZE)
                right = BLOCKSIZE - 1;

            rocsparse_int left = tid - stride;
            if(left < 0)
                left = 0;

            T k1 = sa[tid] / sb[left];
            T k2 = sc[tid] / sb[right];

            T tb   = sb[tid] - sc[left] * k1 - sa[right] * k2;
            T trhs = srhs[tid] - srhs[left] * k1 - srhs[right] * k2;
            T ta   = -sa[left] * k1;
            T tc   = -sc[right] * k2;

            __syncthreads();

            sb[tid]   = tb;
            srhs[tid] = trhs;
            sa[tid]   = ta;
            sc[tid]   = tc;

            stride <<= 1; //stride *= 2;

            __syncthreads();
        }

        if(tid < BLOCKSIZE / 2)
        {
            rocsparse_int i = tid;
            rocsparse_int j = tid + stride;

            if(j < BLOCKSIZE)
            {
                // Solve 2x2 systems
                T det = sb[j] * sb[i] - sc[i] * sa[j];
                det   = static_cast<T>(1) / det;

                sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
                sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
            }
            else
            {
                // Solve 1x1 systems
                sx[i] = srhs[i] / sb[i];
            }
        }

        __syncthreads();

        if(gid < m)
        {
            B[gid + ldb * hipBlockIdx_y] = sx[tid];
        }
    }

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
