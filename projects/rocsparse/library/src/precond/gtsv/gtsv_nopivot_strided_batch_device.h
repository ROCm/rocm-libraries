/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_2x2_kernel(rocsparse_int n,
                                 int64_t       stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // a0 | b0 c0 |
            //    | a1 b1 | c1
            const T a1 = dl[stride * gid + 1];
            const T b0 = d[stride * gid + 0];
            const T b1 = d[stride * gid + 1];
            const T c0 = du[stride * gid + 0];

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
                                 int64_t       stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // Load matrix coefficients
            const T a1 = dl[stride * gid + 1];
            const T a2 = dl[stride * gid + 2];
            const T b0 = d[stride * gid + 0];
            const T b1 = d[stride * gid + 1];
            const T b2 = d[stride * gid + 2];
            const T c0 = du[stride * gid + 0];
            const T c1 = du[stride * gid + 1];

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
                                 int64_t       stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[stride * gid + 1];
            const T a2 = dl[stride * gid + 2];
            const T a3 = dl[stride * gid + 3];
            const T b0 = d[stride * gid + 0];
            const T b1 = d[stride * gid + 1];
            const T b2 = d[stride * gid + 2];
            const T b3 = d[stride * gid + 3];
            const T c0 = du[stride * gid + 0];
            const T c1 = du[stride * gid + 1];
            const T c2 = du[stride * gid + 2];

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
                                 int64_t       stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[stride * gid + 1];
            const T a2 = dl[stride * gid + 2];
            const T a3 = dl[stride * gid + 3];
            const T a4 = dl[stride * gid + 4];
            const T b0 = d[stride * gid + 0];
            const T b1 = d[stride * gid + 1];
            const T b2 = d[stride * gid + 2];
            const T b3 = d[stride * gid + 3];
            const T b4 = d[stride * gid + 4];
            const T c0 = du[stride * gid + 0];
            const T c1 = du[stride * gid + 1];
            const T c2 = du[stride * gid + 2];
            const T c3 = du[stride * gid + 3];

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
                                 int64_t       stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[stride * gid + 1];
            const T a2 = dl[stride * gid + 2];
            const T a3 = dl[stride * gid + 3];
            const T a4 = dl[stride * gid + 4];
            const T a5 = dl[stride * gid + 5];
            const T b0 = d[stride * gid + 0];
            const T b1 = d[stride * gid + 1];
            const T b2 = d[stride * gid + 2];
            const T b3 = d[stride * gid + 3];
            const T b4 = d[stride * gid + 4];
            const T b5 = d[stride * gid + 5];
            const T c0 = du[stride * gid + 0];
            const T c1 = du[stride * gid + 1];
            const T c2 = du[stride * gid + 2];
            const T c3 = du[stride * gid + 3];
            const T c4 = du[stride * gid + 4];

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
                                 int64_t       stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[stride * gid + 1];
            const T a2 = dl[stride * gid + 2];
            const T a3 = dl[stride * gid + 3];
            const T a4 = dl[stride * gid + 4];
            const T a5 = dl[stride * gid + 5];
            const T a6 = dl[stride * gid + 6];
            const T b0 = d[stride * gid + 0];
            const T b1 = d[stride * gid + 1];
            const T b2 = d[stride * gid + 2];
            const T b3 = d[stride * gid + 3];
            const T b4 = d[stride * gid + 4];
            const T b5 = d[stride * gid + 5];
            const T b6 = d[stride * gid + 6];
            const T c0 = du[stride * gid + 0];
            const T c1 = du[stride * gid + 1];
            const T c2 = du[stride * gid + 2];
            const T c3 = du[stride * gid + 3];
            const T c4 = du[stride * gid + 4];
            const T c5 = du[stride * gid + 5];

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
                                    int64_t       stride,
                                    int64_t       ldb,
                                    const T* __restrict__ dl,
                                    const T* __restrict__ d,
                                    const T* __restrict__ du,
                                    T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < n)
        {
            const T* dlower = &dl[stride * gid];
            const T* dmain  = &d[stride * gid];
            const T* dupper = &du[stride * gid];

            T du_prime[M];
            T B_prime[M];

            // Forward sweep
            const T inv_d0 = static_cast<T>(1) / dmain[0];
            du_prime[0]    = dupper[0] * inv_d0;
            for(int i = 1; i < M - 1; i++)
            {
                // denom = dmain[i] - dlower[i] * du_prime[i-1]
                const T inv_denom
                    = static_cast<T>(1) / rocsparse::fma(-dlower[i], du_prime[i - 1], dmain[i]);
                du_prime[i] = dupper[i] * inv_denom;
            }

            B_prime[0] = B[ldb * gid + 0] * inv_d0;
            for(int i = 1; i < M; i++)
            {
                // denom = dmain[i] - dlower[i] * du_prime[i-1]
                const T inv_denom
                    = static_cast<T>(1) / rocsparse::fma(-dlower[i], du_prime[i - 1], dmain[i]);
                // num = B[i] - dlower[i] * B_prime[i-1]
                B_prime[i]
                    = rocsparse::fma(-dlower[i], B_prime[i - 1], B[ldb * gid + i]) * inv_denom;
            }

            // Backward sweep
            B[ldb * gid + M - 1] = B_prime[M - 1];
            for(int i = M - 2; i >= 0; i--)
            {
                B[ldb * gid + i] = rocsparse::fma(-du_prime[i], B[ldb * gid + i + 1], B_prime[i]);
            }
        }
    }

    // Parallel cyclic reduction algorithm
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_batch_pcr_wavefront_kernel(rocsparse_int m,
                                                 rocsparse_int n,
                                                 int64_t       batch_stride,
                                                 int64_t       ldb,
                                                 const T* __restrict__ dl,
                                                 const T* __restrict__ d,
                                                 const T* __restrict__ du,
                                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;

        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        const int iter   = rocsparse::log2_pow2<WF_SIZE / 2>::value;
        int       stride = 1;

        const int batch = (BLOCKSIZE / WF_SIZE) * bid + wid;

        T a = ((lid < m && batch < n) && lid != 0) ? dl[batch_stride * batch + lid]
                                                   : static_cast<T>(0);
        T b = (lid < m && batch < n) ? d[batch_stride * batch + lid] : static_cast<T>(1);
        T c = ((lid < m && batch < n) && lid != (m - 1)) ? du[batch_stride * batch + lid]
                                                         : static_cast<T>(0);
        T x = (lid < m && batch < n) ? B[ldb * batch + lid] : static_cast<T>(0);

        for(int it = 0; it < iter; it++)
        {
            const int right = lid + stride;
            const int left  = lid - stride;

            T a_left = shfl_up(a, stride, WF_SIZE);
            T b_left = shfl_up(b, stride, WF_SIZE);
            T c_left = shfl_up(c, stride, WF_SIZE);

            if(left < 0)
            {
                a_left = static_cast<T>(0);
                b_left = static_cast<T>(0);
                c_left = static_cast<T>(0);
            }

            T a_right = shfl_down(a, stride, WF_SIZE);
            T b_right = shfl_down(b, stride, WF_SIZE);
            T c_right = shfl_down(c, stride, WF_SIZE);

            if(right > (WF_SIZE - 1))
            {
                a_right = static_cast<T>(0);
                b_right = static_cast<T>(0);
                c_right = static_cast<T>(0);
            }

            const T k1 = (left >= 0) ? a / b_left : static_cast<T>(0);
            const T k2 = (right <= WF_SIZE - 1) ? c / b_right : static_cast<T>(0);

            const T a_new = -a_left * k1;
            const T b_new = b - c_left * k1 - a_right * k2;
            const T c_new = -c_right * k2;

            a = a_new;
            b = b_new;
            c = c_new;

            T x_left = shfl_up(x, stride, WF_SIZE);
            if(left < 0)
            {
                x_left = static_cast<T>(0);
            }
            T x_right = shfl_down(x, stride, WF_SIZE);
            if(right > (WF_SIZE - 1))
            {
                x_right = static_cast<T>(0);
            }

            const T x_new = x - x_left * k1 - x_right * k2;

            x = x_new;

            stride <<= 1; //stride *= 2;
        }

        // Solve 2x2 systems (j = lid + stride)
        // bi ci
        // aj bj
        //
        // det = bi * bj - aj * ci
        const T aj = shfl_down(a, stride, WF_SIZE);
        const T bj = shfl_down(b, stride, WF_SIZE);

        const T det = static_cast<T>(1) / (b * bj - aj * c);

        const T xj = shfl_down(x, stride, WF_SIZE);

        if(lid < WF_SIZE / 2) // same as lid < stride
        {
            if(lid < m && batch < n)
            {
                B[ldb * batch + lid] = (bj * x - c * xj) * det;
            }
            if((lid + stride) < m && batch < n)
            {
                B[ldb * batch + lid + stride] = (xj * b - x * aj) * det;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_batch_pcr_shared_kernel(
        rocsparse_int m,
        rocsparse_int n,
        int64_t batch_stride, // rename to stride and then rename other stride to pcr/cr_stride?
        int64_t ldb,
        const T* __restrict__ dl,
        const T* __restrict__ d,
        const T* __restrict__ du,
        T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;

        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        const int iter_BLOCKSIZE = rocsparse::log2_pow2<BLOCKSIZE / 2>::value;
        const int iter_WF_SIZE   = rocsparse::log2_pow2<WF_SIZE / 2>::value;
        const int iter           = iter_BLOCKSIZE - iter_WF_SIZE;

        int stride = 1; // rename to cr_stride

        T a = ((tid < m) && tid != 0) ? dl[batch_stride * bid + tid] : static_cast<T>(0);
        T b = (tid < m) ? d[batch_stride * bid + tid] : static_cast<T>(1);
        T c = ((tid < m) && tid != (m - 1)) ? du[batch_stride * bid + tid] : static_cast<T>(0);
        T x = (tid < m) ? B[ldb * bid + tid] : static_cast<T>(0);

        // Parallel cyclic reduction shared memory
        __shared__ T a_shared[BLOCKSIZE];
        __shared__ T b_shared[BLOCKSIZE];
        __shared__ T c_shared[BLOCKSIZE];
        __shared__ T x_shared[BLOCKSIZE];

        // Fill parallel cyclic reduction shared memory
        a_shared[tid] = a;
        b_shared[tid] = b;
        c_shared[tid] = c;
        x_shared[tid] = x;
        __syncthreads();

        for(int j = 0; j < iter; j++)
        {
            const int right = tid + stride;
            const int left  = tid - stride;

            const T a_left = (left >= 0) ? a_shared[left] : static_cast<T>(0);
            const T b_left = (left >= 0) ? b_shared[left] : static_cast<T>(0);
            const T c_left = (left >= 0) ? c_shared[left] : static_cast<T>(0);

            const T a_right = (right < BLOCKSIZE) ? a_shared[right] : static_cast<T>(0);
            const T b_right = (right < BLOCKSIZE) ? b_shared[right] : static_cast<T>(0);
            const T c_right = (right < BLOCKSIZE) ? c_shared[right] : static_cast<T>(0);

            const T k1 = (b_left != static_cast<T>(0)) ? a / b_left : static_cast<T>(0);
            const T k2 = (b_right != static_cast<T>(0)) ? c / b_right : static_cast<T>(0);

            const T a_new = -a_left * k1;
            const T b_new = b - c_left * k1 - a_right * k2;
            const T c_new = -c_right * k2;

            __syncthreads();
            a_shared[tid] = a_new;
            b_shared[tid] = b_new;
            c_shared[tid] = c_new;

            a = a_new;
            b = b_new;
            c = c_new;

            const T x_left  = (left >= 0) ? x_shared[left] : static_cast<T>(0);
            const T x_right = (right < BLOCKSIZE) ? x_shared[right] : static_cast<T>(0);

            const T x_new = x - x_left * k1 - x_right * k2;

            __syncthreads();
            x_shared[tid] = x_new;

            x = x_new;
            __syncthreads();

            stride *= 2;
        }

        a = a_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
        b = b_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
        c = c_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
        x = x_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
        __syncthreads();

        int stride2 = 1;
        for(int it = 0; it < iter_WF_SIZE; it++)
        {
            const int right = lid + stride2;
            const int left  = lid - stride2;

            T a_left = shfl_up(a, stride2, WF_SIZE);
            T b_left = shfl_up(b, stride2, WF_SIZE);
            T c_left = shfl_up(c, stride2, WF_SIZE);

            if(left < 0)
            {
                a_left = static_cast<T>(0);
                b_left = static_cast<T>(0);
                c_left = static_cast<T>(0);
            }

            T a_right = shfl_down(a, stride2, WF_SIZE);
            T b_right = shfl_down(b, stride2, WF_SIZE);
            T c_right = shfl_down(c, stride2, WF_SIZE);

            if(right > (WF_SIZE - 1))
            {
                a_right = static_cast<T>(0);
                b_right = static_cast<T>(0);
                c_right = static_cast<T>(0);
            }

            const T k1 = (b_left != static_cast<T>(0)) ? a / b_left : static_cast<T>(0);
            const T k2 = (b_right != static_cast<T>(0)) ? c / b_right : static_cast<T>(0);

            const T a_new = -a_left * k1;
            const T b_new = b - c_left * k1 - a_right * k2;
            const T c_new = -c_right * k2;

            a = a_new;
            b = b_new;
            c = c_new;

            T x_left  = shfl_up(x, stride2, WF_SIZE);
            T x_right = shfl_down(x, stride2, WF_SIZE);

            if(left < 0)
            {
                x_left = static_cast<T>(0);
            }

            if(right > (WF_SIZE - 1))
            {
                x_right = static_cast<T>(0);
            }

            const T x_new = x - x_left * k1 - x_right * k2;
            x             = x_new;

            stride2 <<= 1; //stride2 *= 2;
        }

        // Solve 2x2 systems (j = lid + stride2)
        // bi ci
        // aj bj
        //
        // det = bi * bj - aj * ci
        const T aj = shfl_down(a, stride2, WF_SIZE);
        const T bj = shfl_down(b, stride2, WF_SIZE);

        const T det = static_cast<T>(1) / (b * bj - aj * c);

        const T xj = shfl_down(x, stride2, WF_SIZE);

        if(lid < WF_SIZE / 2) // same as lid < stride2
        {
            if(((BLOCKSIZE / WF_SIZE) * lid + wid) < m)
            {
                B[ldb * bid + (BLOCKSIZE / WF_SIZE) * lid + wid] = (bj * x - c * xj) * det;
            }
            if(((BLOCKSIZE / WF_SIZE) * (lid + stride2) + wid) < m)
            {
                B[ldb * bid + (BLOCKSIZE / WF_SIZE) * (lid + stride2) + wid]
                    = (xj * b - x * aj) * det;
            }
        }
    }

    // Combined Parallel cyclic reduction and cyclic reduction algorithm using shared memory
    template <uint32_t BLOCKSIZE, uint32_t PCR_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_batch_crpcr_shared_kernel(rocsparse_int m,
                                                rocsparse_int n,
                                                int64_t       batch_stride,
                                                int64_t       ldb,
                                                const T* __restrict__ dl,
                                                const T* __restrict__ d,
                                                const T* __restrict__ du,
                                                T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;

        const int tot_iter = rocsparse::log2_pow2<(2 * BLOCKSIZE) / 2>::value;
        const int pcr_iter = rocsparse::log2_pow2<PCR_SIZE / 2>::value;
        const int cr_iter  = tot_iter - pcr_iter;

        int stride         = 1;
        int active_threads = BLOCKSIZE;

        // Cyclic reduction shared memory
        __shared__ T sa[2 * BLOCKSIZE];
        __shared__ T sb[2 * BLOCKSIZE];
        __shared__ T sc[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE];

        // Fill cyclic reduction shared memory
        sa[tid] = (tid < m && tid != 0) ? dl[batch_stride * bid + tid] : static_cast<T>(0);
        sa[tid + BLOCKSIZE]
            = (tid + BLOCKSIZE < m) ? dl[batch_stride * bid + tid + BLOCKSIZE] : static_cast<T>(0);
        sb[tid] = (tid < m) ? d[batch_stride * bid + tid] : static_cast<T>(1);
        sb[tid + BLOCKSIZE]
            = (tid + BLOCKSIZE < m) ? d[batch_stride * bid + tid + BLOCKSIZE] : static_cast<T>(1);
        sc[tid]             = (tid < m) ? du[batch_stride * bid + tid] : static_cast<T>(0);
        sc[tid + BLOCKSIZE] = (tid + BLOCKSIZE < m && (tid + BLOCKSIZE) != (m - 1))
                                  ? du[batch_stride * bid + tid + BLOCKSIZE]
                                  : static_cast<T>(0);
        srhs[tid]           = (tid < m) ? B[ldb * bid + tid] : static_cast<T>(0);
        srhs[tid + BLOCKSIZE]
            = (tid + BLOCKSIZE < m) ? B[ldb * bid + tid + BLOCKSIZE] : static_cast<T>(0);

        __syncthreads();

        // Forward reduction using cyclic reduction
        for(int j = 0; j < cr_iter; j++)
        {
            stride *= 2;

            if(tid < active_threads)
            {
                const int index = stride * tid + stride - 1;
                int       left  = index - stride / 2;
                int       right = index + stride / 2;

                if(right >= 2 * BLOCKSIZE)
                {
                    right = 2 * BLOCKSIZE - 1;
                }

                T k1 = sa[index] / sb[left];
                T k2 = sc[index] / sb[right];

                sb[index] = sb[index] - sc[left] * k1 - sa[right] * k2;
                sa[index] = -sa[left] * k1;
                sc[index] = -sc[right] * k2;

                srhs[index] = srhs[index] - srhs[left] * k1 - srhs[right] * k2;
            }

            active_threads /= 2;

            __syncthreads();
        }

        // Parallel cyclic reduction
        const int index      = stride * tid + stride - 1;
        int       pcr_stride = stride;

        for(int j = 0; j < pcr_iter; j++)
        {
            T ta;
            T tb;
            T tc;
            T trhs;

            if(tid < PCR_SIZE)
            {
                rocsparse_int right = index + pcr_stride;
                if(right >= 2 * BLOCKSIZE)
                    right = 2 * BLOCKSIZE - 1;

                rocsparse_int left = index - pcr_stride;
                if(left < 0)
                    left = 0;

                T k1 = sa[index] / sb[left];
                T k2 = sc[index] / sb[right];

                tb = sb[index] - sc[left] * k1 - sa[right] * k2;
                ta = -sa[left] * k1;
                tc = -sc[right] * k2;

                trhs = srhs[index] - srhs[left] * k1 - srhs[right] * k2;
            }

            __syncthreads();
            if(tid < PCR_SIZE)
            {
                sb[index]   = tb;
                sa[index]   = ta;
                sc[index]   = tc;
                srhs[index] = trhs;
            }
            pcr_stride *= 2;
            __syncthreads();
        }

        if(tid < PCR_SIZE / 2)
        {
            const int index = stride * tid + stride - 1;

            // Solve 2x2 systems
            const int i   = index;
            const int j   = index + pcr_stride;
            const T   det = static_cast<T>(1) / (sb[j] * sb[i] - sc[i] * sa[j]);

            const T rhs_i = srhs[i];
            const T rhs_j = srhs[j];

            srhs[i] = (sb[j] * rhs_i - sc[i] * rhs_j) * det;
            srhs[j] = (rhs_j * sb[i] - rhs_i * sa[j]) * det;
        }

        // Backward substitution using cyclic reduction
        active_threads = PCR_SIZE;
        for(int j = 0; j < cr_iter; j++)
        {
            __syncthreads();

            if(tid < active_threads)
            {
                const int index = stride * tid + stride / 2 - 1;
                const int left  = index - stride / 2;
                const int right = index + stride / 2;

                const T rhs_left  = (left >= 0) ? srhs[left] : static_cast<T>(0);
                const T rhs_right = (right < m) ? srhs[right] : static_cast<T>(0);

                srhs[index]
                    = (srhs[index] - sa[index] * rhs_left - sc[index] * rhs_right) / sb[index];
            }

            stride /= 2;
            active_threads *= 2;
        }

        __syncthreads();

        if(tid < m)
        {
            B[ldb * bid + tid] = srhs[tid];
        }
        if(tid + BLOCKSIZE < m)
        {
            B[ldb * bid + tid + BLOCKSIZE] = srhs[tid + BLOCKSIZE];
        }
    }
}
