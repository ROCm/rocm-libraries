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
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_2x2_kernel(rocsparse_int batch_count,
                                 int64_t       batch_stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < batch_count)
        {
            // a0 | b0 c0 |
            //    | a1 b1 | c1
            const T a1 = dl[batch_stride * gid + 1];
            const T b0 = d[batch_stride * gid + 0];
            const T b1 = d[batch_stride * gid + 1];
            const T c0 = du[batch_stride * gid + 0];

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
    void gtsv_nopivot_3x3_kernel(rocsparse_int batch_count,
                                 int64_t       batch_stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < batch_count)
        {
            // Load matrix coefficients
            const T a1 = dl[batch_stride * gid + 1];
            const T a2 = dl[batch_stride * gid + 2];
            const T b0 = d[batch_stride * gid + 0];
            const T b1 = d[batch_stride * gid + 1];
            const T b2 = d[batch_stride * gid + 2];
            const T c0 = du[batch_stride * gid + 0];
            const T c1 = du[batch_stride * gid + 1];

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
    void gtsv_nopivot_4x4_kernel(rocsparse_int batch_count,
                                 int64_t       batch_stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < batch_count)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[batch_stride * gid + 1];
            const T a2 = dl[batch_stride * gid + 2];
            const T a3 = dl[batch_stride * gid + 3];
            const T b0 = d[batch_stride * gid + 0];
            const T b1 = d[batch_stride * gid + 1];
            const T b2 = d[batch_stride * gid + 2];
            const T b3 = d[batch_stride * gid + 3];
            const T c0 = du[batch_stride * gid + 0];
            const T c1 = du[batch_stride * gid + 1];
            const T c2 = du[batch_stride * gid + 2];

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
    void gtsv_nopivot_5x5_kernel(rocsparse_int batch_count,
                                 int64_t       batch_stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < batch_count)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[batch_stride * gid + 1];
            const T a2 = dl[batch_stride * gid + 2];
            const T a3 = dl[batch_stride * gid + 3];
            const T a4 = dl[batch_stride * gid + 4];
            const T b0 = d[batch_stride * gid + 0];
            const T b1 = d[batch_stride * gid + 1];
            const T b2 = d[batch_stride * gid + 2];
            const T b3 = d[batch_stride * gid + 3];
            const T b4 = d[batch_stride * gid + 4];
            const T c0 = du[batch_stride * gid + 0];
            const T c1 = du[batch_stride * gid + 1];
            const T c2 = du[batch_stride * gid + 2];
            const T c3 = du[batch_stride * gid + 3];

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
    void gtsv_nopivot_6x6_kernel(rocsparse_int batch_count,
                                 int64_t       batch_stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < batch_count)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[batch_stride * gid + 1];
            const T a2 = dl[batch_stride * gid + 2];
            const T a3 = dl[batch_stride * gid + 3];
            const T a4 = dl[batch_stride * gid + 4];
            const T a5 = dl[batch_stride * gid + 5];
            const T b0 = d[batch_stride * gid + 0];
            const T b1 = d[batch_stride * gid + 1];
            const T b2 = d[batch_stride * gid + 2];
            const T b3 = d[batch_stride * gid + 3];
            const T b4 = d[batch_stride * gid + 4];
            const T b5 = d[batch_stride * gid + 5];
            const T c0 = du[batch_stride * gid + 0];
            const T c1 = du[batch_stride * gid + 1];
            const T c2 = du[batch_stride * gid + 2];
            const T c3 = du[batch_stride * gid + 3];
            const T c4 = du[batch_stride * gid + 4];

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
    void gtsv_nopivot_7x7_kernel(rocsparse_int batch_count,
                                 int64_t       batch_stride,
                                 int64_t       ldb,
                                 const T* __restrict__ dl,
                                 const T* __restrict__ d,
                                 const T* __restrict__ du,
                                 T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < batch_count)
        {
            // Load matrix coefficients (same for all RHS)
            const T a1 = dl[batch_stride * gid + 1];
            const T a2 = dl[batch_stride * gid + 2];
            const T a3 = dl[batch_stride * gid + 3];
            const T a4 = dl[batch_stride * gid + 4];
            const T a5 = dl[batch_stride * gid + 5];
            const T a6 = dl[batch_stride * gid + 6];
            const T b0 = d[batch_stride * gid + 0];
            const T b1 = d[batch_stride * gid + 1];
            const T b2 = d[batch_stride * gid + 2];
            const T b3 = d[batch_stride * gid + 3];
            const T b4 = d[batch_stride * gid + 4];
            const T b5 = d[batch_stride * gid + 5];
            const T b6 = d[batch_stride * gid + 6];
            const T c0 = du[batch_stride * gid + 0];
            const T c1 = du[batch_stride * gid + 1];
            const T c2 = du[batch_stride * gid + 2];
            const T c3 = du[batch_stride * gid + 3];
            const T c4 = du[batch_stride * gid + 4];
            const T c5 = du[batch_stride * gid + 5];

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
    //   - B:  right-hand side vectors (batch_count systems of size M, stored column-major with stride ldb)
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
    // - Optimal for batch_count >> M where batch_count is the number of systems
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
    void gtsv_nopivot_thomas_kernel(rocsparse_int batch_count,
                                    int64_t       batch_stride,
                                    int64_t       ldb,
                                    const T* __restrict__ dl,
                                    const T* __restrict__ d,
                                    const T* __restrict__ du,
                                    T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = BLOCKSIZE * bid + tid;

        if(gid < batch_count)
        {
            const T* dlower = &dl[batch_stride * gid];
            const T* dmain  = &d[batch_stride * gid];
            const T* dupper = &du[batch_stride * gid];

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
}