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
    // clang-format off
// Consider tridiagonal linear system A * x = rhs where A is m x m. Matrix A is represented by the three
// arrays (the diagonal, upper diagonal, and lower diagonal) each of length m. The first entry in the
// lower diagonal must be zero and the last entry in the upper diagonal must be zero. We solve this linear
// system using the "Spike-Diagonal Pivoting" algorithm as detailed in the thesis:
//
// "Scalable Parallel Tridiagonal Algorithms with diagonal pivoting and their optimizations for many-core
// architectures by L. Chang"
//
// See also:
//
// "L. Chang, J. A. Stratton, H. Kim and W. W. Hwu, "A scalable, numerically stable, high-performance tridiagonal
// solver using GPUs," SC '12: Proceedings of the International Conference on High Performance Computing, Networking,
// Storage and Analysis, Salt Lake City, UT, USA, 2012, pp. 1-11, doi: 10.1109/SC.2012.12."
//
// Here we give a rough outline:
//
// Given the tridiagonal linear system A * x = rhs. We first decompose this into A * x = D * S * x = rhs where
// D is a block diagonal matrix and S is the "spike" matrix. We then define y = S * x which then allows us to
// first solve D * y = rhs and then solve S * x = y. Because each block in the block diagonal matrix D is indendent,
// we can solve each block in parallel. Specifically we use one thread for each diagonal block in D. We could use
// the Thomas algorithm here, however because we want to incorporate some pivoting mechanism, we instead have each thread
// decompose its diagonal block in to L * B * M^T where both L and M are lower triangular matrices and B is a diagonal
// matrix. Specifically if the matrix D has the form:
//
// D = |D1  0  0  0  0  .  0 |
//     |0   D2 0  0  0  .  0 |
//     |0   0  D3 0  0  .  0 |
//     |0   0  0  D4 0  .  0 |
//     |.   .  .  .  .  .  . |
//     |.               .  . |
//     |0   .  .  .  .  .  Dm|
//
// Then D_i = L_i * B_i * M_i^T for i = 1..m. Note that each Di is itself tridiagonal. The matrices L, B, and M can
// be computed by noting that:
//
// D_i = |Pd C | = |Id      0   | |Pd  0 | |Id Pd^-1*C|
//       |A  Tr|   |A*Pd^-1 In-d| |0   Ts| |0  In-d   |
//
// where Pd is either 1x1 or 2x2 and Id is either 1x1 or 2x2 identity matrix. Ts is computed as Ts = Tr - A*Pd^-1*C.
// For example consider one of the block diagonal matrices:
//
// Di = |2 1 0 0 0 0|
//      |1 2 1 0 0 0|
//      |0 1 2 1 0 0|
//      |0 0 1 2 1 0|
//      |0 0 0 1 2 1|
//      |0 0 0 0 1 2|
//
// Then if using no pivoting (i.e. Pd is 1x1) we get:
//
// Pd = 2, Pd^-1 = 1/2, A = |1|, C = |1 0 0 0 0|, and Ts = |3/2 1  0  0  0|
//                          |0|                            |1   2  1  0  0|
//                          |0|                            |0   1  2  1  0|
//                          |0|                            |0   0  1  2  1|
//                          |0|                            |0   0  0  1  2|
//
// We can then recursively perform this on each subsequent Ts until we get:
//
// Di = |1   0   0   0   0   0| |2  0   0   0   0   0  | |1   1/2 0   0   0   0  |
//      |1/2 1   0   0   0   0| |0  3/2 0   0   0   0  | |0   1   2/3 0   0   0  |
//      |0   2/3 1   0   0   0| |0  0   4/3 0   0   0  | |0   0   1   3/4 0   0  |
//      |0   0   3/4 1   0   0| |0  0   0   5/4 0   0  | |0   0   0   1   4/5 0  |
//      |0   0   0   4/5 1   0| |0  0   0   0   6/5 0  | |0   0   0   0   1   5/6|
//      |0   0   0   0   5/6 1| |0  0   0   0   0   7/6| |0   0   0   0   0   1  |
//
// Solving each of these systems then is just a matter of solving L * B * yi = rhsi
// followed by M^T * xi = yi. The determination of whether we should use Pd as 1x1 or 2x2
// is based off the Bunch-Kaufmann pivoting criteria. See cited sources above.
//
// Let us now return to our factoization of the original tridiagonal linear system,
// A * x = D * S * x = rhs. We broke up finding the solution into the two phases. First
// solving D * y = rhs and then secondly solving S * x = y. We now know how to solve
// the first phase which is also the phase that performs the pivoting. We therefore focus on
// solving the "spike" linear system. If the original matrix A is:
//
// A = |2 1 0 0 0 0 0 0| = |2 1 0 0 0 0 0 0| |1   0   v11   0   0   0   0   0|
//     |1 2 1 0 0 0 0 0|   |1 2 0 0 0 0 0 0| |0   1   v12   0   0   0   0   0|
//     |0 1 2 1 0 0 0 0|   |0 0 2 1 0 0 0 0| |0   w21 1     0   v21 0   0   0|
//     |0 0 1 2 1 0 0 0|   |0 0 1 2 0 0 0 0| |0   w22 0     1   v22 0   0   0|
//     |0 0 0 1 2 1 0 0|   |0 0 0 0 2 1 0 0| |0   0   0     w31 1   0   v31 0|
//     |0 0 0 0 1 2 0 0|   |0 0 0 0 1 2 0 0| |0   0   0     w32 0   1   v32 0|
//     |0 0 0 0 0 1 2 1|   |0 0 0 0 0 0 2 1| |0   0   0     0   0   w41 1   0|
//     |0 0 0 0 0 0 1 2|   |0 0 0 0 0 0 1 2| |0   0   0     0   0   w42 0   1|
//                         --------D-------- ----------------S----------------
//
// Here we use 2x2 blocks in D but in practice we use much larger blocks, for example 128x128.
// We can solve for all the v and w unknowns by solving the following:
//
// |2 1||v11| = |0| and |2 1||w21| = |1| etc.
// |1 2||v12|   |1|     |1 2||w22|   |0|
//
// Solving S * x = y then involves recursively decomposing the "spike" matrix like so:
//
// S = |1   0   v11   0   0   0   0   0| = |1  0   v11 0  0  0  0   0| |1 0 0 0 v11' 0 0 0|
//     |0   1   v12   0   0   0   0   0|   |0  1   v12 0  0  0  0   0| |0 1 0 0 v12' 0 0 0|
//     |0   w21 1     0   v21 0   0   0|   |0  w21 1   0  0  0  0   0| |0 0 1 0 v13' 0 0 0|
//     |0   w22 0     1   v22 0   0   0|   |0  w22 0   1  0  0  0   0| |0 0 0 1 v14' 0 0 0|
//     |0   0   0     w31 1   0   v31 0|   |0  0   0   0  1  0  v31 0| |0 0 0 w21' 1 0 0 0|
//     |0   0   0     w32 0   1   v32 0|   |0  0   0   0  0  1  v32 0| |0 0 0 w22' 0 1 0 0|
//     |0   0   0     0   0   w41 1   0|   |0  0   0   0  0  w41 1  0| |0 0 0 w23' 0 0 1 0|
//     |0   0   0     0   0   w42 0   1|   |0  0   0   0  0  w42 0  0| |0 0 0 w24' 0 0 0 1|
//
// In the above the non-prime w and v values (i.e. v11, v12, w21, w22 etc) have been previously
// computed. The primed w and v values (i.e. v11', v12', w21', w22' etc) can be found by solving:
//
// |1   0   v11 0||v11'| = |0| etc...
// |0   1   v12 0||v12'|   |0|
// |0   w21 1   0||v13'|   |0|
// |0   w22 0   1||v14'|   |1|
    // clang-format on

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_transpose_and_pad_array_shared_kernel(rocsparse_int m,
                                                    rocsparse_int m_pad,
                                                    rocsparse_int stride,
                                                    const T* __restrict__ input,
                                                    T* __restrict__ output,
                                                    T pad_value)
    {
        __shared__ T stile[BLOCKSIZE];

        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidx = hipBlockIdx_x;
        rocsparse_int bidy = hipBlockIdx_y;
        rocsparse_int gidx = tidx + BLOCKSIZE * bidx;

        rocsparse_int wid = tidx / (BLOCKSIZE / BLOCKDIM);
        rocsparse_int lid = tidx % (BLOCKSIZE / BLOCKDIM);

        if(gidx < m)
        {
            stile[tidx] = input[gidx + bidy * stride];
        }
        else
        {
            stile[tidx] = pad_value;
        }

        __syncthreads();

        rocsparse_int nblocks = m_pad / BLOCKDIM;
        rocsparse_int k       = (BLOCKSIZE / BLOCKDIM) * bidx + nblocks * wid + lid;

        if(k < m_pad)
        {
            output[k + bidy * m_pad] = stile[BLOCKDIM * lid + wid];
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_transpose_and_pad_array_kernel(rocsparse_int m,
                                             rocsparse_int m_pad,
                                             rocsparse_int stride,
                                             const T* __restrict__ input,
                                             T* __restrict__ output,
                                             T pad_value)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidx = hipBlockIdx_x;
        rocsparse_int bidy = hipBlockIdx_y;

        rocsparse_int gidx = tidx + BLOCKSIZE * bidx;

        rocsparse_int i = (gidx * BLOCKDIM) % m_pad;
        rocsparse_int j = (gidx * BLOCKDIM) / m_pad;
        rocsparse_int k = i + j;

        if(k < m)
        {
            output[gidx + bidy * m_pad] = input[k + bidy * stride];
        }
        else if(k < m_pad)
        {
            output[gidx + bidy * m_pad] = pad_value;
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_transpose_back_array_kernel(rocsparse_int m,
                                          rocsparse_int m_pad,
                                          rocsparse_int stride,
                                          const T* __restrict__ input,
                                          T* __restrict__ output)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidx = hipBlockIdx_x;
        rocsparse_int bidy = hipBlockIdx_y;

        rocsparse_int gidx = tidx + BLOCKSIZE * bidx;

        rocsparse_int i = (gidx * BLOCKDIM) % m_pad;
        rocsparse_int j = (gidx * BLOCKDIM) / m_pad;
        rocsparse_int k = i + j;

        if(k < m)
        {
            output[k + bidy * stride] = input[gidx + bidy * m_pad];
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_LBM_wv_kernel(rocsparse_int m_pad,
                            rocsparse_int n,
                            rocsparse_int ldb,
                            const T* __restrict__ a,
                            const T* __restrict__ b,
                            const T* __restrict__ c,
                            T* __restrict__ w,
                            T* __restrict__ v,
                            T* __restrict__ mt,
                            rocsparse_int* __restrict__ pivot)
    {
        // From Bunch-Kaufman pivoting criteria
        const double kappa = double(0.5) * (rocsparse::sqrt(double(5.0)) - double(1.0));

        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidx = hipBlockIdx_x;
        rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

        rocsparse_int nblocks = m_pad / BLOCKDIM;

        if(gid >= nblocks)
        {
            return;
        }

        T bk                              = b[gid];
        w[gid]                            = a[gid];
        v[gid + (BLOCKDIM - 1) * nblocks] = c[gid + (BLOCKDIM - 1) * nblocks];

        // forward solve (L* B * w = w and L* B * v = v)
        rocsparse_int k = 0;
        while(k < m_pad)
        {
            T ck   = c[k + gid];
            T ak_1 = (k < (BLOCKDIM - 1) * nblocks) ? a[k + nblocks + gid] : static_cast<T>(0);
            T bk_1 = (k < (BLOCKDIM - 1) * nblocks) ? b[k + nblocks + gid] : static_cast<T>(0);
            T ck_1 = (k < (BLOCKDIM - 1) * nblocks) ? c[k + nblocks + gid] : static_cast<T>(0);
            T ak_2 = (k < (BLOCKDIM - 2) * nblocks) ? a[k + 2 * nblocks + gid] : static_cast<T>(0);

            // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
            // pivoting criteria
            double sigma = double(0);
            sigma = rocsparse::max(double(rocsparse::abs(ak_1)), double(rocsparse::abs(ak_2)));
            sigma = rocsparse::max(double(rocsparse::abs(bk_1)), sigma);
            sigma = rocsparse::max(double(rocsparse::abs(ck)), sigma);
            sigma = rocsparse::max(double(rocsparse::abs(ck_1)), sigma);

            // 1x1 pivoting
            if(rocsparse::abs(bk) * sigma >= kappa * rocsparse::abs(ak_1 * ck)
               || k == (BLOCKDIM - 1) * nblocks)
            {
                T iBk = static_cast<T>(1) / bk;

                bk_1 = bk_1 - ak_1 * ck * iBk;
                ak_1 = ak_1 * iBk;
                ck   = ck * iBk;

                T wk = w[k + gid];
                T vk = v[k + gid];

                w[k + gid]     = wk * iBk;
                v[k + gid]     = vk * iBk;
                mt[k + gid]    = ck;
                pivot[k + gid] = 1;

                if(k < (BLOCKDIM - 1) * nblocks)
                {
                    w[k + nblocks + gid] += -ak_1 * wk;
                }

                bk = bk_1;
                k += nblocks;
            }
            // 2x2 pivoting
            else
            {
                T det = bk * bk_1 - ak_1 * ck;
                det   = static_cast<T>(1) / det;

                T wk   = w[k + gid];
                T wk_1 = w[k + nblocks + gid];
                T vk   = v[k + gid];
                T vk_1 = v[k + nblocks + gid];

                w[k + gid]     = (bk_1 * wk - ck * wk_1) * det;
                v[k + gid]     = (bk_1 * vk - ck * vk_1) * det;
                mt[k + gid]    = -ck * ck_1 * det;
                pivot[k + gid] = 2;

                if(k < (BLOCKDIM - 1) * nblocks)
                {
                    w[k + nblocks + gid]     = (-ak_1 * wk + bk * wk_1) * det;
                    v[k + nblocks + gid]     = (-ak_1 * vk + bk * vk_1) * det;
                    mt[k + nblocks + gid]    = bk * ck_1 * det;
                    pivot[k + nblocks + gid] = 2;
                }

                T bk_2 = static_cast<T>(0);

                if(k < (BLOCKDIM - 2) * nblocks)
                {
                    w[k + 2 * nblocks + gid]
                        += -(-ak_1 * ak_2 * det) * wk - (bk * ak_2 * det) * wk_1;

                    bk_2 = b[k + 2 * nblocks + gid];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2 * nblocks;
            }
        }

        __threadfence();

        // at this point k = BLOCKDIM * nblocks
        k -= nblocks;

        k -= nblocks * pivot[k + gid];

        // backward solve (M^T * w = w and M^T * v = v)
        while(k >= 0)
        {
            if(pivot[k + gid] == 1)
            {
                T tmp = mt[k + gid];
                w[k + gid] += -tmp * w[k + nblocks + gid];
                v[k + gid] += -tmp * v[k + nblocks + gid];

                k -= nblocks;
            }
            else
            {
                T tmp1 = mt[k + gid];
                T tmp2 = mt[k - nblocks + gid];

                w[k + gid] += -tmp1 * w[k + nblocks + gid];
                w[k - nblocks + gid] += -tmp2 * w[k + nblocks + gid];
                v[k + gid] += -tmp1 * v[k + nblocks + gid];
                v[k - nblocks + gid] += -tmp2 * v[k + nblocks + gid];

                k -= 2 * nblocks;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t COLS, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_LBM_rhs_kernel(rocsparse_int m_pad,
                             rocsparse_int n,
                             rocsparse_int ldb,
                             const T* __restrict__ a,
                             const T* __restrict__ b,
                             const T* __restrict__ c,
                             T* __restrict__ rhs,
                             const T* __restrict__ mt,
                             const rocsparse_int* __restrict__ pivot)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidx = hipBlockIdx_x;
        rocsparse_int bidy = hipBlockIdx_y;
        rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

        rocsparse_int nblocks = m_pad / BLOCKDIM;

        if(gid >= nblocks)
        {
            return;
        }

        T bk = b[gid];

        // forward solve (L* B * rhs = rhs)
        rocsparse_int k = 0;
        while(k < m_pad)
        {
            T ck   = c[k + gid];
            T ak_1 = (k < (BLOCKDIM - 1) * nblocks) ? a[k + nblocks + gid] : static_cast<T>(0);
            T bk_1 = (k < (BLOCKDIM - 1) * nblocks) ? b[k + nblocks + gid] : static_cast<T>(0);
            T ck_1 = (k < (BLOCKDIM - 1) * nblocks) ? c[k + nblocks + gid] : static_cast<T>(0);
            T ak_2 = (k < (BLOCKDIM - 2) * nblocks) ? a[k + 2 * nblocks + gid] : static_cast<T>(0);

            // 1x1 pivoting
            if(pivot[k + gid] == 1 || k == (BLOCKDIM - 1) * nblocks)
            {
                T iBk = static_cast<T>(1) / bk;

                bk_1 = bk_1 - ak_1 * ck * iBk;

                if(COLS == 8)
                {
                    T rhsk_col0 = rhs[k + gid + m_pad * (COLS * bidy + 0)] * iBk;
                    T rhsk_col1 = rhs[k + gid + m_pad * (COLS * bidy + 1)] * iBk;
                    T rhsk_col2 = rhs[k + gid + m_pad * (COLS * bidy + 2)] * iBk;
                    T rhsk_col3 = rhs[k + gid + m_pad * (COLS * bidy + 3)] * iBk;
                    T rhsk_col4 = rhs[k + gid + m_pad * (COLS * bidy + 4)] * iBk;
                    T rhsk_col5 = rhs[k + gid + m_pad * (COLS * bidy + 5)] * iBk;
                    T rhsk_col6 = rhs[k + gid + m_pad * (COLS * bidy + 6)] * iBk;
                    T rhsk_col7 = rhs[k + gid + m_pad * (COLS * bidy + 7)] * iBk;

                    rhs[k + gid + m_pad * (COLS * bidy + 0)] = rhsk_col0;
                    rhs[k + gid + m_pad * (COLS * bidy + 1)] = rhsk_col1;
                    rhs[k + gid + m_pad * (COLS * bidy + 2)] = rhsk_col2;
                    rhs[k + gid + m_pad * (COLS * bidy + 3)] = rhsk_col3;
                    rhs[k + gid + m_pad * (COLS * bidy + 4)] = rhsk_col4;
                    rhs[k + gid + m_pad * (COLS * bidy + 5)] = rhsk_col5;
                    rhs[k + gid + m_pad * (COLS * bidy + 6)] = rhsk_col6;
                    rhs[k + gid + m_pad * (COLS * bidy + 7)] = rhsk_col7;

                    if(k < (BLOCKDIM - 1) * nblocks)
                    {
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)] = rocsparse::fma(
                            -ak_1, rhsk_col0, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)] = rocsparse::fma(
                            -ak_1, rhsk_col1, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)] = rocsparse::fma(
                            -ak_1, rhsk_col2, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)] = rocsparse::fma(
                            -ak_1, rhsk_col3, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 4)] = rocsparse::fma(
                            -ak_1, rhsk_col4, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 4)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 5)] = rocsparse::fma(
                            -ak_1, rhsk_col5, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 5)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 6)] = rocsparse::fma(
                            -ak_1, rhsk_col6, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 6)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 7)] = rocsparse::fma(
                            -ak_1, rhsk_col7, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 7)]);
                    }
                }
                else if(COLS == 4)
                {
                    T rhsk_col0 = rhs[k + gid + m_pad * (COLS * bidy + 0)] * iBk;
                    T rhsk_col1 = rhs[k + gid + m_pad * (COLS * bidy + 1)] * iBk;
                    T rhsk_col2 = rhs[k + gid + m_pad * (COLS * bidy + 2)] * iBk;
                    T rhsk_col3 = rhs[k + gid + m_pad * (COLS * bidy + 3)] * iBk;

                    rhs[k + gid + m_pad * (COLS * bidy + 0)] = rhsk_col0;
                    rhs[k + gid + m_pad * (COLS * bidy + 1)] = rhsk_col1;
                    rhs[k + gid + m_pad * (COLS * bidy + 2)] = rhsk_col2;
                    rhs[k + gid + m_pad * (COLS * bidy + 3)] = rhsk_col3;

                    if(k < (BLOCKDIM - 1) * nblocks)
                    {
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)] = rocsparse::fma(
                            -ak_1, rhsk_col0, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)] = rocsparse::fma(
                            -ak_1, rhsk_col1, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)] = rocsparse::fma(
                            -ak_1, rhsk_col2, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)] = rocsparse::fma(
                            -ak_1, rhsk_col3, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)]);
                    }
                }
                else if(COLS == 2)
                {
                    T rhsk_col0 = rhs[k + gid + m_pad * (COLS * bidy + 0)] * iBk;
                    T rhsk_col1 = rhs[k + gid + m_pad * (COLS * bidy + 1)] * iBk;

                    rhs[k + gid + m_pad * (COLS * bidy + 0)] = rhsk_col0;
                    rhs[k + gid + m_pad * (COLS * bidy + 1)] = rhsk_col1;

                    if(k < (BLOCKDIM - 1) * nblocks)
                    {
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)] = rocsparse::fma(
                            -ak_1, rhsk_col0, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)]);
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)] = rocsparse::fma(
                            -ak_1, rhsk_col1, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)]);
                    }
                }
                else
                {
                    T rhsk_col0 = rhs[k + gid + m_pad * (COLS * bidy + 0)] * iBk;

                    rhs[k + gid + m_pad * (COLS * bidy + 0)] = rhsk_col0;

                    if(k < (BLOCKDIM - 1) * nblocks)
                    {
                        rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)] = rocsparse::fma(
                            -ak_1, rhsk_col0, rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)]);
                    }
                }

                bk = bk_1;

                k += nblocks;
            }
            // 2x2 pivoting
            else
            {
                T det = bk * bk_1 - ak_1 * ck;
                det   = static_cast<T>(1) / det;

                T bk_2 = static_cast<T>(0);

                if(COLS == 8)
                {
                    T rhsk_col0 = rhs[k + gid + m_pad * (COLS * bidy + 0)] * det;
                    T rhsk_col1 = rhs[k + gid + m_pad * (COLS * bidy + 1)] * det;
                    T rhsk_col2 = rhs[k + gid + m_pad * (COLS * bidy + 2)] * det;
                    T rhsk_col3 = rhs[k + gid + m_pad * (COLS * bidy + 3)] * det;
                    T rhsk_col4 = rhs[k + gid + m_pad * (COLS * bidy + 4)] * det;
                    T rhsk_col5 = rhs[k + gid + m_pad * (COLS * bidy + 5)] * det;
                    T rhsk_col6 = rhs[k + gid + m_pad * (COLS * bidy + 6)] * det;
                    T rhsk_col7 = rhs[k + gid + m_pad * (COLS * bidy + 7)] * det;

                    T rhsk_1_col0 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)] * det;
                    T rhsk_1_col1 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)] * det;
                    T rhsk_1_col2 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)] * det;
                    T rhsk_1_col3 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)] * det;
                    T rhsk_1_col4 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 4)] * det;
                    T rhsk_1_col5 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 5)] * det;
                    T rhsk_1_col6 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 6)] * det;
                    T rhsk_1_col7 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 7)] * det;

                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = (bk_1 * rhsk_col0 - ck * rhsk_1_col0);
                    rhs[k + gid + m_pad * (COLS * bidy + 1)]
                        = (bk_1 * rhsk_col1 - ck * rhsk_1_col1);
                    rhs[k + gid + m_pad * (COLS * bidy + 2)]
                        = (bk_1 * rhsk_col2 - ck * rhsk_1_col2);
                    rhs[k + gid + m_pad * (COLS * bidy + 3)]
                        = (bk_1 * rhsk_col3 - ck * rhsk_1_col3);
                    rhs[k + gid + m_pad * (COLS * bidy + 4)]
                        = (bk_1 * rhsk_col4 - ck * rhsk_1_col4);
                    rhs[k + gid + m_pad * (COLS * bidy + 5)]
                        = (bk_1 * rhsk_col5 - ck * rhsk_1_col5);
                    rhs[k + gid + m_pad * (COLS * bidy + 6)]
                        = (bk_1 * rhsk_col6 - ck * rhsk_1_col6);
                    rhs[k + gid + m_pad * (COLS * bidy + 7)]
                        = (bk_1 * rhsk_col7 - ck * rhsk_1_col7);

                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)]
                        = (-ak_1 * rhsk_col0 + bk * rhsk_1_col0);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)]
                        = (-ak_1 * rhsk_col1 + bk * rhsk_1_col1);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)]
                        = (-ak_1 * rhsk_col2 + bk * rhsk_1_col2);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)]
                        = (-ak_1 * rhsk_col3 + bk * rhsk_1_col3);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 4)]
                        = (-ak_1 * rhsk_col4 + bk * rhsk_1_col4);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 5)]
                        = (-ak_1 * rhsk_col5 + bk * rhsk_1_col5);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 6)]
                        = (-ak_1 * rhsk_col6 + bk * rhsk_1_col6);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 7)]
                        = (-ak_1 * rhsk_col7 + bk * rhsk_1_col7);

                    if(k < (BLOCKDIM - 2) * nblocks)
                    {
                        T tmp1 = -(-ak_1 * ak_2);
                        T tmp2 = (bk * ak_2);

                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 0)]
                            += tmp1 * rhsk_col0 - tmp2 * rhsk_1_col0;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 1)]
                            += tmp1 * rhsk_col1 - tmp2 * rhsk_1_col1;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 2)]
                            += tmp1 * rhsk_col2 - tmp2 * rhsk_1_col2;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 3)]
                            += tmp1 * rhsk_col3 - tmp2 * rhsk_1_col3;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 4)]
                            += tmp1 * rhsk_col4 - tmp2 * rhsk_1_col4;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 5)]
                            += tmp1 * rhsk_col5 - tmp2 * rhsk_1_col5;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 6)]
                            += tmp1 * rhsk_col6 - tmp2 * rhsk_1_col6;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 7)]
                            += tmp1 * rhsk_col7 - tmp2 * rhsk_1_col7;

                        bk_2 = b[k + 2 * nblocks + gid];
                        bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                    }
                }
                else if(COLS == 4)
                {
                    T rhsk_col0 = rhs[k + gid + m_pad * (COLS * bidy + 0)] * det;
                    T rhsk_col1 = rhs[k + gid + m_pad * (COLS * bidy + 1)] * det;
                    T rhsk_col2 = rhs[k + gid + m_pad * (COLS * bidy + 2)] * det;
                    T rhsk_col3 = rhs[k + gid + m_pad * (COLS * bidy + 3)] * det;

                    T rhsk_1_col0 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)] * det;
                    T rhsk_1_col1 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)] * det;
                    T rhsk_1_col2 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)] * det;
                    T rhsk_1_col3 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)] * det;

                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = (bk_1 * rhsk_col0 - ck * rhsk_1_col0);
                    rhs[k + gid + m_pad * (COLS * bidy + 1)]
                        = (bk_1 * rhsk_col1 - ck * rhsk_1_col1);
                    rhs[k + gid + m_pad * (COLS * bidy + 2)]
                        = (bk_1 * rhsk_col2 - ck * rhsk_1_col2);
                    rhs[k + gid + m_pad * (COLS * bidy + 3)]
                        = (bk_1 * rhsk_col3 - ck * rhsk_1_col3);

                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)]
                        = (-ak_1 * rhsk_col0 + bk * rhsk_1_col0);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)]
                        = (-ak_1 * rhsk_col1 + bk * rhsk_1_col1);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)]
                        = (-ak_1 * rhsk_col2 + bk * rhsk_1_col2);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)]
                        = (-ak_1 * rhsk_col3 + bk * rhsk_1_col3);

                    if(k < (BLOCKDIM - 2) * nblocks)
                    {
                        T tmp1 = -(-ak_1 * ak_2);
                        T tmp2 = (bk * ak_2);

                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 0)]
                            += tmp1 * rhsk_col0 - tmp2 * rhsk_1_col0;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 1)]
                            += tmp1 * rhsk_col1 - tmp2 * rhsk_1_col1;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 2)]
                            += tmp1 * rhsk_col2 - tmp2 * rhsk_1_col2;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 3)]
                            += tmp1 * rhsk_col3 - tmp2 * rhsk_1_col3;

                        bk_2 = b[k + 2 * nblocks + gid];
                        bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                    }
                }
                else if(COLS == 2)
                {
                    T rhsk_col0 = rhs[k + gid + m_pad * (COLS * bidy + 0)] * det;
                    T rhsk_col1 = rhs[k + gid + m_pad * (COLS * bidy + 1)] * det;

                    T rhsk_1_col0 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)] * det;
                    T rhsk_1_col1 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)] * det;

                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = (bk_1 * rhsk_col0 - ck * rhsk_1_col0);
                    rhs[k + gid + m_pad * (COLS * bidy + 1)]
                        = (bk_1 * rhsk_col1 - ck * rhsk_1_col1);

                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)]
                        = (-ak_1 * rhsk_col0 + bk * rhsk_1_col0);
                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)]
                        = (-ak_1 * rhsk_col1 + bk * rhsk_1_col1);

                    if(k < (BLOCKDIM - 2) * nblocks)
                    {
                        T tmp1 = -(-ak_1 * ak_2);
                        T tmp2 = (bk * ak_2);
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 0)]
                            += tmp1 * rhsk_col0 - tmp2 * rhsk_1_col0;
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 1)]
                            += tmp1 * rhsk_col1 - tmp2 * rhsk_1_col1;

                        bk_2 = b[k + 2 * nblocks + gid];
                        bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                    }
                }
                else
                {
                    T rhsk_col0 = rhs[k + gid + m_pad * (COLS * bidy + 0)] * det;

                    T rhsk_1_col0 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)] * det;

                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = (bk_1 * rhsk_col0 - ck * rhsk_1_col0);

                    rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)]
                        = (-ak_1 * rhsk_col0 + bk * rhsk_1_col0);

                    if(k < (BLOCKDIM - 2) * nblocks)
                    {
                        rhs[k + 2 * nblocks + gid + m_pad * (COLS * bidy + 0)]
                            += -(-ak_1 * ak_2) * rhsk_col0 - (bk * ak_2) * rhsk_1_col0;

                        bk_2 = b[k + 2 * nblocks + gid];
                        bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                    }
                }

                bk = bk_2;

                k += 2 * nblocks;
            }
        }

        __threadfence();

        // at this point k = BLOCKDIM * nblocks
        k -= nblocks;

        k -= nblocks * pivot[k + gid];

        // backward solve (M^T * rhs = rhs)
        while(k >= 0)
        {
            if(pivot[k + gid] == 1)
            {
                T mt_tmp = -mt[k + gid];

                if(COLS == 8)
                {
                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 0)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 1)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 1)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 2)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 2)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 3)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 3)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 4)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 4)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 4)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 5)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 5)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 5)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 6)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 6)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 6)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 7)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 7)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 7)]);
                }
                else if(COLS == 4)
                {
                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 0)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 1)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 1)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 2)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 2)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 3)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 3)]);
                }
                else if(COLS == 2)
                {
                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 0)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 1)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 1)]);
                }
                else
                {
                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = rocsparse::fma(mt_tmp,
                                         rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)],
                                         rhs[k + gid + m_pad * (COLS * bidy + 0)]);
                }

                k -= nblocks;
            }
            else
            {
                T mt_tmp  = -mt[k + gid];
                T mt_tmp1 = -mt[k - nblocks + gid];

                if(COLS == 8)
                {
                    T tmp0 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)];
                    T tmp1 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)];
                    T tmp2 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)];
                    T tmp3 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)];
                    T tmp4 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 4)];
                    T tmp5 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 5)];
                    T tmp6 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 6)];
                    T tmp7 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 7)];

                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = rocsparse::fma(mt_tmp, tmp0, rhs[k + gid + m_pad * (COLS * bidy + 0)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 1)]
                        = rocsparse::fma(mt_tmp, tmp1, rhs[k + gid + m_pad * (COLS * bidy + 1)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 2)]
                        = rocsparse::fma(mt_tmp, tmp2, rhs[k + gid + m_pad * (COLS * bidy + 2)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 3)]
                        = rocsparse::fma(mt_tmp, tmp3, rhs[k + gid + m_pad * (COLS * bidy + 3)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 4)]
                        = rocsparse::fma(mt_tmp, tmp4, rhs[k + gid + m_pad * (COLS * bidy + 4)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 5)]
                        = rocsparse::fma(mt_tmp, tmp5, rhs[k + gid + m_pad * (COLS * bidy + 5)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 6)]
                        = rocsparse::fma(mt_tmp, tmp6, rhs[k + gid + m_pad * (COLS * bidy + 6)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 7)]
                        = rocsparse::fma(mt_tmp, tmp7, rhs[k + gid + m_pad * (COLS * bidy + 7)]);

                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 0)] = rocsparse::fma(
                        mt_tmp1, tmp0, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 0)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 1)] = rocsparse::fma(
                        mt_tmp1, tmp1, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 1)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 2)] = rocsparse::fma(
                        mt_tmp1, tmp2, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 2)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 3)] = rocsparse::fma(
                        mt_tmp1, tmp3, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 3)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 4)] = rocsparse::fma(
                        mt_tmp1, tmp4, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 4)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 5)] = rocsparse::fma(
                        mt_tmp1, tmp5, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 5)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 6)] = rocsparse::fma(
                        mt_tmp1, tmp6, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 6)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 7)] = rocsparse::fma(
                        mt_tmp1, tmp7, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 7)]);
                }
                else if(COLS == 4)
                {
                    T tmp0 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)];
                    T tmp1 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)];
                    T tmp2 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 2)];
                    T tmp3 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 3)];

                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = rocsparse::fma(mt_tmp, tmp0, rhs[k + gid + m_pad * (COLS * bidy + 0)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 1)]
                        = rocsparse::fma(mt_tmp, tmp1, rhs[k + gid + m_pad * (COLS * bidy + 1)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 2)]
                        = rocsparse::fma(mt_tmp, tmp2, rhs[k + gid + m_pad * (COLS * bidy + 2)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 3)]
                        = rocsparse::fma(mt_tmp, tmp3, rhs[k + gid + m_pad * (COLS * bidy + 3)]);

                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 0)] = rocsparse::fma(
                        mt_tmp1, tmp0, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 0)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 1)] = rocsparse::fma(
                        mt_tmp1, tmp1, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 1)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 2)] = rocsparse::fma(
                        mt_tmp1, tmp2, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 2)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 3)] = rocsparse::fma(
                        mt_tmp1, tmp3, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 3)]);
                }
                else if(COLS == 2)
                {
                    T tmp0 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)];
                    T tmp1 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 1)];

                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = rocsparse::fma(mt_tmp, tmp0, rhs[k + gid + m_pad * (COLS * bidy + 0)]);
                    rhs[k + gid + m_pad * (COLS * bidy + 1)]
                        = rocsparse::fma(mt_tmp, tmp1, rhs[k + gid + m_pad * (COLS * bidy + 1)]);

                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 0)] = rocsparse::fma(
                        mt_tmp1, tmp0, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 0)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 1)] = rocsparse::fma(
                        mt_tmp1, tmp1, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 1)]);
                }
                else
                {
                    T tmp0 = rhs[k + nblocks + gid + m_pad * (COLS * bidy + 0)];

                    rhs[k + gid + m_pad * (COLS * bidy + 0)]
                        = rocsparse::fma(mt_tmp, tmp0, rhs[k + gid + m_pad * (COLS * bidy + 0)]);
                    rhs[k - nblocks + gid + m_pad * (COLS * bidy + 0)] = rocsparse::fma(
                        mt_tmp1, tmp0, rhs[k - nblocks + gid + m_pad * (COLS * bidy + 0)]);
                }

                k -= 2 * nblocks;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_spike_block_level_kernel(rocsparse_int m_pad,
                                       rocsparse_int n,
                                       rocsparse_int ldb,
                                       T* __restrict__ rhs,
                                       const T* __restrict__ w,
                                       const T* __restrict__ v,
                                       T* __restrict__ w2,
                                       T* __restrict__ v2,
                                       T* __restrict__ rhs_scratch,
                                       T* __restrict__ w_scratch,
                                       T* __restrict__ v_scratch)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidx = hipBlockIdx_x;
        rocsparse_int bidy = hipBlockIdx_y;
        rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

        rocsparse_int nblocks = m_pad / BLOCKDIM;

        __shared__ T sw[2 * BLOCKSIZE];
        __shared__ T sv[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE];

        sw[tidx] = (gid < nblocks) ? w[gid] : static_cast<T>(0);
        sw[tidx + BLOCKSIZE]
            = (gid < nblocks) ? w[gid + (BLOCKDIM - 1) * nblocks] : static_cast<T>(0);

        sv[tidx] = (gid < nblocks) ? v[gid] : static_cast<T>(0);
        sv[tidx + BLOCKSIZE]
            = (gid < nblocks) ? v[gid + (BLOCKDIM - 1) * nblocks] : static_cast<T>(0);

        srhs[tidx]             = (gid < nblocks) ? rhs[gid + m_pad * bidy] : static_cast<T>(0);
        srhs[tidx + BLOCKSIZE] = (gid < nblocks)
                                     ? rhs[gid + (BLOCKDIM - 1) * nblocks + m_pad * bidy]
                                     : static_cast<T>(0);

        __syncthreads();

        rocsparse_int stride = 2;

        while(stride <= BLOCKSIZE)
        {
            if(tidx < BLOCKSIZE / stride)
            {
                rocsparse_int index = stride * tidx + stride / 2 - 1;
                rocsparse_int minus = index - stride / 2;
                rocsparse_int plus  = index + stride / 2;

                T det = static_cast<T>(1) - sw[index + 1] * sv[index + BLOCKSIZE];
                det   = static_cast<T>(1) / det;

                T tmp1 = srhs[index + BLOCKSIZE];
                T tmp2 = srhs[index + 1];

                srhs[index + BLOCKSIZE] = (tmp1 - sv[index + BLOCKSIZE] * tmp2) * det;
                srhs[index + 1]         = (tmp2 - tmp1 * sw[index + 1]) * det;
                srhs[minus + 1]         = srhs[minus + 1] - sv[minus + 1] * srhs[index + 1];
                srhs[plus + BLOCKSIZE]
                    = srhs[plus + BLOCKSIZE] - sw[plus + BLOCKSIZE] * srhs[index + BLOCKSIZE];

                sv[index + BLOCKSIZE] = -det * (sv[index + BLOCKSIZE] * sv[index + 1]);
                sv[index + 1]         = det * sv[index + 1];
                sw[index + 1]         = -det * (sw[index + BLOCKSIZE] * sw[index + 1]);
                sw[index + BLOCKSIZE] = det * sw[index + BLOCKSIZE];

                sw[minus + 1] = sw[minus + 1] - sv[minus + 1] * sw[index + 1];
                sv[minus + 1] = -sv[minus + 1] * sv[index + 1];
                sv[plus + BLOCKSIZE]
                    = sv[plus + BLOCKSIZE] - sv[index + BLOCKSIZE] * sw[plus + BLOCKSIZE];
                sw[plus + BLOCKSIZE] = -sw[plus + BLOCKSIZE] * sw[index + BLOCKSIZE];
            }

            stride *= 2;

            __syncthreads();
        }

        if(gid < nblocks)
        {
            if(bidy == 0)
            {
                w2[gid]                            = sw[tidx];
                w2[gid + (BLOCKDIM - 1) * nblocks] = sw[tidx + BLOCKSIZE];
                v2[gid]                            = sv[tidx];
                v2[gid + (BLOCKDIM - 1) * nblocks] = sv[tidx + BLOCKSIZE];
            }

            rhs[gid + m_pad * bidy]                            = srhs[tidx];
            rhs[gid + (BLOCKDIM - 1) * nblocks + m_pad * bidy] = srhs[tidx + BLOCKSIZE];
        }

        if(tidx == 0)
        {
            if(bidy == 0)
            {
                w_scratch[bidx]                = sw[0];
                w_scratch[hipGridDim_x + bidx] = sw[2 * BLOCKSIZE - 1];

                v_scratch[bidx]                = sv[0];
                v_scratch[hipGridDim_x + bidx] = sv[2 * BLOCKSIZE - 1];
            }

            rhs_scratch[bidx + 2 * hipGridDim_x * bidy]                = srhs[0];
            rhs_scratch[hipGridDim_x + bidx + 2 * hipGridDim_x * bidy] = srhs[2 * BLOCKSIZE - 1];
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_solve_spike_grid_level_kernel(rocsparse_int m_pad,
                                            rocsparse_int n,
                                            rocsparse_int ldb,
                                            T* __restrict__ rhs_scratch,
                                            const T* __restrict__ w_scratch,
                                            const T* __restrict__ v_scratch)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidy = hipBlockIdx_y;

        __shared__ T sw[2 * BLOCKSIZE];
        __shared__ T sv[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE];

        sw[tidx]               = w_scratch[tidx];
        sw[tidx + BLOCKSIZE]   = w_scratch[tidx + BLOCKSIZE];
        sv[tidx]               = v_scratch[tidx];
        sv[tidx + BLOCKSIZE]   = v_scratch[tidx + BLOCKSIZE];
        srhs[tidx]             = rhs_scratch[tidx + 2 * BLOCKSIZE * bidy];
        srhs[tidx + BLOCKSIZE] = rhs_scratch[tidx + BLOCKSIZE + 2 * BLOCKSIZE * bidy];

        __syncthreads();

        rocsparse_int stride = 2;

        while(stride <= BLOCKSIZE)
        {
            rocsparse_int i = tidx;
            if(i < BLOCKSIZE / stride)
            {
                rocsparse_int index = stride * i + stride / 2 - 1;
                rocsparse_int minus = index - stride / 2;
                rocsparse_int plus  = index + stride / 2;

                T det = static_cast<T>(1) - sw[index + 1] * sv[index + BLOCKSIZE];
                det   = static_cast<T>(1) / det;

                T tmp1 = srhs[index + BLOCKSIZE];
                T tmp2 = srhs[index + 1];

                srhs[index + BLOCKSIZE] = (tmp1 - sv[index + BLOCKSIZE] * tmp2) * det;
                srhs[index + 1]         = (tmp2 - tmp1 * sw[index + 1]) * det;
                srhs[minus + 1]         = srhs[minus + 1] - sv[minus + 1] * srhs[index + 1];
                srhs[plus + BLOCKSIZE]
                    = srhs[plus + BLOCKSIZE] - sw[plus + BLOCKSIZE] * srhs[index + BLOCKSIZE];

                sv[index + BLOCKSIZE] = -det * (sv[index + BLOCKSIZE] * sv[index + 1]);
                sv[index + 1]         = det * sv[index + 1];
                sw[index + 1]         = -det * (sw[index + BLOCKSIZE] * sw[index + 1]);
                sw[index + BLOCKSIZE] = det * sw[index + BLOCKSIZE];

                sw[minus + 1] = sw[minus + 1] - sv[minus + 1] * sw[index + 1];
                sv[minus + 1] = -sv[minus + 1] * sv[index + 1];
                sv[plus + BLOCKSIZE]
                    = sv[plus + BLOCKSIZE] - sv[index + BLOCKSIZE] * sw[plus + BLOCKSIZE];
                sw[plus + BLOCKSIZE] = -sw[plus + BLOCKSIZE] * sw[index + BLOCKSIZE];
            }

            stride *= 2;

            __syncthreads();
        }

        stride = BLOCKSIZE / 2;
        while(stride >= 2)
        {
            rocsparse_int i = tidx;
            if(i < BLOCKSIZE / stride)
            {
                rocsparse_int index = stride * i + stride / 2 - 1;
                rocsparse_int minus = index - stride / 2;
                rocsparse_int plus  = index + stride / 2 + 1;

                minus = (minus < 0) ? 0 : minus;
                plus  = plus < BLOCKSIZE ? plus : BLOCKSIZE - 1;

                srhs[index + BLOCKSIZE]
                    = srhs[index + BLOCKSIZE] - sw[index + BLOCKSIZE] * srhs[minus + BLOCKSIZE];
                srhs[index + BLOCKSIZE]
                    = srhs[index + BLOCKSIZE] - sv[index + BLOCKSIZE] * srhs[plus];
                srhs[index + 1] = srhs[index + 1] - sw[index + 1] * srhs[minus + BLOCKSIZE];
                srhs[index + 1] = srhs[index + 1] - sv[index + 1] * srhs[plus];
            }

            stride /= 2;

            __syncthreads();
        }

        rhs_scratch[tidx + 2 * BLOCKSIZE * bidy]             = srhs[tidx];
        rhs_scratch[tidx + BLOCKSIZE + 2 * BLOCKSIZE * bidy] = srhs[tidx + BLOCKSIZE];
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_solve_spike_propagate_kernel(rocsparse_int m_pad,
                                           rocsparse_int n,
                                           rocsparse_int ldb,
                                           T* __restrict__ rhs,
                                           const T* __restrict__ w,
                                           const T* __restrict__ v,
                                           const T* __restrict__ rhs_scratch)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidx = hipBlockIdx_x;
        rocsparse_int bidy = hipBlockIdx_y;
        rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

        rocsparse_int nblocks = m_pad / BLOCKDIM;

        __shared__ T sw[2 * BLOCKSIZE];
        __shared__ T sv[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE + 2];

        sw[tidx] = (gid < nblocks) ? w[gid] : static_cast<T>(0);
        sw[tidx + BLOCKSIZE]
            = (gid < nblocks) ? w[gid + (BLOCKDIM - 1) * nblocks] : static_cast<T>(0);

        sv[tidx] = (gid < nblocks) ? v[gid] : static_cast<T>(0);
        sv[tidx + BLOCKSIZE]
            = (gid < nblocks) ? v[gid + (BLOCKDIM - 1) * nblocks] : static_cast<T>(0);

        srhs[tidx + 1] = (gid < nblocks) ? rhs[gid + (BLOCKDIM - 1) * nblocks + m_pad * bidy]
                                         : static_cast<T>(0);
        srhs[tidx + 1 + BLOCKSIZE] = (gid < nblocks) ? rhs[gid + m_pad * bidy] : static_cast<T>(0);

        __syncthreads();

        // load in boundary values from scratch pad
        if(tidx == 0)
        {
            srhs[0] = (bidx > 0) ? rhs_scratch[bidx + hipGridDim_x - 1 + 2 * hipGridDim_x * bidy]
                                 : static_cast<T>(0);
            srhs[2 * BLOCKSIZE + 1] = (bidx < hipGridDim_x - 1)
                                          ? rhs_scratch[bidx + 1 + 2 * hipGridDim_x * bidy]
                                          : static_cast<T>(0);

            srhs[BLOCKSIZE + 1] = rhs_scratch[bidx + 2 * hipGridDim_x * bidy];
            srhs[BLOCKSIZE]     = rhs_scratch[bidx + hipGridDim_x + 2 * hipGridDim_x * bidy];
        }

        __syncthreads();

        rocsparse_int stride = BLOCKSIZE;

        while(stride >= 2)
        {
            if(tidx < BLOCKSIZE / stride)
            {
                rocsparse_int index = stride * tidx + stride / 2 - 1;
                rocsparse_int minus = index - stride / 2;
                rocsparse_int plus  = index + stride / 2;

                srhs[index + 1]
                    = srhs[index + 1] - sv[index + BLOCKSIZE] * srhs[plus + 2 + BLOCKSIZE];
                srhs[index + 1] = srhs[index + 1] - sw[index + BLOCKSIZE] * srhs[minus + 1];
                srhs[index + BLOCKSIZE + 2]
                    = srhs[index + BLOCKSIZE + 2] - sv[index + 1] * srhs[plus + 2 + BLOCKSIZE];
                srhs[index + BLOCKSIZE + 2]
                    = srhs[index + BLOCKSIZE + 2] - sw[index + 1] * srhs[minus + 1];
            }

            stride /= 2;

            __syncthreads();
        }

        if(gid < nblocks)
        {
            rhs[gid + m_pad * bidy]                            = srhs[tidx + 1 + BLOCKSIZE];
            rhs[gid + (BLOCKDIM - 1) * nblocks + m_pad * bidy] = srhs[tidx + 1];
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_spike_backward_substitution_kernel(rocsparse_int m_pad,
                                                 rocsparse_int n,
                                                 rocsparse_int ldb,
                                                 T* __restrict__ rhs,
                                                 const T* __restrict__ w,
                                                 const T* __restrict__ v)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidx = hipBlockIdx_x;
        rocsparse_int bidy = hipBlockIdx_y;
        rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

        rocsparse_int nblocks = m_pad / BLOCKDIM;

        if(gid >= nblocks)
        {
            return;
        }

        T tmp1 = (gid > 0) ? rhs[gid - 1 + (BLOCKDIM - 1) * nblocks + m_pad * bidy]
                           : static_cast<T>(0);
        T tmp2 = (gid + BLOCKDIM < m_pad) ? rhs[gid + 1 + m_pad * bidy] : static_cast<T>(0);

        for(rocsparse_int i = 1; i < BLOCKDIM - 1; i++)
        {
            rhs[gid + i * nblocks + m_pad * bidy]
                = rhs[gid + i * nblocks + m_pad * bidy] - w[gid + i * nblocks] * tmp1;
            rhs[gid + i * nblocks + m_pad * bidy]
                = rhs[gid + i * nblocks + m_pad * bidy] - v[gid + i * nblocks] * tmp2;
        }
    }
}
