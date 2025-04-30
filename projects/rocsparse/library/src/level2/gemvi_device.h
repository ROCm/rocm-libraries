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
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void gemvi_device(I                    m,
                                           I                    n,
                                           T                    alpha,
                                           const T*             A,
                                           int64_t              lda,
                                           I                    nnz,
                                           const T*             x_val,
                                           const I*             x_ind,
                                           T                    beta,
                                           T*                   y,
                                           rocsparse_index_base idx_base)
    {
        const int lid = hipThreadIdx_x & (WFSIZE - 1);
        const int wid = hipThreadIdx_x / WFSIZE;

        // Each threadblock processes WFSIZE rows, where
        // each wavefront processes a column of these rows, e.g.
        // WF 0 processes the first column entry from the list of non-zeros
        // WF 1 processes the second column entry from the list of non-zeros
        // etc.
        const I row = hipBlockIdx_x * WFSIZE + lid;

        // Sub-row sum accumulator
        T sum = static_cast<T>(0);

        // Subsequently, all lanes with id 0 process the first row,
        // all lanes with id 1 process the second row, etc.
        // This guarantees good access pattern into A and x
        if(row < m)
        {
            for(I j = wid; j < nnz; j += BLOCKSIZE / WFSIZE)
            {
                sum = rocsparse::fma(x_val[j], A[(x_ind[j] - idx_base) * lda + row], sum);
            }
        }

        // Having the sub-row sums spread over multiple wavefronts (actually
        // each wavefront contains 64 sub-row sums), we need to use LDS for
        // the row sum reduction.
        __shared__ T sdata[BLOCKSIZE];

        // Write sub-row sum into LDS
        sdata[wid * WFSIZE + lid] = sum;

        // and wait for all threads to finish writing
        __syncthreads();

        // clang-format off
    // Accumulate row sums (from different wavefronts)
    if(WFSIZE == 32)
    {
        if(wid < 16) sdata[wid * WFSIZE + lid] += sdata[(wid + 16) * WFSIZE + lid]; __syncthreads();
    }
    if(wid < 8) sdata[wid * WFSIZE + lid] += sdata[(wid + 8) * WFSIZE + lid]; __syncthreads();
    if(wid < 4) sdata[wid * WFSIZE + lid] += sdata[(wid + 4) * WFSIZE + lid]; __syncthreads();
    if(wid < 2) sdata[wid * WFSIZE + lid] += sdata[(wid + 2) * WFSIZE + lid]; __syncthreads();
    if(wid < 1) sdata[wid * WFSIZE + lid] += sdata[(wid + 1) * WFSIZE + lid];
        // clang-format on

        // Frist wavefront writes (accumulated) 64 row sums back to y
        if(wid == 0 && row < m)
        {
            if(beta != static_cast<T>(0))
            {
                y[row] = rocsparse::fma(alpha, sdata[lid], beta * y[row]);
            }
            else
            {
                y[row] = alpha * sdata[lid];
            }
        }
    }
}
