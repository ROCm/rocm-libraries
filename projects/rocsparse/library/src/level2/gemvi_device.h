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
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t UNROLL, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void gemvi_device_part1(I m,
                                                 I n,
                                                 T alpha,
                                                 const T* __restrict__ A,
                                                 int64_t lda,
                                                 I       nnz,
                                                 const T* __restrict__ x_val,
                                                 const I* __restrict__ x_ind,
                                                 T beta,
                                                 T* __restrict__ y,
                                                 T* __restrict__ workspace,
                                                 rocsparse_index_base idx_base)
    {
        rocsparse_device_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0,
                                "WFSIZE must be a power of two.");
        rocsparse_device_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        rocsparse_device_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        rocsparse_device_assert(((BLOCKSIZE / WFSIZE) & ((BLOCKSIZE / WFSIZE) - 1)) == 0,
                                "The number of wavefronts per block must be a power of two.");
        rocsparse_device_assert(UNROLL > 0, "UNROLL must be positive.");

        const uint32_t lid = hipThreadIdx_x & (WFSIZE - 1);
        const uint32_t wid = hipThreadIdx_x / WFSIZE;

        // Each threadblock processes WFSIZE rows. The sparse vector is split into
        // contiguous chunks, one per wavefront, so the unrolled x_val and x_ind
        // loads are adjacent.
        const I row = hipBlockIdx_x * WFSIZE + lid;

        // Sub-row sum accumulators
        T sum[UNROLL]{};

        I columns[UNROLL];
        T values[UNROLL];

        if(row < m)
        {
            const uint32_t nworkers = (BLOCKSIZE / WFSIZE) * hipGridDim_y; // ncol
            const uint32_t worker   = (BLOCKSIZE / WFSIZE) * hipBlockIdx_y + wid; // col

            const I nworkers_i = static_cast<I>(nworkers);
            const I worker_i   = static_cast<I>(worker);

            // First `remainder` workers receive one extra nonzero.
            const I chunk     = nnz / nworkers_i;
            const I remainder = nnz - chunk * nworkers_i;
            const I extra     = (worker_i < remainder) ? worker_i : remainder;
            const I begin     = worker_i * chunk + extra;
            const I count
                = chunk + ((worker_i < remainder) ? static_cast<I>(1) : static_cast<I>(0));
            const I end = begin + count;

            constexpr I unroll_i = static_cast<I>(UNROLL);

            I j = begin;
            for(; end - j >= unroll_i; j += unroll_i)
            {
#pragma unroll
                for(uint32_t u = 0; u < UNROLL; ++u)
                {
                    const I idx = j + static_cast<I>(u);
                    columns[u]  = (x_ind[idx] - idx_base) * lda + row;
                    values[u]   = x_val[idx];
                }
#pragma unroll
                for(uint32_t u = 0; u < UNROLL; ++u)
                {
                    sum[u] = rocsparse::fma(values[u], A[columns[u]], sum[u]);
                }
            }

#pragma unroll
            for(uint32_t u = 0; u < UNROLL; ++u)
            {
                columns[u] = static_cast<I>(0);
                values[u]  = static_cast<T>(0);
            }
#pragma unroll
            for(uint32_t u = 0; u < UNROLL; ++u)
            {
                const I idx = j + static_cast<I>(u);
                if(idx < end)
                {
                    columns[u] = (x_ind[idx] - idx_base) * lda + row;
                    values[u]  = x_val[idx];
                }
            }
#pragma unroll
            for(uint32_t u = 0; u < UNROLL; ++u)
            {
                const I idx = j + static_cast<I>(u);
                if(idx < end)
                {
                    sum[u] = rocsparse::fma(values[u], A[columns[u]], sum[u]);
                }
            }
        }

        // Collapse the per-thread accumulators before the cross-wavefront reduction.
        T total = sum[0];
        for(uint32_t u = 1; u < UNROLL; u++)
        {
            total += sum[u];
        }

        if constexpr(BLOCKSIZE == WFSIZE)
        {
            if(hipGridDim_y == 1)
            {
                if(row < m)
                {
                    if(beta != static_cast<T>(0))
                    {
                        y[row] = rocsparse::fma(alpha, total, beta * y[row]);
                    }
                    else
                    {
                        y[row] = alpha * total;
                    }
                }
            }
            else
            {
                workspace[WFSIZE * hipGridDim_y * hipBlockIdx_x + WFSIZE * hipBlockIdx_y + lid]
                    = total;
            }

            return;
        }

        __shared__ T sdata[BLOCKSIZE];

        // Write sub-row sum into LDS
        sdata[wid * WFSIZE + lid] = total;

        // and wait for all threads to finish writing
        __syncthreads();

        // Accumulate the per-wavefront sub-row sums (one per wid)
        for(uint32_t s = (BLOCKSIZE / WFSIZE) / 2; s > 0; s >>= 1)
        {
            if(wid < s)
            {
                sdata[wid * WFSIZE + lid] += sdata[(wid + s) * WFSIZE + lid];
            }
            __syncthreads();
        }

        if(hipGridDim_y == 1)
        {
            // First wavefront writes (accumulated) row sums back to y
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
        else
        {
            if(wid == 0)
            {
                workspace[WFSIZE * hipGridDim_y * hipBlockIdx_x + WFSIZE * hipBlockIdx_y + lid]
                    = sdata[lid];
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void gemvi_device_part2(
        I m, int grid_y, T alpha, T beta, const T* __restrict__ workspace, T* __restrict__ y)
    {
        rocsparse_device_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0,
                                "WFSIZE must be a power of two.");
        rocsparse_device_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        rocsparse_device_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        rocsparse_device_assert(((BLOCKSIZE / WFSIZE) & ((BLOCKSIZE / WFSIZE) - 1)) == 0,
                                "The number of wavefronts per block must be a power of two.");

        const uint32_t lid = hipThreadIdx_x & (WFSIZE - 1);
        const uint32_t wid = hipThreadIdx_x / WFSIZE;

        const I row = hipBlockIdx_x * WFSIZE + lid;

        // Number of split-k partial sums per output row produced by part1
        // (equal to the y-dimension of the part1 launch grid).
        const uint32_t     nblocks_part1 = grid_y;
        constexpr uint32_t NWF           = BLOCKSIZE / WFSIZE;

        T sum = static_cast<T>(0);
        for(uint32_t by = static_cast<uint32_t>(wid); by < nblocks_part1; by += NWF)
        {
            sum += workspace[WFSIZE * nblocks_part1 * hipBlockIdx_x + WFSIZE * by + lid];
        }

        // Reduce the per-wavefront partial sums (one set per lid) through LDS,
        // using the same binary-tree reduction as part1.
        __shared__ T sdata[BLOCKSIZE];
        sdata[wid * WFSIZE + lid] = sum;
        __syncthreads();

        for(uint32_t s = NWF / 2; s > 0; s >>= 1)
        {
            if(wid < s)
            {
                sdata[wid * WFSIZE + lid] += sdata[(wid + s) * WFSIZE + lid];
            }
            __syncthreads();
        }

        // First wavefront applies alpha/beta and writes the accumulated row sums to y.
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
