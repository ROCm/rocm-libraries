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
    template <uint32_t BLOCKSIZE, typename J = rocsparse_int>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrcolor_kernel_count_colors(J size,
                                      const J* __restrict__ colors,
                                      J* __restrict__ workspace)
    {
        J gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        J inc = gridDim.x * hipBlockDim_x;

        __shared__ J sdata[BLOCKSIZE];

        J mx = 0;
        for(J idx = gid; idx < size; idx += inc)
        {
            J color = colors[idx];
            if(color > mx)
            {
                mx = color;
            }
        }

        sdata[hipThreadIdx_x] = mx;

        __syncthreads();
        rocsparse::blockreduce_max<BLOCKSIZE>(hipThreadIdx_x, sdata);
        if(hipThreadIdx_x == 0)
        {
            workspace[hipBlockIdx_x] = sdata[0];
        }
    }

    template <uint32_t BLOCKSIZE, typename J = rocsparse_int>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrcolor_kernel_count_colors_finalize(J* __restrict__ workspace)
    {
        __shared__ J sdata[BLOCKSIZE];

        sdata[hipThreadIdx_x] = workspace[hipThreadIdx_x];

        __syncthreads();
        rocsparse::blockreduce_max<BLOCKSIZE>(hipThreadIdx_x, sdata);
        if(hipThreadIdx_x == 0)
        {
            workspace[0] = sdata[0];
        }
    }

    template <uint32_t BLOCKSIZE, typename J = rocsparse_int>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrcolor_kernel_count_uncolored(J size,
                                         const J* __restrict__ colors,
                                         J* __restrict__ workspace)
    {
        J gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        J inc = gridDim.x * hipBlockDim_x;

        __shared__ J sdata[BLOCKSIZE];

        J sum = 0;
        for(J idx = gid; idx < size; idx += inc)
        {
            if(colors[idx] == -1)
            {
                ++sum;
            }
        }

        sdata[hipThreadIdx_x] = sum;

        __syncthreads();
        rocsparse::blockreduce_sum<BLOCKSIZE>(hipThreadIdx_x, sdata);
        if(hipThreadIdx_x == 0)
        {
            workspace[hipBlockIdx_x] = sdata[0];
        }
    }

    template <uint32_t BLOCKSIZE, typename J = rocsparse_int>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrcolor_kernel_count_uncolored_finalize(J* __restrict__ workspace)
    {
        __shared__ J sdata[BLOCKSIZE];

        sdata[hipThreadIdx_x] = workspace[hipThreadIdx_x];

        __syncthreads();
        rocsparse::blockreduce_sum<BLOCKSIZE>(hipThreadIdx_x, sdata);
        if(hipThreadIdx_x == 0)
        {
            workspace[0] = sdata[0];
        }
    }

    ROCSPARSE_DEVICE_ILF uint32_t murmur3_32(uint32_t h)
    {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;

        return h;
    }

    template <uint32_t BLOCKSIZE, typename I = rocsparse_int, typename J = rocsparse_int>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrcolor_kernel_jpl(J m,
                             J color,
                             const I* __restrict__ csr_row_ptr,
                             const J* __restrict__ csr_col_ind,
                             rocsparse_index_base csr_base,
                             J* __restrict__ colors)
    {

        //
        // Each thread processes a vertex
        //
        J row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        //
        // Do not run out of bounds
        //
        if(row >= m)
        {
            return;
        }

        //
        // Assume current vertex is maximum and minimum
        //
        bool min = true, max = true;

        //
        // Do not process already colored vertices
        //
        if(colors[row] != -1)
        {
            return;
        }

        //
        // Get row weight
        //
        uint32_t row_hash = murmur3_32(row);

        //
        // Look at neighbors to check their random number
        //
        const I bound = csr_row_ptr[row + 1] - csr_base;
        for(I j = csr_row_ptr[row] - csr_base; j < bound; ++j)
        {

            //
            // Column to check against
            //
            J col = csr_col_ind[j] - csr_base;

            //
            // Skip diagonal
            //
            if(row == col)
            {
                continue;
            }

            //
            // Current neighbors color (-1 if uncolored)
            //
            J color_nb = colors[col];

            //
            // Skip already colored neighbors
            //
            if(color_nb != -1 && color_nb != color && color_nb != (color + 1))
            {
                continue;
            }

            //
            // Compute column hash
            //
            uint32_t col_hash = murmur3_32(col);

            //
            // Found neighboring vertex with larger weight,
            // vertex cannot be a maximum
            //
            if(row_hash <= col_hash)
            {
                max = false;
            }

            //
            // Found neighboring vertex with smaller weight,
            // vertex cannot be a minimum
            //
            if(row_hash >= col_hash)
            {
                min = false;
            }
        }

        //
        // If vertex is a maximum or a minimum then color it.
        //
        if(max)
        {
            colors[row] = color;
        }
        else if(min)
        {
            colors[row] = color + 1;
        }
    }
}
