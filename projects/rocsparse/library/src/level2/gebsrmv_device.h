/* ************************************************************************
* Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    // General GEBSRMV that works for any GEBSR block dimensions
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void gebsrmvn_general_device(rocsparse_direction dir,
                                                      T                   alpha,
                                                      const rocsparse_int* __restrict__ bsr_row_ptr,
                                                      const rocsparse_int* __restrict__ bsr_col_ind,
                                                      const T* __restrict__ bsr_val,
                                                      rocsparse_int row_block_dim,
                                                      rocsparse_int col_block_dim,
                                                      const T* __restrict__ x,
                                                      T beta,
                                                      T* __restrict__ y,
                                                      rocsparse_index_base idx_base)
    {
        // Lane id
        const rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

        // Wavefront id
        const rocsparse_int wid = hipThreadIdx_x / WFSIZE;

        // Each thread block processes a BSR row
        const rocsparse_int row = hipBlockIdx_x;

        // BSR row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Each wavefront processes a row of the BSR block.
        // If the number of BSR block rows exceed the number of wavefronts, each wavefront
        // processes multiple rows. 'bi' is the row index into the BSR block and 'bj' is
        // the column index.
        // BLOCKSIZE must be the square of WFSIZE.

        // Loop over the rows of the BSR block in chunks of WFSIZE, such that each
        // wavefront will process a row
        for(rocsparse_int bi = wid; bi < row_block_dim; bi += BLOCKSIZE / WFSIZE)
        {
            // BSR block row accumulator
            T sum = static_cast<T>(0);

            // Loop over all BSR blocks in the current row
            for(rocsparse_int j = row_begin; j < row_end; ++j)
            {
                // BSR column index
                const rocsparse_int col = bsr_col_ind[j] - idx_base;

                // Loop over the columns of the BSR block in chunks of WFSIZE, such that
                // each lane will process a single value of the BSR block
                for(rocsparse_int bj = lid; bj < col_block_dim; bj += WFSIZE)
                {
                    // Each lane computes the sum of a specific entry over all BSR blocks in
                    // the current row
                    sum = rocsparse::fma(
                        bsr_val[GEBSR_IND(j, bi, bj, dir)], x[col_block_dim * col + bj], sum);
                }
            }

            // Each wavefront accumulates its BSR block row sum
            sum = rocsparse::wfreduce_sum<WFSIZE>(sum);

            // Last lane of each wavefront writes its result to global memory
            if(lid == WFSIZE - 1)
            {
                if(beta != static_cast<T>(0))
                {
                    y[row * row_block_dim + bi]
                        = rocsparse::fma(beta, y[row * row_block_dim + bi], alpha * sum);
                }
                else
                {
                    y[row * row_block_dim + bi] = alpha * sum;
                }
            }
        }
    }

    // GEBSRMV kernel for GEBSR block dimension of 1 x n
    template <uint32_t BLOCKSIZE, uint32_t COLBSRDIM, uint32_t WFSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void gebsrmvn_1xn_device(rocsparse_int       mb,
                                                  rocsparse_direction dir,
                                                  T                   alpha,
                                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                                  const T* __restrict__ bsr_val,
                                                  const T* __restrict__ x,
                                                  T beta,
                                                  T* __restrict__ y,
                                                  rocsparse_index_base idx_base)
    {
        // Lane id
        const rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

        // Wavefront id
        const rocsparse_int wid = hipThreadIdx_x / WFSIZE;

        // Each thread block processes (BLOCKSIZE / WFSIZE) BSR rows
        const rocsparse_int row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

        // Do not run out of bounds
        if(row >= mb)
        {
            return;
        }

        // BSR row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // BSR block row accumulator
        T sum = static_cast<T>(0);

        // Loop over all BSR blocks in the current row where each lane
        // processes a BSR block
        for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Do not exceed the row
            // Column index into x vector
            const rocsparse_int col = (bsr_col_ind[j] - idx_base) * COLBSRDIM;

            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            for(uint32_t k = 0; k < COLBSRDIM; k++)
            {
                sum = rocsparse::fma(bsr_val[COLBSRDIM * j + k], x[col + k], sum);
            }
        }

        // Each wavefront accumulates its BSR block row sum
        sum = rocsparse::wfreduce_sum<WFSIZE>(sum);

        // Last lane of each wavefront writes the two row sums to global memory
        if(lid == WFSIZE - 1)
        {
            if(beta != static_cast<T>(0))
            {
                y[row] = rocsparse::fma(beta, y[row], alpha * sum);
            }
            else
            {
                y[row] = alpha * sum;
            }
        }
    }

    // GEBSRMV kernel for GEBSR block dimension of 2 x n
    template <uint32_t BLOCKSIZE, uint32_t COLBSRDIM, uint32_t WFSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void gebsrmvn_2xn_device(rocsparse_int       mb,
                                                  rocsparse_direction dir,
                                                  T                   alpha,
                                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                                  const T* __restrict__ bsr_val,
                                                  const T* __restrict__ x,
                                                  T beta,
                                                  T* __restrict__ y,
                                                  rocsparse_index_base idx_base)
    {
        // GEBSR block dimension
        static constexpr int ROWBSRDIM = 2;

        // Lane id
        const rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

        // Wavefront id
        const rocsparse_int wid = hipThreadIdx_x / WFSIZE;

        // Each thread block processes (BLOCKSIZE / WFSIZE) BSR rows
        const rocsparse_int row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

        // Do not run out of bounds
        if(row >= mb)
        {
            return;
        }

        // BSR row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // BSR block row accumulator
        T sum0 = static_cast<T>(0);
        T sum1 = static_cast<T>(0);

        // Loop over all BSR blocks in the current row where each lane
        // processes a BSR block
        if(dir == rocsparse_direction_row)
        {
            for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Do not exceed the row
                // Column index into x vector
                const rocsparse_int col = (bsr_col_ind[j] - idx_base);

                // Compute the sum of the two rows within the BSR blocks of the current
                // BSR row
                for(uint32_t k = 0; k < COLBSRDIM; k++)
                {
                    sum0 = rocsparse::fma(
                        bsr_val[ROWBSRDIM * COLBSRDIM * j + k], x[COLBSRDIM * col + k], sum0);
                    sum1 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + COLBSRDIM + k],
                                          x[COLBSRDIM * col + k],
                                          sum1);
                }
            }
        }
        else
        {
            for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Do not exceed the row
                // Column index into x vector
                const rocsparse_int col = (bsr_col_ind[j] - idx_base);

                // Compute the sum of the two rows within the BSR blocks of the current
                // BSR row
                for(uint32_t k = 0; k < COLBSRDIM; k++)
                {
                    sum0 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + ROWBSRDIM * k],
                                          x[COLBSRDIM * col + k],
                                          sum0);
                    sum1 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + ROWBSRDIM * k + 1],
                                          x[COLBSRDIM * col + k],
                                          sum1);
                }
            }
        }
        // Each wavefront accumulates its BSR block row sum
        sum0 = rocsparse::wfreduce_sum<WFSIZE>(sum0);
        sum1 = rocsparse::wfreduce_sum<WFSIZE>(sum1);

        // Last lane of each wavefront writes the two row sums to global memory
        if(lid == WFSIZE - 1)
        {
            if(beta != static_cast<T>(0))
            {
                y[row * ROWBSRDIM + 0] = rocsparse::fma(beta, y[row * ROWBSRDIM + 0], alpha * sum0);
                y[row * ROWBSRDIM + 1] = rocsparse::fma(beta, y[row * ROWBSRDIM + 1], alpha * sum1);
            }
            else
            {
                y[row * ROWBSRDIM + 0] = alpha * sum0;
                y[row * ROWBSRDIM + 1] = alpha * sum1;
            }
        }
    }

    // GEBSRMV kernel for GEBSR block dimension of 3 x n
    template <uint32_t BLOCKSIZE, uint32_t COLBSRDIM, uint32_t WFSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void gebsrmvn_3xn_device(rocsparse_int       mb,
                                                  rocsparse_direction dir,
                                                  T                   alpha,
                                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                                  const T* __restrict__ bsr_val,
                                                  const T* __restrict__ x,
                                                  T beta,
                                                  T* __restrict__ y,
                                                  rocsparse_index_base idx_base)
    {
        // GEBSR block dimension
        static constexpr int ROWBSRDIM = 3;

        // Lane id
        const rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

        // Wavefront id
        const rocsparse_int wid = hipThreadIdx_x / WFSIZE;

        // Each thread block processes (BLOCKSIZE / WFSIZE) BSR rows
        const rocsparse_int row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

        // Do not run out of bounds
        if(row >= mb)
        {
            return;
        }

        // BSR row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // BSR block row accumulator
        T sum0 = static_cast<T>(0);
        T sum1 = static_cast<T>(0);
        T sum2 = static_cast<T>(0);

        // Loop over all BSR blocks in the current row where each lane
        // processes a BSR block
        if(dir == rocsparse_direction_row)
        {
            for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Do not exceed the row
                // Column index into x vector
                const rocsparse_int col = (bsr_col_ind[j] - idx_base);

                // Compute the sum of the two rows within the BSR blocks of the current
                // BSR row
                for(uint32_t k = 0; k < COLBSRDIM; k++)
                {
                    sum0 = rocsparse::fma(
                        bsr_val[ROWBSRDIM * COLBSRDIM * j + k], x[COLBSRDIM * col + k], sum0);
                    sum1 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + COLBSRDIM + k],
                                          x[COLBSRDIM * col + k],
                                          sum1);
                    sum2 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + 2 * COLBSRDIM + k],
                                          x[COLBSRDIM * col + k],
                                          sum2);
                }
            }
        }
        else
        {
            for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Do not exceed the row
                // Column index into x vector
                const rocsparse_int col = (bsr_col_ind[j] - idx_base);

                // Compute the sum of the two rows within the BSR blocks of the current
                // BSR row
                for(uint32_t k = 0; k < COLBSRDIM; k++)
                {
                    sum0 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + ROWBSRDIM * k],
                                          x[COLBSRDIM * col + k],
                                          sum0);
                    sum1 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + ROWBSRDIM * k + 1],
                                          x[COLBSRDIM * col + k],
                                          sum1);
                    sum2 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + ROWBSRDIM * k + 2],
                                          x[COLBSRDIM * col + k],
                                          sum2);
                }
            }
        }

        // Each wavefront accumulates its BSR block row sum
        sum0 = rocsparse::wfreduce_sum<WFSIZE>(sum0);
        sum1 = rocsparse::wfreduce_sum<WFSIZE>(sum1);
        sum2 = rocsparse::wfreduce_sum<WFSIZE>(sum2);

        // Last lane of each wavefront writes the two row sums to global memory
        if(lid == WFSIZE - 1)
        {
            if(beta != static_cast<T>(0))
            {
                y[row * ROWBSRDIM + 0] = rocsparse::fma(beta, y[row * ROWBSRDIM + 0], alpha * sum0);
                y[row * ROWBSRDIM + 1] = rocsparse::fma(beta, y[row * ROWBSRDIM + 1], alpha * sum1);
                y[row * ROWBSRDIM + 2] = rocsparse::fma(beta, y[row * ROWBSRDIM + 2], alpha * sum2);
            }
            else
            {
                y[row * ROWBSRDIM + 0] = alpha * sum0;
                y[row * ROWBSRDIM + 1] = alpha * sum1;
                y[row * ROWBSRDIM + 2] = alpha * sum2;
            }
        }
    }

    // GEBSRMV kernel for GEBSR block dimension of 4 x n
    template <uint32_t BLOCKSIZE, uint32_t COLBSRDIM, uint32_t WFSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void gebsrmvn_4xn_device(rocsparse_int       mb,
                                                  rocsparse_direction dir,
                                                  T                   alpha,
                                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                                  const T* __restrict__ bsr_val,
                                                  const T* __restrict__ x,
                                                  T beta,
                                                  T* __restrict__ y,
                                                  rocsparse_index_base idx_base)
    {
        // GEBSR block dimension
        static constexpr int ROWBSRDIM = 4;

        // Lane id
        const rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

        // Wavefront id
        const rocsparse_int wid = hipThreadIdx_x / WFSIZE;

        // Each thread block processes (BLOCKSIZE / WFSIZE) BSR rows
        const rocsparse_int row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

        // Do not run out of bounds
        if(row >= mb)
        {
            return;
        }

        // BSR row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // BSR block row accumulator
        T sum0 = static_cast<T>(0);
        T sum1 = static_cast<T>(0);
        T sum2 = static_cast<T>(0);
        T sum3 = static_cast<T>(0);

        // Loop over all BSR blocks in the current row where each lane
        // processes a BSR block
        if(dir == rocsparse_direction_row)
        {
            for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Do not exceed the row
                // Column index into x vector
                const rocsparse_int col = (bsr_col_ind[j] - idx_base);

                // Compute the sum of the two rows within the BSR blocks of the current
                // BSR row
                for(uint32_t k = 0; k < COLBSRDIM; k++)
                {
                    sum0 = rocsparse::fma(
                        bsr_val[ROWBSRDIM * COLBSRDIM * j + k], x[COLBSRDIM * col + k], sum0);
                    sum1 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + COLBSRDIM + k],
                                          x[COLBSRDIM * col + k],
                                          sum1);
                    sum2 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + 2 * COLBSRDIM + k],
                                          x[COLBSRDIM * col + k],
                                          sum2);
                    sum3 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + 3 * COLBSRDIM + k],
                                          x[COLBSRDIM * col + k],
                                          sum3);
                }
            }
        }
        else
        {
            for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Do not exceed the row
                // Column index into x vector
                const rocsparse_int col = (bsr_col_ind[j] - idx_base);

                // Compute the sum of the two rows within the BSR blocks of the current
                // BSR row
                for(uint32_t k = 0; k < COLBSRDIM; k++)
                {
                    sum0 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + ROWBSRDIM * k],
                                          x[COLBSRDIM * col + k],
                                          sum0);
                    sum1 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + ROWBSRDIM * k + 1],
                                          x[COLBSRDIM * col + k],
                                          sum1);
                    sum2 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + ROWBSRDIM * k + 2],
                                          x[COLBSRDIM * col + k],
                                          sum2);
                    sum3 = rocsparse::fma(bsr_val[ROWBSRDIM * COLBSRDIM * j + ROWBSRDIM * k + 3],
                                          x[COLBSRDIM * col + k],
                                          sum3);
                }
            }
        }

        // Each wavefront accumulates its BSR block row sum
        sum0 = rocsparse::wfreduce_sum<WFSIZE>(sum0);
        sum1 = rocsparse::wfreduce_sum<WFSIZE>(sum1);
        sum2 = rocsparse::wfreduce_sum<WFSIZE>(sum2);
        sum3 = rocsparse::wfreduce_sum<WFSIZE>(sum3);

        // Last lane of each wavefront writes the two row sums to global memory
        if(lid == WFSIZE - 1)
        {
            if(beta != static_cast<T>(0))
            {
                y[row * ROWBSRDIM + 0] = rocsparse::fma(beta, y[row * ROWBSRDIM + 0], alpha * sum0);
                y[row * ROWBSRDIM + 1] = rocsparse::fma(beta, y[row * ROWBSRDIM + 1], alpha * sum1);
                y[row * ROWBSRDIM + 2] = rocsparse::fma(beta, y[row * ROWBSRDIM + 2], alpha * sum2);
                y[row * ROWBSRDIM + 3] = rocsparse::fma(beta, y[row * ROWBSRDIM + 3], alpha * sum3);
            }
            else
            {
                y[row * ROWBSRDIM + 0] = alpha * sum0;
                y[row * ROWBSRDIM + 1] = alpha * sum1;
                y[row * ROWBSRDIM + 2] = alpha * sum2;
                y[row * ROWBSRDIM + 3] = alpha * sum3;
            }
        }
    }

    // GEBSRMV kernel for GEBSR block dimension of m x n where m and n <= 8
    template <uint32_t BLOCKSIZE, uint32_t ROWBSRDIM, uint32_t COLBSRDIM, typename T>
    ROCSPARSE_DEVICE_ILF void gebsrmvn_mxn_device(rocsparse_int       mb,
                                                  rocsparse_direction dir,
                                                  T                   alpha,
                                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                                  const T* __restrict__ bsr_val,
                                                  const T* __restrict__ x,
                                                  T beta,
                                                  T* __restrict__ y,
                                                  rocsparse_index_base idx_base)
    {
        // BSR block lane id
        const rocsparse_int lid = hipThreadIdx_x % COLBSRDIM;

        // Each thread block processes a single BSR row
        const rocsparse_int row = hipBlockIdx_x;

        // Offset into x vector
        const rocsparse_int idx = (dir == rocsparse_direction_column)
                                      ? ((hipThreadIdx_x / ROWBSRDIM) % COLBSRDIM)
                                      : lid;

        // Number of BSR blocks processed at the same time
        constexpr uint32_t NBLOCKS = BLOCKSIZE / (ROWBSRDIM * COLBSRDIM);

        // BSR row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // BSR block row accumulator
        T sum = static_cast<T>(0);

        // Loop over all BSR blocks in the current row where each lane
        // processes a BSR block value
        for(rocsparse_int j = row_begin; j < row_end; j += NBLOCKS)
        {
            const rocsparse_int k = j + hipThreadIdx_x / (ROWBSRDIM * COLBSRDIM);

            // Do not exceed the row
            if(k < row_end)
            {
                // Column index into x vector
                const rocsparse_int col = (bsr_col_ind[k] - idx_base) * COLBSRDIM;

                // Compute the sum of the two rows within the BSR blocks of the current
                // BSR row
                sum = rocsparse::fma(
                    bsr_val[j * ROWBSRDIM * COLBSRDIM + hipThreadIdx_x], x[col + idx], sum);
            }
        }

        // Accumulate each row sum of the BSR block
        __shared__ T sdata[ROWBSRDIM * COLBSRDIM * NBLOCKS];

        sdata[hipThreadIdx_x] = sum;

        __syncthreads();

        // Handle case when NBLOCKS is not power of 2 when accumulating the row sum for different blocks
        if((NBLOCKS & (NBLOCKS - 1)) != 0)
        {
            // Find highest power of 2 less than NBLOCKS
            uint32_t POW2 = NBLOCKS;
            POW2--;
            POW2 |= POW2 >> 1;
            POW2 |= POW2 >> 2;
            POW2 |= POW2 >> 4;
            POW2 |= POW2 >> 8;
            POW2++;
            POW2 >>= 1;

            if(hipThreadIdx_x < NBLOCKS * ROWBSRDIM * COLBSRDIM - POW2 * ROWBSRDIM * COLBSRDIM)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + POW2 * ROWBSRDIM * COLBSRDIM];
            __syncthreads();
        }

        // Finish accumulating the row sum for different blocks
        if(NBLOCKS >= 8)
        {
            if(hipThreadIdx_x < 4 * ROWBSRDIM * COLBSRDIM)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4 * ROWBSRDIM * COLBSRDIM];
            __syncthreads();
        }
        if(NBLOCKS >= 4)
        {
            if(hipThreadIdx_x < 2 * ROWBSRDIM * COLBSRDIM)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2 * ROWBSRDIM * COLBSRDIM];
            __syncthreads();
        }
        if(NBLOCKS >= 2)
        {
            if(hipThreadIdx_x < 1 * ROWBSRDIM * COLBSRDIM)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 1 * ROWBSRDIM * COLBSRDIM];
            __syncthreads();
        }

        if(dir == rocsparse_direction_column)
        {
            // Handle case when COLBSRDIM is not power of 2 when reducing the intra block row sum
            if((COLBSRDIM & (COLBSRDIM - 1)) != 0)
            {
                // Find highest power of 2 less than COLBSRDIM
                uint32_t POW2 = COLBSRDIM;
                POW2--;
                POW2 |= POW2 >> 1;
                POW2 |= POW2 >> 2;
                POW2 |= POW2 >> 4;
                POW2 |= POW2 >> 8;
                POW2++;
                POW2 >>= 1;

                if(hipThreadIdx_x < ROWBSRDIM * COLBSRDIM - ROWBSRDIM * POW2)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + ROWBSRDIM * POW2];
                __threadfence_block();
            }

            // Finish reducing the intra block row sum
            if(COLBSRDIM >= 8)
            {
                if(hipThreadIdx_x < ROWBSRDIM * 4)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + ROWBSRDIM * 4];
                __threadfence_block();
            }
            if(COLBSRDIM >= 4)
            {
                if(hipThreadIdx_x < ROWBSRDIM * 2)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + ROWBSRDIM * 2];
                __threadfence_block();
            }
            if(COLBSRDIM >= 2)
            {
                if(hipThreadIdx_x < ROWBSRDIM * 1)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + ROWBSRDIM * 1];
                __threadfence_block();
            }
            if(COLBSRDIM >= 1)
            {
                if(hipThreadIdx_x < ROWBSRDIM * 1)
                    sum = sdata[hipThreadIdx_x];
            }
        }
        else
        {
            // Handle case when COLBSRDIM is not power of 2 when reducing the intra block row sum
            if((COLBSRDIM & (COLBSRDIM - 1)) != 0)
            {
                // Find highest power of 2 less than COLBSRDIM
                uint32_t POW2 = COLBSRDIM;
                POW2--;
                POW2 |= POW2 >> 1;
                POW2 |= POW2 >> 2;
                POW2 |= POW2 >> 4;
                POW2 |= POW2 >> 8;
                POW2++;
                POW2 >>= 1;

                if(lid < COLBSRDIM - POW2)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + POW2];
                __threadfence_block();
            }

            // Finish reducing the intra block row sum
            if(COLBSRDIM >= 8)
            {
                if(lid < 4)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
                __threadfence_block();
            }
            if(COLBSRDIM >= 4)
            {
                if(lid < 2)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];
                __threadfence_block();
            }
            if(COLBSRDIM >= 2)
            {
                if(lid < 1)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 1];
                __threadfence_block();
            }
            if(COLBSRDIM >= 1)
            {
                if(hipThreadIdx_x < ROWBSRDIM)
                    sum = sdata[hipThreadIdx_x * COLBSRDIM];
            }
        }

        // First 8 threads write row sums to global memory
        if(hipThreadIdx_x < ROWBSRDIM)
        {
            if(beta != static_cast<T>(0))
            {
                y[row * ROWBSRDIM + hipThreadIdx_x]
                    = rocsparse::fma(beta, y[row * ROWBSRDIM + hipThreadIdx_x], alpha * sum);
            }
            else
            {
                y[row * ROWBSRDIM + hipThreadIdx_x] = alpha * sum;
            }
        }
    }

    // GEBSRMV kernel for GEBSR block dimension of m x n where m and n <= 16
    template <uint32_t BLOCKSIZE, uint32_t ROWBSRDIM, uint32_t COLBSRDIM, typename T>
    ROCSPARSE_DEVICE_ILF void gebsrmvn_mxn_16_device(rocsparse_int       mb,
                                                     rocsparse_direction dir,
                                                     T                   alpha,
                                                     const rocsparse_int* __restrict__ bsr_row_ptr,
                                                     const rocsparse_int* __restrict__ bsr_col_ind,
                                                     const T* __restrict__ bsr_val,
                                                     const T* __restrict__ x,
                                                     T beta,
                                                     T* __restrict__ y,
                                                     rocsparse_index_base idx_base)
    {
        // BSR block lane id
        const rocsparse_int lid = hipThreadIdx_x % COLBSRDIM;

        // Each thread block processes a single BSR row
        const rocsparse_int row = hipBlockIdx_x;

        // Offset into x vector
        const rocsparse_int idx = (dir == rocsparse_direction_column)
                                      ? ((hipThreadIdx_x / ROWBSRDIM) % COLBSRDIM)
                                      : lid;

        // Number of BSR blocks processed at the same time
        constexpr uint32_t NBLOCKS = BLOCKSIZE / (ROWBSRDIM * COLBSRDIM);

        // BSR row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // BSR block row accumulator
        T sum = static_cast<T>(0);

        // Loop over all BSR blocks in the current row where each lane
        // processes a BSR block value
        for(rocsparse_int j = row_begin; j < row_end; j += NBLOCKS)
        {
            const rocsparse_int k = j + hipThreadIdx_x / (ROWBSRDIM * COLBSRDIM);

            // Do not exceed the row
            if(k < row_end)
            {
                // Column index into x vector
                const rocsparse_int col = (bsr_col_ind[k] - idx_base) * COLBSRDIM;

                // Compute the sum of the two rows within the BSR blocks of the current
                // BSR row
                sum = rocsparse::fma(
                    bsr_val[j * ROWBSRDIM * COLBSRDIM + hipThreadIdx_x], x[col + idx], sum);
            }
        }

        // Accumulate each row sum of the BSR block
        __shared__ T sdata[ROWBSRDIM * COLBSRDIM * NBLOCKS];

        sdata[hipThreadIdx_x] = sum;

        __syncthreads();

        // Handle case when NBLOCKS is not power of 2 when accumulating the row sum for different blocks
        if((NBLOCKS & (NBLOCKS - 1)) != 0)
        {
            // Find highest power of 2 less than NBLOCKS
            uint32_t POW2 = NBLOCKS;
            POW2--;
            POW2 |= POW2 >> 1;
            POW2 |= POW2 >> 2;
            POW2 |= POW2 >> 4;
            POW2 |= POW2 >> 8;
            POW2++;
            POW2 >>= 1;

            if(hipThreadIdx_x < NBLOCKS * ROWBSRDIM * COLBSRDIM - POW2 * ROWBSRDIM * COLBSRDIM)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + POW2 * ROWBSRDIM * COLBSRDIM];
            __syncthreads();
        }

        // Finish accumulating the row sum for different blocks
        if(NBLOCKS >= 8)
        {
            if(hipThreadIdx_x < 4 * ROWBSRDIM * COLBSRDIM)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4 * ROWBSRDIM * COLBSRDIM];
            __syncthreads();
        }
        if(NBLOCKS >= 4)
        {
            if(hipThreadIdx_x < 2 * ROWBSRDIM * COLBSRDIM)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2 * ROWBSRDIM * COLBSRDIM];
            __syncthreads();
        }
        if(NBLOCKS >= 2)
        {
            if(hipThreadIdx_x < 1 * ROWBSRDIM * COLBSRDIM)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 1 * ROWBSRDIM * COLBSRDIM];
            __syncthreads();
        }

        if(dir == rocsparse_direction_column)
        {
            // Handle case when COLBSRDIM is not power of 2 when reducing the intra block row sum
            if((COLBSRDIM & (COLBSRDIM - 1)) != 0)
            {
                // Find highest power of 2 less than COLBSRDIM
                uint32_t POW2 = COLBSRDIM;
                POW2--;
                POW2 |= POW2 >> 1;
                POW2 |= POW2 >> 2;
                POW2 |= POW2 >> 4;
                POW2 |= POW2 >> 8;
                POW2++;
                POW2 >>= 1;

                if(hipThreadIdx_x < ROWBSRDIM * COLBSRDIM - ROWBSRDIM * POW2)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + ROWBSRDIM * POW2];
                __syncthreads();
            }

            // Finish reducing the intra block row sum
            if(COLBSRDIM >= 16)
            {
                if(hipThreadIdx_x < ROWBSRDIM * 8)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + ROWBSRDIM * 8];
                __syncthreads();
            }
            if(COLBSRDIM >= 8)
            {
                if(hipThreadIdx_x < ROWBSRDIM * 4)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + ROWBSRDIM * 4];
                __syncthreads();
            }
            if(COLBSRDIM >= 4)
            {
                if(hipThreadIdx_x < ROWBSRDIM * 2)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + ROWBSRDIM * 2];
                __threadfence_block();
            }
            if(COLBSRDIM >= 2)
            {
                if(hipThreadIdx_x < ROWBSRDIM * 1)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + ROWBSRDIM * 1];
                __threadfence_block();
            }
            if(COLBSRDIM >= 1)
            {
                if(hipThreadIdx_x < ROWBSRDIM * 1)
                    sum = sdata[hipThreadIdx_x];
            }
        }
        else
        {
            // Handle case when COLBSRDIM is not power of 2 when reducing the intra block row sum
            if((COLBSRDIM & (COLBSRDIM - 1)) != 0)
            {
                // Find highest power of 2 less than COLBSRDIM
                uint32_t POW2 = COLBSRDIM;
                POW2--;
                POW2 |= POW2 >> 1;
                POW2 |= POW2 >> 2;
                POW2 |= POW2 >> 4;
                POW2 |= POW2 >> 8;
                POW2++;
                POW2 >>= 1;

                if(lid < COLBSRDIM - POW2)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + POW2];
                __syncthreads();
            }

            // Finish reducing the intra block row sum
            if(COLBSRDIM >= 16)
            {
                if(lid < 8)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 8];
                __syncthreads();
            }
            if(COLBSRDIM >= 8)
            {
                if(lid < 4)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
                __syncthreads();
            }
            if(COLBSRDIM >= 4)
            {
                if(lid < 2)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];
                __syncthreads();
            }
            if(COLBSRDIM >= 2)
            {
                if(lid < 1)
                    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 1];
                __syncthreads();
            }
            if(COLBSRDIM >= 1)
            {
                if(hipThreadIdx_x < ROWBSRDIM)
                    sum = sdata[hipThreadIdx_x * COLBSRDIM];
            }
        }

        // First 16 threads write row sums to global memory
        if(hipThreadIdx_x < ROWBSRDIM)
        {
            if(beta != static_cast<T>(0))
            {
                y[row * ROWBSRDIM + hipThreadIdx_x]
                    = rocsparse::fma(beta, y[row * ROWBSRDIM + hipThreadIdx_x], alpha * sum);
            }
            else
            {
                y[row * ROWBSRDIM + hipThreadIdx_x] = alpha * sum;
            }
        }
    }
}
