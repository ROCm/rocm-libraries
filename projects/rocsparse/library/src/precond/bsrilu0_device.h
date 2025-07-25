/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t BSRDIM, typename T>
    ROCSPARSE_DEVICE_ILF void bsrilu0_2_8_device(rocsparse_direction  dir,
                                                 rocsparse_int        mb,
                                                 const rocsparse_int* bsr_row_ptr,
                                                 const rocsparse_int* bsr_col_ind,
                                                 T*                   bsr_val,
                                                 const rocsparse_int* bsr_diag_ind,
                                                 rocsparse_int        block_dim,
                                                 int*                 done_array,
                                                 const rocsparse_int* map,
                                                 rocsparse_int*       zero_pivot,
                                                 rocsparse_index_base idx_base,
                                                 int                  boost,
                                                 double               boost_tol,
                                                 T                    boost_val)
    {
        // Current row this wavefront is working on
        rocsparse_int row = map[blockIdx.x];

        // Diagonal entry point of the current row
        rocsparse_int row_diag = bsr_diag_ind[row];

        // Row entry point
        rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Zero pivot tracker
        bool pivot = false;

        // Shared memory to cache BSR values
        __shared__ T sdata1[BSRDIM][BSRDIM + 1];
        __shared__ T sdata2[BSRDIM][BSRDIM + 1];

        // Check for structural pivot
        if(row_diag != -1)
        {
            // Process lower diagonal
            for(rocsparse_int j = row_begin; j < row_diag; ++j)
            {
                // Column index of current BSR block
                rocsparse_int bsr_col = bsr_col_ind[j] - idx_base;

                // Load row j into shared memory
                sdata2[threadIdx.y][threadIdx.x]
                    = (threadIdx.x < block_dim && threadIdx.y < block_dim)
                          ? bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                          : static_cast<T>(0);

                // Process all lower matrix BSR blocks

                // Obtain corresponding row entry and exit point that corresponds with the
                // current BSR column. Actually, we skip all lower matrix column indices,
                // therefore starting with the diagonal entry.
                rocsparse_int diag_j    = bsr_diag_ind[bsr_col];
                rocsparse_int row_end_j = bsr_row_ptr[bsr_col + 1] - idx_base;

                // Check for structural pivot
                if(diag_j == -1)
                {
                    pivot = true;
                    break;
                }

                // Spin loop until dependency has been resolved
                while(!__hip_atomic_load(
                    &done_array[bsr_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                    ;

                // Make sure dependencies are visible in global memory
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

                // Load updated BSR block into shared memory
                sdata1[threadIdx.y][threadIdx.x]
                    = (threadIdx.x < block_dim && threadIdx.y < block_dim)
                          ? bsr_val[BSR_IND(diag_j, threadIdx.x, threadIdx.y, dir)]
                          : static_cast<T>(0);

                // Make sure all writes to shared memory are visible
                __threadfence_block();

                // Loop through all rows within the BSR block
                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal entry of the BSR block
                    T diag = sdata1[bi][bi];
                    T val  = sdata2[bi][threadIdx.x];

                    // This has already been checked for zero by previous computations
                    val /= diag;

                    // Make sure val has been read before updating
                    __threadfence_block();

                    // Update
                    if(threadIdx.y == 0)
                    {
                        sdata2[bi][threadIdx.x] = val;
                    }

                    // Do linear combination
                    rocsparse_int bj = bi + 1 + threadIdx.y;
                    if(bj < block_dim)
                    {
                        sdata2[bj][threadIdx.x]
                            = rocsparse::fma(-val, sdata1[bj][bi], sdata2[bj][threadIdx.x]);
                    }

                    __threadfence_block();
                }

                // Write row j back to global memory
                if(threadIdx.x < block_dim && threadIdx.y < block_dim)
                {
                    bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                        = sdata2[threadIdx.y][threadIdx.x];
                }

                // Loop over upper offset pointer and do linear combination for nnz entry
                for(rocsparse_int k = diag_j + 1; k < row_end_j; ++k)
                {
                    rocsparse_int bsr_col_k = bsr_col_ind[k] - idx_base;

                    // Search for matching column index in current row
                    rocsparse_int q         = row_begin + threadIdx.x + threadIdx.y * blockDim.x;
                    rocsparse_int bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;

                    // Check if match has been found by any thread in the wavefront
                    while(bsr_col_j < bsr_col_k)
                    {
                        q += WFSIZE;
                        bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;
                    }

                    // Check if match has been found by any thread in the wavefront
                    int match = __ffsll(__ballot(bsr_col_j == bsr_col_k));

                    // If match has been found, process it
                    if(match)
                    {
                        // Tell all other threads about the matching index
                        rocsparse_int m = rocsparse::shfl(q, match - 1);

                        // Load BSR block from row k into shared memory
                        sdata1[threadIdx.y][threadIdx.x]
                            = (threadIdx.x < block_dim && threadIdx.y < block_dim)
                                  ? bsr_val[BSR_IND(k, threadIdx.x, threadIdx.y, dir)]
                                  : static_cast<T>(0);

                        // Make sure all writes to shared memory are visible
                        __threadfence_block();

                        T sum = static_cast<T>(0);

                        for(rocsparse_int bk = 0; bk < block_dim; ++bk)
                        {
                            sum = rocsparse::fma(
                                sdata2[bk][threadIdx.x], sdata1[threadIdx.y][bk], sum);
                        }

                        // Write back to global row m
                        if(threadIdx.x < block_dim && threadIdx.y < block_dim)
                        {
                            // Do not pre-cache row m as we read/write only once
                            bsr_val[BSR_IND(m, threadIdx.x, threadIdx.y, dir)] -= sum;
                        }
                    }

                    __threadfence_block();
                }
            }

            // Process diagonal
            if(bsr_col_ind[row_diag] - idx_base == row)
            {
                // Load diagonal BSR block into shared memory
                sdata1[threadIdx.y][threadIdx.x]
                    = (threadIdx.x < block_dim && threadIdx.y < block_dim)
                          ? bsr_val[BSR_IND(row_diag, threadIdx.x, threadIdx.y, dir)]
                          : static_cast<T>(0);

                __threadfence_block();

                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal matrix entry
                    T diag = sdata1[bi][bi];

                    // Numeric boost
                    if(boost)
                    {
                        diag = (boost_tol >= rocsparse::abs(diag)) ? boost_val : diag;

                        __threadfence_block();

                        if(threadIdx.x == 0 && threadIdx.y == 0)
                        {
                            sdata1[bi][bi] = diag;
                        }
                    }
                    else
                    {
                        // Check for numeric pivot
                        if(diag == static_cast<T>(0))
                        {
                            pivot = true;
                            continue;
                        }
                    }

                    rocsparse_int bk = bi + 1 + threadIdx.x;
                    if(bk < block_dim)
                    {
                        // Multiplication factor
                        T val = sdata1[bi][bk];
                        val /= diag;

                        // Make sure val has been read before updating
                        __threadfence_block();

                        // Update
                        if(threadIdx.y == 0)
                        {
                            sdata1[bi][bk] = val;
                        }

                        // Do linear combination
                        rocsparse_int bj = bi + 1 + threadIdx.y;
                        if(bj < block_dim)
                        {
                            sdata1[bj][bk] = rocsparse::fma(-val, sdata1[bj][bi], sdata1[bj][bk]);
                        }
                    }
                }

                __threadfence_block();

                // Write diagonal BSR block back to global memory
                if(threadIdx.x < block_dim && threadIdx.y < block_dim)
                {
                    bsr_val[BSR_IND(row_diag, threadIdx.x, threadIdx.y, dir)]
                        = sdata1[threadIdx.y][threadIdx.x];
                }
            }

            // Process upper diagonal BSR blocks
            for(rocsparse_int j = row_diag + 1; j < row_end; ++j)
            {
                __threadfence_block();

                // Load row j into shared memory
                sdata2[threadIdx.y][threadIdx.x]
                    = (threadIdx.x < block_dim && threadIdx.y < block_dim)
                          ? bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                          : static_cast<T>(0);

                __threadfence_block();

                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    rocsparse_int bj = bi + 1 + threadIdx.y;
                    if(bj < block_dim)
                    {
                        sdata2[threadIdx.x][bj] = rocsparse::fma(
                            -sdata1[bi][bj], sdata2[threadIdx.x][bi], sdata2[threadIdx.x][bj]);
                    }
                }

                __threadfence_block();

                // Write row j back to global memory
                if(threadIdx.x < block_dim && threadIdx.y < block_dim)
                {
                    bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                        = sdata2[threadIdx.y][threadIdx.x];
                }
            }
        }
        else
        {
            // Structural pivot found
            pivot = true;
        }

        if(threadIdx.x == 0 && threadIdx.y == 0)
        {
            // First lane writes "we are done" flag
            __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot)
            {
                // Atomically set minimum zero pivot, if found
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t BSRDIM, typename T>
    ROCSPARSE_DEVICE_ILF void bsrilu0_9_32_device(rocsparse_direction  dir,
                                                  rocsparse_int        mb,
                                                  const rocsparse_int* bsr_row_ptr,
                                                  const rocsparse_int* bsr_col_ind,
                                                  T*                   bsr_val,
                                                  const rocsparse_int* bsr_diag_ind,
                                                  rocsparse_int        block_dim,
                                                  int*                 done_array,
                                                  const rocsparse_int* map,
                                                  rocsparse_int*       zero_pivot,
                                                  rocsparse_index_base idx_base,
                                                  int                  boost,
                                                  double               boost_tol,
                                                  T                    boost_val)
    {
        constexpr static uint32_t DIMX = BSRDIM;
        constexpr static uint32_t DIMY = BLOCKSIZE / BSRDIM;

        // Current row this wavefront is working on
        rocsparse_int row = map[blockIdx.x];

        // Diagonal entry point of the current row
        rocsparse_int row_diag = bsr_diag_ind[row];

        // Row entry point
        rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Zero pivot tracker
        bool pivot = false;

        // Shared memory to cache BSR values
        __shared__ T sdata1[BSRDIM][BSRDIM + 1];
        __shared__ T sdata2[BSRDIM][BSRDIM + 1];

        // Check for structural pivot
        if(row_diag != -1)
        {
            // Process lower diagonal
            for(rocsparse_int j = row_begin; j < row_diag; ++j)
            {
                // Column index of current BSR block
                rocsparse_int bsr_col = bsr_col_ind[j] - idx_base;

                // Load row j into shared memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        sdata2[q][p] = bsr_val[BSR_IND(j, p, q, dir)];
                    }
                }

                // Process all lower matrix BSR blocks

                // Obtain corresponding row entry and exit point that corresponds with the
                // current BSR column. Actually, we skip all lower matrix column indices,
                // therefore starting with the diagonal entry.
                rocsparse_int diag_j    = bsr_diag_ind[bsr_col];
                rocsparse_int row_end_j = bsr_row_ptr[bsr_col + 1] - idx_base;

                // Check for structural pivot
                if(diag_j == -1)
                {
                    pivot = true;
                    break;
                }

                // Spin loop until dependency has been resolved
                while(!__hip_atomic_load(
                    &done_array[bsr_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                    ;

                // Make sure dependencies are visible in global memory
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

                // Load updated BSR block into shared memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        sdata1[q][p] = bsr_val[BSR_IND(diag_j, p, q, dir)];
                    }
                }

                // Make sure all writes to shared memory are visible
                __threadfence_block();

                // Loop through all rows within the BSR block
                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal entry of the BSR block
                    T diag = sdata1[bi][bi];

                    for(rocsparse_int bk = threadIdx.x; bk < block_dim; bk += DIMX)
                    {
                        T val = sdata2[bi][bk];

                        // This has already been checked for zero by previous computations
                        val /= diag;

                        // Make sure val has been read before updating
                        __threadfence_block();

                        // Update
                        if(threadIdx.y == 0)
                        {
                            sdata2[bi][bk] = val;
                        }

                        // Do linear combination
                        for(rocsparse_int bj = bi + 1 + threadIdx.y; bj < block_dim; bj += DIMY)
                        {
                            sdata2[bj][bk] = rocsparse::fma(-val, sdata1[bj][bi], sdata2[bj][bk]);
                        }
                    }

                    __threadfence_block();
                }

                // Write row j back to global memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        bsr_val[BSR_IND(j, p, q, dir)] = sdata2[q][p];
                    }
                }

                // Loop over upper offset pointer and do linear combination for nnz entry
                for(rocsparse_int k = diag_j + 1; k < row_end_j; ++k)
                {
                    rocsparse_int bsr_col_k = bsr_col_ind[k] - idx_base;

                    // Search for matching column index in current row
                    rocsparse_int q         = row_begin + threadIdx.x + threadIdx.y * DIMX;
                    rocsparse_int bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;

                    // Check if match has been found by any thread in the wavefront
                    while(bsr_col_j < bsr_col_k)
                    {
                        q += WFSIZE;
                        bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;
                    }

                    // Check if match has been found by any thread in the wavefront
                    int match = __ffsll(__ballot(bsr_col_j == bsr_col_k));

                    // If match has been found, process it
                    if(match)
                    {
                        // Tell all other threads about the matching index
                        rocsparse_int m = rocsparse::shfl(q, match - 1);

                        // Load BSR block from row k into shared memory
                        for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                        {
                            for(rocsparse_int qj = threadIdx.y; qj < block_dim; qj += DIMY)
                            {
                                sdata1[qj][p] = bsr_val[BSR_IND(k, p, qj, dir)];
                            }
                        }

                        // Make sure all writes to shared memory are visible
                        __threadfence_block();

                        for(rocsparse_int bi = threadIdx.x; bi < block_dim; bi += DIMX)
                        {
                            for(rocsparse_int bj = threadIdx.y; bj < block_dim; bj += DIMY)
                            {
                                T sum = static_cast<T>(0);

                                for(rocsparse_int bk = 0; bk < block_dim; ++bk)
                                {
                                    sum = rocsparse::fma(sdata2[bk][bi], sdata1[bj][bk], sum);
                                }

                                // Write back to global row m

                                // Do not pre-cache row m as we read/write only once
                                bsr_val[BSR_IND(m, bi, bj, dir)] -= sum;
                            }
                        }
                    }

                    __threadfence_block();
                }
            }

            // Process diagonal
            if(bsr_col_ind[row_diag] - idx_base == row)
            {
                // Load diagonal BSR block into shared memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        sdata1[q][p] = bsr_val[BSR_IND(row_diag, p, q, dir)];
                    }
                }

                __threadfence_block();

                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal matrix entry
                    T diag = sdata1[bi][bi];

                    // Numeric boost
                    if(boost)
                    {
                        diag = (boost_tol >= rocsparse::abs(diag)) ? boost_val : diag;

                        __threadfence_block();

                        if(threadIdx.x == 0 && threadIdx.y == 0)
                        {
                            sdata1[bi][bi] = diag;
                        }
                    }
                    else
                    {
                        // Check for numeric pivot
                        if(diag == static_cast<T>(0))
                        {
                            pivot = true;
                            continue;
                        }
                    }

                    for(rocsparse_int bk = bi + 1 + threadIdx.x; bk < block_dim; bk += DIMX)
                    {
                        // Multiplication factor
                        T val = sdata1[bi][bk];
                        val /= diag;

                        // Make sure val has been read before updating
                        __threadfence_block();

                        // Update
                        if(threadIdx.y == 0)
                        {
                            sdata1[bi][bk] = val;
                        }

                        // Do linear combination
                        for(rocsparse_int bj = bi + 1 + threadIdx.y; bj < block_dim; bj += DIMY)
                        {
                            sdata1[bj][bk] = rocsparse::fma(-val, sdata1[bj][bi], sdata1[bj][bk]);
                        }
                    }
                }

                __threadfence_block();

                // Write diagonal BSR block back to global memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        bsr_val[BSR_IND(row_diag, p, q, dir)] = sdata1[q][p];
                    }
                }
            }

            // Process upper diagonal BSR blocks
            for(rocsparse_int j = row_diag + 1; j < row_end; ++j)
            {
                __threadfence_block();

                // Load row j into shared memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        sdata2[q][p] = bsr_val[BSR_IND(j, p, q, dir)];
                    }
                }

                __threadfence_block();

                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    for(rocsparse_int bk = threadIdx.x; bk < block_dim; bk += DIMX)
                    {
                        for(rocsparse_int bj = bi + 1 + threadIdx.y; bj < block_dim; bj += DIMY)
                        {
                            sdata2[bk][bj]
                                = rocsparse::fma(-sdata1[bi][bj], sdata2[bk][bi], sdata2[bk][bj]);
                        }
                    }
                }

                __threadfence_block();

                // Write row j back to global memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        bsr_val[BSR_IND(j, p, q, dir)] = sdata2[q][p];
                    }
                }
            }
        }
        else
        {
            // Structural pivot found
            pivot = true;
        }

        if(threadIdx.x == 0 && threadIdx.y == 0)
        {
            // First lane writes "we are done" flag
            __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot)
            {
                // Atomically set minimum zero pivot, if found
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t BSRDIM, typename T>
    ROCSPARSE_DEVICE_ILF void bsrilu0_33_64_device(rocsparse_direction  dir,
                                                   rocsparse_int        mb,
                                                   const rocsparse_int* bsr_row_ptr,
                                                   const rocsparse_int* bsr_col_ind,
                                                   T*                   bsr_val,
                                                   const rocsparse_int* bsr_diag_ind,
                                                   rocsparse_int        block_dim,
                                                   int*                 done_array,
                                                   const rocsparse_int* map,
                                                   rocsparse_int*       zero_pivot,
                                                   rocsparse_index_base idx_base,
                                                   int                  boost,
                                                   double               boost_tol,
                                                   T                    boost_val)
    {
        constexpr static uint32_t DIMX = BSRDIM;
        constexpr static uint32_t DIMY = BLOCKSIZE / BSRDIM;

        // Current row this wavefront is working on
        rocsparse_int row = map[blockIdx.x];

        // Diagonal entry point of the current row
        rocsparse_int row_diag = bsr_diag_ind[row];

        // Row entry point
        rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Zero pivot tracker
        bool pivot = false;

        // Shared memory to cache BSR values
        __shared__ T sdata[BSRDIM][BSRDIM + 1];

        // Check for structural pivot
        if(row_diag != -1)
        {
            // Process lower diagonal
            for(rocsparse_int j = row_begin; j < row_diag; ++j)
            {
                // Column index of current BSR block
                rocsparse_int bsr_col = bsr_col_ind[j] - idx_base;

                // Process all lower matrix BSR blocks

                // Obtain corresponding row entry and exit point that corresponds with the
                // current BSR column. Actually, we skip all lower matrix column indices,
                // therefore starting with the diagonal entry.
                rocsparse_int diag_j    = bsr_diag_ind[bsr_col];
                rocsparse_int row_end_j = bsr_row_ptr[bsr_col + 1] - idx_base;

                // Check for structural pivot
                if(diag_j == -1)
                {
                    pivot = true;
                    break;
                }

                // Spin loop until dependency has been resolved
                while(!__hip_atomic_load(
                    &done_array[bsr_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                    ;

                // Make sure dependencies are visible in global memory
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

                // Load updated BSR block into shared memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        sdata[q][p] = bsr_val[BSR_IND(diag_j, p, q, dir)];
                    }
                }

                // Make sure all writes to shared memory are visible
                __threadfence_block();

                // Loop through all rows within the BSR block
                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal entry of the BSR block
                    T diag = sdata[bi][bi];

                    for(rocsparse_int bk = threadIdx.x; bk < block_dim; bk += DIMX)
                    {
                        T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                        // This has already been checked for zero by previous computations
                        val /= diag;

                        // Update
                        if(threadIdx.y == 0)
                        {
                            bsr_val[BSR_IND(j, bk, bi, dir)] = val;
                        }

                        // Do linear combination
                        for(rocsparse_int bj = bi + 1 + threadIdx.y; bj < block_dim; bj += DIMY)
                        {
                            bsr_val[BSR_IND(j, bk, bj, dir)] = rocsparse::fma(
                                -val, sdata[bj][bi], bsr_val[BSR_IND(j, bk, bj, dir)]);
                        }
                    }
                }

                __threadfence_block();

                // Loop over upper offset pointer and do linear combination for nnz entry
                for(rocsparse_int k = diag_j + 1; k < row_end_j; ++k)
                {
                    rocsparse_int bsr_col_k = bsr_col_ind[k] - idx_base;

                    // Search for matching column index in current row
                    rocsparse_int q         = row_begin + threadIdx.x + threadIdx.y * DIMX;
                    rocsparse_int bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;

                    // Check if match has been found by any thread in the wavefront
                    while(bsr_col_j < bsr_col_k)
                    {
                        q += WFSIZE;
                        bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;
                    }

                    // Check if match has been found by any thread in the wavefront
                    int match = __ffsll(__ballot(bsr_col_j == bsr_col_k));

                    // If match has been found, process it
                    if(match)
                    {
                        // Tell all other threads about the matching index
                        rocsparse_int m = rocsparse::shfl(q, match - 1);

                        // Load BSR block from row k into shared memory
                        for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                        {
                            for(rocsparse_int qj = threadIdx.y; qj < block_dim; qj += DIMY)
                            {
                                sdata[qj][p] = bsr_val[BSR_IND(k, p, qj, dir)];
                            }
                        }

                        // Make sure all writes to shared memory are visible
                        __threadfence_block();

                        for(rocsparse_int bi = threadIdx.x; bi < block_dim; bi += DIMX)
                        {
                            for(rocsparse_int bj = threadIdx.y; bj < block_dim; bj += DIMY)
                            {
                                T sum = static_cast<T>(0);

                                for(rocsparse_int bk = 0; bk < block_dim; ++bk)
                                {
                                    sum = rocsparse::fma(
                                        bsr_val[BSR_IND(j, bi, bk, dir)], sdata[bj][bk], sum);
                                }

                                // Write back to global row m

                                // Do not pre-cache row m as we read/write only once
                                bsr_val[BSR_IND(m, bi, bj, dir)] -= sum;
                            }
                        }
                    }
                }
            }

            __threadfence_block();

            // Process diagonal
            if(bsr_col_ind[row_diag] - idx_base == row)
            {
                // Load diagonal BSR block into shared memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        sdata[q][p] = bsr_val[BSR_IND(row_diag, p, q, dir)];
                    }
                }

                __threadfence_block();

                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal matrix entry
                    T diag = sdata[bi][bi];

                    // Numeric boost
                    if(boost)
                    {
                        diag = (boost_tol >= rocsparse::abs(diag)) ? boost_val : diag;

                        __threadfence_block();

                        if(threadIdx.x == 0 && threadIdx.y == 0)
                        {
                            sdata[bi][bi] = diag;
                        }
                    }
                    else
                    {
                        // Check for numeric pivot
                        if(diag == static_cast<T>(0))
                        {
                            pivot = true;
                            continue;
                        }
                    }

                    for(rocsparse_int bk = bi + 1 + threadIdx.x; bk < block_dim; bk += DIMX)
                    {
                        // Multiplication factor
                        T val = sdata[bi][bk];
                        val /= diag;

                        // Make sure val has been read before updating
                        __threadfence_block();

                        // Update
                        if(threadIdx.y == 0)
                        {
                            sdata[bi][bk] = val;
                        }

                        // Do linear combination
                        for(rocsparse_int bj = bi + 1 + threadIdx.y; bj < block_dim; bj += DIMY)
                        {
                            sdata[bj][bk] = rocsparse::fma(-val, sdata[bj][bi], sdata[bj][bk]);
                        }
                    }
                }

                __threadfence_block();

                // Write diagonal BSR block back to global memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        bsr_val[BSR_IND(row_diag, p, q, dir)] = sdata[q][p];
                    }
                }
            }

            // Process upper diagonal BSR blocks
            for(rocsparse_int j = row_diag + 1; j < row_end; ++j)
            {
                __threadfence_block();

                // Load row j into shared memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        sdata[q][p] = bsr_val[BSR_IND(j, p, q, dir)];
                    }
                }

                __threadfence_block();

                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    for(rocsparse_int bk = threadIdx.x; bk < block_dim; bk += DIMX)
                    {
                        for(rocsparse_int bj = bi + 1 + threadIdx.y; bj < block_dim; bj += DIMY)
                        {
                            sdata[bk][bj] = rocsparse::fma(-bsr_val[BSR_IND(row_diag, bj, bi, dir)],
                                                           sdata[bk][bi],
                                                           sdata[bk][bj]);
                        }
                    }
                }

                __threadfence_block();

                // Write row j back to global memory
                for(rocsparse_int p = threadIdx.x; p < block_dim; p += DIMX)
                {
                    for(rocsparse_int q = threadIdx.y; q < block_dim; q += DIMY)
                    {
                        bsr_val[BSR_IND(j, p, q, dir)] = sdata[q][p];
                    }
                }
            }
        }
        else
        {
            // Structural pivot found
            pivot = true;
        }

        if(threadIdx.x == 0 && threadIdx.y == 0)
        {
            // First lane writes "we are done" flag
            __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot)
            {
                // Atomically set minimum zero pivot, if found
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
    ROCSPARSE_DEVICE_ILF void bsrilu0_general_device(rocsparse_direction  dir,
                                                     rocsparse_int        mb,
                                                     const rocsparse_int* bsr_row_ptr,
                                                     const rocsparse_int* bsr_col_ind,
                                                     T*                   bsr_val,
                                                     const rocsparse_int* bsr_diag_ind,
                                                     rocsparse_int        block_dim,
                                                     int*                 done_array,
                                                     const rocsparse_int* map,
                                                     rocsparse_int*       zero_pivot,
                                                     rocsparse_index_base idx_base,
                                                     int                  boost,
                                                     double               boost_tol,
                                                     T                    boost_val)
    {
        int lid = hipThreadIdx_x & (WFSIZE - 1);
        int wid = hipThreadIdx_x / WFSIZE;

        // Index
        rocsparse_int idx = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= mb)
        {
            return;
        }

        // Current row this wavefront is working on
        rocsparse_int row = map[idx];

        // Diagonal entry point of the current row
        rocsparse_int row_diag = bsr_diag_ind[row];

        // Row entry point
        rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Zero pivot tracker
        bool pivot = false;

        // Check for structural pivot
        if(row_diag != -1)
        {
            // Process lower diagonal
            for(rocsparse_int j = row_begin; j < row_diag; ++j)
            {
                // Column index of current BSR block
                rocsparse_int bsr_col = bsr_col_ind[j] - idx_base;

                // Process all lower matrix BSR blocks

                // Obtain corresponding row entry and exit point that corresponds with the
                // current BSR column. Actually, we skip all lower matrix column indices,
                // therefore starting with the diagonal entry.
                rocsparse_int diag_j    = bsr_diag_ind[bsr_col];
                rocsparse_int row_end_j = bsr_row_ptr[bsr_col + 1] - idx_base;

                // Check for structural pivot
                if(diag_j == -1)
                {
                    pivot = true;
                    break;
                }

                // Spin loop until dependency has been resolved
                (void)rocsparse::spin_loop<SLEEP>(&done_array[bsr_col], __HIP_MEMORY_SCOPE_AGENT);

                // Make sure dependencies are visible in global memory
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

                // Loop through all rows within the BSR block
                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal entry of the BSR block
                    T diag = bsr_val[BSR_IND(diag_j, bi, bi, dir)];

                    // Loop through all rows
                    for(rocsparse_int bk = lid; bk < block_dim; bk += WFSIZE)
                    {
                        T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                        // This has already been checked for zero by previous computations
                        val /= diag;

                        // Update
                        bsr_val[BSR_IND(j, bk, bi, dir)] = val;

                        // Do linear combination

                        // Loop through all columns above the diagonal of the BSR block
                        for(rocsparse_int bj = bi + 1; bj < block_dim; ++bj)
                        {
                            bsr_val[BSR_IND(j, bk, bj, dir)]
                                = rocsparse::fma(-val,
                                                 bsr_val[BSR_IND(diag_j, bi, bj, dir)],
                                                 bsr_val[BSR_IND(j, bk, bj, dir)]);
                        }
                    }
                }

                // Loop over upper offset pointer and do linear combination for nnz entry
                for(rocsparse_int k = diag_j + 1; k < row_end_j; ++k)
                {
                    rocsparse_int bsr_col_k = bsr_col_ind[k] - idx_base;

                    // Search for matching column index in current row
                    rocsparse_int q         = row_begin + lid;
                    rocsparse_int bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;

                    // Check if match has been found by any thread in the wavefront
                    while(bsr_col_j < bsr_col_k)
                    {
                        q += WFSIZE;
                        bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;
                    }

                    // Check if match has been found by any thread in the wavefront
                    int match = __ffsll(__ballot(bsr_col_j == bsr_col_k));

                    // If match has been found, process it
                    if(match)
                    {
                        // Tell all other threads about the matching index
                        rocsparse_int m = rocsparse::shfl(q, match - 1);

                        for(rocsparse_int bi = lid; bi < block_dim; bi += WFSIZE)
                        {
                            for(rocsparse_int bj = 0; bj < block_dim; ++bj)
                            {
                                T sum = static_cast<T>(0);

                                for(rocsparse_int bk = 0; bk < block_dim; ++bk)
                                {
                                    sum = rocsparse::fma(bsr_val[BSR_IND(j, bi, bk, dir)],
                                                         bsr_val[BSR_IND(k, bk, bj, dir)],
                                                         sum);
                                }

                                bsr_val[BSR_IND(m, bi, bj, dir)] -= sum;
                            }
                        }
                    }
                }
            }

            // Process diagonal
            if(bsr_col_ind[row_diag] - idx_base == row)
            {
                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal matrix entry
                    T diag = bsr_val[BSR_IND(row_diag, bi, bi, dir)];

                    // Numeric boost
                    if(boost)
                    {
                        diag = (boost_tol >= rocsparse::abs(diag)) ? boost_val : diag;

                        if(lid == 0)
                        {
                            bsr_val[BSR_IND(row_diag, bi, bi, dir)] = diag;
                        }
                    }
                    else
                    {
                        // Check for numeric pivot
                        if(diag == static_cast<T>(0))
                        {
                            pivot = true;
                            continue;
                        }
                    }

                    for(rocsparse_int bk = bi + 1 + lid; bk < block_dim; bk += WFSIZE)
                    {
                        // Multiplication factor
                        T val = bsr_val[BSR_IND(row_diag, bk, bi, dir)];
                        val /= diag;

                        // Update
                        bsr_val[BSR_IND(row_diag, bk, bi, dir)] = val;

                        // Do linear combination
                        for(rocsparse_int bj = bi + 1; bj < block_dim; ++bj)
                        {
                            bsr_val[BSR_IND(row_diag, bk, bj, dir)]
                                = rocsparse::fma(-val,
                                                 bsr_val[BSR_IND(row_diag, bi, bj, dir)],
                                                 bsr_val[BSR_IND(row_diag, bk, bj, dir)]);
                        }
                    }
                }
            }

            // Process upper diagonal BSR blocks
            for(rocsparse_int j = row_diag + 1; j < row_end; ++j)
            {
                for(rocsparse_int bi = 0; bi < block_dim; ++bi)
                {
                    for(rocsparse_int bk = lid; bk < block_dim; bk += WFSIZE)
                    {
                        for(rocsparse_int bj = bi + 1; bj < block_dim; ++bj)
                        {
                            bsr_val[BSR_IND(j, bj, bk, dir)]
                                = rocsparse::fma(-bsr_val[BSR_IND(row_diag, bj, bi, dir)],
                                                 bsr_val[BSR_IND(j, bi, bk, dir)],
                                                 bsr_val[BSR_IND(j, bj, bk, dir)]);
                        }
                    }
                }
            }
        }
        else
        {
            // Structural pivot found
            pivot = true;
        }

        if(lid == 0)
        {
            // First lane writes "we are done" flag
            __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot)
            {
                // Atomically set minimum zero pivot, if found
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }
}
