/*! \file */
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

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrsv_lower_general_device(rocsparse_int mb,
                                   T             alpha,
                                   const rocsparse_int* __restrict__ bsr_row_ptr,
                                   const rocsparse_int* __restrict__ bsr_col_ind,
                                   const T* __restrict__ bsr_val,
                                   rocsparse_int block_dim,
                                   const T* __restrict__ x,
                                   T* __restrict__ y,
                                   int* __restrict__ done_array,
                                   rocsparse_int* __restrict__ map,
                                   rocsparse_int* __restrict__ zero_pivot,
                                   rocsparse_index_base idx_base,
                                   rocsparse_diag_type  diag_type,
                                   rocsparse_direction  dir)
    {
        const int lid = hipThreadIdx_x & (WFSIZE - 1);
        const int wid = hipThreadIdx_x / WFSIZE;

        // Index into the row map
        const rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= mb)
        {
            return;
        }

        // Get the BSR row this wavefront will operate on
        const rocsparse_int row = map[idx];

        // Current row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Initialize local_col with mb
        rocsparse_int local_col = mb;

        // Initialize y with alpha and x
        for(rocsparse_int bi = lid; bi < block_dim; bi += WFSIZE)
        {
            y[row * block_dim + bi] = alpha * x[row * block_dim + bi];
        }

        // Loop over the current row
        rocsparse_int j;
        for(j = row_begin; j < row_end; ++j)
        {
            // Current column index
            local_col = bsr_col_ind[j] - idx_base;

            // Processing lower triangular

            // Ignore all diagonal entries and above
            if(local_col >= row)
            {
                break;
            }

            // Spin loop until dependency has been resolved
            rocsparse::spin_loop<SLEEP>(&done_array[local_col], __HIP_MEMORY_SCOPE_AGENT);

            // Wait for y to be visible globally
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Local sum computation
            for(rocsparse_int bi = lid; bi < block_dim; bi += WFSIZE)
            {
                // Local sum accumulator
                T local_sum = static_cast<T>(0);

                for(rocsparse_int bj = 0; bj < block_dim; ++bj)
                {
                    local_sum = rocsparse::fma(
                        bsr_val[BSR_IND(j, bi, bj, dir)], y[local_col * block_dim + bj], local_sum);
                }

                // Write local sum to y
                y[row * block_dim + bi] -= local_sum;
            }
        }

        bool pivot = false;

        // Process diagonal
        if(local_col == row)
        {
            for(rocsparse_int bi = 0; bi < block_dim; ++bi)
            {
                // Load diagonal matrix entry
                const T diag = (diag_type == rocsparse_diag_type_non_unit)
                                   ? bsr_val[block_dim * block_dim * j + bi + bi * block_dim]
                                   : static_cast<T>(1);

                // Load result of bi-th BSR row
                T val = y[row * block_dim + bi];
                // Check for numerical pivot
                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    // Divide result of bi-th BSR row by diagonal entry
                    y[row * block_dim + bi] = val /= diag;
                }

                // Update remaining non-diagonal entries
                for(rocsparse_int bj = bi + lid + 1; bj < block_dim; bj += WFSIZE)
                {
                    y[row * block_dim + bj] -= val * bsr_val[BSR_IND(j, bj, bi, dir)];
                }
            }
        }

        // Write "row is done" flag
        if(lid == 0)
        {
            __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrsv_upper_general_device(rocsparse_int mb,
                                   T             alpha,
                                   const rocsparse_int* __restrict__ bsr_row_ptr,
                                   const rocsparse_int* __restrict__ bsr_col_ind,
                                   const T* __restrict__ bsr_val,
                                   rocsparse_int block_dim,
                                   const T* __restrict__ x,
                                   T* __restrict__ y,
                                   int* __restrict__ done_array,
                                   rocsparse_int* __restrict__ map,
                                   rocsparse_int* __restrict__ zero_pivot,
                                   rocsparse_index_base idx_base,
                                   rocsparse_diag_type  diag_type,
                                   rocsparse_direction  dir)
    {
        const int lid = hipThreadIdx_x & (WFSIZE - 1);
        const int wid = hipThreadIdx_x / WFSIZE;

        // Index into the row map
        const rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= mb)
        {
            return;
        }

        // Get the BSR row this wavefront will operate on
        const rocsparse_int row = map[idx];

        // Current row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Initialize local_col with mb
        rocsparse_int local_col = mb;

        // Initialize y with alpha and x
        for(rocsparse_int bi = lid; bi < block_dim; bi += WFSIZE)
        {
            y[row * block_dim + bi] = alpha * x[row * block_dim + bi];
        }

        // Loop over the current row
        rocsparse_int j;
        for(j = row_end - 1; j >= row_begin; --j)
        {
            // Current column index
            local_col = bsr_col_ind[j] - idx_base;

            // Processing upper triangular

            // Ignore all diagonal entries and below
            if(local_col <= row)
            {
                break;
            }

            // Spin loop until dependency has been resolved
            rocsparse::spin_loop<SLEEP>(&done_array[local_col], __HIP_MEMORY_SCOPE_AGENT);

            // Wait for y to be visible globally
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Local sum computation
            for(rocsparse_int bi = lid; bi < block_dim; bi += WFSIZE)
            {
                // Local sum accumulator
                T local_sum = static_cast<T>(0);

                for(rocsparse_int bj = 0; bj < block_dim; ++bj)
                {
                    local_sum = rocsparse::fma(
                        bsr_val[BSR_IND(j, bi, bj, dir)], y[local_col * block_dim + bj], local_sum);
                }

                // Write local sum to y
                y[row * block_dim + bi] -= local_sum;
            }
        }

        bool pivot = false;

        // Process diagonal
        if(local_col == row)
        {
            for(rocsparse_int bi = block_dim - 1; bi >= 0; --bi)
            {
                // Load diagonal matrix entry
                const T diag = (diag_type == rocsparse_diag_type_non_unit)
                                   ? bsr_val[block_dim * block_dim * j + bi + bi * block_dim]
                                   : static_cast<T>(1);

                // Load result of bi-th BSR row
                T val = y[row * block_dim + bi];

                // Check for numerical pivot
                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    // Divide result of bi-th BSR row by diagonal entry
                    y[row * block_dim + bi] = val /= diag;
                }

                // Update remaining non-diagonal entries
                for(rocsparse_int bj = lid; bj < bi; bj += WFSIZE)
                {
                    y[row * block_dim + bj] -= val * bsr_val[BSR_IND(j, bj, bi, dir)];
                }
            }
        }

        // Write "row is done" flag
        if(lid == 0)
        {
            __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, rocsparse_int BSRDIM, bool SLEEP, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrsv_lower_shared_device(rocsparse_int mb,
                                  T             alpha,
                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  rocsparse_int block_dim,
                                  const T* __restrict__ x,
                                  T* __restrict__ y,
                                  int* __restrict__ done_array,
                                  rocsparse_int* __restrict__ map,
                                  rocsparse_int* __restrict__ zero_pivot,
                                  rocsparse_index_base idx_base,
                                  rocsparse_diag_type  diag_type,
                                  rocsparse_direction  dir)
    {
        const int lid = hipThreadIdx_x & (WFSIZE - 1);
        const int wid = hipThreadIdx_x / WFSIZE;

        // Index into the row map
        const rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= mb)
        {
            return;
        }

        // Get the BSR row this wavefront will operate on
        const rocsparse_int row = map[idx];

        // Current row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Initialize local_col with mb
        rocsparse_int local_col = mb;

        // Initialize local summation variable with alpha and x
        T local_sum = alpha * ((lid < block_dim) ? x[row * block_dim + lid] : static_cast<T>(0));

        // Shared memory to hold BSR blocks and updated sums
        __shared__ T sdata1[BLOCKSIZE / WFSIZE * BSRDIM * BSRDIM];
        __shared__ T sdata2[BLOCKSIZE / WFSIZE * BSRDIM];

        T* bsr_values  = &sdata1[wid * BSRDIM * BSRDIM];
        T* bsr_updates = &sdata2[wid * BSRDIM];

        // Loop over the current row
        rocsparse_int j;
        for(j = row_begin; j < row_end; ++j)
        {
            // Current column index
            local_col = bsr_col_ind[j] - idx_base;

            // Load BSR block values
            // Each wavefront loads a full BSR block into shared memory
            // Pad remaining entries with zero
            const int bi = lid & (BSRDIM - 1);
            const int bj = lid / BSRDIM;

            for(rocsparse_int k = bj; k < BSRDIM; k += WFSIZE / BSRDIM)
            {
                bsr_values[bi + k * BSRDIM] = (bi < block_dim && k < block_dim)
                                                  ? bsr_val[BSR_IND(j, bi, k, dir)]
                                                  : static_cast<T>(0);
            }

            // Processing lower triangular

            // Ignore all diagonal entries and above
            if(local_col >= row)
            {
                break;
            }

            // Spin loop until dependency has been resolved
            rocsparse::spin_loop<SLEEP>(&done_array[local_col], __HIP_MEMORY_SCOPE_AGENT);

            // Wait for y to be visible globally
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Load all updated dependencies into shared memory
            if(lid < BSRDIM)
            {
                bsr_updates[lid]
                    = (lid < block_dim) ? y[local_col * block_dim + lid] : static_cast<T>(0);
            }

            __threadfence_block();

            // Local sum computation
            if(lid < block_dim)
            {
                for(rocsparse_int l = 0; l < BSRDIM; ++l)
                {
                    local_sum
                        = rocsparse::fma(-bsr_values[lid + l * BSRDIM], bsr_updates[l], local_sum);
                }
            }
        }

        // Initialize zero pivot
        bool pivot = false;

        // Process diagonal
        if(local_col == row)
        {
            for(rocsparse_int bi = 0; bi < block_dim; ++bi)
            {
                // Load diagonal matrix entry
                const T diag = (diag_type == rocsparse_diag_type_non_unit)
                                   ? bsr_values[bi + bi * BSRDIM]
                                   : static_cast<T>(1);

                // Load result of bi-th BSR row
                T val = rocsparse::shfl(local_sum, bi);

                // Check for numerical pivot
                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    // Divide result of bi-th row by diagonal entry
                    val /= diag;
                }

                // Update remaining non-diagonal entries
                if(lid < block_dim)
                {
                    if(bi < lid)
                    {
                        local_sum = rocsparse::fma(-val, bsr_values[lid + bi * BSRDIM], local_sum);
                    }
                    else if(lid == bi)
                    {
                        local_sum = val;
                    }
                }
            }
        }

        if(lid < block_dim)
        {
            // Store the rows results in y
            y[row * block_dim + lid] = local_sum;
        }

        if(lid == 0)
        {
            // Write "row is done" flag
            __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

            // Find the minimum pivot, if applicable
            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, rocsparse_int BSRDIM, bool SLEEP, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrsv_upper_shared_device(rocsparse_int mb,
                                  T             alpha,
                                  const rocsparse_int* __restrict__ bsr_row_ptr,
                                  const rocsparse_int* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  rocsparse_int block_dim,
                                  const T* __restrict__ x,
                                  T* __restrict__ y,
                                  int* __restrict__ done_array,
                                  rocsparse_int* __restrict__ map,
                                  rocsparse_int* __restrict__ zero_pivot,
                                  rocsparse_index_base idx_base,
                                  rocsparse_diag_type  diag_type,
                                  rocsparse_direction  dir)
    {
        const int lid = hipThreadIdx_x & (WFSIZE - 1);
        const int wid = hipThreadIdx_x / WFSIZE;

        // Index into the row map
        const rocsparse_int idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= mb)
        {
            return;
        }

        // Get the BSR row this wavefront will operate on
        const rocsparse_int row = map[idx];

        // Current row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Initialize local_col with mb
        rocsparse_int local_col = mb;

        // Initialize local summation variable with alpha and x
        T local_sum = alpha * ((lid < block_dim) ? x[row * block_dim + lid] : static_cast<T>(0));

        // Shared memory to hold BSR blocks and updated sums
        __shared__ T sdata1[BLOCKSIZE / WFSIZE * BSRDIM * BSRDIM];
        __shared__ T sdata2[BLOCKSIZE / WFSIZE * BSRDIM];

        T* bsr_values  = &sdata1[wid * BSRDIM * BSRDIM];
        T* bsr_updates = &sdata2[wid * BSRDIM];

        // Loop over the current row
        rocsparse_int j;
        for(j = row_end - 1; j >= row_begin; --j)
        {
            // Current column index
            local_col = bsr_col_ind[j] - idx_base;

            // Load BSR block values
            // Each wavefront loads a full BSR block into shared memory
            // Pad remaining entries with zero
            const int bi = lid & (BSRDIM - 1);
            const int bj = lid / BSRDIM;

            for(rocsparse_int k = bj; k < BSRDIM; k += WFSIZE / BSRDIM)
            {
                bsr_values[bi + k * BSRDIM] = (bi < block_dim && k < block_dim)
                                                  ? bsr_val[BSR_IND(j, bi, k, dir)]
                                                  : static_cast<T>(0);
            }

            // Processing upper triangular

            // Ignore all diagonal entries and below
            if(local_col <= row)
            {
                break;
            }

            // Spin loop until dependency has been resolved
            rocsparse::spin_loop<SLEEP>(&done_array[local_col], __HIP_MEMORY_SCOPE_AGENT);

            // Wait for y to be visible globally
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Load all updated dependencies into shared memory
            if(lid < BSRDIM)
            {
                bsr_updates[lid]
                    = (lid < block_dim) ? y[local_col * block_dim + lid] : static_cast<T>(0);
            }

            __threadfence_block();

            // Local sum computation
            if(lid < block_dim)
            {
                for(rocsparse_int l = 0; l < BSRDIM; ++l)
                {
                    local_sum
                        = rocsparse::fma(-bsr_values[lid + l * BSRDIM], bsr_updates[l], local_sum);
                }
            }
        }

        // Initialize zero pivot
        bool pivot = false;

        // Process diagonal
        if(local_col == row)
        {
            for(rocsparse_int bi = block_dim - 1; bi >= 0; --bi)
            {
                // Load diagonal matrix entry
                const T diag = (diag_type == rocsparse_diag_type_non_unit)
                                   ? bsr_values[bi + bi * BSRDIM]
                                   : static_cast<T>(1);

                // Load result of bi-th BSR row
                T val = rocsparse::shfl(local_sum, bi);

                // Check for numerical pivot
                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    // Divide result of bi-th row by diagonal entry
                    val /= diag;
                }

                // Update remaining non-diagonal entries
                if(lid < block_dim)
                {
                    if(bi > lid)
                    {
                        local_sum = rocsparse::fma(-val, bsr_values[lid + bi * BSRDIM], local_sum);
                    }
                    else if(lid == bi)
                    {
                        local_sum = val;
                    }
                }
            }
        }

        if(lid < block_dim)
        {
            // Store the rows results in y
            y[row * block_dim + lid] = local_sum;
        }

        if(lid == 0)
        {
            // Write "row is done" flag
            __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

            // Find the minimum pivot, if applicable
            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }
}
