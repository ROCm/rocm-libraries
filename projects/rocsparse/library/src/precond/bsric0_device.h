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
    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsric0_2_8_unrolled_kernel(rocsparse_direction direction,
                                    rocsparse_int       mb,
                                    rocsparse_int       block_dim,
                                    const rocsparse_int* __restrict__ bsr_row_ptr,
                                    const rocsparse_int* __restrict__ bsr_col_ind,
                                    T* __restrict__ bsr_val,
                                    const rocsparse_int* __restrict__ bsr_diag_ind,
                                    int* __restrict__ block_done,
                                    const rocsparse_int* __restrict__ block_map,
                                    rocsparse_int* __restrict__ zero_pivot,
                                    rocsparse_index_base idx_base)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int tidy = hipThreadIdx_y;
        rocsparse_int tid  = BSRDIM * tidy + tidx;

        __shared__ rocsparse_int columns[MAX_NNZB];
        __shared__ rocsparse_int index[MAX_NNZB];
        __shared__ rocsparse_int local_index[MAX_NNZB];
        __shared__ T             row_sum[BSRDIM][BSRDIM + 1];
        __shared__ T             temp[BSRDIM][BSRDIM + 1];
        __shared__ T             values[BSRDIM][BSRDIM + 1];
        __shared__ T             local_values[BSRDIM][BSRDIM + 1];

        // Current block row this wavefront is working on
        rocsparse_int block_row = block_map[hipBlockIdx_x];

        // Block diagonal entry point of the current block row
        rocsparse_int block_row_diag = bsr_diag_ind[block_row];

        // If one thread in the warp breaks here, then all threads in
        // the warp break so no divergence
        if(block_row_diag == -1)
        {
            if(tidx == 0 && tidy == 0)
            {
                rocsparse::atomic_min(zero_pivot, block_row + idx_base);

                // Last lane in wavefront writes "we are done" flag for its block row
                __hip_atomic_store(
                    &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }

            return;
        }

        // Block row entry point
        rocsparse_int block_row_begin = bsr_row_ptr[block_row] - idx_base;

        // Write current block row column indices to shared memory
        for(rocsparse_int j = block_row_begin + tid; j < block_row_diag + 1; j += BSRDIM * BSRDIM)
        {
            columns[j - block_row_begin] = bsr_col_ind[j] - idx_base;
        }

        // Block row sum accumulator
        row_sum[tidy][tidx] = static_cast<T>(0);

        __threadfence_block();

        // Loop over non-diagonal block columns of current block row
        for(rocsparse_int j = block_row_begin; j < block_row_diag; j++)
        {
            // Block column index currently being processes
            rocsparse_int block_col = bsr_col_ind[j] - idx_base;

            // Beginning of the row that corresponds to block_col
            rocsparse_int local_block_begin = bsr_row_ptr[block_col] - idx_base;

            // Diagonal entry point of row block_col
            rocsparse_int local_block_diag = bsr_diag_ind[block_col];

            // Structural zero pivot, do not process this row
            if(local_block_diag == -1)
            {
                // If one thread in the warp breaks here, then all threads in
                // the warp break so no divergence
                break;
            }

            if(direction == rocsparse_direction_row)
            {
                values[tidy][tidx] = bsr_val[BSRDIM * BSRDIM * j + BSRDIM * tidy + tidx];
            }
            else
            {
                values[tidy][tidx] = bsr_val[BSRDIM * BSRDIM * j + BSRDIM * tidx + tidy];
            }

            rocsparse_int count = 0;
            rocsparse_int l     = local_block_begin;
            rocsparse_int k     = 0;
            rocsparse_int col_k = columns[k];

            while(l <= local_block_diag && col_k <= block_col)
            {
                rocsparse_int col_l = bsr_col_ind[l] - idx_base;
                col_k               = columns[k];

                if(col_l < col_k)
                {
                    l++;
                }
                else if(col_l > col_k)
                {
                    k++;
                }
                else
                {
                    index[count]       = BSRDIM * BSRDIM * (k + block_row_begin);
                    local_index[count] = BSRDIM * BSRDIM * l;

                    k++;
                    l++;

                    count++;
                }
            }

            __threadfence_block();

            // Spin loop until dependency has been resolved
            while(!__hip_atomic_load(
                &block_done[block_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                ;

            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            if(direction == rocsparse_direction_row)
            {
                local_values[tidy][tidx]
                    = bsr_val[BSRDIM * BSRDIM * local_block_diag + BSRDIM * tidy + tidx];
            }
            else
            {
                local_values[tidy][tidx]
                    = bsr_val[BSRDIM * BSRDIM * local_block_diag + BSRDIM * tidx + tidy];
            }

            __threadfence_block();

            // Local row sum
            T local_sum = static_cast<T>(0);

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            for(rocsparse_int l = 0; l < count - 1; l++)
            {
                rocsparse_int idx2 = local_index[l];
                rocsparse_int idx  = index[l];

                if(direction == rocsparse_direction_row)
                {
                    if(BSRDIM >= 1)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 0];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 0];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 2)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 1];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 1];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 3)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 2];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 2];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 4)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 3];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 3];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 5)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 4];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 4];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 6)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 5];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 5];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 7)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 6];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 6];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 8)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 7];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 7];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                }
                else
                {
                    if(BSRDIM >= 1)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 0 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 0 + tidy];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 2)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 1 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 1 + tidy];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 3)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 2 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 2 + tidy];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 4)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 3 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 3 + tidy];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 5)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 4 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 4 + tidy];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 6)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 5 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 5 + tidy];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 7)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 6 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 6 + tidy];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    if(BSRDIM >= 8)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 7 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 7 + tidy];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                }
            }

            temp[tidy][tidx] = local_sum;

            __threadfence_block();

            for(rocsparse_int k = 0; k < BSRDIM; k++)
            {
                // Current value
                T val = values[tidy][k];

                // Load diagonal entry
                T diag_val = local_values[k][k];

                // Row has numerical zero pivot
                if(diag_val == static_cast<T>(0))
                {
                    if(tidx == 0 && tidy == 0)
                    {
                        // We are looking for the first zero pivot
                        rocsparse::atomic_min(zero_pivot, block_col + idx_base);
                    }

                    diag_val = static_cast<T>(1);
                }

                T local_sum = temp[tidy][k];

                for(rocsparse_int p = 0; p < k; p++)
                {
                    T v1      = local_values[k][p];
                    T v2      = values[tidy][p];
                    local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                }

                // Compute the Cholesky factor and writes it to shared memory
                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                row_sum[tidy][tidx]
                    = rocsparse::fma(val, rocsparse::conj(values[tidx][k]), row_sum[tidy][tidx]);

                __threadfence_block();
            }

            if(direction == rocsparse_direction_row)
            {
                bsr_val[BSRDIM * BSRDIM * j + BSRDIM * tidy + tidx] = values[tidy][tidx];
            }
            else
            {
                bsr_val[BSRDIM * BSRDIM * j + BSRDIM * tidx + tidy] = values[tidy][tidx];
            }

            __threadfence();
        }

        // Load current diagonal block into shared memory
        if(direction == rocsparse_direction_row)
        {
            values[tidy][tidx] = bsr_val[BSRDIM * BSRDIM * block_row_diag + BSRDIM * tidy + tidx];
        }
        else
        {
            values[tidy][tidx] = bsr_val[BSRDIM * BSRDIM * block_row_diag + BSRDIM * tidx + tidy];
        }

        __threadfence_block();

        // Handle diagonal block column of block row.
        for(rocsparse_int k = 0; k < BSRDIM; k++)
        {
            if(k == tidy)
            {
                values[k][k] = rocsparse::sqrt(rocsparse::abs(values[k][k] - row_sum[k][k]));
            }

            __threadfence_block();

            // Load diagonal entry
            T diag_val = values[k][k];

            // Row has numerical zero pivot
            if(diag_val == static_cast<T>(0))
            {
                if(tidx == 0 && tidy == 0)
                {
                    // We are looking for the first zero pivot
                    rocsparse::atomic_min(zero_pivot, block_row + idx_base);
                }

                // Normally would break here but to avoid divergence set diag_val to one and continue
                // The zero pivot has already been set so further computation does not matter
                diag_val = static_cast<T>(1);
            }

            if(k < tidy)
            {
                // Load value
                T val = values[tidy][k];

                // Local row sum
                T local_sum = row_sum[tidy][k];

                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                row_sum[tidy][tidx]
                    = rocsparse::fma(val, rocsparse::conj(values[tidx][k]), row_sum[tidy][tidx]);
            }

            __threadfence_block();
        }

        if(direction == rocsparse_direction_row)
        {
            bsr_val[BSRDIM * BSRDIM * block_row_diag + BSRDIM * tidy + tidx] = values[tidy][tidx];
        }
        else
        {
            bsr_val[BSRDIM * BSRDIM * block_row_diag + BSRDIM * tidx + tidy] = values[tidy][tidx];
        }

        if(tidx == 0 && tidy == 0)
        {
            // Last lane in wavefront writes "we are done" flag for its block row
            __hip_atomic_store(
                &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsric0_2_8_kernel(rocsparse_direction direction,
                           rocsparse_int       mb,
                           rocsparse_int       block_dim,
                           const rocsparse_int* __restrict__ bsr_row_ptr,
                           const rocsparse_int* __restrict__ bsr_col_ind,
                           T* __restrict__ bsr_val,
                           const rocsparse_int* __restrict__ bsr_diag_ind,
                           int* __restrict__ block_done,
                           const rocsparse_int* __restrict__ block_map,
                           rocsparse_int* __restrict__ zero_pivot,
                           rocsparse_index_base idx_base)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int tidy = hipThreadIdx_y;
        rocsparse_int tid  = BSRDIM * tidy + tidx;

        __shared__ rocsparse_int columns[MAX_NNZB];
        __shared__ rocsparse_int index[MAX_NNZB];
        __shared__ rocsparse_int local_index[MAX_NNZB];
        __shared__ T             row_sum[BSRDIM][BSRDIM + 1];
        __shared__ T             temp[BSRDIM][BSRDIM + 1];
        __shared__ T             values[BSRDIM][BSRDIM + 1];
        __shared__ T             local_values[BSRDIM][BSRDIM + 1];

        // Current block row this wavefront is working on
        rocsparse_int block_row = block_map[hipBlockIdx_x];

        // Block diagonal entry point of the current block row
        rocsparse_int block_row_diag = bsr_diag_ind[block_row];

        // If one thread in the warp breaks here, then all threads in
        // the warp break so no divergence
        if(block_row_diag == -1)
        {
            if(tidx == 0 && tidy == 0)
            {
                rocsparse::atomic_min(zero_pivot, block_row + idx_base);

                // Last lane in wavefront writes "we are done" flag for its block row
                __hip_atomic_store(
                    &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }

            return;
        }

        // Block row entry point
        rocsparse_int block_row_begin = bsr_row_ptr[block_row] - idx_base;

        // Write current block row column indices to shared memory
        for(rocsparse_int j = block_row_begin + tid; j < block_row_diag + 1; j += BSRDIM * BSRDIM)
        {
            columns[j - block_row_begin] = bsr_col_ind[j] - idx_base;
        }

        // Block row sum accumulator
        row_sum[tidy][tidx] = static_cast<T>(0);

        __threadfence_block();

        // Loop over non-diagonal block columns of current block row
        for(rocsparse_int j = block_row_begin; j < block_row_diag; j++)
        {
            // Block column index currently being processes
            rocsparse_int block_col = bsr_col_ind[j] - idx_base;

            // Beginning of the row that corresponds to block_col
            rocsparse_int local_block_begin = bsr_row_ptr[block_col] - idx_base;

            // Diagonal entry point of row block_col
            rocsparse_int local_block_diag = bsr_diag_ind[block_col];

            // Structural zero pivot, do not process this row
            if(local_block_diag == -1)
            {
                // If one thread in the warp breaks here, then all threads in
                // the warp break so no divergence
                break;
            }

            if(direction == rocsparse_direction_row)
            {
                values[tidy][tidx]
                    = (tidx < block_dim && tidy < block_dim)
                          ? bsr_val[block_dim * block_dim * j + block_dim * tidy + tidx]
                          : static_cast<T>(0);
            }
            else
            {
                values[tidy][tidx]
                    = (tidx < block_dim && tidy < block_dim)
                          ? bsr_val[block_dim * block_dim * j + block_dim * tidx + tidy]
                          : static_cast<T>(0);
            }

            rocsparse_int count = 0;
            rocsparse_int l     = local_block_begin;
            rocsparse_int k     = 0;
            rocsparse_int col_k = columns[k];

            while(l <= local_block_diag && col_k <= block_col)
            {
                rocsparse_int col_l = bsr_col_ind[l] - idx_base;
                col_k               = columns[k];

                if(col_l < col_k)
                {
                    l++;
                }
                else if(col_l > col_k)
                {
                    k++;
                }
                else
                {
                    // index[count] = BSRDIM * BSRDIM * k;
                    index[count]       = block_dim * block_dim * (k + block_row_begin);
                    local_index[count] = block_dim * block_dim * l;

                    k++;
                    l++;

                    count++;
                }
            }

            __threadfence_block();

            // Spin loop until dependency has been resolved
            while(!__hip_atomic_load(
                &block_done[block_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                ;

            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            if(direction == rocsparse_direction_row)
            {
                local_values[tidy][tidx] = (tidx < block_dim && tidy < block_dim)
                                               ? bsr_val[block_dim * block_dim * local_block_diag
                                                         + block_dim * tidy + tidx]
                                               : static_cast<T>(0);
            }
            else
            {
                local_values[tidy][tidx] = (tidx < block_dim && tidy < block_dim)
                                               ? bsr_val[block_dim * block_dim * local_block_diag
                                                         + block_dim * tidx + tidy]
                                               : static_cast<T>(0);
            }

            __threadfence_block();

            // Local row sum
            T local_sum = static_cast<T>(0);

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            for(rocsparse_int l = 0; l < count - 1; l++)
            {
                rocsparse_int idx2 = local_index[l];
                rocsparse_int idx  = index[l];

                for(rocsparse_int p = 0; p < block_dim; p++)
                {
                    if(direction == rocsparse_direction_row)
                    {
                        T v1      = (tidx < block_dim) ? bsr_val[idx2 + block_dim * tidx + p]
                                                       : static_cast<T>(0);
                        T v2      = (tidy < block_dim) ? bsr_val[idx + block_dim * tidy + p]
                                                       : static_cast<T>(0);
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    else
                    {
                        T v1      = (tidx < block_dim) ? bsr_val[idx2 + block_dim * p + tidx]
                                                       : static_cast<T>(0);
                        T v2      = (tidy < block_dim) ? bsr_val[idx + block_dim * p + tidy]
                                                       : static_cast<T>(0);
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                }
            }

            temp[tidy][tidx] = local_sum;

            __threadfence_block();

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                // Current value
                T val = values[tidy][k];

                // Load diagonal entry
                T diag_val = local_values[k][k];

                // Row has numerical zero pivot
                if(diag_val == static_cast<T>(0))
                {
                    if(tidx == 0 && tidy == 0)
                    {
                        // We are looking for the first zero pivot
                        rocsparse::atomic_min(zero_pivot, block_col + idx_base);
                    }

                    diag_val = static_cast<T>(1);
                }

                T local_sum = temp[tidy][k];

                for(rocsparse_int p = 0; p < k; p++)
                {
                    T v1      = local_values[k][p];
                    T v2      = values[tidy][p];
                    local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                }

                // Compute the Cholesky factor and writes it to shared memory
                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                row_sum[tidy][tidx]
                    = rocsparse::fma(val, rocsparse::conj(values[tidx][k]), row_sum[tidy][tidx]);

                __threadfence_block();
            }

            if(tidx < block_dim && tidy < block_dim)
            {
                if(direction == rocsparse_direction_row)
                {
                    bsr_val[block_dim * block_dim * j + block_dim * tidy + tidx]
                        = values[tidy][tidx];
                }
                else
                {
                    bsr_val[block_dim * block_dim * j + block_dim * tidx + tidy]
                        = values[tidy][tidx];
                }
            }

            __threadfence();
        }

        // Load current diagonal block into shared memory
        if(direction == rocsparse_direction_row)
        {
            values[tidy][tidx]
                = (tidx < block_dim && tidy < block_dim)
                      ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidy + tidx]
                      : static_cast<T>(0);
        }
        else
        {
            values[tidy][tidx]
                = (tidx < block_dim && tidy < block_dim)
                      ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidx + tidy]
                      : static_cast<T>(0);
        }

        __threadfence_block();

        // Handle diagonal block column of block row.
        for(rocsparse_int k = 0; k < block_dim; k++)
        {
            if(k == tidy)
            {
                values[k][k] = rocsparse::sqrt(rocsparse::abs(values[k][k] - row_sum[k][k]));
            }

            __threadfence_block();

            // Load diagonal entry
            T diag_val = values[k][k];

            // Row has numerical zero pivot
            if(diag_val == static_cast<T>(0))
            {
                if(tidx == 0 && tidy == 0)
                {
                    // We are looking for the first zero pivot
                    rocsparse::atomic_min(zero_pivot, block_row + idx_base);
                }

                // Normally would break here but to avoid divergence set diag_val to one and continue
                // The zero pivot has already been set so further computation does not matter
                diag_val = static_cast<T>(1);
            }

            if(k < tidy)
            {
                // Load value
                T val = values[tidy][k];

                // Local row sum
                T local_sum = row_sum[tidy][k];

                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                row_sum[tidy][tidx]
                    = rocsparse::fma(val, rocsparse::conj(values[tidx][k]), row_sum[tidy][tidx]);
            }

            __threadfence_block();
        }

        if(tidx < block_dim && tidy < block_dim)
        {
            if(direction == rocsparse_direction_row)
            {
                bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidy + tidx]
                    = values[tidy][tidx];
            }
            else
            {
                bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidx + tidy]
                    = values[tidy][tidx];
            }
        }

        if(tidx == 0 && tidy == 0)
        {
            // Last lane in wavefront writes "we are done" flag for its block row
            __hip_atomic_store(
                &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsric0_9_16_kernel(rocsparse_direction direction,
                            rocsparse_int       mb,
                            rocsparse_int       block_dim,
                            const rocsparse_int* __restrict__ bsr_row_ptr,
                            const rocsparse_int* __restrict__ bsr_col_ind,
                            T* __restrict__ bsr_val,
                            const rocsparse_int* __restrict__ bsr_diag_ind,
                            int* __restrict__ block_done,
                            const rocsparse_int* __restrict__ block_map,
                            rocsparse_int* __restrict__ zero_pivot,
                            rocsparse_index_base idx_base)
    {
        constexpr static uint32_t DIMX = BLOCKSIZE / BSRDIM;
        constexpr static uint32_t DIMY = BSRDIM;

        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int tidy = hipThreadIdx_y;
        rocsparse_int tid  = DIMX * tidy + tidx;

        __shared__ rocsparse_int columns[MAX_NNZB];
        __shared__ rocsparse_int index[MAX_NNZB];
        __shared__ rocsparse_int local_index[MAX_NNZB];
        __shared__ T             row_sum[BSRDIM][BSRDIM + 1];
        __shared__ T             temp[BSRDIM][BSRDIM + 1];
        __shared__ T             values[BSRDIM][BSRDIM + 1];
        __shared__ T             local_values[BSRDIM][BSRDIM + 1];

        // Current block row this wavefront is working on
        rocsparse_int block_row = block_map[hipBlockIdx_x];

        // Block diagonal entry point of the current block row
        rocsparse_int block_row_diag = bsr_diag_ind[block_row];

        // If one thread in the warp breaks here, then all threads in
        // the warp break so no divergence
        if(block_row_diag == -1)
        {
            if(tidx == 0 && tidy == 0)
            {
                rocsparse::atomic_min(zero_pivot, block_row + idx_base);

                // Last lane in wavefront writes "we are done" flag for its block row
                __hip_atomic_store(
                    &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }

            return;
        }

        // Block row entry point
        rocsparse_int block_row_begin = bsr_row_ptr[block_row] - idx_base;

        // Write current block row column indices to shared memory
        for(rocsparse_int j = block_row_begin + tid; j < block_row_diag + 1; j += DIMX * DIMY)
        {
            columns[j - block_row_begin] = bsr_col_ind[j] - idx_base;
        }

        // Block row sum accumulator
        for(rocsparse_int i = tidx; i < BSRDIM; i += DIMX)
        {
            row_sum[tidy][i] = static_cast<T>(0);
        }

        __threadfence_block();

        // Loop over non-diagonal block columns of current block row
        for(rocsparse_int j = block_row_begin; j < block_row_diag; j++)
        {
            // Block column index currently being processes
            rocsparse_int block_col = bsr_col_ind[j] - idx_base;

            // Beginning of the row that corresponds to block_col
            rocsparse_int local_block_begin = bsr_row_ptr[block_col] - idx_base;

            // Diagonal entry point of row block_col
            rocsparse_int local_block_diag = bsr_diag_ind[block_col];

            // Structural zero pivot, do not process this row
            if(local_block_diag == -1)
            {
                // If one thread in the warp breaks here, then all threads in
                // the warp break so no divergence
                break;
            }

            for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
            {
                if(direction == rocsparse_direction_row)
                {
                    values[tidy][q]
                        = (tidy < block_dim)
                              ? bsr_val[block_dim * block_dim * j + block_dim * tidy + q]
                              : static_cast<T>(0);
                }
                else
                {
                    values[tidy][q]
                        = (tidy < block_dim)
                              ? bsr_val[block_dim * block_dim * j + block_dim * q + tidy]
                              : static_cast<T>(0);
                }

                temp[tidy][q] = static_cast<T>(0);
            }

            rocsparse_int count = 0;
            rocsparse_int l     = local_block_begin;
            rocsparse_int k     = 0;
            rocsparse_int col_k = columns[k];

            while(l <= local_block_diag && col_k <= block_col)
            {
                rocsparse_int col_l = bsr_col_ind[l] - idx_base;
                col_k               = columns[k];

                if(col_l < col_k)
                {
                    l++;
                }
                else if(col_l > col_k)
                {
                    k++;
                }
                else
                {
                    index[count]       = block_dim * block_dim * (k + block_row_begin);
                    local_index[count] = block_dim * block_dim * l;

                    k++;
                    l++;

                    count++;
                }
            }

            __threadfence_block();

            // Spin loop until dependency has been resolved
            while(!__hip_atomic_load(
                &block_done[block_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                ;

            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
            {
                if(direction == rocsparse_direction_row)
                {
                    local_values[tidy][q] = (tidy < block_dim)
                                                ? bsr_val[block_dim * block_dim * local_block_diag
                                                          + block_dim * tidy + q]
                                                : static_cast<T>(0);
                }
                else
                {
                    local_values[tidy][q] = (tidy < block_dim)
                                                ? bsr_val[block_dim * block_dim * local_block_diag
                                                          + block_dim * q + tidy]
                                                : static_cast<T>(0);
                }
            }

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            for(rocsparse_int l = 0; l < count - 1; l++)
            {
                rocsparse_int idx2 = local_index[l];
                rocsparse_int idx  = index[l];

                for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
                {
                    // Local row sum
                    T local_sum = static_cast<T>(0);

                    for(rocsparse_int p = 0; p < block_dim; p++)
                    {
                        if(direction == rocsparse_direction_row)
                        {
                            T v1      = bsr_val[idx2 + block_dim * q + p];
                            T v2      = (tidy < block_dim) ? bsr_val[idx + block_dim * tidy + p]
                                                           : static_cast<T>(0);
                            local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                        }
                        else
                        {
                            T v1      = bsr_val[idx2 + block_dim * p + q];
                            T v2      = (tidy < block_dim) ? bsr_val[idx + block_dim * p + tidy]
                                                           : static_cast<T>(0);
                            local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                        }
                    }

                    temp[tidy][q] += local_sum;
                }
            }

            __threadfence_block();

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                // Current value
                T val = values[tidy][k];

                // Load diagonal entry
                T diag_val = local_values[k][k];

                // Row has numerical zero pivot
                if(diag_val == static_cast<T>(0))
                {
                    if(tidx == 0 && tidy == 0)
                    {
                        // We are looking for the first zero pivot
                        rocsparse::atomic_min(zero_pivot, block_col + idx_base);
                    }

                    diag_val = static_cast<T>(1);
                }

                // Local row sum
                T local_sum = temp[tidy][k];

                for(rocsparse_int p = 0; p < k; p++)
                {
                    T v1      = local_values[k][p];
                    T v2      = values[tidy][p];
                    local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                }

                // Compute the Cholesky factor and writes it to global memory
                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
                {
                    row_sum[tidy][q]
                        = rocsparse::fma(val, rocsparse::conj(values[q][k]), row_sum[tidy][q]);
                }

                __threadfence_block();
            }

            // Write values back to global memory
            for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
            {
                if(tidy < block_dim)
                {
                    if(direction == rocsparse_direction_row)
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * tidy + q] = values[tidy][q];
                    }
                    else
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * q + tidy] = values[tidy][q];
                    }
                }
            }

            __threadfence();
        }

        // Load current diagonal block into shared memory
        for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
        {
            if(direction == rocsparse_direction_row)
            {
                values[tidy][q]
                    = (tidy < block_dim)
                          ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidy + q]
                          : static_cast<T>(0);
            }
            else
            {
                values[tidy][q]
                    = (tidy < block_dim)
                          ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * q + tidy]
                          : static_cast<T>(0);
            }
        }

        __threadfence_block();

        // Handle diagonal block column of block row.
        for(rocsparse_int k = 0; k < block_dim; k++)
        {
            if(k == tidy)
            {
                values[k][k] = rocsparse::sqrt(rocsparse::abs(values[k][k] - row_sum[k][k]));
            }

            __threadfence_block();

            // Load value
            T val = values[tidy][k];

            // Load diagonal entry
            T diag_val = values[k][k];

            // Row has numerical zero pivot
            if(diag_val == static_cast<T>(0))
            {
                if(tidx == 0 && tidy == 0)
                {
                    // We are looking for the first zero pivot
                    rocsparse::atomic_min(zero_pivot, block_row + idx_base);
                }

                // Normally would break here but to avoid divergence set diag_val to one and continue
                // The zero pivot has already been set so further computation does not matter
                diag_val = static_cast<T>(1);
            }

            // Local row sum
            T local_sum = row_sum[tidy][k];

            if(k < tidy)
            {
                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
                {
                    row_sum[tidy][q]
                        = rocsparse::fma(val, rocsparse::conj(values[q][k]), row_sum[tidy][q]);
                }
            }

            __threadfence_block();
        }

        // Write values back to global memory
        for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
        {
            if(tidy < block_dim)
            {
                if(direction == rocsparse_direction_row)
                {
                    bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidy + q]
                        = values[tidy][q];
                }
                else
                {
                    bsr_val[block_dim * block_dim * block_row_diag + block_dim * q + tidy]
                        = values[tidy][q];
                }
            }
        }

        if(tidx == 0 && tidy == 0)
        {
            // Last lane in wavefront writes "we are done" flag for its block row
            __hip_atomic_store(
                &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsric0_17_32_kernel(rocsparse_direction direction,
                             rocsparse_int       mb,
                             rocsparse_int       block_dim,
                             const rocsparse_int* __restrict__ bsr_row_ptr,
                             const rocsparse_int* __restrict__ bsr_col_ind,
                             T* __restrict__ bsr_val,
                             const rocsparse_int* __restrict__ bsr_diag_ind,
                             int* __restrict__ block_done,
                             const rocsparse_int* __restrict__ block_map,
                             rocsparse_int* __restrict__ zero_pivot,
                             rocsparse_index_base idx_base)
    {
        constexpr static uint32_t DIMX = BLOCKSIZE / BSRDIM;
        constexpr static uint32_t DIMY = BSRDIM;

        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int tidy = hipThreadIdx_y;
        rocsparse_int tid  = DIMX * tidy + tidx;

        __shared__ rocsparse_int columns[MAX_NNZB];
        __shared__ rocsparse_int index[MAX_NNZB];
        __shared__ rocsparse_int local_index[MAX_NNZB];
        __shared__ T             row_sum[BSRDIM][BSRDIM + 1];
        __shared__ T             temp[BSRDIM][BSRDIM + 1];
        __shared__ T             values[BSRDIM][BSRDIM + 1];

        // Current block row this wavefront is working on
        rocsparse_int block_row = block_map[hipBlockIdx_x];

        // Block diagonal entry point of the current block row
        rocsparse_int block_row_diag = bsr_diag_ind[block_row];

        // If one thread in the warp breaks here, then all threads in
        // the warp break so no divergence
        if(block_row_diag == -1)
        {
            if(tidx == 0 && tidy == 0)
            {
                rocsparse::atomic_min(zero_pivot, block_row + idx_base);

                // Last lane in wavefront writes "we are done" flag for its block row
                __hip_atomic_store(
                    &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }

            return;
        }

        // Block row entry point
        rocsparse_int block_row_begin = bsr_row_ptr[block_row] - idx_base;

        // Write current block row column indices to shared memory
        for(rocsparse_int j = block_row_begin + tid; j < block_row_diag + 1; j += DIMX * DIMY)
        {
            columns[j - block_row_begin] = bsr_col_ind[j] - idx_base;
        }

        // Block row sum accumulator
        for(rocsparse_int i = tidx; i < BSRDIM; i += DIMX)
        {
            row_sum[tidy][i] = static_cast<T>(0);
        }

        __threadfence_block();

        // Loop over non-diagonal block columns of current block row
        for(rocsparse_int j = block_row_begin; j < block_row_diag; j++)
        {
            // Block column index currently being processes
            rocsparse_int block_col = bsr_col_ind[j] - idx_base;

            // Beginning of the row that corresponds to block_col
            rocsparse_int local_block_begin = bsr_row_ptr[block_col] - idx_base;

            // Diagonal entry point of row block_col
            rocsparse_int local_block_diag = bsr_diag_ind[block_col];

            // Structural zero pivot, do not process this row
            if(local_block_diag == -1)
            {
                // If one thread in the warp breaks here, then all threads in
                // the warp break so no divergence
                break;
            }

            for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
            {
                if(direction == rocsparse_direction_row)
                {
                    values[tidy][q]
                        = (tidy < block_dim)
                              ? bsr_val[block_dim * block_dim * j + block_dim * tidy + q]
                              : static_cast<T>(0);
                }
                else
                {
                    values[tidy][q]
                        = (tidy < block_dim)
                              ? bsr_val[block_dim * block_dim * j + block_dim * q + tidy]
                              : static_cast<T>(0);
                }

                temp[tidy][q] = static_cast<T>(0);
            }

            rocsparse_int count = 0;
            rocsparse_int l     = local_block_begin;
            rocsparse_int k     = 0;
            rocsparse_int col_k = columns[k];

            while(l <= local_block_diag && col_k <= block_col)
            {
                rocsparse_int col_l = bsr_col_ind[l] - idx_base;
                col_k               = columns[k];

                if(col_l < col_k)
                {
                    l++;
                }
                else if(col_l > col_k)
                {
                    k++;
                }
                else
                {
                    index[count]       = block_dim * block_dim * (k + block_row_begin);
                    local_index[count] = block_dim * block_dim * l;

                    k++;
                    l++;

                    count++;
                }
            }

            __threadfence_block();

            // Spin loop until dependency has been resolved
            while(!__hip_atomic_load(
                &block_done[block_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                ;

            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            for(rocsparse_int l = 0; l < count - 1; l++)
            {
                rocsparse_int idx2 = local_index[l];
                rocsparse_int idx  = index[l];

                for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
                {
                    // Local row sum
                    T local_sum = static_cast<T>(0);

                    for(rocsparse_int p = 0; p < block_dim; p++)
                    {
                        if(direction == rocsparse_direction_row)
                        {
                            T v1      = bsr_val[idx2 + block_dim * q + p];
                            T v2      = (tidy < block_dim) ? bsr_val[idx + block_dim * tidy + p]
                                                           : static_cast<T>(0);
                            local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                        }
                        else
                        {
                            T v1      = bsr_val[idx2 + block_dim * p + q];
                            T v2      = (tidy < block_dim) ? bsr_val[idx + block_dim * p + tidy]
                                                           : static_cast<T>(0);
                            local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                        }
                    }

                    temp[tidy][q] += local_sum;
                }
            }

            __threadfence_block();

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                // Current value
                T val = values[tidy][k];

                // Load diagonal entry
                T diag_val = bsr_val[block_dim * block_dim * local_block_diag + block_dim * k + k];

                // Row has numerical zero pivot
                if(diag_val == static_cast<T>(0))
                {
                    if(tidx == 0 && tidy == 0)
                    {
                        // We are looking for the first zero pivot
                        rocsparse::atomic_min(zero_pivot, block_col + idx_base);
                    }

                    diag_val = static_cast<T>(1);
                }

                // Local row sum
                T local_sum = temp[tidy][k];

                for(rocsparse_int p = 0; p < k; p++)
                {
                    if(direction == rocsparse_direction_row)
                    {
                        T v1
                            = bsr_val[block_dim * block_dim * local_block_diag + block_dim * k + p];
                        T v2      = values[tidy][p];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                    else
                    {
                        T v1
                            = bsr_val[block_dim * block_dim * local_block_diag + block_dim * p + k];
                        T v2      = values[tidy][p];
                        local_sum = rocsparse::fma(v1, rocsparse::conj(v2), local_sum);
                    }
                }

                // Compute the Cholesky factor and writes it to global memory
                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
                {
                    row_sum[tidy][q]
                        = rocsparse::fma(val, rocsparse::conj(values[q][k]), row_sum[tidy][q]);
                }

                __threadfence_block();
            }

            // Write values back to global memory
            for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
            {
                if(tidy < block_dim)
                {
                    if(direction == rocsparse_direction_row)
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * tidy + q] = values[tidy][q];
                    }
                    else
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * q + tidy] = values[tidy][q];
                    }
                }
            }

            __threadfence();
        }

        // Load current diagonal block into shared memory
        for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
        {
            if(direction == rocsparse_direction_row)
            {
                values[tidy][q]
                    = (tidy < block_dim)
                          ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidy + q]
                          : static_cast<T>(0);
            }
            else
            {
                values[tidy][q]
                    = (tidy < block_dim)
                          ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * q + tidy]
                          : static_cast<T>(0);
            }
        }

        __threadfence_block();

        // Handle diagonal block column of block row.
        for(rocsparse_int k = 0; k < block_dim; k++)
        {
            if(k == tidy)
            {
                values[k][k] = rocsparse::sqrt(rocsparse::abs(values[k][k] - row_sum[k][k]));
            }

            __threadfence_block();

            // Load value
            T val = values[tidy][k];

            // Load diagonal entry
            T diag_val = values[k][k];

            // Row has numerical zero pivot
            if(diag_val == static_cast<T>(0))
            {
                if(tidx == 0 && tidy == 0)
                {
                    // We are looking for the first zero pivot
                    rocsparse::atomic_min(zero_pivot, block_row + idx_base);
                }

                // Normally would break here but to avoid divergence set diag_val to one and continue
                // The zero pivot has already been set so further computation does not matter
                diag_val = static_cast<T>(1);
            }

            // Local row sum
            T local_sum = row_sum[tidy][k];

            if(k < tidy)
            {
                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
                {
                    row_sum[tidy][q]
                        = rocsparse::fma(val, rocsparse::conj(values[q][k]), row_sum[tidy][q]);
                }
            }

            __threadfence_block();
        }

        // Write values back to global memory
        for(rocsparse_int q = tidx; q < block_dim; q += DIMX)
        {
            if(tidy < block_dim)
            {
                if(direction == rocsparse_direction_row)
                {
                    bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidy + q]
                        = values[tidy][q];
                }
                else
                {
                    bsr_val[block_dim * block_dim * block_row_diag + block_dim * q + tidy]
                        = values[tidy][q];
                }
            }
        }

        if(tidx == 0 && tidy == 0)
        {
            // First lane in wavefront writes "we are done" flag for its block row
            __hip_atomic_store(
                &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsric0_binsearch_kernel(rocsparse_direction direction,
                                 rocsparse_int       mb,
                                 rocsparse_int       block_dim,
                                 const rocsparse_int* __restrict__ bsr_row_ptr,
                                 const rocsparse_int* __restrict__ bsr_col_ind,
                                 T* __restrict__ bsr_val,
                                 const rocsparse_int* __restrict__ bsr_diag_ind,
                                 int* __restrict__ block_done,
                                 const rocsparse_int* __restrict__ block_map,
                                 rocsparse_int* __restrict__ zero_pivot,
                                 rocsparse_index_base idx_base)
    {
        int lid = hipThreadIdx_x & (WFSIZE - 1);
        int wid = hipThreadIdx_x / WFSIZE;

        rocsparse_int idx = hipBlockIdx_x + wid;

        // Current block row this wavefront is working on
        rocsparse_int block_row = block_map[idx];

        // Block diagonal entry point of the current block row
        rocsparse_int block_row_diag = bsr_diag_ind[block_row];

        // If one thread in the warp breaks here, then all threads in
        // the warp break so no divergence
        if(block_row_diag == -1)
        {
            if(lid == WFSIZE - 1)
            {
                rocsparse::atomic_min(zero_pivot, block_row + idx_base);

                // Last lane in wavefront writes "we are done" flag for its block row
                __hip_atomic_store(
                    &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }

            return;
        }

        // Block row entry point
        rocsparse_int block_row_begin = bsr_row_ptr[block_row] - idx_base;
        rocsparse_int block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;

        for(rocsparse_int row = lid; row < block_dim; row += WFSIZE)
        {
            // Row sum accumulator
            T row_sum = static_cast<T>(0);

            // Loop over block columns of current block row
            for(rocsparse_int j = block_row_begin; j < block_row_diag; j++)
            {
                // Block column index currently being processes
                rocsparse_int block_col = bsr_col_ind[j] - idx_base;

                // Beginning of the block row that corresponds to block_col
                rocsparse_int local_block_begin = bsr_row_ptr[block_col] - idx_base;

                // Block diagonal entry point of block row 'block_col'
                rocsparse_int local_block_diag = bsr_diag_ind[block_col];

                // Structural zero pivot, do not process this block row
                if(local_block_diag == -1)
                {
                    // If one thread in the warp breaks here, then all threads in
                    // the warp break so no divergence
                    break;
                }

                // Spin loop until dependency has been resolved

                (void)rocsparse::spin_loop<SLEEP>(&block_done[block_col], __HIP_MEMORY_SCOPE_AGENT);
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

                for(rocsparse_int k = 0; k < block_dim; k++)
                {
                    // Column index currently being processes
                    rocsparse_int col = block_dim * block_col + k;

                    // Load diagonal entry
                    T diag_val
                        = bsr_val[block_dim * block_dim * local_block_diag + block_dim * k + k];

                    // Row has numerical zero pivot
                    if(diag_val == static_cast<T>(0))
                    {
                        if(lid == 0)
                        {
                            // We are looking for the first zero pivot
                            rocsparse::atomic_min(zero_pivot, block_col + idx_base);
                        }

                        // Normally would break here but to avoid divergence set diag_val to one and continue
                        // The zero pivot has already been set so further computation does not matter
                        diag_val = static_cast<T>(1);
                    }

                    T val = static_cast<T>(0);

                    // Corresponding value
                    if(direction == rocsparse_direction_row)
                    {
                        val = bsr_val[block_dim * block_dim * j + block_dim * row + k];
                    }
                    else
                    {
                        val = bsr_val[block_dim * block_dim * j + block_dim * k + row];
                    }

                    // Local row sum
                    T local_sum = static_cast<T>(0);

                    // Loop over the row the current column index depends on
                    // Each lane processes one entry
                    for(rocsparse_int p = local_block_begin; p < local_block_diag + 1; p++)
                    {
                        // Perform a binary search to find matching block columns
                        rocsparse_int l = block_row_begin;
                        rocsparse_int r = block_row_end - 1;
                        rocsparse_int m = (r + l) >> 1;

                        rocsparse_int block_col_j = bsr_col_ind[m] - idx_base;
                        rocsparse_int block_col_p = bsr_col_ind[p] - idx_base;

                        // Binary search for block column
                        while(l < r)
                        {
                            if(block_col_j < block_col_p)
                            {
                                l = m + 1;
                            }
                            else
                            {
                                r = m;
                            }

                            m           = (r + l) >> 1;
                            block_col_j = bsr_col_ind[m] - idx_base;
                        }

                        // Check if a match has been found
                        if(block_col_j == block_col_p)
                        {
                            for(rocsparse_int q = 0; q < block_dim; q++)
                            {
                                if(block_dim * block_col_p + q < col)
                                {
                                    T vp = static_cast<T>(0);
                                    T vj = static_cast<T>(0);
                                    if(direction == rocsparse_direction_row)
                                    {
                                        vp = bsr_val[block_dim * block_dim * p + block_dim * k + q];
                                        vj = bsr_val[block_dim * block_dim * m + block_dim * row
                                                     + q];
                                    }
                                    else
                                    {
                                        vp = bsr_val[block_dim * block_dim * p + block_dim * q + k];
                                        vj = bsr_val[block_dim * block_dim * m + block_dim * q
                                                     + row];
                                    }

                                    // If a match has been found, do linear combination
                                    local_sum = rocsparse::fma(vp, rocsparse::conj(vj), local_sum);
                                }
                            }
                        }
                    }

                    val     = (val - local_sum) / diag_val;
                    row_sum = rocsparse::fma(val, rocsparse::conj(val), row_sum);

                    if(direction == rocsparse_direction_row)
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * row + k] = val;
                    }
                    else
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * k + row] = val;
                    }
                }
            }

            // Handle diagonal block column of block row
            for(rocsparse_int j = 0; j < block_dim; j++)
            {
                rocsparse_int row_diag = block_dim * block_dim * block_row_diag + block_dim * j + j;

                // Check if 'col' row is complete
                if(j == row)
                {
                    bsr_val[row_diag]
                        = rocsparse::sqrt(rocsparse::abs(bsr_val[row_diag] - row_sum));
                }

                // Ensure previous writes to global memory are seen by all threads
                __threadfence();

                // Load diagonal entry
                T diag_val = bsr_val[row_diag];

                // Row has numerical zero pivot
                if(diag_val == static_cast<T>(0))
                {
                    if(lid == 0)
                    {
                        // We are looking for the first zero pivot
                        rocsparse::atomic_min(zero_pivot, block_row + idx_base);
                    }

                    // Normally would break here but to avoid divergence set diag_val to one and continue
                    // The zero pivot has already been set so further computation does not matter
                    diag_val = static_cast<T>(1);
                }

                if(j < row)
                {
                    // Current value
                    T val = static_cast<T>(0);

                    // Corresponding value
                    if(direction == rocsparse_direction_row)
                    {
                        val = bsr_val[block_dim * block_dim * block_row_diag + block_dim * row + j];
                    }
                    else
                    {
                        val = bsr_val[block_dim * block_dim * block_row_diag + block_dim * j + row];
                    }

                    // Local row sum
                    T local_sum = static_cast<T>(0);

                    T vk = static_cast<T>(0);
                    T vj = static_cast<T>(0);
                    for(rocsparse_int k = block_row_begin; k < block_row_diag; k++)
                    {
                        for(rocsparse_int q = 0; q < block_dim; q++)
                        {
                            if(direction == rocsparse_direction_row)
                            {
                                vk = bsr_val[block_dim * block_dim * k + block_dim * j + q];
                                vj = bsr_val[block_dim * block_dim * k + block_dim * row + q];
                            }
                            else
                            {
                                vk = bsr_val[block_dim * block_dim * k + block_dim * q + j];
                                vj = bsr_val[block_dim * block_dim * k + block_dim * q + row];
                            }

                            // If a match has been found, do linear combination
                            local_sum = rocsparse::fma(vk, rocsparse::conj(vj), local_sum);
                        }
                    }

                    for(rocsparse_int q = 0; q < j; q++)
                    {
                        if(direction == rocsparse_direction_row)
                        {
                            vk = bsr_val[block_dim * block_dim * block_row_diag + block_dim * j
                                         + q];
                            vj = bsr_val[block_dim * block_dim * block_row_diag + block_dim * row
                                         + q];
                        }
                        else
                        {
                            vk = bsr_val[block_dim * block_dim * block_row_diag + block_dim * q
                                         + j];
                            vj = bsr_val[block_dim * block_dim * block_row_diag + block_dim * q
                                         + row];
                        }

                        // If a match has been found, do linear combination
                        local_sum = rocsparse::fma(vk, rocsparse::conj(vj), local_sum);
                    }

                    val     = (val - local_sum) / diag_val;
                    row_sum = rocsparse::fma(val, rocsparse::conj(val), row_sum);

                    if(direction == rocsparse_direction_row)
                    {
                        bsr_val[block_dim * block_dim * block_row_diag + block_dim * row + j] = val;
                    }
                    else
                    {
                        bsr_val[block_dim * block_dim * block_row_diag + block_dim * j + row] = val;
                    }
                }

                __threadfence();
            }
        }

        if(lid == WFSIZE - 1)
        {
            // Last lane writes "we are done" flag for current block row
            __hip_atomic_store(
                &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}
