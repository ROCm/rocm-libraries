/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.hpp"
#include "rocsparse_csric0_strided_batched.hpp"
#include "rocsparse_csric0_strided_batched_launch_kernel.hpp"

#include "rocsparse_common.hpp"

namespace rocsparse
{
    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsric0_2_8_unrolled_device(rocsparse_direction direction,
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
    void bsric0_2_8_unrolled_strided_batched_kernel(rocsparse_direction direction,
                                                    rocsparse_int       batch_count,
                                                    rocsparse_int       mb,
                                                    rocsparse_int       block_dim,
                                                    const rocsparse_int* __restrict__ bsr_row_ptr,
                                                    const rocsparse_int* __restrict__ bsr_col_ind,
                                                    T* __restrict__ bsr_val,
                                                    int64_t bsr_val_stride,
                                                    const rocsparse_int* __restrict__ bsr_diag_ind,
                                                    int* __restrict__ block_done,
                                                    int64_t block_done_stride,
                                                    const rocsparse_int* __restrict__ block_map,
                                                    rocsparse_int* __restrict__ zero_pivot,
                                                    int64_t              zero_pivot_stride,
                                                    rocsparse_index_base idx_base)
    {
        const auto i = hipBlockIdx_y;
        bsric0_2_8_unrolled_device<BLOCKSIZE, MAX_NNZB, BSRDIM, T>(direction,
                                                                   batch_count,
                                                                   mb,
                                                                   block_dim,
                                                                   bsr_row_ptr,
                                                                   bsr_col_ind,
                                                                   bsr_val + i * bsr_val_stride,
                                                                   bsr_diag_ind,
                                                                   block_done
                                                                       + i * block_done_stride,
                                                                   block_map,
                                                                   zero_pivot,
                                                                   zero_pivot_stride,
                                                                   idx_base);
    }

    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    void bsric0_2_8_unrolled_strided_batched_launch_kernel(
        rocsparse_direction direction,
        rocsparse_int       batch_count,
        rocsparse_int       mb,
        rocsparse_int       block_dim,
        const rocsparse_int* __restrict__ bsr_row_ptr,
        const rocsparse_int* __restrict__ bsr_col_ind,
        T* __restrict__ bsr_val,
        int64_t bsr_val_stride,
        const rocsparse_int* __restrict__ bsr_diag_ind,
        int* __restrict__ block_done,
        int64_t block_done_stride,
        const rocsparse_int* __restrict__ block_map,
        rocsparse_int* __restrict__ zero_pivot,
        int64_t              zero_pivot_stride,
        rocsparse_index_base idx_base)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsric0_2_8_unrolled_kernel<block_size, maz_nnzb, bsr_block_dim>),
            dim3(mb),
            dim3(bsr_block_dim, bsr_block_dim),
            0,
            handle->stream,
            dir,
            mb,
            block_dim,
            bsr_row_ptr,
            bsr_col_ind,
            bsr_val,
            (const rocsparse_int*)trm_info->get_diag_ind(),
            done_array,
            (const rocsparse_int*)trm_info->get_row_map(),
            (rocsparse_int*)zero_pivot,
            base);
    }

    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    ROCSPARSE_DEVICE_ILF void bsric0_2_8_device(rocsparse_direction direction,
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
    void bsric0_2_8_strided_batched_kernel(rocsparse_direction direction,
                                           rocsparse_int       batch_count,
                                           rocsparse_int       mb,
                                           rocsparse_int       block_dim,
                                           const rocsparse_int* __restrict__ bsr_row_ptr,
                                           const rocsparse_int* __restrict__ bsr_col_ind,
                                           T* __restrict__ bsr_val,
                                           int64_t bsr_val_stride,
                                           const rocsparse_int* __restrict__ bsr_diag_ind,
                                           int* __restrict__ block_done,
                                           int64_t block_done_stride,
                                           const rocsparse_int* __restrict__ block_map,
                                           rocsparse_int* __restrict__ zero_pivot,
                                           int64_t              zero_pivot_stride,
                                           rocsparse_index_base idx_base)
    {
        const auto i = hipBlockIdx_y;
        bsric0_2_8_device<BLOCKSIZE, MAX_NNZB, BSRDIM, T>(direction,
                                                          batch_count,
                                                          mb,
                                                          block_dim,
                                                          bsr_row_ptr,
                                                          bsr_col_ind,
                                                          bsr_val + i * bsr_val_stride,
                                                          bsr_diag_ind,
                                                          block_done + i * block_done_stride,
                                                          block_map,
                                                          zero_pivot,
                                                          zero_pivot_stride,
                                                          idx_base);
    }

    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    ROCSPARSE_DEVICE_ILF void bsric0_9_16_device(rocsparse_direction direction,
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
    void bsric0_9_16_strided_batched_kernel(rocsparse_direction direction,
                                            rocsparse_int       batch_count,
                                            rocsparse_int       mb,
                                            rocsparse_int       block_dim,
                                            const rocsparse_int* __restrict__ bsr_row_ptr,
                                            const rocsparse_int* __restrict__ bsr_col_ind,
                                            T* __restrict__ bsr_val,
                                            int64_t bsr_val_stride,
                                            const rocsparse_int* __restrict__ bsr_diag_ind,
                                            int* __restrict__ block_done,
                                            int64_t block_done_stride,
                                            const rocsparse_int* __restrict__ block_map,
                                            rocsparse_int* __restrict__ zero_pivot,
                                            int64_t              zero_pivot_stride,
                                            rocsparse_index_base idx_base)
    {
        const auto i = hipBlockIdx_y;
        bsric0_9_16_device<BLOCKSIZE, MAX_NNZB, BSRDIM, T>(direction,
                                                           batch_count,
                                                           mb,
                                                           block_dim,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           bsr_val + i * bsr_val_stride,
                                                           bsr_diag_ind,
                                                           block_done + i * block_done_stride,
                                                           block_map,
                                                           zero_pivot,
                                                           zero_pivot_stride,
                                                           idx_base);
    }

    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    ROCSPARSE_DEVICE_ILF void bsric0_17_32_device(rocsparse_direction direction,
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

    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsric0_17_32_strided_batched_kernel(rocsparse_direction direction,
                                             rocsparse_int       batch_count,
                                             rocsparse_int       mb,
                                             rocsparse_int       block_dim,
                                             const rocsparse_int* __restrict__ bsr_row_ptr,
                                             const rocsparse_int* __restrict__ bsr_col_ind,
                                             T* __restrict__ bsr_val,
                                             int64_t bsr_val_stride,
                                             const rocsparse_int* __restrict__ bsr_diag_ind,
                                             int* __restrict__ block_done,
                                             int64_t block_done_stride,
                                             const rocsparse_int* __restrict__ block_map,
                                             rocsparse_int* __restrict__ zero_pivot,
                                             int64_t              zero_pivot_stride,
                                             rocsparse_index_base idx_base)
    {
        const auto i = hipBlockIdx_y;
        bsric0_17_32_device<BLOCKSIZE, MAX_NNZB, BSRDIM, T>(direction,
                                                            batch_count,
                                                            mb,
                                                            block_dim,
                                                            bsr_row_ptr,
                                                            bsr_col_ind,
                                                            bsr_val + i * bsr_val_stride,
                                                            bsr_diag_ind,
                                                            block_done + i * block_done_stride,
                                                            block_map,
                                                            zero_pivot,
                                                            zero_pivot_stride,
                                                            idx_base);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsric0_binsearch_device(rocsparse_direction direction,
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
    template <rocsparse_int BLOCKSIZE, rocsparse_int MAX_NNZB, rocsparse_int BSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsric0_binsearch_strided_batched_kernel(rocsparse_direction direction,
                                                 rocsparse_int       batch_count,
                                                 rocsparse_int       mb,
                                                 rocsparse_int       block_dim,
                                                 const rocsparse_int* __restrict__ bsr_row_ptr,
                                                 const rocsparse_int* __restrict__ bsr_col_ind,
                                                 T* __restrict__ bsr_val,
                                                 int64_t bsr_val_stride,
                                                 const rocsparse_int* __restrict__ bsr_diag_ind,
                                                 int* __restrict__ block_done,
                                                 int64_t block_done_stride,
                                                 const rocsparse_int* __restrict__ block_map,
                                                 rocsparse_int* __restrict__ zero_pivot,
                                                 int64_t              zero_pivot_stride,
                                                 rocsparse_index_base idx_base)
    {
        const auto i = hipBlockIdx_y;
        bsric0_binsearch_device<BLOCKSIZE, MAX_NNZB, BSRDIM, T>(direction,
                                                                batch_count,
                                                                mb,
                                                                block_dim,
                                                                bsr_row_ptr,
                                                                bsr_col_ind,
                                                                bsr_val + i * bsr_val_stride,
                                                                bsr_diag_ind,
                                                                block_done + i * block_done_stride,
                                                                block_map,
                                                                zero_pivot,
                                                                zero_pivot_stride,
                                                                idx_base);
    }

    template <typename T>
    inline void bsric0_launcher(rocsparse_handle       handle,
                                rocsparse_direction    dir,
                                rocsparse_int          mb,
                                rocsparse_int          max_nnzb,
                                rocsparse_index_base   base,
                                T*                     bsr_val,
                                const rocsparse_int*   bsr_row_ptr,
                                const rocsparse_int*   bsr_col_ind,
                                rocsparse_int          block_dim,
                                rocsparse::trm_info_t* trm_info,
                                void*                  zero_pivot,
                                int*                   done_array)
    {
        ROCSPARSE_ROUTINE_TRACE;

        dim3 bsric0_blocks(mb);

        if(handle->wavefront_size == 32)
        {
            LAUNCH_BSRIC_33_inf(T, 32, 32, false);
        }
        else
        {

            const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
            if(gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2)
            {
                LAUNCH_BSRIC_33_inf(T, 64, 64, true);
            }
            else
            {
                if(max_nnzb <= 32)
                {
                    if(block_dim == 1)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 1, 32, 1);
                    }
                    else if(block_dim == 2)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 4, 32, 2);
                    }
                    else if(block_dim == 3)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 9, 32, 3);
                    }
                    else if(block_dim == 4)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 16, 32, 4);
                    }
                    else if(block_dim == 5)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 25, 32, 5);
                    }
                    else if(block_dim == 6)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 36, 32, 6);
                    }
                    else if(block_dim == 7)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 49, 32, 7);
                    }
                    else if(block_dim == 8)
                    {
                        LAUNCH_BSRIC_2_8_UNROLLED(T, 64, 32, 8);
                    }
                    else if(block_dim <= 16)
                    {
                        LAUNCH_BSRIC_9_16(T, 64, 32, 16);
                    }
                    else if(block_dim <= 32)
                    {
                        LAUNCH_BSRIC_17_32(T, 64, 32, 32);
                    }
                    else
                    {
                        LAUNCH_BSRIC_33_inf(T, 64, 64, false);
                    }
                }
                else if(max_nnzb <= 64)
                {
                    if(block_dim <= 8)
                    {
                        LAUNCH_BSRIC_2_8(T, 64, 64, 8);
                    }
                    else if(block_dim <= 16)
                    {
                        LAUNCH_BSRIC_9_16(T, 64, 64, 16);
                    }
                    else if(block_dim <= 32)
                    {
                        LAUNCH_BSRIC_17_32(T, 64, 64, 32);
                    }
                    else
                    {
                        LAUNCH_BSRIC_33_inf(T, 64, 64, false);
                    }
                }
                else if(max_nnzb <= 128)
                {
                    if(block_dim <= 8)
                    {
                        LAUNCH_BSRIC_2_8(T, 64, 128, 8);
                    }
                    else if(block_dim <= 16)
                    {
                        LAUNCH_BSRIC_9_16(T, 64, 128, 16);
                    }
                    else if(block_dim <= 32)
                    {
                        LAUNCH_BSRIC_17_32(T, 64, 128, 32);
                    }
                    else
                    {
                        LAUNCH_BSRIC_33_inf(T, 64, 64, false);
                    }
                }
                else
                {
                    LAUNCH_BSRIC_33_inf(T, 64, 64, false);
                }
            }
        }
    }

}

namespace rocsparse
{

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csric0_strided_batched_hash_kernel(J m,
                                            const I* __restrict__ csr_row_ptr,
                                            const J* __restrict__ csr_col_ind,
                                            T* __restrict__ csr_val,
                                            int64_t csr_val_stride,
                                            const I* __restrict__ csr_diag_ind,
                                            int32_t* __restrict__ done,
                                            int64_t done_stride,
                                            const J* __restrict__ map,
                                            J* __restrict__ zero_pivot,
                                            int64_t zero_pivot_stride,
                                            J* __restrict__ singular_pivot,
                                            int64_t              singular_pivot_stride,
                                            double               tol,
                                            rocsparse_index_base idx_base)
    {
        const auto i = hipBlockIdx_y;
        rocsparse::csric0_hash_device<BLOCKSIZE, WFSIZE, HASH>(m,
                                                               csr_row_ptr,
                                                               csr_col_ind,
                                                               csr_val + i * csr_val_stride,
                                                               csr_diag_ind,
                                                               done + i * done_stride,
                                                               map,
                                                               zero_pivot + i * zero_pivot_stride,
                                                               singular_pivot
                                                                   + i * singular_pivot_stride,
                                                               tol,
                                                               idx_base);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csric0_strided_batched_binsearch_kernel(J m,
                                                 const I* __restrict__ csr_row_ptr,
                                                 const J* __restrict__ csr_col_ind,
                                                 T* __restrict__ csr_val,
                                                 int64_t csr_val_stride,
                                                 const I* __restrict__ csr_diag_ind,
                                                 int32_t* __restrict__ done,
                                                 int64_t done_stride,
                                                 const J* __restrict__ map,
                                                 J* __restrict__ zero_pivot,
                                                 int64_t zero_pivot_stride,
                                                 J* __restrict__ singular_pivot,
                                                 int64_t              singular_pivot_stride,
                                                 double               tol,
                                                 rocsparse_index_base idx_base)
    {
        const auto i = hipBlockIdx_y;
        rocsparse::csric0_binsearch_device<BLOCKSIZE, WFSIZE, SLEEP, T, I, J>(
            m,
            csr_row_ptr,
            csr_col_ind,
            csr_val + i * csr_val_stride,
            csr_diag_ind,
            done + i * done_stride,
            map,
            zero_pivot + i * zero_pivot_stride,
            singular_pivot + i * singular_pivot_stride,
            tol,
            idx_base);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    rocsparse_status
        csric0_strided_batched_launch_hash_kernel(rocsparse_handle handle,
                                                  int64_t          batch_count,
                                                  int64_t          m,
                                                  const void* __restrict__ csr_row_ptr,
                                                  const void* __restrict__ csr_col_ind,
                                                  void* __restrict__ csr_val,
                                                  int64_t csr_val_stride,
                                                  const void* __restrict__ csr_diag_ind,
                                                  int32_t* __restrict__ done,
                                                  int64_t done_stride,
                                                  const void* __restrict__ map,
                                                  void* __restrict__ zero_pivot,
                                                  int64_t zero_pivot_stride,
                                                  void* __restrict__ singular_pivot,
                                                  int64_t              singular_pivot_stride,
                                                  double               tol,
                                                  rocsparse_index_base idx_base)
    {
        dim3 csric0_blocks((m * handle->wavefront_size - 1) / BLOCKSIZE + 1, batch_count);
        dim3 csric0_threads(BLOCKSIZE);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csric0_strided_batched_hash_kernel<BLOCKSIZE, WFSIZE, HASH>),
            csric0_blocks,
            csric0_threads,
            0,
            handle->stream,
            static_cast<J>(m),
            reinterpret_cast<const I*>(csr_row_ptr),
            reinterpret_cast<const J*>(csr_col_ind),
            reinterpret_cast<T*>(csr_val),
            csr_val_stride,
            reinterpret_cast<const I*>(csr_diag_ind),
            done,
            done_stride,
            reinterpret_cast<const J*>(map),
            reinterpret_cast<J*>(zero_pivot),
            zero_pivot_stride,
            reinterpret_cast<J*>(singular_pivot),
            singular_pivot_stride,
            tol,
            idx_base);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    rocsparse_status
        csric0_strided_batched_launch_binsearch_kernel(rocsparse_handle handle,
                                                       int64_t          batch_count,
                                                       int64_t          m,
                                                       const void* __restrict__ csr_row_ptr,
                                                       const void* __restrict__ csr_col_ind,
                                                       void* __restrict__ csr_val,
                                                       int64_t csr_val_stride,
                                                       const void* __restrict__ csr_diag_ind,
                                                       int32_t* __restrict__ done,
                                                       int64_t done_stride,
                                                       const void* __restrict__ map,
                                                       void* __restrict__ zero_pivot,
                                                       int64_t zero_pivot_stride,
                                                       void* __restrict__ singular_pivot,
                                                       int64_t              singular_pivot_stride,
                                                       double               tol,
                                                       rocsparse_index_base idx_base)
    {
        dim3 csric0_blocks((m * handle->wavefront_size - 1) / BLOCKSIZE + 1, batch_count);
        dim3 csric0_threads(BLOCKSIZE);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csric0_strided_batched_binsearch_kernel<BLOCKSIZE, WFSIZE, SLEEP, T, I, J>),
            csric0_blocks,
            csric0_threads,
            0,
            handle->stream,
            m,
            reinterpret_cast<const I*>(csr_row_ptr),
            reinterpret_cast<const J*>(csr_col_ind),
            reinterpret_cast<T*>(csr_val),
            csr_val_stride,
            reinterpret_cast<const I*>(csr_diag_ind),
            done,
            done_stride,
            reinterpret_cast<const J*>(map),
            reinterpret_cast<J*>(zero_pivot),
            zero_pivot_stride,
            reinterpret_cast<J*>(singular_pivot),
            singular_pivot_stride,
            tol,
            idx_base);
        return rocsparse_status_success;
    }

    typedef rocsparse_status (*csric0_launch_kernel_t)(rocsparse_handle handle,
                                                       int64_t          batch_count,
                                                       int64_t          m,
                                                       const void* __restrict__ csr_row_ptr,
                                                       const void* __restrict__ csr_col_ind,
                                                       void* __restrict__ csr_val,
                                                       int64_t csr_val_stride,
                                                       const void* __restrict__ csr_diag_ind,
                                                       int32_t* __restrict__ done,
                                                       int64_t done_stride,
                                                       const void* __restrict__ map,
                                                       void* __restrict__ zero_pivot,
                                                       int64_t zero_pivot_stride,
                                                       void* __restrict__ singular_pivot,
                                                       int64_t              singular_pivot_stride,
                                                       double               tol,
                                                       rocsparse_index_base idx_base);

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              bool     SLEEP,
              typename T,
              typename I,
              typename... P>
    static csric0_launch_kernel_t find_csric0_binsearch_j(rocsparse_indextype j, P... p)
    {
        return (j == rocsparse_indextype_i32)
                   ? rocsparse::csric0_strided_batched_launch_binsearch_kernel<BLOCKSIZE,
                                                                               WF_SIZE,
                                                                               SLEEP,
                                                                               T,
                                                                               I,
                                                                               int32_t>
               : (j == rocsparse_indextype_i64)
                   ? rocsparse::csric0_strided_batched_launch_binsearch_kernel<BLOCKSIZE,
                                                                               WF_SIZE,
                                                                               SLEEP,
                                                                               T,
                                                                               I,
                                                                               int64_t>
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename T, typename... P>
    static csric0_launch_kernel_t find_csric0_binsearch_i(rocsparse_indextype i, P... p)
    {
        return (i == rocsparse_indextype_i32)
                   ? rocsparse::find_csric0_binsearch_j<BLOCKSIZE, WF_SIZE, SLEEP, T, int32_t>(p...)
               : (i == rocsparse_indextype_i64)
                   ? rocsparse::find_csric0_binsearch_j<BLOCKSIZE, WF_SIZE, SLEEP, T, int64_t>(p...)
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename... P>
    static csric0_launch_kernel_t find_csric0_binsearch_t(rocsparse_datatype i, P... p)
    {
        return (i == rocsparse_datatype_f32_r)
                   ? rocsparse::find_csric0_binsearch_i<BLOCKSIZE, WF_SIZE, SLEEP, float>(p...)
               : (i == rocsparse_datatype_f32_c) ? rocsparse::
                       find_csric0_binsearch_i<BLOCKSIZE, WF_SIZE, SLEEP, rocsparse_float_complex>(
                           p...)
               : (i == rocsparse_datatype_f64_c) ? rocsparse::
                       find_csric0_binsearch_i<BLOCKSIZE, WF_SIZE, SLEEP, rocsparse_double_complex>(
                           p...)
               : (i == rocsparse_datatype_f64_r)
                   ? rocsparse::find_csric0_binsearch_i<BLOCKSIZE, WF_SIZE, SLEEP, double>(p...)
                   : nullptr;
    }

    static csric0_launch_kernel_t find_csric0_binsearch_kernel(uint32_t            blocksize_,
                                                               uint32_t            wfsize_,
                                                               bool                sleep_,
                                                               rocsparse_datatype  t_type,
                                                               rocsparse_indextype i_type,
                                                               rocsparse_indextype j_type)
    {
        // Determine gcnArch and ASIC revision
        if(blocksize_ == 256 && ((wfsize_ == 32) || (sleep_ == false)))
        {
            return rocsparse::find_csric0_binsearch_t<256, 32, false>(t_type, i_type, j_type);
        }
        else if(blocksize_ == 256 && ((wfsize_ == 64) || (sleep_ == false)))
        {
            return rocsparse::find_csric0_binsearch_t<256, 64, false>(t_type, i_type, j_type);
        }
        else if(blocksize_ == 256 && ((wfsize_ == 64) || (sleep_ == true)))
        {
            return rocsparse::find_csric0_binsearch_t<256, 64, true>(t_type, i_type, j_type);
        }
        else
        {
            return nullptr;
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename... P>
    static csric0_launch_kernel_t find_csric0_hash_kernel_j(const rocsparse_indextype j, P... p)
    {
        return (j == rocsparse_indextype_i32) ? csric0_strided_batched_launch_hash_kernel<BLOCKSIZE,
                                                                                          WF_SIZE,
                                                                                          HASH,
                                                                                          T,
                                                                                          I,
                                                                                          int32_t>
               : (j == rocsparse_indextype_i64)
                   ? csric0_strided_batched_launch_hash_kernel<BLOCKSIZE,
                                                               WF_SIZE,
                                                               HASH,
                                                               T,
                                                               I,
                                                               int64_t>
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t HASH, typename T, typename... P>
    static csric0_launch_kernel_t find_csric0_hash_kernel_i(const rocsparse_indextype i, P... p)
    {
        return (i == rocsparse_indextype_i32)
                   ? rocsparse::find_csric0_hash_kernel_j<BLOCKSIZE, WF_SIZE, HASH, T, int32_t>(
                       p...)
               : (i == rocsparse_indextype_i64)
                   ? rocsparse::find_csric0_hash_kernel_j<BLOCKSIZE, WF_SIZE, HASH, T, int64_t>(
                       p...)
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t HASH, typename... P>
    static csric0_launch_kernel_t find_csric0_hash_kernel_t(const rocsparse_datatype i, P... p)
    {
        return (i == rocsparse_datatype_f32_r)
                   ? rocsparse::find_csric0_hash_kernel_i<BLOCKSIZE, WF_SIZE, HASH, float>(p...)
               : (i == rocsparse_datatype_f32_c) ? rocsparse::
                       find_csric0_hash_kernel_i<BLOCKSIZE, WF_SIZE, HASH, rocsparse_float_complex>(
                           p...)
               : (i == rocsparse_datatype_f64_c)
                   ? rocsparse::find_csric0_hash_kernel_i<BLOCKSIZE,
                                                          WF_SIZE,
                                                          HASH,
                                                          rocsparse_double_complex>(p...)
               : (i == rocsparse_datatype_f64_r)
                   ? rocsparse::find_csric0_hash_kernel_i<BLOCKSIZE, WF_SIZE, HASH, double>(p...)
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename... P>
    static csric0_launch_kernel_t find_csric0_hash_kernel_mxnnz(const int32_t max_nnz, P... p)
    {
        return (max_nnz <= 32)   ? rocsparse::find_csric0_hash_kernel_t<BLOCKSIZE, WF_SIZE, 1>(p...)
               : (max_nnz <= 64) ? rocsparse::find_csric0_hash_kernel_t<BLOCKSIZE, WF_SIZE, 2>(p...)
               : (max_nnz <= 128)
                   ? rocsparse::find_csric0_hash_kernel_t<BLOCKSIZE, WF_SIZE, 4>(p...)
               : (max_nnz <= 256)
                   ? rocsparse::find_csric0_hash_kernel_t<BLOCKSIZE, WF_SIZE, 8>(p...)
               : (max_nnz <= 512)
                   ? rocsparse::find_csric0_hash_kernel_t<BLOCKSIZE, WF_SIZE, 16>(p...)
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, typename... P>
    static csric0_launch_kernel_t find_csric0_hash_kernel_wf(const int32_t i, P... p)
    {
        return (i == 32)   ? rocsparse::find_csric0_hash_kernel_mxnnz<BLOCKSIZE, 32>(p...)
               : (i == 64) ? rocsparse::find_csric0_hash_kernel_mxnnz<BLOCKSIZE, 64>(p...)
                           : nullptr;
    }

    static csric0_launch_kernel_t find_csric0_hash_kernel(const uint32_t            blocksize_,
                                                          const uint32_t            wfsize,
                                                          const uint32_t            max_nnz,
                                                          const rocsparse_datatype  t_type,
                                                          const rocsparse_indextype i_type,
                                                          const rocsparse_indextype j_type)
    {
        if(blocksize_ == 256)
        {
            return rocsparse::find_csric0_hash_kernel_wf<256>(
                wfsize, max_nnz, t_type, i_type, j_type);
        }
        else
        {
            return nullptr;
        }
    }

    rocsparse_status csric0_strided_batched_launch_kernel(rocsparse_handle    handle,
                                                          int64_t             batch_count,
                                                          int64_t             m,
                                                          rocsparse_indextype csr_ptr_row_indextype,
                                                          const void* __restrict__ csr_row_ptr,
                                                          rocsparse_indextype j_type,
                                                          const void* __restrict__ csr_col_ind,
                                                          rocsparse_datatype t_type,
                                                          void* __restrict__ csr_val,
                                                          int64_t csr_val_stride,
                                                          const void* __restrict__ csr_diag_ind,
                                                          int32_t* __restrict__ done,
                                                          int64_t done_stride,
                                                          const void* __restrict__ map,
                                                          void* __restrict__ zero_pivot,
                                                          int64_t zero_pivot_stride,
                                                          void* __restrict__ singular_pivot,
                                                          int64_t singular_pivot_stride,
                                                          double  tol,
                                                          rocsparse_index_base idx_base,
                                                          int64_t              max_nnz);

}

rocsparse_status
    rocsparse::csric0_strided_batched_launch_kernel(rocsparse_handle    handle,
                                                    int64_t             batch_count,
                                                    int64_t             m,
                                                    rocsparse_indextype csr_ptr_row_indextype,
                                                    const void* __restrict__ csr_row_ptr,
                                                    rocsparse_indextype csr_col_ind_indextype,
                                                    const void* __restrict__ csr_col_ind,
                                                    rocsparse_datatype csr_val_datatype,
                                                    void* __restrict__ csr_val,
                                                    int64_t csr_val_stride,
                                                    const void* __restrict__ csr_diag_ind,
                                                    int32_t* __restrict__ done,
                                                    int64_t done_stride,
                                                    const void* __restrict__ map,
                                                    void* __restrict__ zero_pivot,
                                                    int64_t zero_pivot_stride,
                                                    void* __restrict__ singular_pivot,
                                                    int64_t              singular_pivot_stride,
                                                    double               tol,
                                                    rocsparse_index_base idx_base,
                                                    int64_t              max_nnz)
{
    const bool sleep
        = (rocsparse::handle_get_arch_name(handle) == rocpsarse_arch_names::gfx908 && //
           handle->asic_rev < 2);

    auto launch_kernel
        = (sleep || max_nnz > 512)
              ? rocsparse::find_csric0_binsearch_kernel(256,
                                                        (sleep) ? 64 : handle->wavefront_size,
                                                        sleep,
                                                        csr_val_datatype,
                                                        csr_ptr_row_indextype,
                                                        csr_col_ind_indextype)
              : rocsparse::find_csric0_hash_kernel(256,
                                                   handle->wavefront_size,
                                                   max_nnz,
                                                   csr_val_datatype,
                                                   csr_ptr_row_indextype,
                                                   csr_col_ind_indextype);

    RETURN_IF_ROCSPARSE_ERROR(launch_kernel(handle,
                                            batch_count,
                                            m,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            csr_val_stride,
                                            csr_diag_ind,
                                            done,
                                            done_stride,
                                            map,
                                            zero_pivot,
                                            zero_pivot_stride,
                                            singular_pivot,
                                            singular_pivot_stride,
                                            tol,
                                            idx_base));
    return rocsparse_status_success;
}
