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
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I, typename J, typename T>
    ROCSPARSE_DEVICE_ILF void csrsm_device(rocsparse_operation transB,
                                           J                   m,
                                           J                   nrhs,
                                           T                   alpha,
                                           const I* __restrict__ csr_row_ptr,
                                           const J* __restrict__ csr_col_ind,
                                           const T* __restrict__ csr_val,
                                           T* __restrict__ B,
                                           int64_t ldb,
                                           int* __restrict__ done_array,
                                           J* __restrict__ map,
                                           J* __restrict__ zero_pivot,
                                           rocsparse_index_base idx_base,
                                           rocsparse_fill_mode  fill_mode,
                                           rocsparse_diag_type  diag_type)
    {
        // Index into the row map
        const J idx = hipBlockIdx_x % m;

        // Shared memory to hold columns and values
        __shared__ J scsr_col_ind[BLOCKSIZE];
        __shared__ T scsr_val[BLOCKSIZE];

        // Get the row this warp will operate on
        const J row = map[idx];

        // Current row entry point and exit point
        const I row_begin = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        // Column index into B
        const J col_B = (hipBlockIdx_x / m) * BLOCKSIZE + hipThreadIdx_x;

        // Index into B (i,j)
        const int64_t idx_B = row * ldb + col_B;

        // Index into done array
        const J id = (hipBlockIdx_x / m) * m;

        // Initialize local sum with alpha and X
        T local_sum = static_cast<T>(0);
        if(transB == rocsparse_operation_conjugate_transpose)
        {
            local_sum = (col_B < nrhs) ? alpha * rocsparse::conj(B[idx_B]) : static_cast<T>(0);
        }
        else
        {
            local_sum = (col_B < nrhs) ? alpha * B[idx_B] : static_cast<T>(0);
        }

        // Initialize diagonal entry
        T diagonal = static_cast<T>(1);

        for(I j = row_begin; j < row_end; ++j)
        {
            // Project j onto [0, BLOCKSIZE-1]
            const J k = (j - row_begin) & (BLOCKSIZE - 1);

            // Preload column indices and values into shared memory
            // This happens only once for each chunk of BLOCKSIZE elements
            if(k == 0)
            {
                scsr_col_ind[hipThreadIdx_x] = (hipThreadIdx_x < row_end - j)
                                                   ? csr_col_ind[hipThreadIdx_x + j] - idx_base
                                                   : -1;
                scsr_val[hipThreadIdx_x]
                    = (hipThreadIdx_x < row_end - j) ? csr_val[hipThreadIdx_x + j] : -1;
            }

            // Wait for preload to finish
            __syncthreads();

            // Current column this lane operates on
            const J local_col = scsr_col_ind[k];

            // Local value this lane operates with
            T local_val = scsr_val[k];

            // Check for numerical zero
            if(local_val == static_cast<T>(0) && local_col == row
               && diag_type == rocsparse_diag_type_non_unit)
            {
                // Numerical zero pivot found, avoid division by 0
                // and store index for later use.
                if(hipThreadIdx_x == 0)
                {
                    rocsparse::atomic_min(zero_pivot, row + idx_base);
                }

                local_val = static_cast<T>(1);
            }

            // Differentiate upper and lower triangular mode
            if(fill_mode == rocsparse_fill_mode_upper)
            {
                // Processing upper triangular

                // Ignore all entries that are below the diagonal
                if(local_col < row)
                {
                    continue;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    // If diagonal type is non unit, do division by diagonal entry
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        diagonal = static_cast<T>(1) / local_val;
                    }

                    // Skip diagonal entry
                    continue;
                }
            }
            else if(fill_mode == rocsparse_fill_mode_lower)
            {
                // Processing lower triangular

                // Ignore all entries that are above the diagonal
                if(local_col > row)
                {
                    break;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    // If diagonal type is non unit, do division by diagonal entry
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        diagonal = static_cast<T>(1) / local_val;
                    }

                    // Skip diagonal entry
                    break;
                }
            }

            // Spin loop until dependency has been resolved
            if(hipThreadIdx_x == 0)
            {
                rocsparse::spin_loop<SLEEP>(&done_array[local_col + id], __HIP_MEMORY_SCOPE_AGENT);
            }

            // Wait for spin looping thread to finish as the whole block depends on this row
            __syncthreads();

            // Make sure updated B is visible globally
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Index into X
            const int64_t idx_X = local_col * ldb + col_B;

            // Local sum computation for each lane
            local_sum = (col_B < nrhs) ? rocsparse::fma(-local_val, B[idx_X], local_sum)
                                       : static_cast<T>(0);
        }

        // If we have non unit diagonal, take the diagonal into account
        // For unit diagonal, this would be multiplication with one
        if(diag_type == rocsparse_diag_type_non_unit)
        {
            local_sum = local_sum * diagonal;
        }

        // Store result in B
        if(col_B < nrhs)
        {
            B[idx_B] = local_sum;
        }

        // Make sure B is written to global memory before setting row is done flag
        __threadfence();

        // Wait for all threads to finish the threadfence before we mark the row "done"
        __syncthreads();

        if(hipThreadIdx_x == 0)
        {
            // Write the "row is done" flag
            __hip_atomic_store(
                &done_array[row + id], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}
