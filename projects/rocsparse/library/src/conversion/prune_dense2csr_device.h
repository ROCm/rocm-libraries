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
#include "handle.h"

namespace rocsparse
{
    ROCSPARSE_KERNEL(1)
    void nnz_total_device_kernel(rocsparse_int m,
                                 const rocsparse_int* __restrict__ csr_row_ptr,
                                 rocsparse_int* __restrict__ nnz_total_dev_host_ptr)
    {
        if(hipThreadIdx_x == 0)
        {
            *nnz_total_dev_host_ptr = csr_row_ptr[m] - csr_row_ptr[0];
        }
    }

    template <rocsparse_int DIM_X, rocsparse_int DIM_Y, typename T>
    ROCSPARSE_DEVICE_ILF void prune_dense2csr_nnz_device(rocsparse_int m,
                                                         rocsparse_int n,
                                                         const T* __restrict__ A,
                                                         int64_t lda,
                                                         T       threshold,
                                                         rocsparse_int* __restrict__ nnz_per_row)
    {
        rocsparse_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
        rocsparse_int tx        = thread_id % DIM_X;
        rocsparse_int ty        = thread_id / DIM_X;
        rocsparse_int ind       = hipBlockIdx_x * DIM_X * 4 + tx;
        rocsparse_int n_tail    = n % (4 * DIM_Y);
        rocsparse_int col;
        rocsparse_int res_A[4];

        __shared__ rocsparse_int sdata[DIM_X * 4 * DIM_Y];

        for(rocsparse_int k = 0; k < 4; ++k)
        {
            res_A[k] = 0;
        }

        for(col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y)
        {
            for(rocsparse_int k = 0; k < 4; ++k)
            {
                if(ind + k * DIM_X < m)
                {
                    for(rocsparse_int j = 0; j < 4; ++j)
                    {
                        res_A[k]
                            += (rocsparse::abs(A[ind + k * DIM_X + (col + j) * lda]) > threshold)
                                   ? 1
                                   : 0;
                    }
                }
            }
        }

        if(n_tail > 0)
        {
            for(rocsparse_int k = 0; k < 4; ++k)
            {
                if(ind + k * DIM_X < m)
                {
                    for(rocsparse_int j = 0; j < 4; ++j)
                    {
                        if(col + j < n)
                        {
                            res_A[k] += (rocsparse::abs(A[ind + k * DIM_X + (col + j) * lda])
                                         > threshold)
                                            ? 1
                                            : 0;
                        }
                    }
                }
            }
        }

        for(rocsparse_int k = 0; k < 4; ++k)
        {
            sdata[tx + k * DIM_X + ty * DIM_X * 4] = res_A[k];
        }

        __syncthreads();

        ind = hipBlockIdx_x * DIM_X * 4 + thread_id;
        if(thread_id < DIM_X * 4)
        {
            for(rocsparse_int j = 1; j < DIM_Y; j++)
            {
                sdata[thread_id] += sdata[thread_id + DIM_X * 4 * j];
            }

            if(ind < m)
            {
                nnz_per_row[ind] = sdata[thread_id];
            }
        }
    }

    template <rocsparse_int NUMROWS_PER_BLOCK, rocsparse_int WF_SIZE, typename T>
    ROCSPARSE_DEVICE_ILF void prune_dense2csr_device(rocsparse_index_base base,
                                                     rocsparse_int        m,
                                                     rocsparse_int        n,
                                                     const T* __restrict__ dense_val,
                                                     int64_t ld,
                                                     T       threshold,
                                                     T* __restrict__ csr_val,
                                                     const rocsparse_int* __restrict__ csr_row_ptr,
                                                     rocsparse_int* __restrict__ csr_col_ind)
    {
        const rocsparse_int wavefront_index = hipThreadIdx_x / WF_SIZE,
                            lane_index      = hipThreadIdx_x % WF_SIZE;
        const uint64_t      filter          = 0xffffffffffffffff >> (63 - lane_index);
        const rocsparse_int row_index       = NUMROWS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

        if(row_index < m)
        {
            rocsparse_int shift = csr_row_ptr[row_index] - base;

            // The warp handles the entire row.
            for(rocsparse_int column_index = lane_index; column_index < n; column_index += WF_SIZE)
            {
                // Synchronize for cache considerations.
                __syncthreads();

                // Get value.
                const T value = dense_val[row_index + column_index * ld];

                // Predicate.
                const bool predicate = rocsparse::abs(value) > threshold;

                // Mask of the wavefront.
                const uint64_t wavefront_mask = __ballot(predicate);

                // Get the number of previous non-zero in the row.
                const uint64_t count_previous_nnzs = __popcll(wavefront_mask & filter);

                if(predicate)
                {
                    // Calculate local index.
                    const uint64_t local_index_in_warp = count_previous_nnzs - 1;

                    // Populate the sparse matrix.
                    csr_val[shift + local_index_in_warp]     = value;
                    csr_col_ind[shift + local_index_in_warp] = column_index + base;
                }

                // Broadcast the update of the shift to all 64 threads for the next set of 64 columns.
                // Choose the last lane since that it contains the size of the sparse row (even if its predicate is false).
                shift += __shfl(static_cast<int>(count_previous_nnzs), WF_SIZE - 1);
            }
        }
    }
}
