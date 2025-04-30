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

#include <limits>

#include "common.h"

namespace rocsparse
{
    template <rocsparse_int BLOCK_SIZE>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void compute_nnz_from_row_ptr_array_kernel(rocsparse_int m,
                                               const rocsparse_int* __restrict__ csr_row_ptr,
                                               rocsparse_int* nnz)
    {
        rocsparse_int thread_id = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

        if(thread_id == 0)
        {
            *nnz = csr_row_ptr[m] - csr_row_ptr[0];
        }
    }

    template <rocsparse_int BLOCK_SIZE,
              rocsparse_int SEGMENTS_PER_BLOCK,
              rocsparse_int SEGMENT_SIZE,
              rocsparse_int WF_SIZE,
              typename T>
    ROCSPARSE_DEVICE_ILF void nnz_compress_device(rocsparse_int        m,
                                                  rocsparse_index_base idx_base_A,
                                                  const T* __restrict__ csr_val_A,
                                                  const rocsparse_int* __restrict__ csr_row_ptr_A,
                                                  rocsparse_int* __restrict__ nnz_per_row,
                                                  T tol)
    {
        const rocsparse_int segment_id      = hipThreadIdx_x / SEGMENT_SIZE;
        const rocsparse_int segment_lane_id = hipThreadIdx_x % SEGMENT_SIZE;

        const rocsparse_int row_index = SEGMENTS_PER_BLOCK * hipBlockIdx_x + segment_id;

        if(row_index < m)
        {
            const rocsparse_int start_A = csr_row_ptr_A[row_index] - idx_base_A;
            const rocsparse_int end_A   = csr_row_ptr_A[row_index + 1] - idx_base_A;

            rocsparse_int count = 0;

            // One segment per row
            for(rocsparse_int i = start_A + segment_lane_id; i < end_A; i += SEGMENT_SIZE)
            {
                const T value = csr_val_A[i];
                if(rocsparse::abs(value) > rocsparse::real(tol)
                   && rocsparse::abs(value) > std::numeric_limits<float>::min())
                {
                    count++;
                }
            }

            // last thread in segment will contain the total count after this call
            count = rocsparse::wfreduce_sum<SEGMENT_SIZE>(count);

            // broadcast count from last thread in segment to all threads in segment
            count = __shfl(count, SEGMENT_SIZE - 1, SEGMENT_SIZE);

            nnz_per_row[row_index] = count;
        }
    }
}
