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

#include "handle.h"
#include <hip/hip_runtime.h>

namespace rocsparse
{
    template <rocsparse_int NUMROWS_PER_BLOCK,
              rocsparse_int WF_SIZE,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL(WF_SIZE* NUMROWS_PER_BLOCK)
    void dense2csr_kernel(rocsparse_index_base base,
                          rocsparse_order      order,
                          J                    m,
                          J                    n,
                          const T* __restrict__ dense_val,
                          int64_t ld,
                          T* __restrict__ csr_val,
                          I* __restrict__ csr_row_ptr,
                          J* __restrict__ csr_col_ind)
    {
        const rocsparse_int wavefront_index = hipThreadIdx_x / WF_SIZE;
        const J             lane_index      = hipThreadIdx_x & (WF_SIZE - 1);
        const uint64_t      filter          = 0xffffffffffffffff >> (63 - lane_index);
        const J             row_index       = NUMROWS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

        if(row_index < m)
        {
            I shift = csr_row_ptr[row_index] - base;
            //
            // The warp handles the entire row.
            //
            for(J column_index = lane_index; column_index < n; column_index += WF_SIZE)
            {
                //
                // Synchronize for cache considerations.
                //
                __syncthreads();

                //
                // Get value.
                //
                T value = static_cast<T>(0);
                if(order == rocsparse_order_column)
                {
                    value = dense_val[row_index + column_index * ld];
                }
                else
                {
                    value = dense_val[column_index + row_index * ld];
                }

                //
                // Predicate.
                //
                const bool predicate = value != 0;

                //
                // Mask of the wavefront.
                //
                const uint64_t wavefront_mask = __ballot(predicate);

                // Get the number of previous non-zeros in the row for each lane.
                const int count_previous_nnzs = __popcll(wavefront_mask & filter);

                // Get the number of non-zeros in the row.
                const int count_total_nnzs = __popcll(wavefront_mask);

                if(predicate)
                {
                    //
                    // Calculate local index.
                    //
                    const int local_index_in_warp = count_previous_nnzs - 1;

                    //
                    // Populate the sparse matrix.
                    //
                    csr_val[shift + local_index_in_warp]     = value;
                    csr_col_ind[shift + local_index_in_warp] = column_index + base;
                }

                shift += count_total_nnzs;
            }
        }
    }
    template <rocsparse_int NUMCOLUMNS_PER_BLOCK,
              rocsparse_int WF_SIZE,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL(WF_SIZE* NUMCOLUMNS_PER_BLOCK)
    void dense2csc_kernel(rocsparse_index_base base,
                          rocsparse_order      order,
                          J                    m,
                          J                    n,
                          const T* __restrict__ dense_val,
                          int64_t ld,
                          T* __restrict__ csc_val,
                          I* __restrict__ csc_col_ptr,
                          J* __restrict__ csc_row_ind)
    {
        const rocsparse_int wavefront_index = hipThreadIdx_x / WF_SIZE;
        const J             lane_index      = hipThreadIdx_x & (WF_SIZE - 1);
        const uint64_t      filter          = 0xffffffffffffffff >> (63 - lane_index);
        const J             column_index = NUMCOLUMNS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

        if(column_index < n)
        {
            I shift = csc_col_ptr[column_index] - base;
            //
            // The warp handles the entire column.
            //
            for(J row_index = lane_index; row_index < m; row_index += WF_SIZE)
            {
                //
                // Get value.
                //
                T value = static_cast<T>(0);
                if(order == rocsparse_order_column)
                {
                    value = dense_val[row_index + column_index * ld];
                }
                else
                {
                    value = dense_val[column_index + row_index * ld];
                }

                //
                // Predicate.
                //
                const bool predicate = value != 0;

                //
                // Mask of the wavefront.
                //
                const uint64_t wavefront_mask = __ballot(predicate);

                // Get the number of previous non-zeros in the row for each lane.
                const int count_previous_nnzs = __popcll(wavefront_mask & filter);

                // Get the number of non-zeros in the row.
                const int count_total_nnzs = __popcll(wavefront_mask);

                //
                // Synchronize for cache considerations.
                //
                __syncthreads();

                if(predicate)
                {
                    //
                    // Calculate local index.
                    //
                    const int local_index = count_previous_nnzs - 1;

                    //
                    // Populate the sparse matrix.
                    //
                    csc_val[shift + local_index]     = value;
                    csc_row_ind[shift + local_index] = row_index + base;
                }

                shift += count_total_nnzs;
            }
        }
    }
}
