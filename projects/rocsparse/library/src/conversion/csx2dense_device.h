/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    void csr2dense_kernel(rocsparse_int base,
                          J             m,
                          J             n,
                          const T* __restrict__ csr_val,
                          const I* __restrict__ csr_row_ptr,
                          const J* __restrict__ csr_col_ind,
                          T* __restrict__ dense_val,
                          int64_t         ld,
                          rocsparse_order order)
    {
        const rocsparse_int wavefront_index = hipThreadIdx_x / WF_SIZE,
                            lane_index      = hipThreadIdx_x % WF_SIZE;
        const J row_index                   = NUMROWS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

        if(row_index < m)
        {
            const I shift    = csr_row_ptr[row_index] - base;
            const I size_row = csr_row_ptr[row_index + 1] - base - shift;

            //
            // One wavefront executes one sparse row.
            //
            for(I index = lane_index; index < size_row; index += WF_SIZE)
            {
                __syncthreads();
                const J column_index = csr_col_ind[shift + index] - base;

                if(order == rocsparse_order_column)
                {
                    dense_val[column_index * ld + row_index] = csr_val[shift + index];
                }
                else
                {
                    dense_val[row_index * ld + column_index] = csr_val[shift + index];
                }
            }
        }
    }

    template <rocsparse_int NUMCOLUMNS_PER_BLOCK,
              rocsparse_int WF_SIZE,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL(WF_SIZE* NUMCOLUMNS_PER_BLOCK)
    void csc2dense_kernel(rocsparse_int base,
                          J             m,
                          J             n,
                          const T* __restrict__ csc_val,
                          const I* __restrict__ csc_col_ptr,
                          const J* __restrict__ csc_row_ind,
                          T* __restrict__ dense_val,
                          int64_t         ld,
                          rocsparse_order order)
    {
        const rocsparse_int wavefront_index = hipThreadIdx_x / WF_SIZE,
                            lane_index      = hipThreadIdx_x % WF_SIZE;
        const J column_index = NUMCOLUMNS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

        if(column_index < n)
        {
            const I shift       = csc_col_ptr[column_index] - base;
            const I size_column = csc_col_ptr[column_index + 1] - base - shift;

            //
            // One wavefront executes one sparse column.
            //
            for(I index = lane_index; index < size_column; index += WF_SIZE)
            {
                const J row_index = csc_row_ind[shift + index] - base;

                if(order == rocsparse_order_column)
                {
                    dense_val[column_index * ld + row_index] = csc_val[shift + index];
                }
                else
                {
                    dense_val[row_index * ld + column_index] = csc_val[shift + index];
                }
            }
        }
    }
}
