/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <rocsparse_int NUM_ELL_COLUMNS_PER_BLOCK,
              rocsparse_int WF_SIZE,
              typename I,
              typename T>
    ROCSPARSE_KERNEL(WF_SIZE* NUM_ELL_COLUMNS_PER_BLOCK)
    void ell2dense_kernel(rocsparse_index_base ell_base,
                          I                    m,
                          I                    n,
                          I                    ell_width,
                          const T* __restrict__ ell_val,
                          const I* __restrict__ ell_col_ind,
                          T* __restrict__ dense_val,
                          int64_t         ld,
                          rocsparse_order order)
    {
        const auto wavefront_index  = hipThreadIdx_x / WF_SIZE;
        const auto lane_index       = hipThreadIdx_x % WF_SIZE;
        const auto ell_column_index = NUM_ELL_COLUMNS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

        if(ell_column_index < ell_width)
        {
            //
            // One wavefront executes one ell column.
            //
            for(I row_index = lane_index; row_index < m; row_index += WF_SIZE)
            {
                const auto ell_idx      = ELL_IND(row_index, ell_column_index, m, ell_width);
                const auto column_index = ell_col_ind[ell_idx] - ell_base;

                if(column_index >= 0 && column_index < n)
                {
                    if(order == rocsparse_order_column)
                    {
                        dense_val[column_index * ld + row_index] = ell_val[ell_idx];
                    }
                    else
                    {
                        dense_val[row_index * ld + column_index] = ell_val[ell_idx];
                    }
                }
            }
        }
    }
}
