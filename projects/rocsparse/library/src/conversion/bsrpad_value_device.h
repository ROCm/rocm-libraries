/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <rocsparse_int BLOCK_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void bsrpad_value_kernel_sorted(rocsparse_int        m,
                                    rocsparse_int        mb,
                                    rocsparse_int        block_dim,
                                    T                    value,
                                    rocsparse_index_base bsr_base,
                                    T* __restrict__ bsr_val,
                                    const rocsparse_int* __restrict__ bsr_row_ptr,
                                    const rocsparse_int* __restrict__ bsr_col_ind)
    {

        rocsparse_int block_id = hipBlockIdx_x;
        rocsparse_int local_id = hipThreadIdx_x;

        rocsparse_int block_index = -1;

        rocsparse_int i = block_id * BLOCK_SIZE + local_id;

        rocsparse_int start = bsr_row_ptr[mb - 1] - bsr_base;
        rocsparse_int end   = bsr_row_ptr[mb] - bsr_base;

        //find block
        if((end - start) > 0)
        {
            if((bsr_col_ind[end - 1] - bsr_base) == (mb - 1))
            {
                block_index = end - 1;
            }
        }

        if(block_index >= 0)
        {
            if(i < block_dim)
            {
                rocsparse_int start_local_index = m % block_dim;
                if(start_local_index > 0)
                {
                    if(i >= start_local_index)
                    {
                        bsr_val[block_index * block_dim * block_dim + i * block_dim + i] = value;
                    }
                }
            }
        }
    }

    template <rocsparse_int BLOCK_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void bsrpad_value_kernel_unsorted(rocsparse_int        m,
                                      rocsparse_int        mb,
                                      rocsparse_int        block_dim,
                                      T                    value,
                                      rocsparse_index_base bsr_base,
                                      T* __restrict__ bsr_val,
                                      const rocsparse_int* __restrict__ bsr_row_ptr,
                                      const rocsparse_int* __restrict__ bsr_col_ind)
    {
        rocsparse_int block_id = hipBlockIdx_x;
        rocsparse_int local_id = hipThreadIdx_x;

        __shared__ rocsparse_int block_index;

        rocsparse_int i = block_id * BLOCK_SIZE + local_id;

        if(local_id == 0)
        {
            block_index = -1;
        }

        rocsparse_int start = bsr_row_ptr[mb - 1] - bsr_base;
        rocsparse_int end   = bsr_row_ptr[mb] - bsr_base;

        __syncthreads();

        //find block
        for(rocsparse_int index = start + local_id; index < end; index += BLOCK_SIZE)
        {
            if(bsr_col_ind[index] - bsr_base == mb - 1)
            {
                block_index = index;
            }
        }

        __syncthreads();

        if(block_index >= 0)
        {
            if(i < block_dim)
            {
                rocsparse_int start_local_index = m % block_dim;
                if(start_local_index > 0)
                {
                    if(i >= start_local_index)
                    {
                        bsr_val[block_index * block_dim * block_dim + i * block_dim + i] = value;
                    }
                }
            }
        }
    }
}
