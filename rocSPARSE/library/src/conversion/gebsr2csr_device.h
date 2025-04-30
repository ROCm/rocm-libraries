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

#include <hip/hip_runtime.h>

namespace rocsparse
{
    template <rocsparse_int BLOCK_SIZE,
              rocsparse_int ROW_BLOCK_DIM,
              rocsparse_int COL_BLOCK_DIM,
              typename T>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void gebsr2csr_block_per_row_1_32_kernel(rocsparse_direction  dir,
                                             rocsparse_int        mb,
                                             rocsparse_int        nb,
                                             rocsparse_index_base bsr_base,
                                             const T* __restrict__ bsr_val,
                                             const rocsparse_int* __restrict__ bsr_row_ptr,
                                             const rocsparse_int* __restrict__ bsr_col_ind,
                                             rocsparse_int        row_block_dim,
                                             rocsparse_int        col_block_dim,
                                             rocsparse_index_base csr_base,
                                             T* __restrict__ csr_val,
                                             rocsparse_int* __restrict__ csr_row_ptr,
                                             rocsparse_int* __restrict__ csr_col_ind)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int bid = hipBlockIdx_x;

        rocsparse_int start = bsr_row_ptr[bid] - bsr_base;
        rocsparse_int end   = bsr_row_ptr[bid + 1] - bsr_base;

        if(bid == 0 && tid == 0)
        {
            csr_row_ptr[0] = csr_base;
        }

        rocsparse_int lid = tid & (ROW_BLOCK_DIM * COL_BLOCK_DIM - 1);
        rocsparse_int wid = tid / (ROW_BLOCK_DIM * COL_BLOCK_DIM);

        rocsparse_int c = lid & (COL_BLOCK_DIM - 1);
        rocsparse_int r = lid / COL_BLOCK_DIM;

        if(r >= row_block_dim || c >= col_block_dim)
        {
            return;
        }

        rocsparse_int prev
            = row_block_dim * col_block_dim * start + col_block_dim * (end - start) * r;
        rocsparse_int current = col_block_dim * (end - start);

        csr_row_ptr[row_block_dim * bid + r + 1] = prev + current + csr_base;

        for(rocsparse_int i = start + wid; i < end;
            i += (BLOCK_SIZE / (ROW_BLOCK_DIM * COL_BLOCK_DIM)))
        {
            rocsparse_int col    = bsr_col_ind[i] - bsr_base;
            rocsparse_int offset = prev + col_block_dim * (i - start) + c;

            csr_col_ind[offset] = col_block_dim * col + c + csr_base;

            if(dir == rocsparse_direction_row)
            {
                csr_val[offset]
                    = bsr_val[row_block_dim * col_block_dim * i + col_block_dim * r + c];
            }
            else
            {
                csr_val[offset]
                    = bsr_val[row_block_dim * col_block_dim * i + row_block_dim * c + r];
            }
        }
    }

    template <rocsparse_int BLOCK_SIZE,
              rocsparse_int ROW_BLOCK_DIM,
              rocsparse_int COL_BLOCK_DIM,
              rocsparse_int SUB_ROW_BLOCK_DIM,
              rocsparse_int SUB_COL_BLOCK_DIM,
              typename T>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void gebsr2csr_block_per_row_33_128_kernel(rocsparse_direction  dir,
                                               rocsparse_int        mb,
                                               rocsparse_int        nb,
                                               rocsparse_index_base bsr_base,
                                               const T* __restrict__ bsr_val,
                                               const rocsparse_int* __restrict__ bsr_row_ptr,
                                               const rocsparse_int* __restrict__ bsr_col_ind,
                                               rocsparse_int        row_block_dim,
                                               rocsparse_int        col_block_dim,
                                               rocsparse_index_base csr_base,
                                               T* __restrict__ csr_val,
                                               rocsparse_int* __restrict__ csr_row_ptr,
                                               rocsparse_int* __restrict__ csr_col_ind)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int bid = hipBlockIdx_x;

        rocsparse_int start = bsr_row_ptr[bid] - bsr_base;
        rocsparse_int end   = bsr_row_ptr[bid + 1] - bsr_base;

        if(bid == 0 && tid == 0)
        {
            csr_row_ptr[0] = csr_base;
        }

        for(rocsparse_int y = 0; y < (ROW_BLOCK_DIM / SUB_ROW_BLOCK_DIM); y++)
        {
            rocsparse_int r = (tid / SUB_COL_BLOCK_DIM) + SUB_ROW_BLOCK_DIM * y;

            if(r < row_block_dim)
            {
                rocsparse_int prev
                    = row_block_dim * col_block_dim * start + col_block_dim * (end - start) * r;
                rocsparse_int current = col_block_dim * (end - start);

                csr_row_ptr[row_block_dim * bid + r + 1] = prev + current + csr_base;
            }
        }

        for(rocsparse_int i = start; i < end; i++)
        {
            rocsparse_int col = bsr_col_ind[i] - bsr_base;

            for(rocsparse_int y = 0; y < (ROW_BLOCK_DIM / SUB_ROW_BLOCK_DIM); y++)
            {
                for(rocsparse_int x = 0; x < (COL_BLOCK_DIM / SUB_COL_BLOCK_DIM); x++)
                {
                    rocsparse_int c = (tid & (SUB_COL_BLOCK_DIM - 1)) + SUB_COL_BLOCK_DIM * x;
                    rocsparse_int r = (tid / SUB_COL_BLOCK_DIM) + SUB_ROW_BLOCK_DIM * y;

                    if(r < row_block_dim && c < col_block_dim)
                    {
                        rocsparse_int prev = row_block_dim * col_block_dim * start
                                             + col_block_dim * (end - start) * r;

                        rocsparse_int offset = prev + col_block_dim * (i - start) + c;

                        csr_col_ind[offset] = col_block_dim * col + c + csr_base;

                        if(dir == rocsparse_direction_row)
                        {
                            csr_val[offset] = bsr_val[row_block_dim * col_block_dim * i
                                                      + col_block_dim * r + c];
                        }
                        else
                        {
                            csr_val[offset] = bsr_val[row_block_dim * col_block_dim * i
                                                      + row_block_dim * c + r];
                        }
                    }
                }
            }
        }
    }

    template <rocsparse_int BLOCK_SIZE, rocsparse_int WF_SIZE>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void gebsr2csr_nnz_kernel(rocsparse_int        mb,
                              rocsparse_int        nb,
                              rocsparse_index_base bsr_base,
                              const rocsparse_int* __restrict__ bsr_row_ptr,
                              const rocsparse_int* __restrict__ bsr_col_ind,
                              rocsparse_int        row_block_dim,
                              rocsparse_int        col_block_dim,
                              rocsparse_index_base csr_base,
                              rocsparse_int* __restrict__ csr_row_ptr,
                              rocsparse_int* __restrict__ csr_col_ind)
    {
        rocsparse_int entries_in_block = row_block_dim * col_block_dim;

        rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
        rocsparse_int warp_id   = thread_id / WF_SIZE;
        rocsparse_int lane_id   = thread_id % WF_SIZE;

        if(warp_id >= mb * row_block_dim)
        { // one warp per row in matrix
            return;
        }

        rocsparse_int block_row    = warp_id / row_block_dim; // block row in bsr matrix
        rocsparse_int row_in_block = warp_id % row_block_dim; // local row in bsr row block

        rocsparse_int bsr_row_start = bsr_row_ptr[block_row] - bsr_base;
        rocsparse_int bsr_row_end   = bsr_row_ptr[block_row + 1] - bsr_base;

        rocsparse_int entries_in_row = (bsr_row_end - bsr_row_start) * col_block_dim;
        rocsparse_int number_of_entries_in_prev_rows
            = bsr_row_start * entries_in_block + row_in_block * entries_in_row;

        if(warp_id == 0)
        {
            csr_row_ptr[0] = csr_base;
        }

        csr_row_ptr[warp_id + 1] = number_of_entries_in_prev_rows + entries_in_row + csr_base;

        for(rocsparse_int i = bsr_row_start + lane_id; i < bsr_row_end; i += WF_SIZE)
        {

            rocsparse_int col = bsr_col_ind[i] - bsr_base;
            rocsparse_int offset
                = number_of_entries_in_prev_rows + col_block_dim * (i - bsr_row_start);

            for(rocsparse_int j = 0; j < col_block_dim; j++)
            {
                csr_col_ind[offset + j] = col_block_dim * col + j + csr_base;
            }
        }
    }
}
