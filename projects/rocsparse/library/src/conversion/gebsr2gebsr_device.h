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
    template <rocsparse_int BLOCK_SIZE, rocsparse_int WF_SEGMENT_SIZE>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void gebsr2gebsr_nnz_fast_kernel(rocsparse_int        mb_A,
                                     rocsparse_int        nb_A,
                                     rocsparse_index_base base_A,
                                     const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                     const rocsparse_int* __restrict__ bsr_col_ind_A,
                                     rocsparse_int        row_block_dim_A,
                                     rocsparse_int        col_block_dim_A,
                                     rocsparse_int        mb_C,
                                     rocsparse_int        nb_C,
                                     rocsparse_index_base base_C,
                                     rocsparse_int* __restrict__ bsr_row_ptr_C,
                                     rocsparse_int row_block_dim_C,
                                     rocsparse_int col_block_dim_C)
    {
        constexpr rocsparse_int SEGMENTS_PER_BLOCK = (BLOCK_SIZE / WF_SEGMENT_SIZE);

        rocsparse_int block_id = hipBlockIdx_x;

        rocsparse_int wf_segment_id      = (hipThreadIdx_x / WF_SEGMENT_SIZE) % SEGMENTS_PER_BLOCK;
        rocsparse_int wf_segment_lane_id = hipThreadIdx_x % WF_SEGMENT_SIZE;

        rocsparse_int row = SEGMENTS_PER_BLOCK * row_block_dim_C * block_id
                            + row_block_dim_C * wf_segment_id + wf_segment_lane_id;

        rocsparse_int block_row = row / row_block_dim_A;

        rocsparse_int block_row_start = 0;
        rocsparse_int block_row_end   = 0;

        if(block_row < mb_A && wf_segment_lane_id < row_block_dim_C)
        {
            block_row_start = bsr_row_ptr_A[block_row] - base_A;
            block_row_end   = bsr_row_ptr_A[block_row + 1] - base_A;
        }

        rocsparse_int block_col    = 0;
        rocsparse_int nnzb_per_row = 0;

        while(block_col < nb_C)
        {
            rocsparse_int min_block_col_index = nb_C;

            bool should_break = false;
            for(rocsparse_int i = block_row_start; i < block_row_end; i++)
            {
                rocsparse_int temp = (bsr_col_ind_A[i] - base_A) * col_block_dim_A;

                for(rocsparse_int j = 0; j < col_block_dim_A; j++)
                {
                    rocsparse_int block_col_index = (temp + j) / col_block_dim_C;

                    if(block_col_index >= block_col)
                    {
                        min_block_col_index = block_col_index;
                        block_row_start     = i;
                        should_break        = true;
                        break;
                    }
                }

                if(should_break)
                {
                    break;
                }
            }

            // last thread in segment will contain the min after this call
            rocsparse::wfreduce_min<WF_SEGMENT_SIZE>(&min_block_col_index);

            // broadcast min_block_col_index from last thread in segment to all threads in segment
            min_block_col_index = __shfl(min_block_col_index, WF_SEGMENT_SIZE - 1, WF_SEGMENT_SIZE);

            // update block_col for all threads in segment
            block_col = min_block_col_index + 1;

            if(min_block_col_index < nb_C && wf_segment_lane_id == WF_SEGMENT_SIZE - 1)
            {
                nnzb_per_row++;
            }
        }

        if(SEGMENTS_PER_BLOCK * block_id + wf_segment_id < mb_C
           && wf_segment_lane_id == WF_SEGMENT_SIZE - 1)
        {
            bsr_row_ptr_C[0]                                                 = base_C;
            bsr_row_ptr_C[SEGMENTS_PER_BLOCK * block_id + wf_segment_id + 1] = nnzb_per_row;
        }
    }

    template <rocsparse_direction DIRECTION,
              rocsparse_int       BLOCK_SIZE,
              rocsparse_int       WF_SEGMENT_SIZE,
              typename T>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void gebsr2gebsr_fast_kernel(rocsparse_int        mb_A,
                                 rocsparse_int        nb_A,
                                 rocsparse_index_base base_A,
                                 const T* __restrict__ bsr_val_A,
                                 const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                 const rocsparse_int* __restrict__ bsr_col_ind_A,
                                 rocsparse_int        row_block_dim_A,
                                 rocsparse_int        col_block_dim_A,
                                 rocsparse_int        mb_C,
                                 rocsparse_int        nb_C,
                                 rocsparse_index_base base_C,
                                 T* __restrict__ bsr_val_C,
                                 rocsparse_int* __restrict__ bsr_row_ptr_C,
                                 rocsparse_int* __restrict__ bsr_col_ind_C,
                                 rocsparse_int row_block_dim_C,
                                 rocsparse_int col_block_dim_C)
    {
        constexpr rocsparse_int SEGMENTS_PER_BLOCK = (BLOCK_SIZE / WF_SEGMENT_SIZE);

        rocsparse_int block_id = hipBlockIdx_x;

        rocsparse_int wf_segment_id      = (hipThreadIdx_x / WF_SEGMENT_SIZE) % SEGMENTS_PER_BLOCK;
        rocsparse_int wf_segment_lane_id = hipThreadIdx_x % WF_SEGMENT_SIZE;

        rocsparse_int row = SEGMENTS_PER_BLOCK * row_block_dim_C * block_id
                            + row_block_dim_C * wf_segment_id + wf_segment_lane_id;

        rocsparse_int block_row = row / row_block_dim_A;

        rocsparse_int block_row_start = 0;
        rocsparse_int block_row_end   = 0;

        if(block_row < mb_A && wf_segment_lane_id < row_block_dim_C)
        {
            block_row_start = bsr_row_ptr_A[block_row] - base_A;
            block_row_end   = bsr_row_ptr_A[block_row + 1] - base_A;
        }

        rocsparse_int bsr_row_start = 0;

        if(SEGMENTS_PER_BLOCK * block_id + wf_segment_id < mb_C)
        {
            bsr_row_start = bsr_row_ptr_C[SEGMENTS_PER_BLOCK * block_id + wf_segment_id] - base_C;
        }

        rocsparse_int block_col    = 0;
        rocsparse_int nnzb_per_row = 0;

        while(block_col < nb_C)
        {
            rocsparse_int min_block_col = nb_C;

            bool should_break = false;
            for(rocsparse_int i = block_row_start; i < block_row_end; i++)
            {
                rocsparse_int temp = (bsr_col_ind_A[i] - base_A) * col_block_dim_A;

                for(rocsparse_int j = 0; j < col_block_dim_A; j++)
                {
                    rocsparse_int bcol = (temp + j) / col_block_dim_C;

                    if(bcol >= block_col)
                    {
                        min_block_col   = bcol;
                        block_row_start = i;
                        should_break    = true;
                        break;
                    }
                }

                if(should_break)
                {
                    break;
                }
            }

            // find minimum CSR column index across all threads in this segment and store in last thread of segment
            rocsparse::wfreduce_min<WF_SEGMENT_SIZE>(&min_block_col);

            // have last thread in segment write to CSR column indices array
            if(min_block_col < nb_C && wf_segment_lane_id == WF_SEGMENT_SIZE - 1)
            {
                bsr_col_ind_C[bsr_row_start + nnzb_per_row] = min_block_col + base_C;
                nnzb_per_row++;
            }

            // broadcast CSR minimum column index from last thread in segment to all threads in segment
            min_block_col = __shfl(min_block_col, WF_SEGMENT_SIZE - 1, WF_SEGMENT_SIZE);

            // broadcast nnzb_per_row from last thread in segment to all threads in segment
            nnzb_per_row = __shfl(nnzb_per_row, WF_SEGMENT_SIZE - 1, WF_SEGMENT_SIZE);

            rocsparse_int k
                = row_block_dim_C * col_block_dim_C * (bsr_row_start + nnzb_per_row - 1);

            should_break = false;
            for(rocsparse_int i = block_row_start; i < block_row_end; i++)
            {
                rocsparse_int temp = (bsr_col_ind_A[i] - base_A) * col_block_dim_A;

                for(rocsparse_int j = 0; j < col_block_dim_A; j++)
                {
                    rocsparse_int col  = temp + j;
                    rocsparse_int bcol = col / col_block_dim_C;

                    if(bcol == min_block_col)
                    {
                        if(DIRECTION == rocsparse_direction_row)
                        {
                            rocsparse_int indexC
                                = k + col_block_dim_C * wf_segment_lane_id + col % col_block_dim_C;
                            rocsparse_int indexA = row_block_dim_A * col_block_dim_A * i
                                                   + col_block_dim_A * (row % row_block_dim_A) + j;

                            bsr_val_C[indexC] = bsr_val_A[indexA];
                        }
                        else
                        {
                            rocsparse_int indexC = k + row_block_dim_C * (col % col_block_dim_C)
                                                   + wf_segment_lane_id;
                            rocsparse_int indexA = row_block_dim_A * col_block_dim_A * i
                                                   + row_block_dim_A * j + (row % row_block_dim_A);

                            bsr_val_C[indexC] = bsr_val_A[indexA];
                        }
                    }

                    if(bcol > min_block_col)
                    {
                        should_break = true;
                        break;
                    }
                }

                if(should_break)
                {
                    break;
                }
            }

            // update block_col for all threads in segment
            block_col = min_block_col + 1;
        }
    }

    template <uint32_t BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gebsr2gebsr_compute_nnz_total_kernel(rocsparse_int mb,
                                              const rocsparse_int* __restrict__ bsr_row_ptr,
                                              rocsparse_int* __restrict__ nnz_total_dev_host_ptr)
    {
        rocsparse_int thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

        if(thread_id == 0)
        {
            *nnz_total_dev_host_ptr = bsr_row_ptr[mb] - bsr_row_ptr[0];
        }
    }
}
