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

        if(hipBlockIdx_x == 0 && tid == 0)
        {
            csr_row_ptr[0] = csr_base;
        }

        rocsparse_int lid = tid & (ROW_BLOCK_DIM * COL_BLOCK_DIM - 1);
        rocsparse_int wid = tid / (ROW_BLOCK_DIM * COL_BLOCK_DIM);

        rocsparse_int c = lid & (COL_BLOCK_DIM - 1);
        rocsparse_int r = lid / COL_BLOCK_DIM;

        // Hoisted out of the block-row loop below: r and c do not depend on the block
        // row, so a thread that falls outside the block has nothing to do for any of
        // them. This kernel has no barrier, so returning early here is safe.
        if(r >= row_block_dim || c >= col_block_dim)
        {
            return;
        }

        // Grid-stride over the block rows: grid.x is clamped against maxGridSize[0],
        // so one grid sweep only covers hipGridDim_x of them. The bound is block
        // uniform (hipBlockIdx_x, hipGridDim_x and the kernel argument mb), and the
        // kernel holds no __shared__ state and has no __syncthreads(), so iterations
        // are independent.
        for(int64_t bid64 = hipBlockIdx_x; bid64 < mb; bid64 += hipGridDim_x)
        {
            const rocsparse_int bid = static_cast<rocsparse_int>(bid64);

            rocsparse_int start = bsr_row_ptr[bid] - bsr_base;
            rocsparse_int end   = bsr_row_ptr[bid + 1] - bsr_base;

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

        if(hipBlockIdx_x == 0 && tid == 0)
        {
            csr_row_ptr[0] = csr_base;
        }

        // Grid-stride over the block rows: grid.x is clamped against maxGridSize[0],
        // so one grid sweep only covers hipGridDim_x of them. The bound is block
        // uniform (hipBlockIdx_x, hipGridDim_x and the kernel argument mb), and the
        // kernel holds no __shared__ state and has no __syncthreads(), so iterations
        // are independent.
        for(int64_t bid64 = hipBlockIdx_x; bid64 < mb; bid64 += hipGridDim_x)
        {
            const rocsparse_int bid = static_cast<rocsparse_int>(bid64);

            rocsparse_int start = bsr_row_ptr[bid] - bsr_base;
            rocsparse_int end   = bsr_row_ptr[bid + 1] - bsr_base;

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
        const rocsparse_int entries_in_block = row_block_dim * col_block_dim;

        // AISPARSE-684. One wavefront per row of the CSR matrix. The bound used to be
        // the rocsparse_int expression `mb * row_block_dim`, formed exactly as the
        // host formed it when it sized the grid, so the launch was undersized and the
        // guard it was checked against was corrupted by the same signed overflow --
        // undefined behaviour, and in practice a partially written csr_row_ptr /
        // csr_col_ind with no error reported. Both sides are 64-bit now.
        const int64_t       num_rows          = static_cast<int64_t>(mb) * row_block_dim;
        const int64_t       rows_per_block    = BLOCK_SIZE / WF_SIZE;
        const int64_t       row_in_block_grid = hipThreadIdx_x / WF_SIZE;
        const rocsparse_int lane_id           = hipThreadIdx_x % WF_SIZE;

        // Block-uniform stride bound: row_base is built from hipBlockIdx_x,
        // hipGridDim_x, a kernel argument (mb, row_block_dim) and compile-time
        // constants only -- no hipThreadIdx_x -- so every thread of a block runs the
        // same number of iterations. This kernel contains no __syncthreads() and the
        // per-row guard is a `continue` rather than a `return`, so a wavefront that
        // falls past the end of the last partial block still reaches the next
        // iteration of the stride loop.
        for(int64_t row_base = rows_per_block * hipBlockIdx_x; row_base < num_rows;
            row_base += rows_per_block * hipGridDim_x)
        {
            const int64_t warp_id = row_base + row_in_block_grid;

            if(warp_id >= num_rows)
            { // one warp per row in matrix
                continue;
            }

            // block row in bsr matrix, and local row in bsr row block. Both are
            // bounded by mb and row_block_dim respectively, so they fit rocsparse_int
            // whenever the caller's own arrays do.
            const rocsparse_int block_row    = static_cast<rocsparse_int>(warp_id / row_block_dim);
            const rocsparse_int row_in_block = static_cast<rocsparse_int>(warp_id % row_block_dim);

            const rocsparse_int bsr_row_start = bsr_row_ptr[block_row] - bsr_base;
            const rocsparse_int bsr_row_end   = bsr_row_ptr[block_row + 1] - bsr_base;

            const rocsparse_int entries_in_row = (bsr_row_end - bsr_row_start) * col_block_dim;
            const rocsparse_int number_of_entries_in_prev_rows
                = bsr_row_start * entries_in_block + row_in_block * entries_in_row;

            if(warp_id == 0)
            {
                csr_row_ptr[0] = csr_base;
            }

            csr_row_ptr[warp_id + 1] = number_of_entries_in_prev_rows + entries_in_row + csr_base;

            for(rocsparse_int i = bsr_row_start + lane_id; i < bsr_row_end; i += WF_SIZE)
            {

                const rocsparse_int col = bsr_col_ind[i] - bsr_base;
                const rocsparse_int offset
                    = number_of_entries_in_prev_rows + col_block_dim * (i - bsr_row_start);

                for(rocsparse_int j = 0; j < col_block_dim; j++)
                {
                    csr_col_ind[offset + j] = col_block_dim * col + j + csr_base;
                }
            }
        }
    }
}
