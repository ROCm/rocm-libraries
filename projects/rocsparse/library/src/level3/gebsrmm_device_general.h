/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.hpp"

namespace rocsparse
{
    template <rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y, typename T>
    ROCSPARSE_DEVICE_ILF void
        gebsrmm_general_blockdim_device(rocsparse_direction direction,
                                        rocsparse_operation trans_B,
                                        rocsparse_int       Mb,
                                        rocsparse_int       N,
                                        T                   alpha,
                                        const rocsparse_int* __restrict__ bsr_row_ptr,
                                        const rocsparse_int* __restrict__ bsr_col_ind,
                                        const T* __restrict__ bsr_val,
                                        rocsparse_int row_block_dim,
                                        rocsparse_int col_block_dim,
                                        const T* __restrict__ B,
                                        int64_t ldb,
                                        T       beta,
                                        T* __restrict__ C,
                                        int64_t              ldc,
                                        rocsparse_index_base idx_base)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int tidy = hipThreadIdx_y;

        rocsparse_int block_row = hipBlockIdx_x;

        rocsparse_int block_row_start = 0;
        rocsparse_int block_row_end   = 0;
        if(block_row < Mb)
        {
            block_row_start = bsr_row_ptr[block_row] - idx_base;
            block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;
        }

        __shared__ T shared_B[BSR_BLOCK_DIM * BLK_SIZE_Y];
        __shared__ T shared_A[BSR_BLOCK_DIM * BSR_BLOCK_DIM];

        rocsparse_int global_col = tidy + hipBlockIdx_y * BLK_SIZE_Y;

        // Clamp global_col to [0, N-1] for B pointer arithmetic.
        // Out-of-bounds threads (global_col >= N) use col 0 as a safe landing
        // address; the loaded value is discarded (multiplied by zero via shared_A
        // or ignored during the final C write).  This prevents speculative GPU
        // loads from reading past the end of B even when the compiler or hardware
        // hoists the load above the bounds predicate.
        bool          col_in_range = (global_col < N);
        rocsparse_int safe_col     = col_in_range ? global_col : 0;

        int64_t colB = safe_col * ldb; // always a valid column offset into B
        int64_t colC = safe_col * ldc; // clamped: C read (fma) is also guarded this way

        for(rocsparse_int x = 0; x < row_block_dim; x += BSR_BLOCK_DIM)
        {
            rocsparse_int global_row = tidx + x + hipBlockIdx_x * row_block_dim;

            T sum = static_cast<T>(0);

            for(rocsparse_int k = block_row_start; k < block_row_end; k++)
            {
                rocsparse_int block_col = (bsr_col_ind[k] - idx_base);

                for(rocsparse_int y = 0; y < col_block_dim; y += BLK_SIZE_Y)
                {

                    // Always load from a clamped (always valid) address.
                    // AMD GPU hardware may speculatively issue loads from both branches
                    // of a conditional, so we must ensure the address is in-range
                    // unconditionally.  safe_col clamps global_col to [0, N-1], so colB
                    // always points to a valid column.  We then mask out the loaded value
                    // with a scalar multiply by the in-range predicate (0 or 1) so
                    // out-of-bounds threads contribute zero to the shared tile.
                    //
                    // Separate row guard: (tidx+y) < col_block_dim prevents reading past
                    // the row dimension of B — this access is harmless even when clamped
                    // because safe_row stays within [0, col_block_dim-1].
                    rocsparse_int safe_row = (tidx + y) < col_block_dim ? (tidx + y) : 0;
                    T row_mask = (tidx + y) < col_block_dim ? static_cast<T>(1) : static_cast<T>(0);
                    T col_mask = col_in_range ? static_cast<T>(1) : static_cast<T>(0);

                    if(trans_B == rocsparse_operation_none)
                    {
                        T bval = B[col_block_dim * block_col + safe_row + colB];
                        shared_B[BSR_BLOCK_DIM * tidy + tidx] = bval * row_mask * col_mask;
                    }
                    else
                    {
                        T bval = B[safe_col + ldb * (col_block_dim * block_col + safe_row)];
                        shared_B[BSR_BLOCK_DIM * tidy + tidx] = bval * row_mask * col_mask;
                    }

                    // Same clamping pattern for bsr_val (shared_A).
                    // Threads where (tidx+x) >= row_block_dim or (tidy+y) >= col_block_dim
                    // are out of the block — clamp indices and mask the result to zero.
                    rocsparse_int safe_tidx_x = (tidx + x) < row_block_dim ? (tidx + x) : 0;
                    rocsparse_int safe_tidy_y = (tidy + y) < col_block_dim ? (tidy + y) : 0;
                    T a_mask = ((tidx + x) < row_block_dim && (tidy + y) < col_block_dim)
                                   ? static_cast<T>(1)
                                   : static_cast<T>(0);

                    if(direction == rocsparse_direction_row)
                    {
                        T aval = bsr_val[row_block_dim * col_block_dim * k
                                         + col_block_dim * safe_tidx_x + safe_tidy_y];
                        shared_A[BSR_BLOCK_DIM * tidy + tidx] = aval * a_mask;
                    }
                    else
                    {
                        T aval = bsr_val[row_block_dim * col_block_dim * k
                                         + row_block_dim * safe_tidy_y + safe_tidx_x];
                        shared_A[BSR_BLOCK_DIM * tidy + tidx] = aval * a_mask;
                    }

                    __syncthreads();

                    for(rocsparse_int j = 0; j < BSR_BLOCK_DIM; j++)
                    {
                        sum = rocsparse::fma(shared_A[BSR_BLOCK_DIM * j + tidx],
                                             shared_B[BSR_BLOCK_DIM * tidy + j],
                                             sum);
                    }

                    __syncthreads();
                }
            }

            if(block_row < Mb && global_col < N && (tidx + x) < row_block_dim)
            {
                if(beta == static_cast<T>(0))
                {
                    C[global_row + colC] = alpha * sum;
                }
                else
                {
                    C[global_row + colC] = rocsparse::fma(beta, C[global_row + colC], alpha * sum);
                }
            }
        }
    }
}
