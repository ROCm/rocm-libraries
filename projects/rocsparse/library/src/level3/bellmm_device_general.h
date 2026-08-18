/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <rocsparse_int BELL_BLOCK_DIM,
              rocsparse_int BLK_SIZE_Y,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void bellmm_general_blockdim_device(rocsparse_operation trans_A,
                                                             rocsparse_operation trans_B,
                                                             I                   Mb,
                                                             I                   N,
                                                             T                   alpha,
                                                             I                   bell_cols,
                                                             I                   bell_block_dim,
                                                             const I* __restrict__ bell_col_ind,
                                                             const A* __restrict__ bell_val,
                                                             const B* __restrict__ dense_B,
                                                             int64_t         ldb,
                                                             rocsparse_order order_B,
                                                             T               beta,
                                                             C* __restrict__ dense_C,
                                                             int64_t              ldc,
                                                             rocsparse_order      order_C,
                                                             rocsparse_index_base idx_base)
    {
        // Each thread block is responsible for one block-row of A (hipBlockIdx_x) and a tile of
        // BLK_SIZE_Y columns of the dense matrices (hipBlockIdx_y). Within the block, hipThreadIdx_x
        // selects the row inside the A block (stepping by BELL_BLOCK_DIM to cover block dimensions
        // larger than the thread block) and hipThreadIdx_y selects the dense column.
        const I block_row = hipBlockIdx_x;
        if(block_row >= Mb)
        {
            return;
        }

        const bool conj_A     = (trans_A == rocsparse_operation_conjugate_transpose);
        const bool conj_B     = (trans_B == rocsparse_operation_conjugate_transpose);
        const bool do_trans_B = (trans_B != rocsparse_operation_none);

        const I block_dim       = bell_block_dim;
        const I ell_block_width = bell_cols / block_dim;

        // Dense column handled by this thread.
        const I n = hipThreadIdx_y + hipBlockIdx_y * BLK_SIZE_Y;

        for(I x = 0; x < block_dim; x += BELL_BLOCK_DIM)
        {
            // Local row inside the A block.
            const I r = hipThreadIdx_x + x;

            if(r >= block_dim || n >= N)
            {
                continue;
            }

            const I C_row = block_row * block_dim + r;

            T sum = static_cast<T>(0);

            // Walk the non-zero blocks of this block-row.
            for(I ei = 0; ei < ell_block_width; ei++)
            {
                const I bc = bell_col_ind[static_cast<size_t>(block_row) * ell_block_width + ei]
                             - idx_base;

                // Padded (empty) ELL slot.
                if(bc < 0)
                {
                    break;
                }

                for(I c = 0; c < block_dim; c++)
                {
                    // A_block[r, c] from the cuSPARSE row-major value layout:
                    //   val[(block_row * block_dim + r) * bell_cols + ei * block_dim + c]
                    const int64_t a_idx
                        = (static_cast<int64_t>(block_row) * block_dim + r) * bell_cols
                          + static_cast<int64_t>(ei) * block_dim + c;
                    const T a = rocsparse::conj_val(static_cast<T>(bell_val[a_idx]), conj_A);

                    // B element at logical position (bc * block_dim + c, n).
                    const I b_row = bc * block_dim + c;
                    int64_t b_idx;
                    if(!do_trans_B)
                    {
                        b_idx = (order_B == rocsparse_order_column)
                                    ? b_row + static_cast<int64_t>(n) * ldb
                                    : static_cast<int64_t>(b_row) * ldb + n;
                    }
                    else
                    {
                        b_idx = (order_B == rocsparse_order_column)
                                    ? n + static_cast<int64_t>(b_row) * ldb
                                    : static_cast<int64_t>(n) * ldb + b_row;
                    }
                    const T b = rocsparse::conj_val(static_cast<T>(dense_B[b_idx]), conj_B);

                    sum = rocsparse::fma<T>(a, b, sum);
                }
            }

            const int64_t idx_C = (order_C == rocsparse_order_column)
                                      ? C_row + static_cast<int64_t>(n) * ldc
                                      : static_cast<int64_t>(C_row) * ldc + n;

            if(beta == static_cast<T>(0))
            {
                dense_C[idx_C] = alpha * sum;
            }
            else
            {
                dense_C[idx_C]
                    = rocsparse::fma<T>(beta, static_cast<T>(dense_C[idx_C]), alpha * sum);
            }
        }
    }
}
