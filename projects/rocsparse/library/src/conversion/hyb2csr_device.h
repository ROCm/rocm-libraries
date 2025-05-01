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
    template <uint32_t BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void hyb2csr_nnz_kernel(rocsparse_int        m,
                            rocsparse_int        n,
                            rocsparse_int        ell_nnz,
                            rocsparse_int        ell_width,
                            const rocsparse_int* ell_col_ind,
                            rocsparse_int        coo_nnz,
                            const rocsparse_int* coo_row_ptr,
                            rocsparse_int*       row_nnz,
                            rocsparse_index_base idx_base)
    {
        // Each thread processes one row
        rocsparse_int row = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        // Do not run out of bounds
        if(row >= m)
        {
            return;
        }

        // If COO part is available
        rocsparse_int nnz = (coo_nnz > 0) ? coo_row_ptr[row + 1] - coo_row_ptr[row] : 0;

        // If ELL part is available
        if(ell_nnz > 0)
        {
            for(rocsparse_int p = 0; p < ell_width; ++p)
            {
                rocsparse_int idx = ELL_IND(row, p, m, ell_width);
                rocsparse_int col = ell_col_ind[idx] - idx_base;

                if(col >= 0 && col < n)
                {
                    ++nnz;
                }
                else
                {
                    break;
                }
            }
        }

        // Write total non-zeros to global memory
        row_nnz[row] = nnz;
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void hyb2csr_fill_kernel(rocsparse_int        m,
                             rocsparse_int        n,
                             rocsparse_int        ell_nnz,
                             rocsparse_int        ell_width,
                             const rocsparse_int* ell_col_ind,
                             const T*             ell_val,
                             rocsparse_int        coo_nnz,
                             const rocsparse_int* coo_row_ptr,
                             const rocsparse_int* coo_col_ind,
                             const T*             coo_val,
                             const rocsparse_int* csr_row_ptr,
                             rocsparse_int*       csr_col_ind,
                             T*                   csr_val,
                             rocsparse_index_base idx_base)
    {
        // Each thread processes one row
        rocsparse_int row = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        // Do not run out of bounds
        if(row >= m)
        {
            return;
        }

        // Correct number of columns by index base
        n += idx_base;

        // Offset into CSR matrix
        rocsparse_int csr_idx = csr_row_ptr[row] - idx_base;

        // Process ELL part if available
        if(ell_nnz > 0)
        {
            for(rocsparse_int p = 0; p < ell_width; ++p)
            {
                rocsparse_int ell_idx = ELL_IND(row, p, m, ell_width);
                rocsparse_int col     = ell_col_ind[ell_idx];

                // Fill CSR matrix with ELL entries
                if(col >= idx_base && col < n)
                {
                    csr_col_ind[csr_idx] = col;
                    csr_val[csr_idx]     = ell_val[ell_idx];

                    ++csr_idx;
                }
                else
                {
                    break;
                }
            }
        }

        // Process COO part if available
        if(coo_nnz > 0)
        {
            // Loop over COO entries from current row, using the previously
            // computed row pointer array
            rocsparse_int coo_row_begin = coo_row_ptr[row] - idx_base;
            rocsparse_int coo_row_end   = coo_row_ptr[row + 1] - idx_base;

            // Fill CSR matrix with COO entries
            for(rocsparse_int j = coo_row_begin; j < coo_row_end; ++j)
            {
                csr_col_ind[csr_idx] = coo_col_ind[j];
                csr_val[csr_idx]     = coo_val[j];

                ++csr_idx;
            }
        }
    }
}
