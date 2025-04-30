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

namespace rocsparse
{
    // Compute non-zero entries per CSR row to obtain the COO nnz per row.
    template <uint32_t BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void hyb_coo_nnz(rocsparse_int        m,
                     rocsparse_int        ell_width,
                     const rocsparse_int* csr_row_ptr,
                     rocsparse_int*       coo_row_nnz,
                     rocsparse_index_base idx_base)
    {
        rocsparse_int gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid < m)
        {
            rocsparse_int row_nnz = csr_row_ptr[gid + 1] - csr_row_ptr[gid];

            if(row_nnz > ell_width)
            {
                row_nnz              = row_nnz - ell_width;
                coo_row_nnz[gid + 1] = row_nnz;
            }
            else
            {
                coo_row_nnz[gid + 1] = 0;
            }
        }

        if(gid == 0)
        {
            coo_row_nnz[0] = idx_base;
        }
    }

    // CSR to HYB format conversion kernel
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2hyb_kernel(rocsparse_int        m,
                        const T*             csr_val,
                        const rocsparse_int* csr_row_ptr,
                        const rocsparse_int* csr_col_ind,
                        rocsparse_int        ell_width,
                        rocsparse_int*       ell_col_ind,
                        T*                   ell_val,
                        rocsparse_int*       coo_row_ind,
                        rocsparse_int*       coo_col_ind,
                        T*                   coo_val,
                        rocsparse_int*       workspace,
                        rocsparse_index_base idx_base)
    {
        rocsparse_int ai = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(ai >= m)
        {
            return;
        }

        rocsparse_int p = 0;

        rocsparse_int row_begin = csr_row_ptr[ai] - idx_base;
        rocsparse_int row_end   = csr_row_ptr[ai + 1] - idx_base;
        rocsparse_int coo_idx   = coo_row_ind ? workspace[ai] - idx_base : 0;

        // Fill HYB matrix
        for(rocsparse_int aj = row_begin; aj < row_end; ++aj)
        {
            if(p < ell_width)
            {
                // Fill ELL part
                rocsparse_int idx = ELL_IND(ai, p++, m, ell_width);
                ell_col_ind[idx]  = csr_col_ind[aj];
                ell_val[idx]      = csr_val[aj];
            }
            else
            {
                // Fill COO part
                coo_row_ind[coo_idx] = ai + idx_base;
                coo_col_ind[coo_idx] = csr_col_ind[aj];
                coo_val[coo_idx]     = csr_val[aj];
                ++coo_idx;
            }
        }

        // Pad remaining ELL structure
        for(rocsparse_int aj = row_end - row_begin; aj < ell_width; ++aj)
        {
            rocsparse_int idx = ELL_IND(ai, aj, m, ell_width);
            ell_col_ind[idx]  = -1;
            ell_val[idx]      = static_cast<T>(0);
        }
    }
}
