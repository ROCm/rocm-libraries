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
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ell2csr_index_base(I* __restrict__ nnz)
    {
        --(*nnz);
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ell2csr_nnz_per_row(J m,
                             J n,
                             J ell_width,
                             const J* __restrict__ ell_col_ind,
                             rocsparse_index_base ell_base,
                             I* __restrict__ csr_row_ptr,
                             rocsparse_index_base csr_base)
    {
        const J ai = ((J)BLOCKSIZE) * hipBlockIdx_x + hipThreadIdx_x;

        if(ai >= m)
        {
            return;
        }

        if(ai == 0)
        {
            csr_row_ptr[0] = csr_base;
        }

        I nnz = 0;

        for(rocsparse_int p = 0; p < ell_width; ++p)
        {
            const I idx = ELL_IND(ai, p, m, ell_width);
            const J col = ell_col_ind[idx] - ell_base;
            if(col >= 0 && col < n)
            {
                ++nnz;
            }
            else
            {
                break;
            }
        }
        csr_row_ptr[ai + 1] = nnz;
    }

    template <uint32_t BLOCKSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ell2csr_fill(J m,
                      J n,
                      J ell_width,
                      const J* __restrict__ ell_col_ind,
                      const T* __restrict__ ell_val,
                      rocsparse_index_base ell_base,
                      const I* __restrict__ csr_row_ptr,
                      J* __restrict__ csr_col_ind,
                      T* __restrict__ csr_val,
                      rocsparse_index_base csr_base)
    {
        const J ai = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;

        if(ai >= m)
        {
            return;
        }

        I csr_idx = csr_row_ptr[ai] - csr_base;
        for(J p = 0; p < ell_width; ++p)
        {
            const I ell_idx = ELL_IND(ai, p, m, ell_width);
            const J ell_col = ell_col_ind[ell_idx] - ell_base;
            if(ell_col >= 0 && ell_col < n)
            {
                csr_col_ind[csr_idx] = ell_col + csr_base;
                csr_val[csr_idx]     = ell_val[ell_idx];
                ++csr_idx;
            }
            else
            {
                break;
            }
        }
    }
}
