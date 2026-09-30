/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_HIP_KERNELS_CONVERSION_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_CONVERSION_HPP_

#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution
{

    // Compute non-zero entries per row
    template <typename I, typename J>
    __global__ void kernel_hyb_coo_nnz(I m,
                                       I ell_width,
                                       const J* __restrict__ csr_row_ptr,
                                       J* __restrict__ coo_row_nnz)
    {
        I gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(gid >= m)
        {
            return;
        }

        I row_nnz        = csr_row_ptr[gid + 1] - csr_row_ptr[gid] - ell_width;
        coo_row_nnz[gid] = row_nnz > 0 ? row_nnz : 0;
    }

    // CSR to HYB format conversion kernel
    template <typename T, typename I, typename J>
    __global__ void kernel_hyb_csr2hyb(I m,
                                       const T* __restrict__ csr_val,
                                       const J* __restrict__ csr_row_ptr,
                                       const I* __restrict__ csr_col_ind,
                                       I ell_width,
                                       I* __restrict__ ell_col_ind,
                                       T* __restrict__ ell_val,
                                       I* __restrict__ coo_row_ind,
                                       I* __restrict__ coo_col_ind,
                                       T* __restrict__ coo_val,
                                       J* __restrict__ workspace)
    {
        I ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= m)
        {
            return;
        }

        I p = 0;

        J row_begin = csr_row_ptr[ai];
        J row_end   = csr_row_ptr[ai + 1];
        J coo_idx   = coo_row_ind ? workspace[ai] : 0;

        // Fill HYB matrix
        for(J aj = row_begin; aj < row_end; ++aj)
        {
            if(p < ell_width)
            {
                // Fill ELL part
                I idx            = ELL_IND(ai, p++, m, ell_width);
                ell_col_ind[idx] = csr_col_ind[aj];
                ell_val[idx]     = csr_val[aj];
            }
            else
            {
                // Fill COO part
                coo_row_ind[coo_idx] = ai;
                coo_col_ind[coo_idx] = csr_col_ind[aj];
                coo_val[coo_idx]     = csr_val[aj];
                ++coo_idx;
            }
        }

        // Pad remaining ELL structure
        for(I aj = row_end - row_begin; aj < ell_width; ++aj)
        {
            I idx            = ELL_IND(ai, aj, m, ell_width);
            ell_col_ind[idx] = -1;
            ell_val[idx]     = static_cast<T>(0);
        }
    }

    template <typename I, typename J>
    __global__ void kernel_dia_diag_idx(I nrow,
                                        J* __restrict__ row_offset,
                                        I* __restrict__ col,
                                        I* __restrict__ diag_idx)
    {
        I row = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

        if(row >= nrow)
        {
            return;
        }

        for(J j = row_offset[row]; j < row_offset[row + 1]; ++j)
        {
            I idx         = col[j] - row + nrow;
            diag_idx[idx] = 1;
        }
    }

    template <typename I>
    __global__ void kernel_dia_fill_offset(I nrow,
                                           I ncol,
                                           I* __restrict__ diag_idx,
                                           const I* __restrict__ offset_map,
                                           I* __restrict__ offset)
    {
        I i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

        if(i >= nrow + ncol)
        {
            return;
        }

        if(diag_idx[i] == 1)
        {
            offset[offset_map[i]] = i - nrow;
            diag_idx[i]           = offset_map[i];
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_dia_convert(I nrow,
                                       I ndiag,
                                       const J* __restrict__ row_offset,
                                       const I* __restrict__ col,
                                       const T* __restrict__ val,
                                       const I* __restrict__ diag_idx,
                                       T* __restrict__ dia_val)
    {
        I row = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

        if(row >= nrow)
        {
            return;
        }

        for(J j = row_offset[row]; j < row_offset[row + 1]; ++j)
        {
            I idx = col[j] - row + nrow;

            dia_val[DIA_IND(row, diag_idx[idx], nrow, ndiag)] = val[j];
        }
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_CONVERSION_HPP_
