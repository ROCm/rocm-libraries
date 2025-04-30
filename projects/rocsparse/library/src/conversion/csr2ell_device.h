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

#include "common.h"
#include "handle.h"

namespace rocsparse
{
    // Compute non-zero entries per CSR row and do a block reduction over the maximum
    // Store result in a workspace for final reduction on part2
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ell_width_kernel_part1(J m, const I* csr_row_ptr, J* workspace)
    {
        const uint32_t tid = hipThreadIdx_x;
        const uint32_t gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        __shared__ J sdata[BLOCKSIZE];
        sdata[tid] = 0;

        for(uint32_t idx = gid; idx < m; idx += hipGridDim_x * BLOCKSIZE)
        {
            sdata[tid] = rocsparse::max(sdata[tid], J(csr_row_ptr[idx + 1] - csr_row_ptr[idx]));
        }

        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(tid, sdata);

        if(tid == 0)
        {
            workspace[hipBlockIdx_x] = sdata[0];
        }
    }

    // Part2 kernel for final reduction over the maximum CSR nnz row entries
    template <uint32_t BLOCKSIZE, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ell_width_kernel_part2(J m, J* workspace)
    {
        const uint32_t tid = hipThreadIdx_x;

        __shared__ J sdata[BLOCKSIZE];
        sdata[tid] = 0;

        for(uint32_t i = tid; i < m; i += BLOCKSIZE)
        {
            sdata[tid] = rocsparse::max(sdata[tid], workspace[i]);
        }

        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(tid, sdata);

        if(tid == 0)
        {
            workspace[0] = sdata[0];
        }
    }

    // CSR to ELL format conversion kernel
    template <uint32_t BLOCKSIZE, typename T, typename I, typename J>
    __device__ void csr2ell_device(J                    m,
                                   const T*             csr_val,
                                   const I*             csr_row_ptr,
                                   const J*             csr_col_ind,
                                   rocsparse_index_base csr_idx_base,
                                   J                    ell_width,
                                   J*                   ell_col_ind,
                                   T*                   ell_val,
                                   rocsparse_index_base ell_idx_base)
    {
        const J ai = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(ai >= m)
        {
            return;
        }

        J p = 0;

        const I row_begin = csr_row_ptr[ai] - csr_idx_base;
        const I row_end   = csr_row_ptr[ai + 1] - csr_idx_base;

        // Fill ELL matrix
        for(rocsparse_int aj = row_begin; aj < row_end; ++aj)
        {
            if(p >= ell_width)
            {
                break;
            }

            rocsparse_int idx = ELL_IND(ai, p++, m, ell_width);
            ell_col_ind[idx]  = csr_col_ind[aj] - csr_idx_base + ell_idx_base;
            ell_val[idx]      = csr_val[aj];
        }

        // Pad remaining ELL structure
        for(J aj = row_end - row_begin; aj < ell_width; ++aj)
        {
            const I idx      = ELL_IND(ai, aj, m, ell_width);
            ell_col_ind[idx] = -1;
            ell_val[idx]     = static_cast<T>(0);
        }
    }

    // CSR to ELL format conversion kernel
    template <uint32_t BLOCKSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2ell_kernel(J                    m,
                        const T*             csr_val,
                        const I*             csr_row_ptr,
                        const J*             csr_col_ind,
                        rocsparse_index_base csr_idx_base,
                        J                    ell_width,
                        J*                   ell_col_ind,
                        T*                   ell_val,
                        rocsparse_index_base ell_idx_base)
    {
        csr2ell_device<BLOCKSIZE, T, I, J>(m,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_idx_base,
                                           ell_width,
                                           ell_col_ind,
                                           ell_val,
                                           ell_idx_base);
    }
    // CSR to ELL format conversion kernel
    template <uint32_t BLOCKSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2ell_strided_batched_kernel(J                    m,
                                        const T*             csr_val,
                                        int64_t              csr_val_stride,
                                        const I*             csr_row_ptr,
                                        const J*             csr_col_ind,
                                        rocsparse_index_base csr_idx_base,
                                        J                    ell_width,
                                        J*                   ell_col_ind,
                                        T*                   ell_val,
                                        int64_t              ell_val_stride,
                                        rocsparse_index_base ell_idx_base)
    {
        const T* batch_csr_val = csr_val + csr_val_stride * blockIdx.y;
        T*       batch_ell_val = ell_val + ell_val_stride * blockIdx.y;
        csr2ell_device<BLOCKSIZE, T, I, J>(m,
                                           batch_csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_idx_base,
                                           ell_width,
                                           ell_col_ind,
                                           batch_ell_val,
                                           ell_idx_base);
    }
}
