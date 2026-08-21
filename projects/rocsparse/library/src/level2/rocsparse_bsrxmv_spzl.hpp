/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_control.hpp"

#include "rocsparse_utility.hpp"

#include "rocsparse_common.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrxmv_scale_array(I mb,
                            I size_of_mask,
                            I block_dim,
                            const I* __restrict__ bsr_mask_ptr,
                            T* __restrict__ y,
                            T                    beta,
                            rocsparse_index_base idx_base)
    {
        // Compute the total number of scalar entries in 64-bit to avoid a signed
        // 32-bit overflow of block_dim * mb (see AISPARSE-659).
        const int64_t nentries = (bsr_mask_ptr == nullptr)
                                     ? static_cast<int64_t>(block_dim) * mb
                                     : static_cast<int64_t>(block_dim) * size_of_mask;

        // Grid-stride loop so a grid clamped against maxGridSize[0] still covers
        // the full range.
        const int64_t stride = static_cast<int64_t>(BLOCKSIZE) * hipGridDim_x;
        for(int64_t idx = static_cast<int64_t>(hipThreadIdx_x)
                          + static_cast<int64_t>(BLOCKSIZE) * hipBlockIdx_x;
            idx < nentries;
            idx += stride)
        {
            if(bsr_mask_ptr == nullptr)
            {
                y[idx] *= beta;
            }
            else
            {
                const int64_t shift
                    = (static_cast<int64_t>(bsr_mask_ptr[idx / block_dim]) - idx_base) * block_dim;

                y[shift + (idx % block_dim)] *= beta;
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrxmv_scale_array(I mb,
                            I size_of_mask,
                            I block_dim,
                            const I* __restrict__ bsr_mask_ptr,
                            T* __restrict__ y,
                            const T*             beta,
                            rocsparse_index_base idx_base)
    {
        if(*beta != static_cast<T>(1))
        {
            // Compute the total number of scalar entries in 64-bit to avoid a
            // signed 32-bit overflow of block_dim * mb (see AISPARSE-659).
            const int64_t nentries = (bsr_mask_ptr == nullptr)
                                         ? static_cast<int64_t>(block_dim) * mb
                                         : static_cast<int64_t>(block_dim) * size_of_mask;

            // Grid-stride loop so a grid clamped against maxGridSize[0] still
            // covers the full range.
            const int64_t stride = static_cast<int64_t>(BLOCKSIZE) * hipGridDim_x;
            for(int64_t idx = static_cast<int64_t>(hipThreadIdx_x)
                              + static_cast<int64_t>(BLOCKSIZE) * hipBlockIdx_x;
                idx < nentries;
                idx += stride)
            {
                if(bsr_mask_ptr == nullptr)
                {
                    y[idx] *= (*beta);
                }
                else
                {
                    const int64_t shift
                        = (static_cast<int64_t>(bsr_mask_ptr[idx / block_dim]) - idx_base)
                          * block_dim;

                    y[shift + (idx % block_dim)] *= (*beta);
                }
            }
        }
    }

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    void bsrxmvn_2x2(rocsparse_handle     handle,
                     rocsparse_direction  dir,
                     J                    mb,
                     I                    nnzb,
                     const T*             alpha_device_host,
                     J                    size_of_mask,
                     const J*             bsr_mask_ptr,
                     const I*             bsr_row_ptr,
                     const I*             bsr_end_ptr,
                     const J*             bsr_col_ind,
                     const A*             bsr_val,
                     const X*             x,
                     const T*             beta_device_host,
                     Y*                   y,
                     rocsparse_index_base base);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    void bsrxmvn_3x3(rocsparse_handle     handle,
                     rocsparse_direction  dir,
                     J                    mb,
                     I                    nnzb,
                     const T*             alpha_device_host,
                     J                    size_of_mask,
                     const J*             bsr_mask_ptr,
                     const I*             bsr_row_ptr,
                     const I*             bsr_end_ptr,
                     const J*             bsr_col_ind,
                     const A*             bsr_val,
                     const X*             x,
                     const T*             beta_device_host,
                     Y*                   y,
                     rocsparse_index_base base);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    void bsrxmvn_4x4(rocsparse_handle     handle,
                     rocsparse_direction  dir,
                     J                    mb,
                     I                    nnzb,
                     const T*             alpha_device_host,
                     J                    size_of_mask,
                     const J*             bsr_mask_ptr,
                     const I*             bsr_row_ptr,
                     const I*             bsr_end_ptr,
                     const J*             bsr_col_ind,
                     const A*             bsr_val,
                     const X*             x,
                     const T*             beta_device_host,
                     Y*                   y,
                     rocsparse_index_base base);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    void bsrxmvn_5x5(rocsparse_handle     handle,
                     rocsparse_direction  dir,
                     J                    mb,
                     I                    nnzb,
                     const T*             alpha_device_host,
                     J                    size_of_mask,
                     const J*             bsr_mask_ptr,
                     const I*             bsr_row_ptr,
                     const I*             bsr_end_ptr,
                     const J*             bsr_col_ind,
                     const A*             bsr_val,
                     const X*             x,
                     const T*             beta_device_host,
                     Y*                   y,
                     rocsparse_index_base base);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    void bsrxmvn_8x8(rocsparse_handle     handle,
                     rocsparse_direction  dir,
                     J                    mb,
                     I                    nnzb,
                     const T*             alpha_device_host,
                     J                    size_of_mask,
                     const J*             bsr_mask_ptr,
                     const I*             bsr_row_ptr,
                     const I*             bsr_end_ptr,
                     const J*             bsr_col_ind,
                     const A*             bsr_val,
                     const X*             x,
                     const T*             beta_device_host,
                     Y*                   y,
                     rocsparse_index_base base);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    void bsrxmvn_16x16(rocsparse_handle     handle,
                       rocsparse_direction  dir,
                       J                    mb,
                       I                    nnzb,
                       const T*             alpha_device_host,
                       J                    size_of_mask,
                       const J*             bsr_mask_ptr,
                       const I*             bsr_row_ptr,
                       const I*             bsr_end_ptr,
                       const J*             bsr_col_ind,
                       const A*             bsr_val,
                       const X*             x,
                       const T*             beta_device_host,
                       Y*                   y,
                       rocsparse_index_base base);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    void bsrxmvn_17_32(rocsparse_handle     handle,
                       rocsparse_direction  dir,
                       J                    mb,
                       I                    nnzb,
                       const T*             alpha_device_host,
                       J                    size_of_mask,
                       const J*             bsr_mask_ptr,
                       const I*             bsr_row_ptr,
                       const I*             bsr_end_ptr,
                       const J*             bsr_col_ind,
                       const A*             bsr_val,
                       J                    bsr_dim,
                       const X*             x,
                       const T*             beta_device_host,
                       Y*                   y,
                       rocsparse_index_base base);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    void bsrxmvn_general(rocsparse_handle     handle,
                         rocsparse_direction  dir,
                         J                    mb,
                         const T*             alpha_device_host,
                         J                    size_of_mask,
                         const J*             bsr_mask_ptr,
                         const I*             bsr_row_ptr,
                         const I*             bsr_end_ptr,
                         const J*             bsr_col_ind,
                         const A*             bsr_val,
                         J                    bsr_dim,
                         const X*             x,
                         const T*             beta_device_host,
                         Y*                   y,
                         rocsparse_index_base base);
}
