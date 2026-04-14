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
        I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        // Do not run out of bounds
        if(bsr_mask_ptr == nullptr)
        {
            if(idx >= block_dim * mb)
            {
                return;
            }

            y[idx] *= beta;
        }
        else
        {
            if(idx >= block_dim * size_of_mask)
            {
                return;
            }

            I shift = (bsr_mask_ptr[idx / block_dim] - idx_base) * block_dim;

            y[shift + (idx % block_dim)] *= beta;
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
        I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(*beta != static_cast<T>(1))
        {
            // Do not run out of bounds
            if(bsr_mask_ptr == nullptr)
            {
                if(idx >= block_dim * mb)
                {
                    return;
                }

                y[idx] *= (*beta);
            }
            else
            {
                if(idx >= block_dim * size_of_mask)
                {
                    return;
                }

                I shift = (bsr_mask_ptr[idx / block_dim] - idx_base) * block_dim;

                y[shift + (idx % block_dim)] *= (*beta);
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
