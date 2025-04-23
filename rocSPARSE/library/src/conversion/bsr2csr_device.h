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

#include "common.h"

namespace rocsparse
{
    template <uint32_t BLOCK_SIZE, uint32_t BLOCK_DIM, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void bsr2csr_block_per_row_2_7_kernel(rocsparse_direction  direction,
                                          J                    mb,
                                          J                    nb,
                                          rocsparse_index_base bsr_base,
                                          const T* __restrict__ bsr_val,
                                          const I* __restrict__ bsr_row_ptr,
                                          const J* __restrict__ bsr_col_ind,
                                          J                    block_dim,
                                          rocsparse_index_base csr_base,
                                          T* __restrict__ csr_val,
                                          I* __restrict__ csr_row_ptr,
                                          J* __restrict__ csr_col_ind)
    {
        // Find next largest power of 2
        uint32_t BLOCK_DIM2 = fnp2(BLOCK_DIM);

        J tid = hipThreadIdx_x;
        J bid = hipBlockIdx_x;

        I start = bsr_row_ptr[bid] - bsr_base;
        I end   = bsr_row_ptr[bid + 1] - bsr_base;

        if(bid == 0 && tid == 0)
        {
            csr_row_ptr[0] = csr_base;
        }

        J lid = tid & (BLOCK_DIM2 - 1);
        J wid = tid / BLOCK_DIM2;

        J r = lid;

        if(r >= BLOCK_DIM)
        {
            return;
        }

        I prev    = BLOCK_DIM * BLOCK_DIM * start + BLOCK_DIM * (end - start) * r;
        I current = BLOCK_DIM * (end - start);

        csr_row_ptr[BLOCK_DIM * bid + r + 1] = prev + current + csr_base;

        for(I i = start + wid; i < end; i += (BLOCK_SIZE / BLOCK_DIM2))
        {
            J col    = bsr_col_ind[i] - bsr_base;
            I offset = prev + BLOCK_DIM * (i - start);

            for(J j = 0; j < BLOCK_DIM; j++)
            {
                csr_col_ind[offset + j] = BLOCK_DIM * col + j + csr_base;

                if(direction == rocsparse_direction_row)
                {
                    csr_val[offset + j] = bsr_val[BLOCK_DIM * BLOCK_DIM * i + r * BLOCK_DIM + j];
                }
                else
                {
                    csr_val[offset + j] = bsr_val[BLOCK_DIM * BLOCK_DIM * i + r + BLOCK_DIM * j];
                }
            }
        }
    }

    template <uint32_t BLOCK_SIZE, uint32_t BLOCK_DIM, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void bsr2csr_block_per_row_8_32_kernel(rocsparse_direction  direction,
                                           J                    mb,
                                           J                    nb,
                                           rocsparse_index_base bsr_base,
                                           const T* __restrict__ bsr_val,
                                           const I* __restrict__ bsr_row_ptr,
                                           const J* __restrict__ bsr_col_ind,
                                           J                    block_dim,
                                           rocsparse_index_base csr_base,
                                           T* __restrict__ csr_val,
                                           I* __restrict__ csr_row_ptr,
                                           J* __restrict__ csr_col_ind)
    {
        J tid = hipThreadIdx_x;
        J bid = hipBlockIdx_x;

        I start = bsr_row_ptr[bid] - bsr_base;
        I end   = bsr_row_ptr[bid + 1] - bsr_base;

        if(bid == 0 && tid == 0)
        {
            csr_row_ptr[0] = csr_base;
        }

        J lid = tid & (BLOCK_DIM * BLOCK_DIM - 1);
        J wid = tid / (BLOCK_DIM * BLOCK_DIM);

        J c = lid & (BLOCK_DIM - 1);
        J r = lid / BLOCK_DIM;

        if(r >= block_dim || c >= block_dim)
        {
            return;
        }

        I prev    = block_dim * block_dim * start + block_dim * (end - start) * r;
        I current = block_dim * (end - start);

        csr_row_ptr[block_dim * bid + r + 1] = prev + current + csr_base;

        for(I i = start + wid; i < end; i += (BLOCK_SIZE / (BLOCK_DIM * BLOCK_DIM)))
        {
            J col    = bsr_col_ind[i] - bsr_base;
            I offset = prev + block_dim * (i - start) + c;

            csr_col_ind[offset] = block_dim * col + c + csr_base;

            if(direction == rocsparse_direction_row)
            {
                csr_val[offset] = bsr_val[block_dim * block_dim * i + block_dim * r + c];
            }
            else
            {
                csr_val[offset] = bsr_val[block_dim * block_dim * i + block_dim * c + r];
            }
        }
    }

    template <uint32_t BLOCK_SIZE,
              uint32_t BLOCK_DIM,
              uint32_t SUB_BLOCK_DIM,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void bsr2csr_block_per_row_33_256_kernel(rocsparse_direction  direction,
                                             J                    mb,
                                             J                    nb,
                                             rocsparse_index_base bsr_base,
                                             const T* __restrict__ bsr_val,
                                             const I* __restrict__ bsr_row_ptr,
                                             const J* __restrict__ bsr_col_ind,
                                             J                    block_dim,
                                             rocsparse_index_base csr_base,
                                             T* __restrict__ csr_val,
                                             I* __restrict__ csr_row_ptr,
                                             J* __restrict__ csr_col_ind)
    {
        J tid = hipThreadIdx_x;
        J bid = hipBlockIdx_x;

        I start = bsr_row_ptr[bid] - bsr_base;
        I end   = bsr_row_ptr[bid + 1] - bsr_base;

        if(bid == 0 && tid == 0)
        {
            csr_row_ptr[0] = csr_base;
        }

        for(J y = 0; y < (BLOCK_DIM / SUB_BLOCK_DIM); y++)
        {
            J r = (tid / SUB_BLOCK_DIM) + SUB_BLOCK_DIM * y;

            if(r < block_dim)
            {
                I prev    = block_dim * block_dim * start + block_dim * (end - start) * r;
                I current = block_dim * (end - start);

                csr_row_ptr[block_dim * bid + r + 1] = prev + current + csr_base;
            }
        }

        for(I i = start; i < end; i++)
        {
            J col = bsr_col_ind[i] - bsr_base;

            for(J y = 0; y < (BLOCK_DIM / SUB_BLOCK_DIM); y++)
            {
                for(J x = 0; x < (BLOCK_DIM / SUB_BLOCK_DIM); x++)
                {
                    J c = (tid & (SUB_BLOCK_DIM - 1)) + SUB_BLOCK_DIM * x;
                    J r = (tid / SUB_BLOCK_DIM) + SUB_BLOCK_DIM * y;

                    if(r < block_dim && c < block_dim)
                    {
                        I prev = block_dim * block_dim * start + block_dim * (end - start) * r;

                        I offset = prev + block_dim * (i - start) + c;

                        csr_col_ind[offset] = block_dim * col + c + csr_base;

                        if(direction == rocsparse_direction_row)
                        {
                            csr_val[offset]
                                = bsr_val[block_dim * block_dim * i + block_dim * r + c];
                        }
                        else
                        {
                            csr_val[offset]
                                = bsr_val[block_dim * block_dim * i + block_dim * c + r];
                        }
                    }
                }
            }
        }
    }

    template <uint32_t BLOCK_SIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void bsr2csr_block_dim_equals_one_kernel(J                    mb,
                                             J                    nb,
                                             rocsparse_index_base bsr_base,
                                             const T* __restrict__ bsr_val,
                                             const I* __restrict__ bsr_row_ptr,
                                             const J* __restrict__ bsr_col_ind,
                                             rocsparse_index_base csr_base,
                                             T* __restrict__ csr_val,
                                             I* __restrict__ csr_row_ptr,
                                             J* __restrict__ csr_col_ind)
    {
        J tid = hipThreadIdx_x + BLOCK_SIZE * hipBlockIdx_x;

        if(tid < mb)
        {
            if(tid == 0)
            {
                csr_row_ptr[0] = (bsr_row_ptr[0] - bsr_base) + csr_base;
            }

            csr_row_ptr[tid + 1] = (bsr_row_ptr[tid + 1] - bsr_base) + csr_base;
        }

        I nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];

        J index = tid;
        while(index < nnzb)
        {
            csr_col_ind[index] = (bsr_col_ind[index] - bsr_base) + csr_base;
            csr_val[index]     = bsr_val[index];

            index += BLOCK_SIZE * hipGridDim_x;
        }
    }
}
