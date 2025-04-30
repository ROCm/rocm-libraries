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

#include "common.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t BLOCKDIM, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2bsr_nnz_wavefront_per_row_multipass_kernel(J                    m,
                                                        J                    n,
                                                        J                    mb,
                                                        J                    nb,
                                                        J                    block_dim,
                                                        rocsparse_index_base csr_base,
                                                        const I* __restrict__ csr_row_ptr,
                                                        const J* __restrict__ csr_col_ind,
                                                        rocsparse_index_base bsr_base,
                                                        I* __restrict__ bsr_row_ptr)
    {
        int bid = hipBlockIdx_x;
        int tid = hipThreadIdx_x;

        // Lane id
        int lid = tid & (WFSIZE - 1);
        // Wavefront id
        int wid = tid / WFSIZE;

        int c = lid & (WFSIZE / BLOCKDIM - 1);
        int r = lid / (WFSIZE / BLOCKDIM);

        J row = (BLOCKSIZE / WFSIZE) * block_dim * bid + block_dim * wid + r;

        __shared__ bool found[BLOCKSIZE / WFSIZE];
        __shared__ J    nnzb_per_row[BLOCKSIZE / WFSIZE];

        nnzb_per_row[wid] = 0;

        __syncthreads();

        I row_begin = (row < m && r < block_dim) ? csr_row_ptr[row] - csr_base : 0;
        I row_end   = (row < m && r < block_dim) ? csr_row_ptr[row + 1] - csr_base : 0;

        I next_k = row_begin;

        // Begin of the current row chunk (this is the column index of the current row)
        I chunk_begin = 0;

        // Loop over the row chunks until the end of the row has been reached (which is
        // the number of total columns)
        while(chunk_begin < nb)
        {
            // Initialize row nnz table and accumulator
            found[wid] = 0;

            // Wait for all threads to finish initialization
            __threadfence_block();

            // Initialize the beginning of the next chunk
            J min_block_col = nb;

            I index_k = row_end;

            for(I k = next_k + c; k < row_end; k += (WFSIZE / BLOCKDIM))
            {
                J block_col = (csr_col_ind[k] - csr_base) / block_dim;

                if(block_col == chunk_begin)
                {
                    found[wid] = 1;
                }
                else
                {
                    index_k       = k;
                    min_block_col = rocsparse::min(min_block_col, block_col);
                    break;
                }
            }

            __threadfence_block();

            rocsparse::wfreduce_min<(WFSIZE / BLOCKDIM)>(&index_k);
            next_k = __shfl(index_k, (WFSIZE / BLOCKDIM) - 1, (WFSIZE / BLOCKDIM));

            if(found[wid] && lid == 0)
            {
                nnzb_per_row[wid]++;
            }

            rocsparse::wfreduce_min<WFSIZE>(&min_block_col);
            chunk_begin = __shfl(min_block_col, WFSIZE - 1, WFSIZE);

            __threadfence_block();
        }

        if(lid == 0)
        {
            bsr_row_ptr[0] = bsr_base;
            if(((BLOCKSIZE / WFSIZE) * bid + wid) < mb)
            {
                bsr_row_ptr[(BLOCKSIZE / WFSIZE) * bid + wid + 1] = nnzb_per_row[wid];
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2bsr_nnz_block_per_row_multipass_kernel(J                    m,
                                                    J                    n,
                                                    J                    mb,
                                                    J                    nb,
                                                    J                    block_dim,
                                                    rocsparse_index_base csr_base,
                                                    const I* __restrict__ csr_row_ptr,
                                                    const J* __restrict__ csr_col_ind,
                                                    rocsparse_index_base bsr_base,
                                                    I* __restrict__ bsr_row_ptr)
    {
        int bid = hipBlockIdx_x;
        int tid = hipThreadIdx_x;

        // Lane id
        int lid = tid & (BLOCKSIZE / BLOCKDIM - 1);
        // Wavefront id
        int wid = tid / (BLOCKSIZE / BLOCKDIM);

        J row = block_dim * bid + wid;

        __shared__ bool found;
        __shared__ J    nnzb_per_row;
        __shared__ J    shared[BLOCKSIZE];

        nnzb_per_row = 0;
        __syncthreads();

        I row_begin = (row < m && wid < block_dim) ? csr_row_ptr[row] - csr_base : 0;
        I row_end   = (row < m && wid < block_dim) ? csr_row_ptr[row + 1] - csr_base : 0;

        I next_k = row_begin;

        // Begin of the current row chunk (this is the column index of the current row)
        I chunk_begin = 0;

        // Loop over the row chunks until the end of the row has been reached (which is
        // the number of total columns)
        while(chunk_begin < nb)
        {
            found = 0;

            // Wait for all threads to finish initialization
            __syncthreads();

            // Initialize the beginning of the next chunk
            J min_block_col = nb;

            I index_k = row_end;

            for(I k = next_k + lid; k < row_end; k += (BLOCKSIZE / BLOCKDIM))
            {
                J block_col = (csr_col_ind[k] - csr_base) / block_dim;

                if(block_col == chunk_begin)
                {
                    found = 1;
                }
                else
                {
                    index_k       = k;
                    min_block_col = rocsparse::min(min_block_col, block_col);
                    break;
                }
            }

            __syncthreads();

            rocsparse::wfreduce_min<BLOCKSIZE / BLOCKDIM>(&index_k);
            next_k = __shfl(index_k, (BLOCKSIZE / BLOCKDIM) - 1, BLOCKSIZE / BLOCKDIM);

            if(found && tid == 0)
            {
                nnzb_per_row++;
            }

            shared[tid] = min_block_col;
            __syncthreads();

            rocsparse::blockreduce_min<BLOCKSIZE>(tid, shared);

            chunk_begin = shared[0];
            __syncthreads();
        }

        if(tid == 0)
        {
            bsr_row_ptr[0]       = bsr_base;
            bsr_row_ptr[bid + 1] = nnzb_per_row;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2bsr_nnz_65_inf_kernel(J                    m,
                                   J                    n,
                                   J                    mb,
                                   J                    nb,
                                   J                    block_dim,
                                   J                    rows_per_segment,
                                   rocsparse_index_base csr_base,
                                   const I* __restrict__ csr_row_ptr,
                                   const J* __restrict__ csr_col_ind,
                                   rocsparse_index_base bsr_base,
                                   I* __restrict__ bsr_row_ptr,
                                   I* __restrict__ temp1)
    {
        J block_id = hipBlockIdx_x;
        J lane_id  = hipThreadIdx_x;

        J block_col    = 0;
        J nnzb_per_row = 0;

        // temp array used as global scratch pad
        I* row_start
            = temp1 + (2 * rows_per_segment * BLOCKSIZE * block_id) + rows_per_segment * lane_id;
        I* row_end = temp1 + (2 * rows_per_segment * BLOCKSIZE * block_id)
                     + rows_per_segment * BLOCKSIZE + rows_per_segment * lane_id;

        for(J j = 0; j < rows_per_segment; j++)
        {
            row_start[j] = 0;
            row_end[j]   = 0;

            J row_index = block_dim * block_id + BLOCKSIZE * j + lane_id;

            if(row_index < m && (BLOCKSIZE * j + lane_id) < block_dim)
            {
                row_start[j] = csr_row_ptr[row_index] - csr_base;
                row_end[j]   = csr_row_ptr[row_index + 1] - csr_base;
            }
        }

        while(block_col < nb)
        {
            // Find minimum column index that is also greater than or equal to col
            J min_block_col_index = nb;

            for(J j = 0; j < rows_per_segment; j++)
            {
                for(I i = row_start[j]; i < row_end[j]; i++)
                {
                    J block_col_index = (csr_col_ind[i] - csr_base) / block_dim;

                    if(block_col_index >= block_col)
                    {
                        if(block_col_index <= min_block_col_index)
                        {
                            min_block_col_index = block_col_index;
                        }

                        row_start[j] = i;

                        break;
                    }
                }
            }

            // last thread in segment will contain the min after this call
            rocsparse::wfreduce_min<BLOCKSIZE>(&min_block_col_index);

            // broadcast min_block_col_index from last thread in segment to all threads in segment
            min_block_col_index = __shfl(min_block_col_index, BLOCKSIZE - 1, BLOCKSIZE);

            block_col = min_block_col_index + 1;

            if(lane_id == BLOCKSIZE - 1)
            {
                if(min_block_col_index < nb)
                {
                    nnzb_per_row++;
                }
            }
        }

        if(block_id < mb && lane_id == BLOCKSIZE - 1)
        {
            bsr_row_ptr[0]            = bsr_base;
            bsr_row_ptr[block_id + 1] = nnzb_per_row;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2bsr_nnz_block_dim_equals_one_kernel(J                    m,
                                                 rocsparse_index_base csr_base,
                                                 const I* __restrict__ csr_row_ptr,
                                                 rocsparse_index_base bsr_base,
                                                 I* __restrict__ bsr_row_ptr,
                                                 I* __restrict__ bsr_nnz)
    {
        J thread_id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

        if(thread_id < m + 1)
        {
            bsr_row_ptr[thread_id] = (csr_row_ptr[thread_id] - csr_base) + bsr_base;
        }

        if(thread_id == 0)
        {
            *bsr_nnz = csr_row_ptr[m] - csr_row_ptr[0];
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2bsr_nnz_block_dim_equals_one_kernel(J                    m,
                                                 rocsparse_index_base csr_base,
                                                 const I* __restrict__ csr_row_ptr,
                                                 rocsparse_index_base bsr_base,
                                                 I* __restrict__ bsr_row_ptr)
    {
        J tid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(tid < m + 1)
        {
            bsr_row_ptr[tid] = (csr_row_ptr[tid] - csr_base) + bsr_base;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2bsr_nnz_compute_nnz_total_kernel(J mb,
                                              const I* __restrict__ bsr_row_ptr,
                                              I* __restrict__ bsr_nnz)
    {
        J tid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(tid == 0)
        {
            *bsr_nnz = bsr_row_ptr[mb] - bsr_row_ptr[0];
        }
    }
}
