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
    template <uint32_t BLOCKSIZE, uint32_t ROW_BLOCKDIM, rocsparse_int WFSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2gebsr_nnz_wavefront_per_row_multipass_kernel(
        rocsparse_int        m,
        rocsparse_int        n,
        rocsparse_int        mb,
        rocsparse_int        nb,
        rocsparse_int        row_block_dim,
        rocsparse_int        col_block_dim,
        rocsparse_index_base csr_base,
        const rocsparse_int* __restrict__ csr_row_ptr,
        const rocsparse_int* __restrict__ csr_col_ind,
        rocsparse_index_base bsr_base,
        rocsparse_int* __restrict__ bsr_row_ptr)
    {
        int bid = hipBlockIdx_x;
        int tid = hipThreadIdx_x;

        // Lane id
        int lid = tid & (WFSIZE - 1);
        // Wavefront id
        int wid = tid / (WFSIZE);

        int c = lid & (WFSIZE / ROW_BLOCKDIM - 1);
        int r = lid / (WFSIZE / ROW_BLOCKDIM);

        rocsparse_int row = (BLOCKSIZE / WFSIZE) * row_block_dim * bid + row_block_dim * wid + r;

        __shared__ bool          found[BLOCKSIZE / WFSIZE];
        __shared__ rocsparse_int nnzb_per_row[BLOCKSIZE / WFSIZE];

        nnzb_per_row[wid] = 0;

        __syncthreads();

        rocsparse_int row_begin = (row < m && r < row_block_dim) ? csr_row_ptr[row] - csr_base : 0;
        rocsparse_int row_end
            = (row < m && r < row_block_dim) ? csr_row_ptr[row + 1] - csr_base : 0;

        rocsparse_int next_k = row_begin;

        // Begin of the current row chunk (this is the column index of the current row)
        rocsparse_int chunk_begin = 0;

        // Loop over the row chunks until the end of the row has been reached (which is
        // the number of total columns)
        while(chunk_begin < nb)
        {
            // Initialize row nnz table and accumulator
            found[wid] = 0;

            // Wait for all threads to finish initialization
            __threadfence_block();

            // Initialize the beginning of the next chunk
            rocsparse_int min_block_col = nb;

            rocsparse_int index_k = row_end;

            for(rocsparse_int k = next_k + c; k < row_end; k += WFSIZE / ROW_BLOCKDIM)
            {
                rocsparse_int block_col = (csr_col_ind[k] - csr_base) / col_block_dim;

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

            rocsparse::wfreduce_min<WFSIZE / ROW_BLOCKDIM>(&index_k);
            next_k = __shfl(index_k, WFSIZE / ROW_BLOCKDIM - 1, WFSIZE / ROW_BLOCKDIM);

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

    template <uint32_t BLOCKSIZE, uint32_t ROW_BLOCKDIM>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2gebsr_nnz_block_per_row_multipass_kernel(rocsparse_int        m,
                                                      rocsparse_int        n,
                                                      rocsparse_int        mb,
                                                      rocsparse_int        nb,
                                                      rocsparse_int        row_block_dim,
                                                      rocsparse_int        col_block_dim,
                                                      rocsparse_index_base csr_base,
                                                      const rocsparse_int* __restrict__ csr_row_ptr,
                                                      const rocsparse_int* __restrict__ csr_col_ind,
                                                      rocsparse_index_base bsr_base,
                                                      rocsparse_int* __restrict__ bsr_row_ptr)
    {
        int bid = hipBlockIdx_x;
        int tid = hipThreadIdx_x;

        // Lane id
        int lid = tid & (BLOCKSIZE / ROW_BLOCKDIM - 1);
        // Wavefront id
        int wid = tid / (BLOCKSIZE / ROW_BLOCKDIM);

        rocsparse_int row = row_block_dim * bid + wid;

        __shared__ bool          found;
        __shared__ rocsparse_int nnzb_per_row;
        __shared__ rocsparse_int shared[BLOCKSIZE];

        nnzb_per_row = 0;
        __syncthreads();

        rocsparse_int row_begin
            = (row < m && wid < row_block_dim) ? csr_row_ptr[row] - csr_base : 0;
        rocsparse_int row_end
            = (row < m && wid < row_block_dim) ? csr_row_ptr[row + 1] - csr_base : 0;

        rocsparse_int next_k = row_begin;

        // Begin of the current row chunk (this is the column index of the current row)
        rocsparse_int chunk_begin = 0;

        // Loop over the row chunks until the end of the row has been reached (which is
        // the number of total columns)
        while(chunk_begin < nb)
        {
            found = 0;

            // Wait for all threads to finish initialization
            __syncthreads();

            // Initialize the beginning of the next chunk
            rocsparse_int min_block_col = nb;

            rocsparse_int index_k = row_end;

            for(rocsparse_int k = next_k + lid; k < row_end; k += (BLOCKSIZE / ROW_BLOCKDIM))
            {
                rocsparse_int block_col = (csr_col_ind[k] - csr_base) / col_block_dim;

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

            rocsparse::wfreduce_min<BLOCKSIZE / ROW_BLOCKDIM>(&index_k);
            next_k = __shfl(index_k, (BLOCKSIZE / ROW_BLOCKDIM) - 1, BLOCKSIZE / ROW_BLOCKDIM);

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

    template <rocsparse_int BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2gebsr_nnz_65_inf_kernel(rocsparse_int        m,
                                     rocsparse_int        n,
                                     rocsparse_int        mb,
                                     rocsparse_int        nb,
                                     rocsparse_int        row_block_dim,
                                     rocsparse_int        col_block_dim,
                                     rocsparse_int        rows_per_segment,
                                     rocsparse_index_base csr_base,
                                     const rocsparse_int* __restrict__ csr_row_ptr,
                                     const rocsparse_int* __restrict__ csr_col_ind,
                                     rocsparse_index_base bsr_base,
                                     rocsparse_int* __restrict__ bsr_row_ptr,
                                     rocsparse_int* __restrict__ temp1)
    {
        rocsparse_int block_id = hipBlockIdx_x;
        rocsparse_int lane_id  = hipThreadIdx_x;

        rocsparse_int block_col    = 0;
        rocsparse_int nnzb_per_row = 0;

        // temp array used as global scratch pad
        rocsparse_int* row_start
            = temp1 + (2 * rows_per_segment * BLOCKSIZE * block_id) + rows_per_segment * lane_id;
        rocsparse_int* row_end = temp1 + (2 * rows_per_segment * BLOCKSIZE * block_id)
                                 + rows_per_segment * BLOCKSIZE + rows_per_segment * lane_id;

        for(rocsparse_int j = 0; j < rows_per_segment; j++)
        {
            row_start[j] = 0;
            row_end[j]   = 0;

            rocsparse_int row_index = row_block_dim * block_id + BLOCKSIZE * j + lane_id;

            if(row_index < m && (BLOCKSIZE * j + lane_id) < row_block_dim)
            {
                row_start[j] = csr_row_ptr[row_index] - csr_base;
                row_end[j]   = csr_row_ptr[row_index + 1] - csr_base;
            }
        }

        while(block_col < nb)
        {
            // Find minimum column index that is also greater than or equal to col
            rocsparse_int min_block_col_index = nb;
            for(rocsparse_int j = 0; j < rows_per_segment; j++)
            {
                for(rocsparse_int i = row_start[j]; i < row_end[j]; i++)
                {
                    rocsparse_int block_col_index = (csr_col_ind[i] - csr_base) / col_block_dim;

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

            //
            // last thread in segment will contain the min after this call
            //
            rocsparse::wfreduce_min<BLOCKSIZE>(&min_block_col_index);

            //
            // broadcast min_block_col_index from last thread in segment to all threads in segment
            //
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

    template <rocsparse_int BLOCKSIZE,
              rocsparse_int ROW_BLOCKDIM,
              rocsparse_int COL_BLOCKDIM,
              rocsparse_int WFSIZE,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2gebsr_wavefront_per_row_multipass_kernel(rocsparse_direction        dir,
                                                      rocsparse_int              m,
                                                      rocsparse_int              n,
                                                      rocsparse_int              mb,
                                                      rocsparse_int              nb,
                                                      rocsparse_int              row_block_dim,
                                                      rocsparse_int              col_block_dim,
                                                      const rocsparse_index_base csr_base,
                                                      const T* __restrict__ csr_val,
                                                      const rocsparse_int* __restrict__ csr_row_ptr,
                                                      const rocsparse_int* __restrict__ csr_col_ind,
                                                      const rocsparse_index_base bsr_base,
                                                      T* __restrict__ bsr_val,
                                                      rocsparse_int* __restrict__ bsr_row_ptr,
                                                      rocsparse_int* __restrict__ bsr_col_ind)
    {
        int bid = hipBlockIdx_x;
        int tid = hipThreadIdx_x;

        // Lane id
        int lid = tid & (WFSIZE - 1);
        // Wavefront id
        int wid = tid / WFSIZE;

        int c = lid & (WFSIZE / ROW_BLOCKDIM - 1);
        int r = lid / (WFSIZE / ROW_BLOCKDIM);

        rocsparse_int block_row = (BLOCKSIZE / WFSIZE) * bid + wid;
        rocsparse_int row = (BLOCKSIZE / WFSIZE) * row_block_dim * bid + row_block_dim * wid + r;

        __shared__ bool table[BLOCKSIZE / WFSIZE];
        __shared__ T    data[(BLOCKSIZE / WFSIZE) * ROW_BLOCKDIM * COL_BLOCKDIM];

        rocsparse_int row_begin = (row < m && r < row_block_dim) ? csr_row_ptr[row] - csr_base : 0;
        rocsparse_int row_end
            = (row < m && r < row_block_dim) ? csr_row_ptr[row + 1] - csr_base : 0;

        rocsparse_int block_row_begin = (block_row < mb) ? bsr_row_ptr[block_row] - bsr_base : 0;

        rocsparse_int next_k = row_begin;

        // Begin of the current row chunk (this is the column index of the current row)
        rocsparse_int chunk_begin = 0;

        // Loop over the row chunks until the end of the row has been reached (which is
        // the number of total columns)
        while(chunk_begin < nb)
        {
            // Initialize row nnz table and accumulator
            table[wid] = 0;
            for(uint32_t i = c; i < COL_BLOCKDIM; i += WFSIZE / ROW_BLOCKDIM)
            {
                data[ROW_BLOCKDIM * COL_BLOCKDIM * wid + COL_BLOCKDIM * r + i] = static_cast<T>(0);
            }

            // Wait for all threads to finish initialization
            __threadfence_block();

            // Initialize the beginning of the next chunk
            rocsparse_int min_block_col = nb;

            rocsparse_int index_k = row_end;

            for(rocsparse_int k = next_k + c; k < row_end; k += WFSIZE / ROW_BLOCKDIM)
            {
                rocsparse_int col       = (csr_col_ind[k] - csr_base);
                rocsparse_int block_col = col / col_block_dim;

                if(block_col == chunk_begin)
                {
                    table[wid] = 1;
                    data[ROW_BLOCKDIM * COL_BLOCKDIM * wid + COL_BLOCKDIM * r
                         + (col % col_block_dim)]
                        = csr_val[k];
                }
                else
                {
                    index_k       = k;
                    min_block_col = rocsparse::min(min_block_col, block_col);
                    break;
                }
            }

            __threadfence_block();

            rocsparse::wfreduce_min<WFSIZE / ROW_BLOCKDIM>(&index_k);
            next_k = __shfl(index_k, (WFSIZE / ROW_BLOCKDIM) - 1, WFSIZE / ROW_BLOCKDIM);

            int offset = 0;
            if(table[wid])
            {
                bsr_col_ind[block_row_begin] = chunk_begin + bsr_base;

                for(uint32_t i = 0; i < COL_BLOCKDIM; i += WFSIZE / ROW_BLOCKDIM)
                {
                    if(r < row_block_dim && (c + i) < col_block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            bsr_val[int64_t(row_block_dim) * col_block_dim * block_row_begin
                                    + col_block_dim * r + c + i]
                                = data[ROW_BLOCKDIM * COL_BLOCKDIM * wid + COL_BLOCKDIM * r + c
                                       + i];
                        }
                        else
                        {
                            bsr_val[int64_t(row_block_dim) * col_block_dim * block_row_begin
                                    + row_block_dim * (c + i) + r]
                                = data[ROW_BLOCKDIM * COL_BLOCKDIM * wid + COL_BLOCKDIM * r + c
                                       + i];
                        }
                    }
                }

                offset++;
            }

            __threadfence_block();

            block_row_begin += offset;

            rocsparse::wfreduce_min<WFSIZE>(&min_block_col);
            chunk_begin = __shfl(min_block_col, WFSIZE - 1, WFSIZE);

            __threadfence_block();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t ROW_BLOCKDIM, uint32_t COL_BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2gebsr_block_per_row_multipass_kernel(rocsparse_direction        dir,
                                                  rocsparse_int              m,
                                                  rocsparse_int              n,
                                                  rocsparse_int              mb,
                                                  rocsparse_int              nb,
                                                  rocsparse_int              row_block_dim,
                                                  rocsparse_int              col_block_dim,
                                                  const rocsparse_index_base csr_base,
                                                  const T* __restrict__ csr_val,
                                                  const rocsparse_int* __restrict__ csr_row_ptr,
                                                  const rocsparse_int* __restrict__ csr_col_ind,
                                                  const rocsparse_index_base bsr_base,
                                                  T* __restrict__ bsr_val,
                                                  rocsparse_int* __restrict__ bsr_row_ptr,
                                                  rocsparse_int* __restrict__ bsr_col_ind)
    {
        int bid = hipBlockIdx_x;
        int tid = hipThreadIdx_x;

        // Lane id
        int lid = tid & (BLOCKSIZE / ROW_BLOCKDIM - 1);
        // Wavefront id
        int wid = tid / (BLOCKSIZE / ROW_BLOCKDIM);

        rocsparse_int block_row = bid;
        rocsparse_int row       = row_block_dim * bid + wid;

        __shared__ bool table;
        __shared__ T    data[ROW_BLOCKDIM * COL_BLOCKDIM];

        rocsparse_int row_begin
            = (row < m && wid < row_block_dim) ? csr_row_ptr[row] - csr_base : 0;
        rocsparse_int row_end
            = (row < m && wid < row_block_dim) ? csr_row_ptr[row + 1] - csr_base : 0;

        rocsparse_int block_row_begin = bsr_row_ptr[block_row] - bsr_base;

        rocsparse_int next_k = row_begin;

        // Begin of the current row chunk (this is the column index of the current row)
        rocsparse_int chunk_begin = 0;

        // Loop over the row chunks until the end of the row has been reached (which is
        // the number of total columns)
        while(chunk_begin < nb)
        {
            table = 0;
            for(uint32_t i = 0; i < COL_BLOCKDIM; i += BLOCKSIZE / ROW_BLOCKDIM)
            {
                data[COL_BLOCKDIM * wid + i + lid] = static_cast<T>(0);
            }

            // Wait for all threads to finish initialization
            __syncthreads();

            // Initialize the beginning of the next chunk
            rocsparse_int min_block_col = nb;

            rocsparse_int index_k = row_end;

            for(rocsparse_int k = next_k + lid; k < row_end; k += (BLOCKSIZE / ROW_BLOCKDIM))
            {
                rocsparse_int col       = (csr_col_ind[k] - csr_base);
                rocsparse_int block_col = col / col_block_dim;

                if(block_col == chunk_begin)
                {
                    table                                            = 1;
                    data[COL_BLOCKDIM * wid + (col % col_block_dim)] = csr_val[k];
                }
                else
                {
                    index_k       = k;
                    min_block_col = rocsparse::min(min_block_col, block_col);
                    break;
                }
            }

            __syncthreads();

            rocsparse::wfreduce_min<BLOCKSIZE / ROW_BLOCKDIM>(&index_k);
            next_k = __shfl(index_k, (BLOCKSIZE / ROW_BLOCKDIM) - 1, BLOCKSIZE / ROW_BLOCKDIM);

            int offset = 0;
            if(table)
            {
                bsr_col_ind[block_row_begin] = chunk_begin + bsr_base;

                for(uint32_t i = 0; i < COL_BLOCKDIM; i += BLOCKSIZE / ROW_BLOCKDIM)
                {
                    if((i + lid) < col_block_dim && wid < row_block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            bsr_val[int64_t(row_block_dim) * col_block_dim * block_row_begin
                                    + col_block_dim * wid + i + lid]
                                = data[COL_BLOCKDIM * wid + i + lid];
                        }
                        else
                        {
                            bsr_val[int64_t(row_block_dim) * col_block_dim * block_row_begin
                                    + row_block_dim * (i + lid) + wid]
                                = data[COL_BLOCKDIM * wid + i + lid];
                        }
                    }
                }

                offset++;
            }

            __syncthreads();

            rocsparse_int* shared = reinterpret_cast<rocsparse_int*>(data);

            shared[tid] = min_block_col;

            __syncthreads();
            block_row_begin += offset;

            rocsparse::blockreduce_min<BLOCKSIZE>(tid, shared);

            chunk_begin = shared[0];
            __syncthreads();
        }
    }

    template <rocsparse_int BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2gebsr_65_inf_kernel(rocsparse_direction        direction,
                                 rocsparse_int              m,
                                 rocsparse_int              n,
                                 rocsparse_int              mb,
                                 rocsparse_int              nb,
                                 rocsparse_int              row_block_dim,
                                 rocsparse_int              col_block_dim,
                                 rocsparse_int              rows_per_segment,
                                 const rocsparse_index_base csr_base,
                                 const T* __restrict__ csr_val,
                                 const rocsparse_int* __restrict__ csr_row_ptr,
                                 const rocsparse_int* __restrict__ csr_col_ind,
                                 const rocsparse_index_base bsr_base,
                                 T* __restrict__ bsr_val,
                                 rocsparse_int* __restrict__ bsr_row_ptr,
                                 rocsparse_int* __restrict__ bsr_col_ind,
                                 rocsparse_int* __restrict__ temp1,
                                 T* __restrict__ temp2)
    {
        rocsparse_int block_id = hipBlockIdx_x;
        rocsparse_int lane_id  = hipThreadIdx_x;

        rocsparse_int bsr_row_start = 0;

        if(block_id < mb)
        {
            bsr_row_start = bsr_row_ptr[block_id] - bsr_base;
        }

        rocsparse_int csr_col       = 0;
        rocsparse_int bsr_block_col = 0;
        rocsparse_int nnzb_per_row  = 0;

        // temp arrays used as global scratch pad
        rocsparse_int* row_start
            = temp1 + (3 * rows_per_segment * BLOCKSIZE * block_id) + rows_per_segment * lane_id;
        rocsparse_int* row_end = temp1 + (3 * rows_per_segment * BLOCKSIZE * block_id)
                                 + rows_per_segment * BLOCKSIZE + rows_per_segment * lane_id;
        rocsparse_int* csr_col_index = temp1 + (3 * rows_per_segment * BLOCKSIZE * block_id)
                                       + 2 * rows_per_segment * BLOCKSIZE
                                       + rows_per_segment * lane_id;
        T* csr_value
            = temp2 + (rows_per_segment * BLOCKSIZE * block_id) + rows_per_segment * lane_id;

        for(rocsparse_int j = 0; j < rows_per_segment; j++)
        {
            row_start[j] = 0;
            row_end[j]   = 0;

            rocsparse_int row_index = row_block_dim * block_id + BLOCKSIZE * j + lane_id;

            if(row_index < m && (BLOCKSIZE * j + lane_id) < row_block_dim)
            {
                row_start[j] = csr_row_ptr[row_index] - csr_base;
                row_end[j]   = csr_row_ptr[row_index + 1] - csr_base;
            }
        }

        while(csr_col < n)
        {
            T             min_csr_value     = 0;
            rocsparse_int min_csr_col_index = n;

            for(rocsparse_int j = 0; j < rows_per_segment; j++)
            {
                csr_value[j]     = 0;
                csr_col_index[j] = n;

                for(rocsparse_int i = row_start[j]; i < row_end[j]; i++)
                {
                    csr_value[j]     = csr_val[i];
                    csr_col_index[j] = csr_col_ind[i] - csr_base;

                    if(csr_col_index[j] >= csr_col)
                    {
                        if(csr_col_index[j] <= min_csr_col_index)
                        {
                            min_csr_value     = csr_value[j];
                            min_csr_col_index = csr_col_index[j];
                        }

                        row_start[j] = i;

                        break;
                    }
                }
            }

            // find minimum CSR column index across all threads in this segment and store in last thread of segment
            rocsparse::wfreduce_min<BLOCKSIZE>(&min_csr_col_index);

            // have last thread in segment write to BSR column indices array
            if(min_csr_col_index < n && lane_id == BLOCKSIZE - 1)
            {
                if((min_csr_col_index / col_block_dim) >= bsr_block_col)
                {
                    bsr_col_ind[bsr_row_start + nnzb_per_row]
                        = min_csr_col_index / col_block_dim + bsr_base;

                    nnzb_per_row++;
                    bsr_block_col = (min_csr_col_index / col_block_dim) + 1;
                }
            }

            // broadcast CSR minimum column index from last thread in segment to all threads in segment
            min_csr_col_index = __shfl(min_csr_col_index, BLOCKSIZE - 1, BLOCKSIZE);

            // broadcast nnzb_per_row from last thread in segment to all threads in segment
            nnzb_per_row = __shfl(nnzb_per_row, BLOCKSIZE - 1, BLOCKSIZE);

            // Write BSR values
            for(rocsparse_int j = 0; j < rows_per_segment; j++)
            {
                if(csr_col_index[j] < n
                   && csr_col_index[j] / col_block_dim == min_csr_col_index / col_block_dim)
                {
                    if(direction == rocsparse_direction_row)
                    {
                        int64_t k = int64_t(bsr_row_start + nnzb_per_row - 1) * row_block_dim
                                        * col_block_dim
                                    + (BLOCKSIZE * j + lane_id) * col_block_dim
                                    + csr_col_index[j] % col_block_dim;

                        bsr_val[k] = csr_value[j];
                    }
                    else
                    {
                        int64_t k = int64_t(bsr_row_start + nnzb_per_row - 1) * row_block_dim
                                        * col_block_dim
                                    + (csr_col_index[j] % col_block_dim) * row_block_dim
                                    + (BLOCKSIZE * j + lane_id);
                        bsr_val[k] = csr_value[j];
                    }
                }
            }

            // update csr_col for all threads in segment
            csr_col = min_csr_col_index + 1;
        }
    }

    template <uint32_t BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2gebsr_nnz_kernel_bm1(rocsparse_int        m,
                                  rocsparse_index_base csr_base,
                                  const rocsparse_int* __restrict__ csr_row_ptr,
                                  const rocsparse_int* __restrict__ csr_col_ind,
                                  rocsparse_index_base bsr_base,
                                  rocsparse_int* __restrict__ bsr_row_ptr,
                                  rocsparse_int col_block_dim)
    {
        rocsparse_int thread_id = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;
        if(thread_id < m)
        {
            rocsparse_int count = 0, pbj = -1;
            for(rocsparse_int k = csr_row_ptr[thread_id] - csr_base;
                k < csr_row_ptr[thread_id + 1] - csr_base;
                ++k)
            {
                rocsparse_int j  = csr_col_ind[k] - csr_base;
                rocsparse_int bj = j / col_block_dim;
                if(bj != pbj)
                {
                    pbj = bj;
                    ++count;
                }
            }
            bsr_row_ptr[0]             = bsr_base;
            bsr_row_ptr[thread_id + 1] = count;
        }
    }

    template <rocsparse_int BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2gebsr_kernel_bm1(rocsparse_int              m,
                              rocsparse_int              n,
                              rocsparse_int              mb,
                              rocsparse_int              nb,
                              const rocsparse_index_base csr_base,
                              const T*                   csr_val,
                              const rocsparse_int*       csr_row_ptr,
                              const rocsparse_int*       csr_col_ind,
                              rocsparse_direction        bsr_direction,
                              const rocsparse_index_base bsr_base,
                              T*                         bsr_val,
                              const rocsparse_int*       bsr_row_ptr,
                              rocsparse_int*             bsr_col_ind,
                              rocsparse_int              row_block_dim,
                              rocsparse_int              col_block_dim)
    {

        rocsparse_int thread_id = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;
        if(thread_id < m)
        {
            rocsparse_int pbj             = -1;
            rocsparse_int bsr_col_ind_off = bsr_row_ptr[thread_id] - bsr_base - 1;
            rocsparse_int bsr_val_off     = bsr_col_ind_off;
            for(rocsparse_int k = csr_row_ptr[thread_id] - csr_base;
                k < csr_row_ptr[thread_id + 1] - csr_base;
                ++k)
            {
                rocsparse_int j  = csr_col_ind[k] - csr_base;
                rocsparse_int bj = j / col_block_dim;
                rocsparse_int lj = j % col_block_dim;
                if(bj != pbj)
                {
                    pbj = bj;

                    bsr_col_ind[++bsr_col_ind_off] = bj + bsr_base;
                    ++bsr_val_off;
                }

                if(bsr_direction != rocsparse_direction_row)
                {
                    bsr_val[int64_t(bsr_val_off) * row_block_dim * col_block_dim + col_block_dim * 0
                            + lj]
                        = csr_val[k];
                }
                else
                {
                    bsr_val[int64_t(bsr_val_off) * row_block_dim * col_block_dim
                            + row_block_dim * lj + 0]
                        = csr_val[k];
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr2gebsr_nnz_compute_nnz_total_kernel(rocsparse_int mb,
                                                const rocsparse_int* __restrict__ bsr_row_ptr,
                                                rocsparse_int* __restrict__ bsr_nnz)
    {
        rocsparse_int thread_id = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(thread_id == 0)
        {
            *bsr_nnz = bsr_row_ptr[mb] - bsr_row_ptr[0];
        }
    }
}
