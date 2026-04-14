/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t WFSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrgeam_wf_per_row_multipass_2_3_device(rocsparse_direction dir,
                                                rocsparse_int       mb,
                                                rocsparse_int       nb,
                                                rocsparse_int       block_dim,
                                                T                   alpha,
                                                const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                                const rocsparse_int* __restrict__ bsr_col_ind_A,
                                                const T* __restrict__ bsr_val_A,
                                                T beta,
                                                const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                                const rocsparse_int* __restrict__ bsr_col_ind_B,
                                                const T* __restrict__ bsr_val_B,
                                                const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                                rocsparse_int* __restrict__ bsr_col_ind_C,
                                                T* __restrict__ bsr_val_C,
                                                rocsparse_index_base idx_base_A,
                                                rocsparse_index_base idx_base_B,
                                                rocsparse_index_base idx_base_C)
    {
        // Lane id
        rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

        // Wavefront id
        rocsparse_int wid = hipThreadIdx_x / WFSIZE;

        // Each wavefront processes a row
        rocsparse_int row = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= mb)
        {
            return;
        }

        // Row entry marker and value accumulator
        __shared__ bool stable[BLOCKSIZE];
        __shared__ T    sdata[BLOCKDIM * BLOCKDIM * BLOCKSIZE];

        bool* table = &stable[wid * WFSIZE];
        T*    data  = &sdata[wid * BLOCKDIM * BLOCKDIM * WFSIZE];

        // Get row entry and exit point of A
        rocsparse_int row_begin_A = bsr_row_ptr_A[row] - idx_base_A;
        rocsparse_int row_end_A   = bsr_row_ptr_A[row + 1] - idx_base_A;

        // Get row entry and exit point of B
        rocsparse_int row_begin_B = bsr_row_ptr_B[row] - idx_base_B;
        rocsparse_int row_end_B   = bsr_row_ptr_B[row + 1] - idx_base_B;

        // Get row entry point of C
        rocsparse_int row_begin_C = bsr_row_ptr_C[row] - idx_base_C;

        // Load the first column of the current row from A and B to set the starting
        // point for the first chunk
        rocsparse_int col_A
            = (row_begin_A < row_end_A) ? bsr_col_ind_A[row_begin_A] - idx_base_A : nb;
        rocsparse_int col_B
            = (row_begin_B < row_end_B) ? bsr_col_ind_B[row_begin_B] - idx_base_B : nb;

        // Begin of the current row chunk
        rocsparse_int chunk_begin = rocsparse::min(col_A, col_B);

        // Initialize the index for column access into A and B
        row_begin_A += lid;
        row_begin_B += lid;

        // Loop over the chunks until the end of both rows (A and B) has been reached (which
        // is the number of total columns n)
        while(true)
        {
            // Initialize row nnz table and value accumulator
            table[lid] = false;
            for(uint32_t r = 0; r < BLOCKDIM; r++)
            {
                for(uint32_t c = 0; c < BLOCKDIM; c++)
                {
                    data[BLOCKDIM * BLOCKDIM * lid + BLOCKDIM * r + c] = static_cast<T>(0);
                }
            }

            __threadfence_block();

            // Initialize the beginning of the next chunk
            rocsparse_int min_col = nb;

            // Loop over all columns of A, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_A < row_end_A; row_begin_A += WFSIZE)
            {
                // Get the column of A
                rocsparse_int col = bsr_col_ind_A[row_begin_A] - idx_base_A;

                // Get the column of A shifted by the chunk_begin
                rocsparse_int shf_A = col - chunk_begin;

                // Check if this column of A is within the chunk
                if(shf_A < WFSIZE)
                {
                    // Mark nnz
                    table[shf_A] = true;

                    // Initialize with value of A
                    if(dir == rocsparse_direction_row)
                    {
                        for(uint32_t r = 0; r < BLOCKDIM; r++)
                        {
                            for(uint32_t c = 0; c < BLOCKDIM; c++)
                            {
                                data[BLOCKDIM * BLOCKDIM * shf_A + BLOCKDIM * r + c]
                                    = alpha
                                      * bsr_val_A[BLOCKDIM * BLOCKDIM * row_begin_A + BLOCKDIM * r
                                                  + c];
                            }
                        }
                    }
                    else
                    {
                        for(uint32_t r = 0; r < BLOCKDIM; r++)
                        {
                            for(uint32_t c = 0; c < BLOCKDIM; c++)
                            {
                                data[BLOCKDIM * BLOCKDIM * shf_A + BLOCKDIM * r + c]
                                    = alpha
                                      * bsr_val_A[BLOCKDIM * BLOCKDIM * row_begin_A + BLOCKDIM * c
                                                  + r];
                            }
                        }
                    }
                }
                else
                {
                    // Store the first column index of A that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col);
                    break;
                }
            }

            __threadfence_block();

            // Loop over all columns of B, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_B < row_end_B; row_begin_B += WFSIZE)
            {
                // Get the column of B
                rocsparse_int col = bsr_col_ind_B[row_begin_B] - idx_base_B;

                // Get the column of B shifted by the chunk_begin
                rocsparse_int shf_B = col - chunk_begin;

                // Check if this column of B is within the chunk
                if(shf_B < WFSIZE)
                {
                    // Mark nnz
                    table[shf_B] = true;

                    // Add values of B
                    if(dir == rocsparse_direction_row)
                    {
                        for(uint32_t r = 0; r < BLOCKDIM; r++)
                        {
                            for(int c = 0; c < BLOCKDIM; c++)
                            {
                                data[BLOCKDIM * BLOCKDIM * shf_B + BLOCKDIM * r + c]
                                    = rocsparse::fma(
                                        beta,
                                        bsr_val_B[BLOCKDIM * BLOCKDIM * row_begin_B + BLOCKDIM * r
                                                  + c],
                                        data[BLOCKDIM * BLOCKDIM * shf_B + BLOCKDIM * r + c]);
                            }
                        }
                    }
                    else
                    {
                        for(uint32_t r = 0; r < BLOCKDIM; r++)
                        {
                            for(int c = 0; c < BLOCKDIM; c++)
                            {
                                data[BLOCKDIM * BLOCKDIM * shf_B + BLOCKDIM * r + c]
                                    = rocsparse::fma(
                                        beta,
                                        bsr_val_B[BLOCKDIM * BLOCKDIM * row_begin_B + BLOCKDIM * c
                                                  + r],
                                        data[BLOCKDIM * BLOCKDIM * shf_B + BLOCKDIM * r + c]);
                            }
                        }
                    }
                }
                else
                {
                    // Store the first column index of B that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col);
                    break;
                }
            }

            __threadfence_block();

            // Each lane checks whether there is an non-zero entry to fill or not
            bool has_nnzb = table[lid];

            // Obtain the bitmask that marks the position of each non-zero entry
            uint64_t mask = __ballot(has_nnzb);

            // If the lane has an nnz assign, it must be filled into C
            if(has_nnzb)
            {
                // Compute the lane's fill position in C
                rocsparse_int offset = rocsparse::popc<WFSIZE>(mask, lid);

                // Fill C
                bsr_col_ind_C[row_begin_C + offset - 1] = lid + chunk_begin + idx_base_C;
                if(dir == rocsparse_direction_row)
                {
                    for(uint32_t r = 0; r < BLOCKDIM; r++)
                    {
                        for(int c = 0; c < BLOCKDIM; c++)
                        {
                            bsr_val_C[BLOCKDIM * BLOCKDIM * (row_begin_C + offset - 1)
                                      + BLOCKDIM * r + c]
                                = data[BLOCKDIM * BLOCKDIM * lid + BLOCKDIM * r + c];
                        }
                    }
                }
                else
                {
                    for(uint32_t r = 0; r < BLOCKDIM; r++)
                    {
                        for(int c = 0; c < BLOCKDIM; c++)
                        {
                            bsr_val_C[BLOCKDIM * BLOCKDIM * (row_begin_C + offset - 1)
                                      + BLOCKDIM * r + c]
                                = data[BLOCKDIM * BLOCKDIM * lid + BLOCKDIM * c + r];
                        }
                    }
                }
            }

            // Shift the row entry to C by the number of total nnz of the current row
            row_begin_C += __popcll(mask);

            // Gather wavefront-wide minimum for the next chunks starting column index
            // Using shfl_xor here so that each thread in the wavefront obtains the final
            // result
            for(uint32_t i = WFSIZE >> 1; i > 0; i >>= 1)
            {
                min_col = rocsparse::min(min_col, __shfl_xor(min_col, i));
            }

            // Each thread sets the new chunk beginning
            chunk_begin = min_col;

            // Once the chunk beginning has reached the total number of columns nb,
            // we are done
            if(chunk_begin >= nb)
            {
                break;
            }
        }
    }

    // Compute matrix addition, where each row is processed by a wavefront.
    // Splitting row into several chunks such that we can use shared memory to store whether
    // a column index is populated or not.
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrgeam_wf_per_row_multipass_device(rocsparse_direction dir,
                                            rocsparse_int       mb,
                                            rocsparse_int       nb,
                                            rocsparse_int       block_dim,
                                            T                   alpha,
                                            const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                            const rocsparse_int* __restrict__ bsr_col_ind_A,
                                            const T* __restrict__ bsr_val_A,
                                            T beta,
                                            const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                            const rocsparse_int* __restrict__ bsr_col_ind_B,
                                            const T* __restrict__ bsr_val_B,
                                            const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                            rocsparse_int* __restrict__ bsr_col_ind_C,
                                            T* __restrict__ bsr_val_C,
                                            rocsparse_index_base idx_base_A,
                                            rocsparse_index_base idx_base_B,
                                            rocsparse_index_base idx_base_C)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int bid = hipBlockIdx_x;

        rocsparse_int lid = tid & (BLOCKDIM * BLOCKDIM - 1);
        rocsparse_int wid = tid / (BLOCKDIM * BLOCKDIM);

        rocsparse_int r = lid & (BLOCKDIM - 1);
        rocsparse_int c = lid / BLOCKDIM;

        // Each block processes a row
        rocsparse_int row = (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)) * bid + wid;

        // Row entry marker and value accumulator
        __shared__ int sdone[BLOCKSIZE / (BLOCKDIM * BLOCKDIM)];
        __shared__ int stable[BLOCKSIZE / (BLOCKDIM * BLOCKDIM)];
        __shared__ T   sdata[BLOCKSIZE];

        // Get row entry and exit point of A
        rocsparse_int row_begin_A = (row < mb) ? bsr_row_ptr_A[row] - idx_base_A : 0;
        rocsparse_int row_end_A   = (row < mb) ? bsr_row_ptr_A[row + 1] - idx_base_A : 0;

        // Get row entry and exit point of B
        rocsparse_int row_begin_B = (row < mb) ? bsr_row_ptr_B[row] - idx_base_B : 0;
        rocsparse_int row_end_B   = (row < mb) ? bsr_row_ptr_B[row + 1] - idx_base_B : 0;

        // Get row entry point of C
        rocsparse_int row_begin_C = (row < mb) ? bsr_row_ptr_C[row] - idx_base_C : 0;

        // Load the first column of the current row from A and B to set the starting
        // point for the first chunk
        rocsparse_int col_A
            = (row_begin_A < row_end_A) ? bsr_col_ind_A[row_begin_A] - idx_base_A : nb;
        rocsparse_int col_B
            = (row_begin_B < row_end_B) ? bsr_col_ind_B[row_begin_B] - idx_base_B : nb;

        // Begin of the current row chunk
        rocsparse_int chunk_begin = rocsparse::min(col_A, col_B);

        sdone[wid] = (row < mb) ? 0 : 1;

        __syncthreads();

        // Loop over the chunks until the end of both rows (A and B) has been reached (which
        // is the number of total columns nb)
        while(sdone[0] < (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
        {
            // Initialize row nnz table and value accumulator
            stable[wid] = 0;
            sdata[tid]  = static_cast<T>(0);

            __syncthreads();

            // Initialize the beginning of the next chunk
            rocsparse_int min_col = nb;

            // Loop over all columns of A, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_A < row_end_A; row_begin_A++)
            {
                // Get the column of A
                rocsparse_int col = bsr_col_ind_A[row_begin_A] - idx_base_A;

                // Get the column of A shifted by the chunk_begin
                rocsparse_int shf_A = col - chunk_begin;

                // Check if this column of A is within the chunk
                if(shf_A < 1)
                {
                    // Mark nnz
                    stable[wid] = 1;

                    // Initialize with value of A
                    if(c < block_dim && r < block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * r + c]
                                = alpha
                                  * bsr_val_A[block_dim * block_dim * row_begin_A + block_dim * r
                                              + c];
                        }
                        else
                        {
                            sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * c + r]
                                = alpha
                                  * bsr_val_A[block_dim * block_dim * row_begin_A + block_dim * c
                                              + r];
                        }
                    }
                }
                else
                {
                    // Store the first column index of A that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col);
                    break;
                }
            }

            __syncthreads();

            // Loop over all columns of B, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_B < row_end_B; row_begin_B++)
            {
                // Get the column of B
                rocsparse_int col = bsr_col_ind_B[row_begin_B] - idx_base_B;

                // Get the column of B shifted by the chunk_begin
                rocsparse_int shf_B = col - chunk_begin;

                // Check if this column of B is within the chunk
                if(shf_B < 1)
                {
                    // Mark nnz
                    stable[wid] = 1;

                    // Add values of B
                    if(c < block_dim && r < block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * r + c] = rocsparse::fma(
                                beta,
                                bsr_val_B[block_dim * block_dim * row_begin_B + block_dim * r + c],
                                sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * r + c]);
                        }
                        else
                        {
                            sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * c + r] = rocsparse::fma(
                                beta,
                                bsr_val_B[block_dim * block_dim * row_begin_B + block_dim * c + r],
                                sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * c + r]);
                        }
                    }
                }
                else
                {
                    // Store the first column index of B that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col);
                    break;
                }
            }

            __syncthreads();

            if(stable[wid])
            {
                bsr_col_ind_C[row_begin_C] = chunk_begin + idx_base_C;

                if(c < block_dim && r < block_dim)
                {
                    if(dir == rocsparse_direction_row)
                    {
                        bsr_val_C[block_dim * block_dim * row_begin_C + block_dim * r + c]
                            = sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * r + c];
                    }
                    else
                    {
                        bsr_val_C[block_dim * block_dim * row_begin_C + block_dim * c + r]
                            = sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * c + r];
                    }
                }
            }

            __syncthreads();

            row_begin_C++;

            // Each thread sets the new chunk beginning
            chunk_begin = min_col;

            __syncthreads();

            // Once the chunk beginning has reached the total number of columns nb,
            // we are done
            if(chunk_begin >= nb)
            {
                sdone[wid] = 1;
            }
            else
            {
                sdone[wid] = 0;
            }

            __syncthreads();

            rocsparse::blockreduce_sum<BLOCKSIZE / (BLOCKDIM * BLOCKDIM)>(tid, sdone);
        }
    }

    // Compute matrix addition, where each row is processed by a block.
    // Splitting row into several chunks such that we can use shared memory to store whether
    // a column index is populated or not.
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrgeam_block_per_row_multipass_device(rocsparse_direction dir,
                                               rocsparse_int       mb,
                                               rocsparse_int       nb,
                                               rocsparse_int       block_dim,
                                               T                   alpha,
                                               const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                               const rocsparse_int* __restrict__ bsr_col_ind_A,
                                               const T* __restrict__ bsr_val_A,
                                               T beta,
                                               const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                               const rocsparse_int* __restrict__ bsr_col_ind_B,
                                               const T* __restrict__ bsr_val_B,
                                               const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                               rocsparse_int* __restrict__ bsr_col_ind_C,
                                               T* __restrict__ bsr_val_C,
                                               rocsparse_index_base idx_base_A,
                                               rocsparse_index_base idx_base_B,
                                               rocsparse_index_base idx_base_C)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int bid = hipBlockIdx_x;

        rocsparse_int lid = tid & (BLOCKDIM * BLOCKDIM - 1);
        rocsparse_int wid = tid / (BLOCKDIM * BLOCKDIM);

        rocsparse_int r = lid & (BLOCKDIM - 1);
        rocsparse_int c = lid / BLOCKDIM;

        // Each block processes a row
        rocsparse_int row = bid;

        // Row entry marker and value accumulator
        __shared__ int stable[BLOCKSIZE / (BLOCKDIM * BLOCKDIM)];
        __shared__ T   sdata[BLOCKSIZE];

        // Get row entry and exit point of A
        rocsparse_int row_begin_A = bsr_row_ptr_A[row] - idx_base_A;
        rocsparse_int row_end_A   = bsr_row_ptr_A[row + 1] - idx_base_A;

        // Get row entry and exit point of B
        rocsparse_int row_begin_B = bsr_row_ptr_B[row] - idx_base_B;
        rocsparse_int row_end_B   = bsr_row_ptr_B[row + 1] - idx_base_B;

        // Get row entry point of C
        rocsparse_int row_begin_C = bsr_row_ptr_C[row] - idx_base_C;

        // Load the first column of the current row from A and B to set the starting
        // point for the first chunk
        rocsparse_int col_A
            = (row_begin_A < row_end_A) ? bsr_col_ind_A[row_begin_A] - idx_base_A : nb;
        rocsparse_int col_B
            = (row_begin_B < row_end_B) ? bsr_col_ind_B[row_begin_B] - idx_base_B : nb;

        // Begin of the current row chunk
        rocsparse_int chunk_begin = rocsparse::min(col_A, col_B);

        // Initialize the index for column access into A and B
        row_begin_A += wid;
        row_begin_B += wid;

        // Loop over the chunks until the end of both rows (A and B) has been reached (which
        // is the number of total columns nb)
        while(true)
        {
            // Initialize row nnz table and value accumulator
            stable[wid] = 0;
            sdata[tid]  = static_cast<T>(0);

            __syncthreads();

            // Initialize the beginning of the next chunk
            rocsparse_int min_col = nb;

            // Loop over all columns of A, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_A < row_end_A; row_begin_A += (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
            {
                // Get the column of A
                rocsparse_int col = bsr_col_ind_A[row_begin_A] - idx_base_A;

                // Get the column of A shifted by the chunk_begin
                rocsparse_int shf_A = col - chunk_begin;

                // Check if this column of A is within the chunk
                if(shf_A < (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
                {
                    // Mark nnz
                    stable[shf_A] = 1;

                    // Initialize with value of A
                    if(c < block_dim && r < block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            sdata[BLOCKDIM * BLOCKDIM * shf_A + BLOCKDIM * r + c]
                                = alpha
                                  * bsr_val_A[block_dim * block_dim * row_begin_A + block_dim * r
                                              + c];
                        }
                        else
                        {
                            sdata[BLOCKDIM * BLOCKDIM * shf_A + BLOCKDIM * c + r]
                                = alpha
                                  * bsr_val_A[block_dim * block_dim * row_begin_A + block_dim * c
                                              + r];
                        }
                    }
                }
                else
                {
                    // Store the first column index of A that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col);
                    break;
                }
            }

            __syncthreads();

            // Loop over all columns of B, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_B < row_end_B; row_begin_B += (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
            {
                // Get the column of B
                rocsparse_int col = bsr_col_ind_B[row_begin_B] - idx_base_B;

                // Get the column of B shifted by the chunk_begin
                rocsparse_int shf_B = col - chunk_begin;

                // Check if this column of B is within the chunk
                if(shf_B < (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
                {
                    // Mark nnz
                    stable[shf_B] = 1;

                    // Add values of B
                    if(c < block_dim && r < block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            sdata[BLOCKDIM * BLOCKDIM * shf_B + BLOCKDIM * r + c] = rocsparse::fma(
                                beta,
                                bsr_val_B[block_dim * block_dim * row_begin_B + block_dim * r + c],
                                sdata[BLOCKDIM * BLOCKDIM * shf_B + BLOCKDIM * r + c]);
                        }
                        else
                        {
                            sdata[BLOCKDIM * BLOCKDIM * shf_B + BLOCKDIM * c + r] = rocsparse::fma(
                                beta,
                                bsr_val_B[block_dim * block_dim * row_begin_B + block_dim * c + r],
                                sdata[BLOCKDIM * BLOCKDIM * shf_B + BLOCKDIM * c + r]);
                        }
                    }
                }
                else
                {
                    // Store the first column index of B that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col);
                    break;
                }
            }

            __syncthreads();

            if((BLOCKSIZE / (BLOCKDIM * BLOCKDIM)) == 1)
            {
                if(stable[wid])
                {
                    rocsparse_int idx = row_begin_C + stable[wid] - 1;

                    bsr_col_ind_C[idx] = wid + chunk_begin + idx_base_C;

                    if(c < block_dim && r < block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            bsr_val_C[block_dim * block_dim * idx + block_dim * r + c]
                                = sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * r + c];
                        }
                        else
                        {
                            bsr_val_C[block_dim * block_dim * idx + block_dim * c + r]
                                = sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * c + r];
                        }
                    }
                }

                __syncthreads();

                row_begin_C += stable[wid];

                // Each thread sets the new chunk beginning
                chunk_begin = min_col;

                __syncthreads();

                // Once the chunk beginning has reached the total number of columns nb,
                // we are done
                if(chunk_begin >= nb)
                {
                    break;
                }
            }
            else
            {
                int temp = stable[wid];
                __syncthreads();

                // Segmented wavefront reduction
                for(uint32_t j = 1; j < (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)); j <<= 1)
                {
                    if(wid >= j)
                    {
                        temp = temp + stable[wid - j];
                    }
                    __syncthreads();
                    stable[wid] = temp;
                    __syncthreads();
                }

                int total_offset = stable[(BLOCKSIZE / (BLOCKDIM * BLOCKDIM)) - 1];
                int prev         = (wid >= 1) ? stable[wid - 1] : 0;

                __syncthreads();

                if(wid >= 1)
                {
                    if(temp == prev)
                    {
                        stable[wid] = 0;
                    }
                }

                __syncthreads();

                if(stable[wid])
                {
                    rocsparse_int idx = row_begin_C + stable[wid] - 1;

                    bsr_col_ind_C[idx] = wid + chunk_begin + idx_base_C;

                    if(c < block_dim && r < block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            bsr_val_C[block_dim * block_dim * idx + block_dim * r + c]
                                = sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * r + c];
                        }
                        else
                        {
                            bsr_val_C[block_dim * block_dim * idx + block_dim * c + r]
                                = sdata[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * c + r];
                        }
                    }
                }

                __syncthreads();

                row_begin_C += total_offset;

                // Compute block-wide minimum for the next chunks starting column index
                stable[wid] = min_col;

                __syncthreads();

                rocsparse::blockreduce_min<(BLOCKSIZE / (BLOCKDIM * BLOCKDIM))>(tid, stable);

                min_col = stable[0];

                __syncthreads();

                // Each thread sets the new chunk beginning
                chunk_begin = min_col;

                // Once the chunk beginning has reached the total number of columns nb,
                // we are done
                if(chunk_begin >= nb)
                {
                    break;
                }
            }
        }
    }

    // Compute matrix addition, where each row is processed by a block.
    // Splitting row into several chunks such that we can use shared memory to store whether
    // a column index is populated or not.
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrgeam_block_per_row_multipass_device2(rocsparse_direction dir,
                                                rocsparse_int       mb,
                                                rocsparse_int       nb,
                                                rocsparse_int       block_dim,
                                                T                   alpha,
                                                const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                                const rocsparse_int* __restrict__ bsr_col_ind_A,
                                                const T* __restrict__ bsr_val_A,
                                                T beta,
                                                const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                                const rocsparse_int* __restrict__ bsr_col_ind_B,
                                                const T* __restrict__ bsr_val_B,
                                                const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                                rocsparse_int* __restrict__ bsr_col_ind_C,
                                                T* __restrict__ bsr_val_C,
                                                rocsparse_index_base idx_base_A,
                                                rocsparse_index_base idx_base_B,
                                                rocsparse_index_base idx_base_C)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int bid = hipBlockIdx_x;

        rocsparse_int lid = tid & (BLOCKSIZE / BLOCKDIM - 1);
        rocsparse_int wid = tid / (BLOCKSIZE / BLOCKDIM);

        // Each block processes a row
        rocsparse_int row = bid;

        // Row entry marker and value accumulator
        __shared__ int table;
        __shared__ T   data[BLOCKDIM * BLOCKDIM];

        // Get row entry and exit point of A
        rocsparse_int row_begin_A = bsr_row_ptr_A[row] - idx_base_A;
        rocsparse_int row_end_A   = bsr_row_ptr_A[row + 1] - idx_base_A;

        // Get row entry and exit point of B
        rocsparse_int row_begin_B = bsr_row_ptr_B[row] - idx_base_B;
        rocsparse_int row_end_B   = bsr_row_ptr_B[row + 1] - idx_base_B;

        // Get row entry point of C
        rocsparse_int row_begin_C = bsr_row_ptr_C[row] - idx_base_C;

        // Load the first column of the current row from A and B to set the starting
        // point for the first chunk
        rocsparse_int col_A
            = (row_begin_A < row_end_A) ? bsr_col_ind_A[row_begin_A] - idx_base_A : nb;
        rocsparse_int col_B
            = (row_begin_B < row_end_B) ? bsr_col_ind_B[row_begin_B] - idx_base_B : nb;

        // Begin of the current row chunk
        rocsparse_int chunk_begin = rocsparse::min(col_A, col_B);

        // Loop over the chunks until the end of both rows (A and B) has been reached (which
        // is the number of total columns nb)
        while(true)
        {
            // Initialize row nnz table and value accumulator
            table = 0;
            for(uint32_t i = 0; i < BLOCKDIM; i += BLOCKSIZE / BLOCKDIM)
            {
                data[BLOCKDIM * wid + i + lid] = static_cast<T>(0);
            }

            __syncthreads();

            // Initialize the beginning of the next chunk
            rocsparse_int min_col = nb;

            // Loop over all columns of A, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_A < row_end_A; row_begin_A++)
            {
                // Get the column of A
                rocsparse_int col = bsr_col_ind_A[row_begin_A] - idx_base_A;

                // Check if this column of A is within the chunk
                if(col == chunk_begin)
                {
                    table = 1;

                    // Initialize with value of A
                    for(uint32_t i = 0; i < BLOCKDIM; i += BLOCKSIZE / BLOCKDIM)
                    {
                        if((i + lid) < block_dim && wid < block_dim)
                        {
                            if(dir == rocsparse_direction_row)
                            {
                                data[BLOCKDIM * wid + (i + lid)]
                                    = alpha
                                      * bsr_val_A[block_dim * block_dim * row_begin_A
                                                  + block_dim * wid + (i + lid)];
                            }
                            else
                            {
                                data[BLOCKDIM * (i + lid) + wid]
                                    = alpha
                                      * bsr_val_A[block_dim * block_dim * row_begin_A
                                                  + block_dim * (i + lid) + wid];
                            }
                        }
                    }
                }
                else
                {
                    // Store the first column index of A that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col);
                    break;
                }
            }

            __syncthreads();

            // Loop over all columns of B, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_B < row_end_B; row_begin_B++)
            {
                // Get the column of B
                rocsparse_int col = bsr_col_ind_B[row_begin_B] - idx_base_B;

                // Check if this column of B is within the chunk
                if(col == chunk_begin)
                {
                    table = 1;

                    // Add values of B
                    for(uint32_t i = 0; i < BLOCKDIM; i += BLOCKSIZE / BLOCKDIM)
                    {
                        if((i + lid) < block_dim && wid < block_dim)
                        {
                            if(dir == rocsparse_direction_row)
                            {
                                data[BLOCKDIM * wid + (i + lid)]
                                    = rocsparse::fma(beta,
                                                     bsr_val_B[block_dim * block_dim * row_begin_B
                                                               + block_dim * wid + (i + lid)],
                                                     data[BLOCKDIM * wid + (i + lid)]);
                            }
                            else
                            {
                                data[BLOCKDIM * (i + lid) + wid]
                                    = rocsparse::fma(beta,
                                                     bsr_val_B[block_dim * block_dim * row_begin_B
                                                               + block_dim * (i + lid) + wid],
                                                     data[BLOCKDIM * (i + lid) + wid]);
                            }
                        }
                    }
                }
                else
                {
                    // Store the first column index of B that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col);
                    break;
                }
            }

            __syncthreads();

            if(table)
            {
                bsr_col_ind_C[row_begin_C] = chunk_begin + idx_base_C;

                for(uint32_t i = 0; i < BLOCKDIM; i += BLOCKSIZE / BLOCKDIM)
                {
                    if((i + lid) < block_dim && wid < block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            bsr_val_C[block_dim * block_dim * row_begin_C + block_dim * wid
                                      + (i + lid)]
                                = data[BLOCKDIM * wid + (i + lid)];
                        }
                        else
                        {
                            bsr_val_C[block_dim * block_dim * row_begin_C + block_dim * (i + lid)
                                      + wid]
                                = data[BLOCKDIM * (i + lid) + wid];
                        }
                    }
                }
            }

            __syncthreads();

            row_begin_C += table;

            // Each thread sets the new chunk beginning
            chunk_begin = min_col;

            __syncthreads();

            // Once the chunk beginning has reached the total number of columns nb,
            // we are done
            if(chunk_begin >= nb)
            {
                break;
            }
        }
    }
}
