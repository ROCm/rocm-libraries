/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void
        csrgeam_fill_symbolic_multipass_device(int64_t m,
                                               int64_t n,
                                               const I* __restrict__ csr_row_ptr_A,
                                               const J* __restrict__ csr_col_ind_A,
                                               const I* __restrict__ csr_row_ptr_B,
                                               const J* __restrict__ csr_col_ind_B,
                                               const I* __restrict__ csr_row_ptr_C,
                                               J* __restrict__ csr_col_ind_C,
                                               rocsparse_index_base idx_base_A,
                                               rocsparse_index_base idx_base_B,
                                               rocsparse_index_base idx_base_C)
    {
        // Lane id
        const uint32_t lid = hipThreadIdx_x & (WFSIZE - 1);

        // Wavefront id
        const uint32_t wid = hipThreadIdx_x / WFSIZE;

        // Each wavefront processes a row
        const int64_t row = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= m)
        {
            return;
        }

        // Row entry marker and value accumulator
        __shared__ bool stable[BLOCKSIZE];

        bool* table = &stable[wid * WFSIZE];

        // Get row entry and exit point of A
        I row_begin_A = csr_row_ptr_A[row] - idx_base_A;
        I row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

        // Get row entry and exit point of B
        I row_begin_B = csr_row_ptr_B[row] - idx_base_B;
        I row_end_B   = csr_row_ptr_B[row + 1] - idx_base_B;

        // Get row entry point of C
        I row_begin_C = csr_row_ptr_C[row] - idx_base_C;

        // Load the first column of the current row from A and B to set the starting
        // point for the first chunk
        const J col_A = (row_begin_A < row_end_A) ? csr_col_ind_A[row_begin_A] - idx_base_A : n;
        const J col_B = (row_begin_B < row_end_B) ? csr_col_ind_B[row_begin_B] - idx_base_B : n;

        // Begin of the current row chunk
        J chunk_begin = rocsparse::min(col_A, col_B);

        // Initialize the index for column access into A and B
        row_begin_A += lid;
        row_begin_B += lid;

        // Loop over the chunks until the end of both rows (A and B) has been reached (which
        // is the number of total columns n)
        while(true)
        {
            // Initialize row nnz table and value accumulator
            table[lid] = false;

            __threadfence_block();

            // Initialize the beginning of the next chunk
            J min_col = n;

            // Loop over all columns of A, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_A < row_end_A; row_begin_A += WFSIZE)
            {
                // Get the column of A
                const J col = csr_col_ind_A[row_begin_A] - idx_base_A;

                // Get the column of A shifted by the chunk_begin
                const J shf_A = col - chunk_begin;

                // Check if this column of A is within the chunk
                if(shf_A < WFSIZE)
                {
                    // Mark nnz
                    table[shf_A] = true;
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
                const J col = csr_col_ind_B[row_begin_B] - idx_base_B;

                // Get the column of B shifted by the chunk_begin
                const J shf_B = col - chunk_begin;

                // Check if this column of B is within the chunk
                if(shf_B < WFSIZE)
                {
                    // Mark nnz
                    table[shf_B] = true;
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
            const bool has_nnz = table[lid];

            // Obtain the bitmask that marks the position of each non-zero entry
            uint64_t mask = __ballot(has_nnz);

            // If the lane has an nnz assign, it must be filled into C
            if(has_nnz)
            {
                // Compute the lane's fill position in C
                uint32_t offset = rocsparse::popc<WFSIZE>(mask, lid);

                // Fill C
                csr_col_ind_C[row_begin_C + offset - 1] = lid + chunk_begin + idx_base_C;
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

            // Once the chunk beginning has reached the total number of columns n,
            // we are done
            if(chunk_begin >= n)
            {
                break;
            }
        }
    }
}
