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
    // Hash operation to insert key into hash table
    // Returns true if key has been added
    template <uint32_t HASHVAL, uint32_t HASHSIZE, typename I>
    ROCSPARSE_DEVICE_ILF bool insert_key(I key, I* __restrict__ table)
    {
        constexpr I empty = -1;

        // Compute hash
        I hash = (key * HASHVAL) & (HASHSIZE - 1);

        // Loop until key has been inserted
        while(true)
        {
            // Load table[hash] exactly once in case it gets set by another thread
            const I temp = table[hash];

            if(temp == key)
            {
                // Element already present
                return false;
            }
            else if(temp == empty)
            {
                // If empty, add element with atomic
                if(rocsparse::atomic_cas(&table[hash], empty, key) == empty)
                {
                    // Increment number of insertions
                    return true;
                }
            }
            else
            {
                // Linear probing, when hash is collided, try next entry
                hash = (hash + 1) & (HASHSIZE - 1);
            }
        }

        return false;
    }

    // Hash operation to insert key into hash table
    // Returns true if key has been added
    template <uint32_t HASHVAL, uint32_t HASHSIZE, typename I, typename J>
    ROCSPARSE_DEVICE_ILF bool
        insert_key(J key, I* __restrict__ table, I* __restrict__ local_idxs, I local_idx)
    {
        constexpr I empty = -1;

        // Compute hash
        I hash = (key * HASHVAL) & (HASHSIZE - 1);
        // Loop until key has been inserted
        while(true)
        {
            // Load table[hash] exactly once in case it gets set by another thread
            const I temp = table[hash];

            if(temp == key)
            {
                // Element already present
                return false;
            }
            else if(temp == empty)
            {
                rocsparse::atomic_cas(&table[hash], empty, key);
                rocsparse::atomic_cas(&local_idxs[hash], empty, local_idx);
                return true;
            }
            else
            {
                // Linear probing, when hash is collided, try next entry
                hash = (hash + 1) & (HASHSIZE - 1);
            }
        }
        return false;
    }
    // Hash operation to insert pair into hash table
    template <uint32_t HASHVAL, uint32_t HASHSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void
        insert_pair(I key, T val, I* __restrict__ table, T* __restrict__ data, I empty)
    {
        // Compute hash
        I hash = (key * HASHVAL) & (HASHSIZE - 1);

        // Loop until pair has been inserted
        while(true)
        {
            // Load table[hash] exactly once in case it gets set by another thread
            const I temp = table[hash];

            if(temp == key)
            {
                // Element already present, add value to exsiting entry
                rocsparse::atomic_add(&data[hash], val);
                break;
            }
            else if(temp == empty)
            {
                // If empty, add element with atomic
                if(rocsparse::atomic_cas(&table[hash], empty, key) == empty)
                {
                    // Add value
                    rocsparse::atomic_add(&data[hash], val);
                    break;
                }
            }
            else
            {
                // Linear probing, when hash is collided, try next entry
                hash = (hash + 1) & (HASHSIZE - 1);
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void
        csrgemm_numeric_fill_wf_per_row_device(J m,
                                               J nk,
                                               const J* __restrict__ offset,
                                               const J* __restrict__ perm,
                                               T alpha,
                                               const I* __restrict__ csr_row_ptr_A,
                                               const J* __restrict__ csr_col_ind_A,
                                               const T* __restrict__ csr_val_A,
                                               const I* __restrict__ csr_row_ptr_B,
                                               const J* __restrict__ csr_col_ind_B,
                                               const T* __restrict__ csr_val_B,
                                               T beta,
                                               const I* __restrict__ csr_row_ptr_D,
                                               const J* __restrict__ csr_col_ind_D,
                                               const T* __restrict__ csr_val_D,
                                               const I* __restrict__ csr_row_ptr_C,
                                               const J* __restrict__ csr_col_ind_C,
                                               T* __restrict__ csr_val_C,
                                               rocsparse_index_base idx_base_A,
                                               rocsparse_index_base idx_base_B,
                                               rocsparse_index_base idx_base_C,
                                               rocsparse_index_base idx_base_D,
                                               bool                 mul,
                                               bool                 add)
    {

        // Lane id
        int lid = hipThreadIdx_x & (WFSIZE - 1);
        // Wavefront id
        int wid = hipThreadIdx_x / WFSIZE;

        // Each (sub)wavefront processes a row
        J row = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Hash table in shared memory
        __shared__ J stable[BLOCKSIZE / WFSIZE * HASHSIZE];
        __shared__ T sdata[BLOCKSIZE / WFSIZE * HASHSIZE];

        // Local hash table
        J* table = &stable[wid * HASHSIZE];
        T* data  = &sdata[wid * HASHSIZE];

        // Initialize hash table
        for(uint32_t i = lid; i < HASHSIZE; i += WFSIZE)
        {
            table[i] = nk;
            data[i]  = static_cast<T>(0);
        }

        __threadfence_block();

        // Bounds check
        if(row >= m)
        {
            return;
        }

        // Apply permutation, if available
        row = perm ? perm[row + *offset] : row;

        // alpha * A * B part
        if(mul)
        {
            // Get row boundaries of the current row in A
            I row_begin_A = csr_row_ptr_A[row] - idx_base_A;
            I row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

            // Loop over columns of A in current row
            for(I j = row_begin_A + lid; j < row_end_A; j += WFSIZE)
            {
                // Column of A in current row
                J col_A = csr_col_ind_A[j] - idx_base_A;
                // Value of A in current row
                T val_A = alpha * csr_val_A[j];

                // Loop over columns of B in row col_A
                I row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                I row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                // Insert all columns of B into hash table
                for(I k = row_begin_B; k < row_end_B; ++k)
                {
                    // Insert key value pair into hash table
                    insert_pair<HASHVAL, HASHSIZE>(
                        csr_col_ind_B[k] - idx_base_B, val_A * csr_val_B[k], table, data, nk);
                }
            }
        }

        // beta * D part
        if(add)
        {
            // Get row boundaries of the current row in D
            I row_begin_D = csr_row_ptr_D[row] - idx_base_D;
            I row_end_D   = csr_row_ptr_D[row + 1] - idx_base_D;

            // Loop over columns of D in current row and insert all columns of D into hash table
            for(I j = row_begin_D + lid; j < row_end_D; j += WFSIZE)
            {
                // Insert key value pair into hash table
                insert_pair<HASHVAL, HASHSIZE>(
                    csr_col_ind_D[j] - idx_base_D, beta * csr_val_D[j], table, data, nk);
            }
        }

        __threadfence_block();

        // Entry point of current row into C
        I row_begin_C = csr_row_ptr_C[row] - idx_base_C;

        // Loop over hash table
        for(uint32_t i = lid; i < HASHSIZE; i += WFSIZE)
        {
            // Get column from hash table to fill it into C
            J col_C = table[i];

            // Skip hash table entry if not present
            if(col_C >= nk)
            {
                continue;
            }

            // Initialize index into C
            I idx_C = row_begin_C;

            // Initialize index into hash table
            uint32_t hash_idx = 0;

            // Loop through hash table to find the (sorted) index into C for the
            // current column index
            // Checking the whole hash table is actually faster for these hash
            // table sizes, compared to hash table compression
            while(hash_idx < HASHSIZE)
            {
                // Increment index into C if column entry is greater than table entry
                if(col_C > table[hash_idx])
                {
                    ++idx_C;
                }

                // Goto next hash table index
                ++hash_idx;
            }

            // Accumulate value to the obtained position in C
            // csr_col_ind_C[idx_C] = col_C + idx_base_C;
            csr_val_C[idx_C] = data[i];
        }
    }

    // Compute column entries and accumulate values, where each row is processed by a single block
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              uint32_t WARPSIZE,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void
        csrgemm_numeric_fill_block_per_row_device(J nk,
                                                  const J* __restrict__ offset_,
                                                  const J* __restrict__ perm,
                                                  T alpha,
                                                  const I* __restrict__ csr_row_ptr_A,
                                                  const J* __restrict__ csr_col_ind_A,
                                                  const T* __restrict__ csr_val_A,
                                                  const I* __restrict__ csr_row_ptr_B,
                                                  const J* __restrict__ csr_col_ind_B,
                                                  const T* __restrict__ csr_val_B,
                                                  T beta,
                                                  const I* __restrict__ csr_row_ptr_D,
                                                  const J* __restrict__ csr_col_ind_D,
                                                  const T* __restrict__ csr_val_D,
                                                  const I* __restrict__ csr_row_ptr_C,
                                                  const J* __restrict__ csr_col_ind_C,
                                                  T* __restrict__ csr_val_C,
                                                  rocsparse_index_base idx_base_A,
                                                  rocsparse_index_base idx_base_B,
                                                  rocsparse_index_base idx_base_C,
                                                  rocsparse_index_base idx_base_D,
                                                  bool                 mul,
                                                  bool                 add)
    {
        // Lane id
        int lid = hipThreadIdx_x & (WFSIZE - 1);
        // Wavefront id
        int wid = hipThreadIdx_x / WFSIZE;

        // Hash table in shared memory
        extern __shared__ char shared_memory[];
        J*                     table = (J*)shared_memory;
        T*                     data  = (T*)(shared_memory + sizeof(J) * HASHSIZE);

        // Initialize hash table
        for(uint32_t i = hipThreadIdx_x; i < HASHSIZE; i += BLOCKSIZE)
        {
            table[i] = nk;
            data[i]  = static_cast<T>(0);
        }

        // Wait for all threads to finish initialization
        __syncthreads();

        // Each block processes a row (apply permutation)
        J row = perm[hipBlockIdx_x + *offset_];

        // alpha * A * B part
        if(mul)
        {
            // Get row boundaries of the current row in A
            I row_begin_A = csr_row_ptr_A[row] - idx_base_A;
            I row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

            // Loop over columns of A in current row
            for(I j = row_begin_A + wid; j < row_end_A; j += BLOCKSIZE / WFSIZE)
            {
                // Column of A in current row
                J col_A = csr_col_ind_A[j] - idx_base_A;
                // Value of A in current row
                T val_A = alpha * csr_val_A[j];

                // Loop over columns of B in row col_A
                I row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                I row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                for(I k = row_begin_B + lid; k < row_end_B; k += WFSIZE)
                {
                    // Insert key value pair into hash table
                    insert_pair<HASHVAL, HASHSIZE>(
                        csr_col_ind_B[k] - idx_base_B, val_A * csr_val_B[k], table, data, nk);
                }
            }
        }

        // beta * D part
        if(add)
        {
            // Get row boundaries of the current row in D
            I row_begin_D = csr_row_ptr_D[row] - idx_base_D;
            I row_end_D   = csr_row_ptr_D[row + 1] - idx_base_D;

            // Loop over columns of D in current row and insert all columns of D into hash table
            for(I j = row_begin_D + hipThreadIdx_x; j < row_end_D; j += BLOCKSIZE)
            {
                // Insert key value pair into hash table
                insert_pair<HASHVAL, HASHSIZE>(
                    csr_col_ind_D[j] - idx_base_D, beta * csr_val_D[j], table, data, nk);
            }
        }

        // Wait for hash operations to finish
        __syncthreads();

        // Compress hash table, such that valid entries come first
        J* scan_offsets = (J*)(shared_memory + (sizeof(J) + sizeof(T)) * HASHSIZE);

        // Offset into hash table
        J hash_offset = 0;

        // Loop over the hash table and do the compression
        for(uint32_t i = hipThreadIdx_x; i < HASHSIZE; i += BLOCKSIZE)
        {
            // Get column and value from hash table
            J col_C = table[i];
            T val_C = data[i];

            // Boolean to store if thread owns a non-zero element
            bool has_nnz = col_C < nk;

            // Each thread obtains a bit mask of all wavefront-wide non-zero entries
            // to compute its wavefront-wide non-zero offset
            uint64_t mask = __ballot(has_nnz);

            // The number of bits set to 1 is the amount of wavefront-wide non-zeros
            int nnz = __popcll(mask);

            // Obtain the lane mask, where all bits lesser equal the lane id are set to 1
            // e.g. for lane id 7, lanemask_le = 0b11111111
            // HIP implements only __lanemask_lt() unfortunately ...
            uint64_t lanemask_le = UINT64_MAX >> (sizeof(uint64_t) * CHAR_BIT - (__lane_id() + 1));

            // Compute the intra wavefront offset of the lane id by bitwise AND with the lane mask
            int offset = __popcll(lanemask_le & mask);

            // Need to sync here to make sure reading from data array has finished
            __syncthreads();

            // Each wavefront writes its offset / nnz into shared memory so we can compute the
            // scan offset
            scan_offsets[hipThreadIdx_x / WARPSIZE] = nnz;

            // Wait for all wavefronts to finish writing
            __syncthreads();

            // Each thread accumulates the offset of all previous wavefronts to obtain its offset
            for(uint32_t j = 1; j < BLOCKSIZE / WARPSIZE; ++j)
            {
                if(hipThreadIdx_x >= j * WARPSIZE)
                {
                    offset += scan_offsets[j - 1];
                }
            }

            // Offset depends on all previously added non-zeros and need to be shifted by
            // 1 (zero-based indexing)
            J idx = hash_offset + offset - 1;

            // Only threads with a non-zero value write their values
            if(has_nnz)
            {
                table[idx] = col_C;
                data[idx]  = val_C;
            }

            // Last thread in block writes the block-wide offset such that all subsequent
            // entries are shifted by this offset
            if(hipThreadIdx_x == BLOCKSIZE - 1)
            {
                scan_offsets[BLOCKSIZE / WARPSIZE - 1] = offset;
            }

            // Wait for last thread in block to finish writing
            __syncthreads();

            // Each thread reads the block-wide offset and adds it to its local offset
            hash_offset += scan_offsets[BLOCKSIZE / WARPSIZE - 1];
        }

        // Entry point into row of C
        I row_begin_C = csr_row_ptr_C[row] - idx_base_C;
        I row_end_C   = csr_row_ptr_C[row + 1] - idx_base_C;
        J row_nnz     = row_end_C - row_begin_C;

        // Loop over all valid entries in hash table
        for(J i = hipThreadIdx_x; i < row_nnz; i += BLOCKSIZE)
        {
            J col_C = table[i];
            T val_C = data[i];

            // Index into C
            I idx_C = row_begin_C;

            // Loop through hash table to find the (sorted) index into C for the
            // current column index
            for(J j = 0; j < row_nnz; ++j)
            {
                // Increment index into C if column entry is greater than table entry
                if(col_C > table[j])
                {
                    ++idx_C;
                }
            }

            // Write column and accumulated value to the obtain position in C
            //        csr_col_ind_C[idx_C] = col_C + idx_base_C;
            csr_val_C[idx_C] = val_C;
        }
    }

    // Compute column entries and accumulate values, where each row is processed by a single
    // block. Splitting row into several chunks such that we can use shared memory to store
    // whether a column index is populated or not. Each row has at least 4097 non-zero
    // entries to compute.
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t CHUNKSIZE,
              uint32_t WARPSIZE,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void
        csrgemm_numeric_fill_block_per_row_multipass_device(J n,
                                                            const J* __restrict__ offset_,
                                                            const J* __restrict__ perm,
                                                            T alpha,
                                                            const I* __restrict__ csr_row_ptr_A,
                                                            const J* __restrict__ csr_col_ind_A,
                                                            const T* __restrict__ csr_val_A,
                                                            const I* __restrict__ csr_row_ptr_B,
                                                            const J* __restrict__ csr_col_ind_B,
                                                            const T* __restrict__ csr_val_B,
                                                            T beta,
                                                            const I* __restrict__ csr_row_ptr_D,
                                                            const J* __restrict__ csr_col_ind_D,
                                                            const T* __restrict__ csr_val_D,
                                                            const I* __restrict__ csr_row_ptr_C,
                                                            const J* __restrict__ csr_col_ind_C,
                                                            T* __restrict__ csr_val_C,
                                                            I* __restrict__ workspace_B,
                                                            rocsparse_index_base idx_base_A,
                                                            rocsparse_index_base idx_base_B,
                                                            rocsparse_index_base idx_base_C,
                                                            rocsparse_index_base idx_base_D,
                                                            bool                 mul,
                                                            bool                 add)
    {
        // Lane id
        int lid = hipThreadIdx_x & (WFSIZE - 1);
        // Wavefront id
        int wid = hipThreadIdx_x / WFSIZE;

        // Each block processes a row (apply permutation)
        J row = perm[hipBlockIdx_x + *offset_];

        // Row entry marker and value accumulator
        __shared__ bool table[CHUNKSIZE];
        __shared__ T    data[CHUNKSIZE];

        // Shared memory to determine the minimum of all column indices of B that exceed the
        // current chunk
        __shared__ J next_chunk;

        // Begin of the current row chunk (this is the column index of the current row)
        J chunk_begin = 0;
        J chunk_end   = CHUNKSIZE;

        // Get row boundaries of the current row in A
        I row_begin_A = (mul) ? csr_row_ptr_A[row] - idx_base_A : 0;
        I row_end_A   = (mul) ? csr_row_ptr_A[row + 1] - idx_base_A : 0;

        // Entry point into columns of C
        I row_begin_C = csr_row_ptr_C[row] - idx_base_C;

        // Loop over the row chunks until the end of the row has been reached (which is
        // the number of total columns)
        while(chunk_begin < n)
        {
            // Initialize row nnz table and accumulator
            for(uint32_t i = hipThreadIdx_x; i < CHUNKSIZE; i += BLOCKSIZE)
            {
                table[i] = 0;
                data[i]  = static_cast<T>(0);
            }

            // Initialize next chunk column index
            if(hipThreadIdx_x == 0)
            {
                next_chunk = n;
            }

            // Wait for all threads to finish initialization
            __syncthreads();

            // Initialize the beginning of the next chunk
            J min_col = n;

            // alpha * A * B part
            if(mul)
            {
                // Loop over columns of A in current row
                for(I j = row_begin_A + wid; j < row_end_A; j += BLOCKSIZE / WFSIZE)
                {
                    // Column of A in current row
                    J col_A = csr_col_ind_A[j] - idx_base_A;

                    // Value of A in current row
                    T val_A = alpha * csr_val_A[j];

                    // Loop over columns of B in row col_A
                    I row_begin_B
                        = (chunk_begin == 0) ? csr_row_ptr_B[col_A] - idx_base_B : workspace_B[j];
                    I row_end_B = csr_row_ptr_B[col_A + 1] - idx_base_B;

                    // Keep track of the first k where the column index of B is exceeding
                    // the current chunks end point
                    I next_k = row_begin_B + lid;

                    // Loop over columns of B in row col_A
                    for(I k = next_k; k < row_end_B; k += WFSIZE)
                    {
                        // Column of B in row col_A
                        J col_B = csr_col_ind_B[k] - idx_base_B;

                        if(col_B >= chunk_begin && col_B < chunk_end)
                        {
                            // Mark nnz table if entry at col_B
                            table[col_B - chunk_begin] = 1;

                            // Atomically accumulate the intermediate products
                            rocsparse::atomic_add(&data[col_B - chunk_begin], val_A * csr_val_B[k]);
                        }
                        else if(col_B >= chunk_end)
                        {
                            // If column index exceeds chunks end point, store k as starting
                            // point of the columns of B for the next pass
                            next_k = k;

                            // Store the first column index of B that exceeds the current chunk
                            min_col = rocsparse::min(min_col, col_B);
                            break;
                        }
                    }

                    // Obtain the minimum of all k that exceed the current chunks end point
                    rocsparse::wfreduce_min<WFSIZE>(&next_k);

                    // Store the minimum globally for the next chunk
                    if(lid == WFSIZE - 1)
                    {
                        workspace_B[j] = next_k;
                    }
                }
            }

            // beta * D part
            if(add)
            {
                // Get row boundaries of the current row in D
                I row_begin_D = csr_row_ptr_D[row] - idx_base_D;
                I row_end_D   = csr_row_ptr_D[row + 1] - idx_base_D;

                // Loop over columns of D in current row
                for(I j = row_begin_D + hipThreadIdx_x; j < row_end_D; j += BLOCKSIZE)
                {
                    // Column of D in row col_A
                    J col_D = csr_col_ind_D[j] - idx_base_D;

                    if(col_D >= chunk_begin && col_D < chunk_end)
                    {
                        // Mark nnz table if entry at col_D
                        table[col_D - chunk_begin] = 1;

                        // Atomically accumulate the entry of D
                        rocsparse::atomic_add(&data[col_D - chunk_begin], beta * csr_val_D[j]);
                    }
                    else if(col_D >= chunk_end)
                    {
                        // Store the first column index of D that exceeds the current chunk
                        min_col = rocsparse::min(min_col, col_D);
                        break;
                    }

                    // Performance can potentially improved by adding another temporary
                    // workspace of dimension sizeof(J) * nnz, which is significant!
                }
            }

            // Gather wavefront-wide minimum for the next chunks starting column index
            rocsparse::wfreduce_min<WFSIZE>(&min_col);

            // Last thread in each wavefront finds block-wide minimum atomically
            if(lid == WFSIZE - 1)
            {
                // Atomically determine the new chunks beginning (minimum column index of B
                // that is larger than the current chunks end point)
                rocsparse::atomic_min(&next_chunk, min_col);
            }

            // Wait for all threads to finish
            __syncthreads();

            // We can re-use the shared memory to communicate the scan offsets of each
            // wavefront
            int* scan_offsets = reinterpret_cast<int*>(data);

            // "Pseudo compress" the table array such that we can copy the values over into C
            // In fact, we do an exclusive scan to obtain the index where each non-zero has
            // to be copied to
            for(uint32_t i = hipThreadIdx_x; i < CHUNKSIZE; i += BLOCKSIZE)
            {
                // Each thread loads its marker and value to know whether it has to process a
                // non-zero entry or not
                bool has_nnz = table[i];
                T    value   = data[i];

                // Each thread obtains a bit mask of all wavefront-wide non-zero entries
                // to compute its wavefront-wide non-zero offset in C
                uint64_t mask = __ballot(has_nnz);

                // The number of bits set to 1 is the amount of wavefront-wide non-zeros
                int nnz = __popcll(mask);

                // Obtain the lane mask, where all bits lesser equal the lane id are set to 1
                // e.g. for lane id 7, lanemask_le = 0b11111111
                // HIP implements only __lanemask_lt() unfortunately ...
                uint64_t lanemask_le
                    = UINT64_MAX >> (sizeof(uint64_t) * CHAR_BIT - (__lane_id() + 1));

                // Compute the intra wavefront offset of the lane id by bitwise AND with the lane mask
                int offset = __popcll(lanemask_le & mask);

                // Need to sync here to make sure reading from data array has finished
                __syncthreads();

                // Each wavefront writes its offset / nnz into shared memory so we can compute the
                // scan offset
                scan_offsets[hipThreadIdx_x / WARPSIZE] = nnz;

                // Wait for all wavefronts to finish writing
                __syncthreads();

                // Each thread accumulates the offset of all previous wavefronts to obtain its
                // offset into C
                for(uint32_t j = 1; j < BLOCKSIZE / WARPSIZE; ++j)
                {
                    if(hipThreadIdx_x >= j * WARPSIZE)
                    {
                        offset += scan_offsets[j - 1];
                    }
                }

                // Offset into C depends on all previously added non-zeros and need to be shifted by
                // 1 (zero-based indexing)
                I idx = row_begin_C + offset - 1;

                // Only threads with a non-zero value write to C
                if(has_nnz)
                {
                    //                csr_col_ind_C[idx] = i + chunk_begin + idx_base_C;
                    csr_val_C[idx] = value;
                }

                // Last thread in block writes the block-wide offset into C such that all subsequent
                // entries are shifted by this offset
                if(hipThreadIdx_x == BLOCKSIZE - 1)
                {
                    scan_offsets[BLOCKSIZE / WARPSIZE - 1] = offset;
                }

                // Wait for last thread in block to finish writing
                __syncthreads();

                // Each thread reads the block-wide offset and adds it to its local offset into C
                row_begin_C += scan_offsets[BLOCKSIZE / WARPSIZE - 1];
            }

            // Each thread loads the new chunk beginning and end point
            chunk_begin = next_chunk;
            chunk_end   = chunk_begin + CHUNKSIZE;

            // Wait for all threads to finish load from shared memory
            __syncthreads();
        }
    }
}
