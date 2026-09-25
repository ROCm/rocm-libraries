/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    // NOTE: 'table' is a block-shared hash table read via plain loads that must
    // observe other threads' atomic_cas writes; it must NOT be __restrict__.
    ROCSPARSE_DEVICE_ILF bool insert_key(I key, I* table)
    {
        static_assert(HASHSIZE > 0 && (HASHSIZE & (HASHSIZE - 1)) == 0,
                      "HASHSIZE must be a power of two.");
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
    ROCSPARSE_DEVICE_ILF bool insert_key(J key, I* table, I* __restrict__ local_idxs, I local_idx)
    {
        static_assert(HASHSIZE > 0 && (HASHSIZE & (HASHSIZE - 1)) == 0,
                      "HASHSIZE must be a power of two.");
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
    ROCSPARSE_DEVICE_ILF void insert_pair(I key, T val, I* table, T* __restrict__ data, I empty)
    {
        static_assert(HASHSIZE > 0 && (HASHSIZE & (HASHSIZE - 1)) == 0,
                      "HASHSIZE must be a power of two.");
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
                rocsparse::atomic_add(data, hash, HASHSIZE, val);
                break;
            }
            else if(temp == empty)
            {
                // If empty, add element with atomic
                if(rocsparse::atomic_cas(&table[hash], empty, key) == empty)
                {
                    // Add value
                    rocsparse::atomic_add(data, hash, HASHSIZE, val);
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
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        static_assert(HASHSIZE > 0 && (HASHSIZE & (HASHSIZE - 1)) == 0,
                      "HASHSIZE must be a power of two.");
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
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        static_assert(HASHSIZE > 0 && (HASHSIZE & (HASHSIZE - 1)) == 0,
                      "HASHSIZE must be a power of two.");
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
}
