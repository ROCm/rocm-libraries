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
    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I>
    // NOTE: 'data' points into block-shared LDS used for a cross-thread segmented
    // reduction; it must NOT be __restrict__ (noalias would let the compiler forward
    // reads across __syncthreads() and drop other threads' contributions).
    ROCSPARSE_DEVICE_ILF void csrgemm_symbolic_group_reduce(int tid, I* data)
    {
        // clang-format off
    if(BLOCKSIZE > 512 && tid < 512) for(uint32_t i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid + 512) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE > 256 && tid < 256) for(uint32_t i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid + 256) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE > 128 && tid < 128) for(uint32_t i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid + 128) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >  64 && tid <  64) for(uint32_t i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +  64) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >  32 && tid <  32) for(uint32_t i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +  32) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >  16 && tid <  16) for(uint32_t i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +  16) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >   8 && tid <   8) for(uint32_t i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +   8) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >   4 && tid <   4) for(uint32_t i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +   4) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >   2 && tid <   2) for(uint32_t i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +   2) * GROUPS + i]; __syncthreads();
    if(BLOCKSIZE >   1 && tid <   1) for(uint32_t i = 0; i < GROUPS; ++i) data[tid * GROUPS + i] += data[(tid +   1) * GROUPS + i]; __syncthreads();
        // clang-format on
    }

    template <uint32_t HASHSIZE, typename J>
    constexpr bool exceeding_smem_nnz(uint32_t shared_mem_optin)
    {
        return (sizeof(J) * HASHSIZE) > shared_mem_optin;
    }

    template <uint32_t HASHSIZE, typename J>
    constexpr bool exceeding_smem_symbolic(uint32_t shared_mem_optin)
    {
        return ((sizeof(J) * HASHSIZE + sizeof(J) * (1024 / 32 + 1)) > shared_mem_optin);
    }

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_group_reduce_part1(J m,
                                             I* __restrict__ int_prod,
                                             J* __restrict__ group_size,
                                             uint32_t shared_mem_optin)
    {
        J row = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        // Shared memory for block reduction
        __shared__ J sdata[BLOCKSIZE * GROUPS];

        // Initialize shared memory
        for(uint32_t i = 0; i < GROUPS; ++i)
        {
            sdata[hipThreadIdx_x * GROUPS + i] = 0;
        }

        __threadfence_block();

        // Loop over rows
        for(; row < m; row += hipGridDim_x * BLOCKSIZE)
        {
            I nprod = int_prod[row];

            // clang-format off
             if(nprod <=    32) { ++sdata[hipThreadIdx_x * GROUPS + 0]; int_prod[row] = 0; }
        else if(nprod <=    64) { ++sdata[hipThreadIdx_x * GROUPS + 1]; int_prod[row] = 1; }
        else if(nprod <=   512) { ++sdata[hipThreadIdx_x * GROUPS + 2]; int_prod[row] = 2; }
        else if(nprod <=  1024) { ++sdata[hipThreadIdx_x * GROUPS + 3]; int_prod[row] = 3; }
        else if(nprod <=  2048) { ++sdata[hipThreadIdx_x * GROUPS + 4]; int_prod[row] = 4; }
        else if(nprod <=  4096) { ++sdata[hipThreadIdx_x * GROUPS + 5]; int_prod[row] = 5; }
        else if(nprod <=  8192) { ++sdata[hipThreadIdx_x * GROUPS + 6]; int_prod[row] = 6; }
        else if(nprod <=  16384 && !exceeding_smem_nnz<16384, J>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 7]; int_prod[row] = 7; }
        else if(nprod <=  32768 && !exceeding_smem_nnz<32768, J>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 8]; int_prod[row] = 8; }
        else if(nprod <=  65536 && !exceeding_smem_nnz<65536, J>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 9]; int_prod[row] = 9; }
        else                    { ++sdata[hipThreadIdx_x * GROUPS + 10]; int_prod[row] = 10; }
            // clang-format on
        }

        // Wait for all threads to finish
        __syncthreads();

        // Reduce block
        csrgemm_symbolic_group_reduce<BLOCKSIZE, GROUPS>(hipThreadIdx_x, sdata);

        // Write result
        if(hipThreadIdx_x < GROUPS)
        {
            group_size[hipBlockIdx_x * GROUPS + hipThreadIdx_x] = sdata[hipThreadIdx_x];
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_group_reduce_part2(J m,
                                             const I* __restrict__ csr_row_ptr,
                                             J* __restrict__ group_size,
                                             int* __restrict__ workspace,
                                             uint32_t shared_mem_optin)
    {
        J row = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        // Shared memory for block reduction
        __shared__ J sdata[BLOCKSIZE * GROUPS];

        // Initialize shared memory
        for(uint32_t i = 0; i < GROUPS; ++i)
        {
            sdata[hipThreadIdx_x * GROUPS + i] = 0;
        }

        __threadfence_block();

        // Loop over rows
        for(; row < m; row += hipGridDim_x * BLOCKSIZE)
        {
            I nnz = csr_row_ptr[row + 1] - csr_row_ptr[row];

            // clang-format off
             if(nnz <=    16) { ++sdata[hipThreadIdx_x * GROUPS + 0]; workspace[row] = 0; }
        else if(nnz <=    32) { ++sdata[hipThreadIdx_x * GROUPS + 1]; workspace[row] = 1; }
        else if(nnz <=   256) { ++sdata[hipThreadIdx_x * GROUPS + 2]; workspace[row] = 2; }
        else if(nnz <=   512) { ++sdata[hipThreadIdx_x * GROUPS + 3]; workspace[row] = 3; }
        else if(nnz <=  1024) { ++sdata[hipThreadIdx_x * GROUPS + 4]; workspace[row] = 4; }
        else if(nnz <=  2048) { ++sdata[hipThreadIdx_x * GROUPS + 5]; workspace[row] = 5; }
        else if(nnz <=  4096 && !exceeding_smem_symbolic<4096, J>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 6]; workspace[row] = 6; }
        else if(nnz <=  8192 && !exceeding_smem_symbolic<8192, J>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 7]; workspace[row] = 7; }
        else if(nnz <=  16384 && !exceeding_smem_symbolic<16384, J>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 8]; workspace[row] = 8; }
        else if(nnz <=  32768 && !exceeding_smem_symbolic<32768, J>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 9]; workspace[row] = 9; }
        else                  { ++sdata[hipThreadIdx_x * GROUPS + 10]; workspace[row] = 10; }
            // clang-format on
        }

        // Wait for all threads to finish
        __syncthreads();

        // Reduce block
        csrgemm_symbolic_group_reduce<BLOCKSIZE, GROUPS>(hipThreadIdx_x, sdata);

        // Write result
        if(hipThreadIdx_x < GROUPS)
        {
            group_size[hipBlockIdx_x * GROUPS + hipThreadIdx_x] = sdata[hipThreadIdx_x];
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_group_reduce_part3(I* __restrict__ group_size)
    {
        // Shared memory for block reduction
        __shared__ I sdata[BLOCKSIZE * GROUPS];

        // Copy global data to shared memory
        for(uint32_t i = hipThreadIdx_x; i < BLOCKSIZE * GROUPS; i += BLOCKSIZE)
        {
            sdata[i] = group_size[i];
        }

        // Wait for all threads to finish
        __syncthreads();

        // Reduce block
        csrgemm_symbolic_group_reduce<BLOCKSIZE, GROUPS>(hipThreadIdx_x, sdata);

        // Write result back to global memory
        if(hipThreadIdx_x < GROUPS)
        {
            group_size[hipThreadIdx_x] = sdata[hipThreadIdx_x];
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_max_row_nnz_part1(J m,
                                            const I* __restrict__ csr_row_ptr,
                                            J* __restrict__ workspace)
    {
        static_assert(BLOCKSIZE > 0 && (BLOCKSIZE & (BLOCKSIZE - 1)) == 0,
                      "BLOCKSIZE must be a power of two.");
        J row = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        // Initialize local maximum
        J local_max = 0;

        // Loop over rows
        for(; row < m; row += hipGridDim_x * BLOCKSIZE)
        {
            // Determine local maximum
            local_max = rocsparse::max(local_max, J(csr_row_ptr[row + 1] - csr_row_ptr[row]));
        }

        // Shared memory for block reduction
        __shared__ J sdata[BLOCKSIZE];

        // Write local maximum into shared memory
        sdata[hipThreadIdx_x] = local_max;

        // Wait for all threads to finish
        __syncthreads();

        // Reduce block
        rocsparse::blockreduce_max<BLOCKSIZE>(hipThreadIdx_x, sdata);

        // Write result
        if(hipThreadIdx_x == 0)
        {
            workspace[hipBlockIdx_x] = sdata[0];
        }
    }

    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_max_row_nnz_part2(I* __restrict__ workspace)
    {
        static_assert(BLOCKSIZE > 0 && (BLOCKSIZE & (BLOCKSIZE - 1)) == 0,
                      "BLOCKSIZE must be a power of two.");
        // Shared memory for block reduction
        __shared__ I sdata[BLOCKSIZE];

        // Initialize shared memory with workspace entry
        sdata[hipThreadIdx_x] = workspace[hipThreadIdx_x];

        // Wait for all threads to finish
        __syncthreads();

        // Reduce block
        rocsparse::blockreduce_max<BLOCKSIZE>(hipThreadIdx_x, sdata);

        // Write result
        if(hipThreadIdx_x == 0)
        {
            workspace[0] = sdata[0];
        }
    }

    // Hash operation to insert key into hash table
    // Returns true if key has been added
    template <uint32_t HASHVAL, uint32_t HASHSIZE, typename I>
    // NOTE: 'table' is a block-shared hash table read via plain loads that must
    // observe other threads' atomic_cas writes; it must NOT be __restrict__.
    ROCSPARSE_DEVICE_ILF bool insert_key(I key, I* table, I empty)
    {
        static_assert(HASHSIZE > 0 && (HASHSIZE & (HASHSIZE - 1)) == 0,
                      "HASHSIZE must be a power of two.");
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

    // Compute column entries and accumulate values, where each row is processed by a single wavefront
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J>
    ROCSPARSE_DEVICE_ILF void
        csrgemm_symbolic_fill_wf_per_row_device(J m,
                                                J nk,
                                                const J* __restrict__ offset,
                                                const J* __restrict__ perm,
                                                const I* __restrict__ csr_row_ptr_A,
                                                const J* __restrict__ csr_col_ind_A,
                                                const I* __restrict__ csr_row_ptr_B,
                                                const J* __restrict__ csr_col_ind_B,
                                                const I* __restrict__ csr_row_ptr_D,
                                                const J* __restrict__ csr_col_ind_D,
                                                const I* __restrict__ csr_row_ptr_C,
                                                J* __restrict__ csr_col_ind_C,
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

        // Local hash table
        J* table = &stable[wid * HASHSIZE];

        // Initialize hash table
        for(uint32_t i = lid; i < HASHSIZE; i += WFSIZE)
        {
            table[i] = nk;
        }

        __threadfence_block();

        // Bounds check
        if(row >= m)
        {
            return;
        }

        //
        // Apply permutation, if available
        //
        row = perm ? perm[row + *offset] : row;

        //
        // Build hash tables.
        //

        //
        // alpha * A * B part
        //
        if(mul == true)
        {
            // Get row boundaries of the current row in A
            const I row_begin_A = csr_row_ptr_A[row] - idx_base_A;
            const I row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

            // Loop over columns of A in current row
            for(I j = row_begin_A + lid; j < row_end_A; j += WFSIZE)
            {
                // Column of A in current row
                const J col_A = csr_col_ind_A[j] - idx_base_A;

                // Loop over columns of B in row col_A
                const I row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                const I row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                // Insert all columns of B into hash table
                for(I k = row_begin_B; k < row_end_B; ++k)
                {
                    // Insert key into hash table
                    insert_key<HASHVAL, HASHSIZE>(csr_col_ind_B[k] - idx_base_B, table, nk);
                }
            }
        }

        // beta * D part
        if(add == true)
        {
            // Get row boundaries of the current row in D
            I row_begin_D = csr_row_ptr_D[row] - idx_base_D;
            I row_end_D   = csr_row_ptr_D[row + 1] - idx_base_D;

            // Loop over columns of D in current row and insert all columns of D into hash table
            for(I j = row_begin_D + lid; j < row_end_D; j += WFSIZE)
            {
                // Insert key pair into hash table
                insert_key<HASHVAL, HASHSIZE>(csr_col_ind_D[j] - idx_base_D, table, nk);
            }
        }

        __threadfence_block();

        // Entry point of current row into C
        I row_begin_C = csr_row_ptr_C[row] - idx_base_C;
        I row_end_C   = csr_row_ptr_C[row + 1] - idx_base_C;

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

            // Write column and accumulated value to the obtained position in C
            if(idx_C >= row_begin_C && idx_C < row_end_C)
            {
                csr_col_ind_C[idx_C] = col_C + idx_base_C;
            }
        }
    }

    // Compute column entries and accumulate values, where each row is processed by a single block
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              uint32_t WARPSIZE,
              typename I,
              typename J>
    ROCSPARSE_DEVICE_ILF void
        csrgemm_symbolic_fill_block_per_row_device(J nk,
                                                   const J* __restrict__ offset_,
                                                   const J* __restrict__ perm,
                                                   const I* __restrict__ csr_row_ptr_A,
                                                   const J* __restrict__ csr_col_ind_A,
                                                   const I* __restrict__ csr_row_ptr_B,
                                                   const J* __restrict__ csr_col_ind_B,
                                                   const I* __restrict__ csr_row_ptr_D,
                                                   const J* __restrict__ csr_col_ind_D,
                                                   const I* __restrict__ csr_row_ptr_C,
                                                   J* __restrict__ csr_col_ind_C,
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

        // Initialize hash table
        for(uint32_t i = hipThreadIdx_x; i < HASHSIZE; i += BLOCKSIZE)
        {
            table[i] = nk;
        }

        // Wait for all threads to finish initialization
        __syncthreads();

        // Each block processes a row (apply permutation)
        J row = perm[hipBlockIdx_x + *offset_];

        // alpha * A * B part
        if(mul == true)
        {
            // Get row boundaries of the current row in A
            I row_begin_A = csr_row_ptr_A[row] - idx_base_A;
            I row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

            // Loop over columns of A in current row
            for(I j = row_begin_A + wid; j < row_end_A; j += BLOCKSIZE / WFSIZE)
            {
                // Column of A in current row
                J col_A = csr_col_ind_A[j] - idx_base_A;

                // Loop over columns of B in row col_A
                I row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                I row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                for(I k = row_begin_B + lid; k < row_end_B; k += WFSIZE)
                {
                    // Insert key value pair into hash table
                    insert_key<HASHVAL, HASHSIZE>(csr_col_ind_B[k] - idx_base_B, table, nk);
                }
            }
        }

        // beta * D part
        if(add == true)
        {
            // Get row boundaries of the current row in D
            I row_begin_D = csr_row_ptr_D[row] - idx_base_D;
            I row_end_D   = csr_row_ptr_D[row + 1] - idx_base_D;

            // Loop over columns of D in current row and insert all columns of D into hash table
            for(I j = row_begin_D + hipThreadIdx_x; j < row_end_D; j += BLOCKSIZE)
            {
                // Insert key value pair into hash table
                insert_key<HASHVAL, HASHSIZE>(csr_col_ind_D[j] - idx_base_D, table, nk);
            }
        }

        // Wait for hash operations to finish
        __syncthreads();

        // Compress hash table, such that valid entries come first
        J* scan_offsets = (J*)(shared_memory + sizeof(J) * HASHSIZE);

        // Offset into hash table
        J hash_offset = 0;

        // Loop over the hash table and do the compression
        for(uint32_t i = hipThreadIdx_x; i < HASHSIZE; i += BLOCKSIZE)
        {
            // Get column and value from hash table
            J col_C = table[i];

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
            if(idx_C >= row_begin_C && idx_C < row_end_C)
            {
                csr_col_ind_C[idx_C] = col_C + idx_base_C;
            }
        }
    }
}
