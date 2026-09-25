/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    // Decrement
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_index_base(I* nnz)
    {
        --(*nnz);
    }

    // Copy an array
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_copy(I size,
                      const J* __restrict__ in,
                      J* __restrict__ out,
                      rocsparse_index_base idx_base_in,
                      rocsparse_index_base idx_base_out)
    {
        I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(idx >= size)
        {
            return;
        }

        out[idx] = in[idx] - idx_base_in + idx_base_out;
    }

    // Copy and scale an array
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void csrgemm_copy_scale_device(I size, T alpha, const T* in, T* out)
    {
        I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(idx >= size)
        {
            return;
        }

        out[idx] = alpha * in[idx];
    }

    // Compute number of intermediate products of each row
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_intermediate_products(J m,
                                       const I* __restrict__ csr_row_ptr_A,
                                       const J* __restrict__ csr_col_ind_A,
                                       const I* __restrict__ csr_row_ptr_B,
                                       const I* __restrict__ csr_row_ptr_D,
                                       I* __restrict__ int_prod,
                                       rocsparse_index_base idx_base_A,
                                       bool                 mul,
                                       bool                 add)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        // Lane id
        int lid = hipThreadIdx_x & (WFSIZE - 1);

        // Each (sub)wavefront processes a row
        J row = (hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x) / WFSIZE;

        // Bounds check
        if(row >= m)
        {
            return;
        }

        // Initialize intermediate product counter of current row
        I nprod = 0;

        // alpha * A * B part
        if(mul == true)
        {
            // Row begin and row end of A matrix
            I row_begin_A = csr_row_ptr_A[row] - idx_base_A;
            I row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

            // Loop over columns of A in current row
            for(I j = row_begin_A + lid; j < row_end_A; j += WFSIZE)
            {
                // Current column of A
                J col_A = csr_col_ind_A[j] - idx_base_A;

                // Accumulate non zero entries of B in row col_A
                nprod += (csr_row_ptr_B[col_A + 1] - csr_row_ptr_B[col_A]);
            }

            // Gather nprod
            nprod = rocsparse::wfreduce_sum<WFSIZE>(nprod);
        }

        // Last lane writes result
        if(lid == WFSIZE - 1)
        {
            // beta * D part
            if(add == true)
            {
                nprod += (csr_row_ptr_D[row + 1] - csr_row_ptr_D[row]);
            }

            // Write number of intermediate products of the current row
            int_prod[row] = nprod;
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I>
    // NOTE: 'data' points into block-shared LDS and is used for a cross-thread
    // segmented reduction (each thread reads slots written by other threads across
    // __syncthreads()). It must NOT be __restrict__: clang lowers __restrict__ to
    // LLVM noalias, which is interpreted to exclude other-thread writes, letting the
    // compiler forward the pre-barrier value and drop neighbor contributions.
    ROCSPARSE_DEVICE_ILF void csrgemm_group_reduce(int tid, I* data)
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

    template <uint32_t HASHSIZE, typename J, typename T>
    constexpr bool exceeding_smem(uint32_t shared_mem_optin)
    {
        return (((sizeof(J) + sizeof(T)) * HASHSIZE + sizeof(J) * (1024 / 32 + 1))
                > shared_mem_optin);
    }

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_group_reduce_part1(J m,
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
        csrgemm_group_reduce<BLOCKSIZE, GROUPS>(hipThreadIdx_x, sdata);

        // Write result
        if(hipThreadIdx_x < GROUPS)
        {
            group_size[hipBlockIdx_x * GROUPS + hipThreadIdx_x] = sdata[hipThreadIdx_x];
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_group_reduce_part2(J m,
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
        else if(nnz <=  4096 && !exceeding_smem<4096, J, T>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 6]; workspace[row] = 6; }
        else if(nnz <=  8192 && !exceeding_smem<8192, J, T>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 7]; workspace[row] = 7; }
        else if(nnz <=  16384 && !exceeding_smem<16384, J, T>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 8]; workspace[row] = 8; }
        else if(nnz <=  32768 && !exceeding_smem<32768, J, T>(shared_mem_optin)) { ++sdata[hipThreadIdx_x * GROUPS + 9]; workspace[row] = 9; }
        else                  { ++sdata[hipThreadIdx_x * GROUPS + 10]; workspace[row] = 10; }
            // clang-format on
        }

        // Wait for all threads to finish
        __syncthreads();

        // Reduce block
        csrgemm_group_reduce<BLOCKSIZE, GROUPS>(hipThreadIdx_x, sdata);

        // Write result
        if(hipThreadIdx_x < GROUPS)
        {
            group_size[hipBlockIdx_x * GROUPS + hipThreadIdx_x] = sdata[hipThreadIdx_x];
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_group_reduce_part3(I* __restrict__ group_size)
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
        csrgemm_group_reduce<BLOCKSIZE, GROUPS>(hipThreadIdx_x, sdata);

        // Write result back to global memory
        if(hipThreadIdx_x < GROUPS)
        {
            group_size[hipThreadIdx_x] = sdata[hipThreadIdx_x];
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_max_row_nnz_part1(J m,
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
    void csrgemm_max_row_nnz_part2(I* __restrict__ workspace)
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
    // NOTE: 'table' is a block-shared hash table. The plain load `table[hash]` below
    // must observe atomic_cas writes from other threads, so 'table' must NOT be
    // __restrict__ (noalias would let the compiler hoist the load out of the loop).
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
                if(rocsparse::atomic_cas<I>(&table[hash], empty, key) == empty)
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
                if(rocsparse::atomic_cas<I>(&table[hash], empty, key) == empty)
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

    // Compute non-zero entries per row, where each row is processed by a single wavefront
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_nnz_wf_per_row(J m,
                                const J* __restrict__ offset,
                                const J* __restrict__ perm,
                                const I* __restrict__ csr_row_ptr_A,
                                const J* __restrict__ csr_col_ind_A,
                                const I* __restrict__ csr_row_ptr_B,
                                const J* __restrict__ csr_col_ind_B,
                                const I* __restrict__ csr_row_ptr_D,
                                const J* __restrict__ csr_col_ind_D,
                                I* __restrict__ row_nnz,
                                rocsparse_index_base idx_base_A,
                                rocsparse_index_base idx_base_B,
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
            table[i] = -1;
        }

        __threadfence_block();

        // Bounds check
        if(row >= m)
        {
            return;
        }

        // Apply permutation, if available
        row = perm ? perm[row + *offset] : row;

        // Initialize row nnz
        J nnz = 0;

        // alpha * A * B part
        if(mul == true)
        {
            // Get row boundaries of the current row in A
            I row_begin_A = csr_row_ptr_A[row] - idx_base_A;
            I row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

            // Loop over columns of A in current row
            for(I j = row_begin_A + lid; j < row_end_A; j += WFSIZE)
            {
                // Column of A in current row
                J col_A = csr_col_ind_A[j] - idx_base_A;

                // Loop over columns of B in row col_A
                I row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                I row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                // Insert all columns of B into hash table
                for(I k = row_begin_B; k < row_end_B; ++k)
                {
                    // Count the actual insertions to obtain row nnz of C
                    nnz += insert_key<HASHVAL, HASHSIZE>(csr_col_ind_B[k] - idx_base_B, table);
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
                // Count the actual insertions to obtain row nnz of C
                nnz += insert_key<HASHVAL, HASHSIZE>(csr_col_ind_D[j] - idx_base_D, table);
            }
        }

        // Accumulate all row nnz within each (sub)wavefront to obtain the total row nnz
        // of the current row
        nnz = rocsparse::wfreduce_sum<WFSIZE>(nnz);

        // Write result to global memory
        if(lid == WFSIZE - 1)
        {
            row_nnz[row] = nnz;
        }
    }

    // Compute non-zero entries per row, where each row is processed by a single block
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_nnz_block_per_row(const J* __restrict__ offset,
                                   const J* __restrict__ perm,
                                   const I* __restrict__ csr_row_ptr_A,
                                   const J* __restrict__ csr_col_ind_A,
                                   const I* __restrict__ csr_row_ptr_B,
                                   const J* __restrict__ csr_col_ind_B,
                                   const I* __restrict__ csr_row_ptr_D,
                                   const J* __restrict__ csr_col_ind_D,
                                   I* __restrict__ row_nnz,
                                   rocsparse_index_base idx_base_A,
                                   rocsparse_index_base idx_base_B,
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

        // Each block processes a row (apply permutation)
        J row = perm[hipBlockIdx_x + *offset];

        // Hash table in shared memory
        extern __shared__ char shared_memory[];
        J*                     table = (J*)shared_memory;

        // Initialize hash table
        for(uint32_t i = hipThreadIdx_x; i < HASHSIZE; i += BLOCKSIZE)
        {
            table[i] = -1;
        }

        // Wait for all threads to finish initialization
        __syncthreads();

        // Initialize row nnz
        J nnz = 0;

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
                    // Count the actual insertions to obtain row nnz of C
                    nnz += insert_key<HASHVAL, HASHSIZE>(csr_col_ind_B[k] - idx_base_B, table);
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
            for(I j = row_begin_D + wid; j < row_end_D; j += BLOCKSIZE / WFSIZE)
            {
                // Count the actual insertions to obtain row nnz of C
                nnz += insert_key<HASHVAL, HASHSIZE>(csr_col_ind_D[j] - idx_base_D, table);
            }
        }

        // Wait for all threads to finish hash operation
        __syncthreads();

        // Accumulate all row nnz within each (sub)wavefront to obtain the total row nnz
        // of the current row
        nnz = rocsparse::wfreduce_sum<WFSIZE>(nnz);

        // Write result to shared memory for final reduction by first wavefront
        if(lid == WFSIZE - 1)
        {
            table[wid] = nnz;
        }

        // Wait for all threads to finish reduction
        __syncthreads();

        // Gather row nnz for the whole block
        nnz = (hipThreadIdx_x < BLOCKSIZE / WFSIZE) ? table[hipThreadIdx_x] : 0;

        // First wavefront computes final sum
        nnz = rocsparse::wfreduce_sum<BLOCKSIZE / WFSIZE>(nnz);

        // Write result to global memory
        if(hipThreadIdx_x == BLOCKSIZE / WFSIZE - 1)
        {
            row_nnz[row] = nnz;
        }
    }

    // Compute column entries and accumulate values, where each row is processed by a single wavefront
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrgemm_fill_wf_per_row_device(J m,
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
                                                             J* __restrict__ csr_col_ind_C,
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
        if(mul == true)
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
        if(add == true)
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
                csr_val_C[idx_C]     = data[i];
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
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrgemm_fill_block_per_row_device(J nk,
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
                                                                J* __restrict__ csr_col_ind_C,
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
        if(add == true)
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
            if(idx_C >= row_begin_C && idx_C < row_end_C)
            {
                csr_col_ind_C[idx_C] = col_C + idx_base_C;
                csr_val_C[idx_C]     = val_C;
            }
        }
    }
}
