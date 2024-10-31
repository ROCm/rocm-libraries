/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrgemm_symbolic_calc.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "control.h"
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_csrgemm.hpp"
#include "utility.h"

#include "common.h"
#include "rocsparse_primitives.h"

namespace rocsparse
{
    // Compute column entries and accumulate values, where each row is processed by a single
    // block. Splitting row into several chunks such that we can use shared memory to store
    // whether a column index is populated or not. Each row has at least 4097 non-zero
    // entries to compute.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t CHUNKSIZE, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void
        csrgemm_symbolic_fill_block_per_row_multipass_device(J n,
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
        __shared__ int  data[CHUNKSIZE];

        // Shared memory to determine the minimum of all column indices of B that exceed the
        // current chunk
        __shared__ J next_chunk;

        // Begin of the current row chunk (this is the column index of the current row)
        J chunk_begin = 0;
        J chunk_end   = CHUNKSIZE;

        // Get row boundaries of the current row in A
        I row_begin_A = (mul == true) ? csr_row_ptr_A[row] - idx_base_A : 0;
        I row_end_A   = (mul == true) ? csr_row_ptr_A[row + 1] - idx_base_A : 0;

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
            if(mul == true)
            {
                // Loop over columns of A in current row
                for(I j = row_begin_A + wid; j < row_end_A; j += BLOCKSIZE / WFSIZE)
                {
                    // Column of A in current row
                    J col_A = csr_col_ind_A[j] - idx_base_A;

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
            if(add == true)
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
                        // rocsparse::atomic_add(&data[col_D - chunk_begin], beta * csr_val_D[j]);
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

                // Each thread obtains a bit mask of all wavefront-wide non-zero entries
                // to compute its wavefront-wide non-zero offset in C
                uint64_t mask = __ballot(has_nnz == true);

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
                scan_offsets[hipThreadIdx_x / warpSize] = nnz;

                // Wait for all wavefronts to finish writing
                __syncthreads();

                // Each thread accumulates the offset of all previous wavefronts to obtain its
                // offset into C
                for(uint32_t j = 1; j < BLOCKSIZE / warpSize; ++j)
                {
                    if(hipThreadIdx_x >= j * warpSize)
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
                    csr_col_ind_C[idx] = i + chunk_begin + idx_base_C;
                }

                // Last thread in block writes the block-wide offset into C such that all subsequent
                // entries are shifted by this offset
                if(hipThreadIdx_x == BLOCKSIZE - 1)
                {
                    scan_offsets[BLOCKSIZE / warpSize - 1] = offset;
                }

                // Wait for last thread in block to finish writing
                __syncthreads();

                // Each thread reads the block-wide offset and adds it to its local offset into C
                row_begin_C += scan_offsets[BLOCKSIZE / warpSize - 1];
            }

            // Each thread loads the new chunk beginning and end point
            chunk_begin = next_chunk;
            chunk_end   = chunk_begin + CHUNKSIZE;

            // Wait for all threads to finish load from shared memory
            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t CHUNKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_fill_block_per_row_multipass(J n,
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
                                                       I* __restrict__ workspace_B,
                                                       rocsparse_index_base idx_base_A,
                                                       rocsparse_index_base idx_base_B,
                                                       rocsparse_index_base idx_base_C,
                                                       rocsparse_index_base idx_base_D,
                                                       bool                 mul,
                                                       bool                 add)
    {
        rocsparse::csrgemm_symbolic_fill_block_per_row_multipass_device<BLOCKSIZE,
                                                                        WFSIZE,
                                                                        CHUNKSIZE>(n,
                                                                                   offset,
                                                                                   perm,
                                                                                   csr_row_ptr_A,
                                                                                   csr_col_ind_A,
                                                                                   csr_row_ptr_B,
                                                                                   csr_col_ind_B,
                                                                                   csr_row_ptr_D,
                                                                                   csr_col_ind_D,
                                                                                   csr_row_ptr_C,
                                                                                   csr_col_ind_C,
                                                                                   workspace_B,
                                                                                   idx_base_A,
                                                                                   idx_base_B,
                                                                                   idx_base_C,
                                                                                   idx_base_D,
                                                                                   mul,
                                                                                   add);
    }

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I>
    ROCSPARSE_DEVICE_ILF void csrgemm_symbolic_group_reduce(int tid, I* __restrict__ data)
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

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_group_reduce_part1(J m,
                                             I* __restrict__ int_prod,
                                             J* __restrict__ group_size)
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
        else                    { ++sdata[hipThreadIdx_x * GROUPS + 7]; int_prod[row] = 7; }
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

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, bool EXCEEDS_SMEM, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_group_reduce_part2(J m,
                                             const I* __restrict__ csr_row_ptr,
                                             J* __restrict__ group_size,
                                             int* __restrict__ workspace)
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
        else if(nnz <=  4096 && !EXCEEDS_SMEM) { ++sdata[hipThreadIdx_x * GROUPS + 6]; workspace[row] = 6; }
        else                  { ++sdata[hipThreadIdx_x * GROUPS + 7]; workspace[row] = 7; }
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
    ROCSPARSE_DEVICE_ILF bool insert_key(I key, I* __restrict__ table, I empty)
    {
        // Compute hash
        I hash = (key * HASHVAL) & (HASHSIZE - 1);

        // Loop until key has been inserted
        while(true)
        {
            if(table[hash] == key)
            {
                // Element already present
                return false;
            }
            else if(table[hash] == empty)
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
            csr_col_ind_C[idx_C] = col_C + idx_base_C;
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_fill_wf_per_row(J m,
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
        rocsparse::csrgemm_symbolic_fill_wf_per_row_device<BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(
            m,
            nk,
            offset,
            perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    // Compute column entries and accumulate values, where each row is processed by a single block
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
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
        // Lane id
        int lid = hipThreadIdx_x & (WFSIZE - 1);
        // Wavefront id
        int wid = hipThreadIdx_x / WFSIZE;

        // Hash table in shared memory
        __shared__ J table[HASHSIZE];

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
        __shared__ J scan_offsets[BLOCKSIZE / warpSize + 1];

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
            scan_offsets[hipThreadIdx_x / warpSize] = nnz;

            // Wait for all wavefronts to finish writing
            __syncthreads();

            // Each thread accumulates the offset of all previous wavefronts to obtain its offset
            for(uint32_t j = 1; j < BLOCKSIZE / warpSize; ++j)
            {
                if(hipThreadIdx_x >= j * warpSize)
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
                scan_offsets[BLOCKSIZE / warpSize - 1] = offset;
            }

            // Wait for last thread in block to finish writing
            __syncthreads();

            // Each thread reads the block-wide offset and adds it to its local offset
            hash_offset += scan_offsets[BLOCKSIZE / warpSize - 1];
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
            csr_col_ind_C[idx_C] = col_C + idx_base_C;
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_fill_block_per_row(J nk,
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
        rocsparse::csrgemm_symbolic_fill_block_per_row_device<BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(
            nk,
            offset,
            perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    template <typename I, typename J>
    static inline rocsparse_status csrgemm_launcher(rocsparse_handle     handle,
                                                    J                    group_size,
                                                    const J*             group_offset,
                                                    const J*             perm,
                                                    J                    m,
                                                    J                    n,
                                                    J                    k,
                                                    const I*             csr_row_ptr_A,
                                                    const J*             csr_col_ind_A,
                                                    const I*             csr_row_ptr_B,
                                                    const J*             csr_col_ind_B,
                                                    const I*             csr_row_ptr_D,
                                                    const J*             csr_col_ind_D,
                                                    const I*             csr_row_ptr_C,
                                                    J*                   csr_col_ind_C,
                                                    rocsparse_index_base base_A,
                                                    rocsparse_index_base base_B,
                                                    rocsparse_index_base base_C,
                                                    rocsparse_index_base base_D,
                                                    bool                 mul,
                                                    bool                 add)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 64
#define CSRGEMM_HASHSIZE 4096
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_symbolic_fill_block_per_row<CSRGEMM_DIM,
                                                            CSRGEMM_SUB,
                                                            CSRGEMM_HASHSIZE,
                                                            CSRGEMM_FLL_HASH>),
            dim3(group_size),
            dim3(CSRGEMM_DIM),
            0,
            handle->stream,
            rocsparse::max(k, n),
            group_offset,
            perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            base_A,
            base_B,
            base_C,
            base_D,
            mul,
            add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM

        ROCSPARSE_RETURN_STATUS(success);
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_symbolic_calc_preprocess_template(rocsparse_handle handle,
                                                                      const J          m,
                                                                      const I* csr_row_ptr_C,
                                                                      void*    temp_buffer)
{

    // Stream
    hipStream_t stream = handle->stream;

    // Flag for exceeding shared memory
    constexpr bool exceeding_smem
        = (std::is_same<I, int64_t>::value && std::is_same<J, int64_t>::value);

    // Temporary buffer
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // Determine maximum non-zero entries per row of all rows
    J* workspace = reinterpret_cast<J*>(buffer);

#define CSRGEMM_DIM 256
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_symbolic_max_row_nnz_part1<CSRGEMM_DIM>),
                                       dim3(CSRGEMM_DIM),
                                       dim3(CSRGEMM_DIM),
                                       0,
                                       stream,
                                       m,
                                       csr_row_ptr_C,
                                       workspace);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_symbolic_max_row_nnz_part2<CSRGEMM_DIM>),
                                       dim3(1),
                                       dim3(CSRGEMM_DIM),
                                       0,
                                       stream,
                                       workspace);
#undef CSRGEMM_DIM

    J nnz_max;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&nnz_max, workspace, sizeof(J), hipMemcpyDeviceToHost, stream));
    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Group offset buffer
    J* d_group_offset = reinterpret_cast<J*>(buffer);

    buffer += sizeof(J) * 256;
    // Group size buffer

    // If maximum of row nnz exceeds 16, we process the rows in groups of
    // similar sized row nnz

    J* d_group_size = reinterpret_cast<J*>(buffer);
    buffer += sizeof(J) * 256 * CSRGEMM_MAXGROUPS;
    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_size, 0, sizeof(J) * CSRGEMM_MAXGROUPS, stream));
    if(nnz_max > 16)
    {
        // Group size buffer

        // Permutation temporary arrays
        J* tmp_vals = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        J* tmp_perm = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        int* tmp_keys = reinterpret_cast<int*>(buffer);
        buffer += ((sizeof(int) * m - 1) / 256 + 1) * 256;

        int* tmp_groups = reinterpret_cast<int*>(buffer);
        buffer += ((sizeof(int) * m - 1) / 256 + 1) * 256;

        // Determine number of rows per group
#define CSRGEMM_DIM 256
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_symbolic_group_reduce_part2<CSRGEMM_DIM,
                                                            CSRGEMM_MAXGROUPS,
                                                            exceeding_smem>),
            dim3(CSRGEMM_DIM),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            m,
            csr_row_ptr_C,
            d_group_size,
            tmp_groups);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_symbolic_group_reduce_part3<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
            dim3(1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            d_group_size);
#undef CSRGEMM_DIM
        size_t rocprim_size;
        // Exclusive sum to obtain group offsets
        void* rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<J, J>(
            handle, static_cast<J>(0), CSRGEMM_MAXGROUPS, &rocprim_size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::exclusive_scan(handle,
                                                                        d_group_size,
                                                                        d_group_offset,
                                                                        static_cast<J>(0),
                                                                        CSRGEMM_MAXGROUPS,
                                                                        rocprim_size,
                                                                        rocprim_buffer));

        // Create identity permutation for group access
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::create_identity_permutation_template(handle, m, tmp_perm));

        rocsparse::primitives::double_buffer<int> d_keys(tmp_groups, tmp_keys);
        rocsparse::primitives::double_buffer<J>   d_vals(tmp_perm, tmp_vals);

        // Sort pairs (by groups)
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<int, J>(
            handle, m, 0, 3, &rocprim_size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, d_keys, d_vals, m, 0, 3, rocprim_size, rocprim_buffer));

        // Release tmp_groups buffer
        // buffer -= ((sizeof(int) * m - 1) / 256 + 1) * 256;

        // Release tmp_keys buffer
        // buffer -= ((sizeof(int) * m - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(d_group_size, &m, sizeof(J), hipMemcpyHostToDevice, stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_offset, 0, sizeof(J), stream));
    }
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        d_group_size + CSRGEMM_MAXGROUPS, &nnz_max, sizeof(J), hipMemcpyHostToDevice, stream));
    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Compute columns and accumulate values for each group
    ROCSPARSE_RETURN_STATUS(success);
}

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_symbolic_calc_template(rocsparse_handle          handle,
                                                           rocsparse_operation       trans_A,
                                                           rocsparse_operation       trans_B,
                                                           J                         m,
                                                           J                         n,
                                                           J                         k,
                                                           const rocsparse_mat_descr descr_A,
                                                           I                         nnz_A,
                                                           const I*                  csr_row_ptr_A,
                                                           const J*                  csr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           I                         nnz_B,
                                                           const I*                  csr_row_ptr_B,
                                                           const J*                  csr_col_ind_B,
                                                           const rocsparse_mat_descr descr_D,
                                                           I                         nnz_D,
                                                           const I*                  csr_row_ptr_D,
                                                           const J*                  csr_col_ind_D,
                                                           const rocsparse_mat_descr descr_C,
                                                           I                         nnz_C,
                                                           const I*                  csr_row_ptr_C,
                                                           J*                        csr_col_ind_C,
                                                           const rocsparse_mat_info  info_C,
                                                           void*                     temp_buffer)
{
    const J*    d_group_offset = (J*)temp_buffer;
    const J*    d_perm         = nullptr;
    const char* bb             = reinterpret_cast<const char*>(temp_buffer);
    bb += sizeof(J) * 256;
    const J* d_group_size = reinterpret_cast<const J*>(bb);

    J h_group_size[CSRGEMM_MAXGROUPS + 1];

    // Copy group sizes to host
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(h_group_size,
                                       d_group_size,
                                       sizeof(J) * (CSRGEMM_MAXGROUPS + 1),
                                       hipMemcpyDeviceToHost,
                                       handle->stream));
    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    J nnz_max = h_group_size[CSRGEMM_MAXGROUPS];
    if(nnz_max > 16)
    {

        bb += sizeof(J) * 256 * CSRGEMM_MAXGROUPS;
        d_perm = reinterpret_cast<const J*>(bb);
    }
    else
    {
        d_perm = nullptr;
    }

    hipStream_t stream = handle->stream;

    // Index base
    rocsparse_index_base base_A
        = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
    rocsparse_index_base base_B
        = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
    rocsparse_index_base base_D
        = info_C->csrgemm_info->add ? descr_D->base : rocsparse_index_base_zero;

    // Flag for exceeding shared memory
    constexpr bool exceeding_smem
        = (std::is_same<I, int64_t>::value && std::is_same<J, int64_t>::value);

    // Group 0: 0 - 16 non-zeros per row
    if(h_group_size[0] > 0)
    {

#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 16
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_symbolic_fill_wf_per_row<CSRGEMM_DIM,
                                                         CSRGEMM_SUB,
                                                         CSRGEMM_HASHSIZE,
                                                         CSRGEMM_FLL_HASH>),
            dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            h_group_size[0],
            rocsparse::max(k, n),
            &d_group_offset[0],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 1: 17 - 32 non-zeros per row
    if(h_group_size[1] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 32

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_symbolic_fill_wf_per_row<CSRGEMM_DIM,
                                                         CSRGEMM_SUB,
                                                         CSRGEMM_HASHSIZE,
                                                         CSRGEMM_FLL_HASH>),
            dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            h_group_size[1],
            rocsparse::max(k, n),
            &d_group_offset[1],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 2: 33 - 256 non-zeros per row
    if(h_group_size[2] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 256
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_symbolic_fill_block_per_row<CSRGEMM_DIM,
                                                            CSRGEMM_SUB,
                                                            CSRGEMM_HASHSIZE,
                                                            CSRGEMM_FLL_HASH>),
            dim3(h_group_size[2]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            rocsparse::max(k, n),
            &d_group_offset[2],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 3: 257 - 512 non-zeros per row
    if(h_group_size[3] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 512
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_symbolic_fill_block_per_row<CSRGEMM_DIM,
                                                            CSRGEMM_SUB,
                                                            CSRGEMM_HASHSIZE,
                                                            CSRGEMM_FLL_HASH>),
            dim3(h_group_size[3]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            rocsparse::max(k, n),
            &d_group_offset[3],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 4: 513 - 1024 non-zeros per row
    if(h_group_size[4] > 0)
    {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 1024
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_symbolic_fill_block_per_row<CSRGEMM_DIM,
                                                            CSRGEMM_SUB,
                                                            CSRGEMM_HASHSIZE,
                                                            CSRGEMM_FLL_HASH>),
            dim3(h_group_size[4]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            rocsparse::max(k, n),
            &d_group_offset[4],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 5: 1025 - 2048 non-zeros per row
    if(h_group_size[5] > 0)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 2048
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_symbolic_fill_block_per_row<CSRGEMM_DIM,
                                                            CSRGEMM_SUB,
                                                            CSRGEMM_HASHSIZE,
                                                            CSRGEMM_FLL_HASH>),
            dim3(h_group_size[5]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            rocsparse::max(k, n),
            &d_group_offset[5],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 6: 2049 - 4096 non-zeros per row
    if(h_group_size[6] > 0 && !exceeding_smem)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_launcher(handle,
                                                              h_group_size[6],
                                                              &d_group_offset[6],
                                                              d_perm,
                                                              m,
                                                              n,
                                                              k,
                                                              csr_row_ptr_A,
                                                              csr_col_ind_A,
                                                              csr_row_ptr_B,
                                                              csr_col_ind_B,
                                                              csr_row_ptr_D,
                                                              csr_col_ind_D,
                                                              csr_row_ptr_C,
                                                              csr_col_ind_C,
                                                              base_A,
                                                              base_B,
                                                              descr_C->base,
                                                              base_D,
                                                              info_C->csrgemm_info->mul,
                                                              info_C->csrgemm_info->add));
    }

    // Group 7: more than 4096 non-zeros per row
    if(h_group_size[7] > 0)
    {
        // Matrices B and D must be sorted in order to run this path
        if(descr_B->storage_mode == rocsparse_storage_mode_unsorted
           || (info_C->csrgemm_info->add ? descr_D->storage_mode == rocsparse_storage_mode_unsorted
                                         : false))
        {
            return rocsparse_status_requires_sorted_storage;
        }

#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_CHUNKSIZE 2048
        I* workspace_B = nullptr;

        if(info_C->csrgemm_info->mul == true)
        {
            // Allocate additional buffer for C = A * B
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync((void**)&workspace_B, sizeof(I) * nnz_A, handle->stream));
        }

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_symbolic_fill_block_per_row_multipass<CSRGEMM_DIM,
                                                                      CSRGEMM_SUB,
                                                                      CSRGEMM_CHUNKSIZE>),
            dim3(h_group_size[7]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            n,
            &d_group_offset[7],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            workspace_B,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);

        if(info_C->csrgemm_info->mul == true)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(workspace_B, handle->stream));
        }
#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    ROCSPARSE_RETURN_STATUS(success);
}

#define INSTANTIATE(I, J)                                                               \
    template rocsparse_status rocsparse::csrgemm_symbolic_calc_preprocess_template(     \
        rocsparse_handle handle, const J m, const I* csr_row_ptr_C, void* temp_buffer); \
                                                                                        \
    template rocsparse_status rocsparse::csrgemm_symbolic_calc_template(                \
        rocsparse_handle          handle,                                               \
        rocsparse_operation       trans_A,                                              \
        rocsparse_operation       trans_B,                                              \
        J                         m,                                                    \
        J                         n,                                                    \
        J                         k,                                                    \
        const rocsparse_mat_descr descr_A,                                              \
        I                         nnz_A,                                                \
        const I*                  csr_row_ptr_A,                                        \
        const J*                  csr_col_ind_A,                                        \
        const rocsparse_mat_descr descr_B,                                              \
        I                         nnz_B,                                                \
        const I*                  csr_row_ptr_B,                                        \
        const J*                  csr_col_ind_B,                                        \
        const rocsparse_mat_descr descr_D,                                              \
        I                         nnz_D,                                                \
        const I*                  csr_row_ptr_D,                                        \
        const J*                  csr_col_ind_D,                                        \
        const rocsparse_mat_descr descr_C,                                              \
        I                         nnz_C,                                                \
        const I*                  csr_row_ptr_C,                                        \
        J*                        csr_col_ind_C,                                        \
        const rocsparse_mat_info  info_C,                                               \
        void*                     temp_buffer)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);

#undef INSTANTIATE
