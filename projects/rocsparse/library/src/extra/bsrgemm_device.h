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
    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I>
    ROCSPARSE_DEVICE_ILF void bsrgemm_group_reduce(int tid, I* __restrict__ data)
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

    template <uint32_t BLOCKDIM, uint32_t NNZB, typename T>
    ROCSPARSE_DEVICE_ILF constexpr bool exceeds_shared_memory()
    {
        if(BLOCKDIM == 2)
        {
            if(NNZB >= 512)
            {
                return std::is_same<T, rocsparse_double_complex>();
            }
        }
        return false;
    }

    // Need to pass BLOCKDIM here and adjust number of groups based on block dim???
    template <uint32_t BLOCKSIZE,
              uint32_t GROUPS,
              uint32_t BLOCKDIM,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgemm_group_reduce_part2(J mb,
                                    const I* __restrict__ bsr_row_ptr,
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
        for(; row < mb; row += hipGridDim_x * BLOCKSIZE)
        {
            I nnzb = bsr_row_ptr[row + 1] - bsr_row_ptr[row];

            // clang-format off
             if(nnzb <=    8 && !exceeds_shared_memory<BLOCKDIM, 8, T>()) { ++sdata[hipThreadIdx_x * GROUPS + 0]; workspace[row] = 0; }
        else if(nnzb <=   16 && !exceeds_shared_memory<BLOCKDIM, 16, T>()) { ++sdata[hipThreadIdx_x * GROUPS + 1]; workspace[row] = 1; }
        else if(nnzb <=   32 && !exceeds_shared_memory<BLOCKDIM, 32, T>()) { ++sdata[hipThreadIdx_x * GROUPS + 2]; workspace[row] = 2; }
        else if(nnzb <=   64 && !exceeds_shared_memory<BLOCKDIM, 64, T>()) { ++sdata[hipThreadIdx_x * GROUPS + 3]; workspace[row] = 3; }
        else if(nnzb <=  128 && !exceeds_shared_memory<BLOCKDIM, 128, T>()) { ++sdata[hipThreadIdx_x * GROUPS + 4]; workspace[row] = 4; }
        else if(nnzb <=  256 && !exceeds_shared_memory<BLOCKDIM, 256, T>()) { ++sdata[hipThreadIdx_x * GROUPS + 5]; workspace[row] = 5; }
        else if(nnzb <=  512 && !exceeds_shared_memory<BLOCKDIM, 512, T>()) { ++sdata[hipThreadIdx_x * GROUPS + 6]; workspace[row] = 6; }
        else                  { ++sdata[hipThreadIdx_x * GROUPS + 7]; workspace[row] = 7; }
            // clang-format on
        }

        // Wait for all threads to finish
        __syncthreads();

        // Reduce block
        bsrgemm_group_reduce<BLOCKSIZE, GROUPS>(hipThreadIdx_x, sdata);

        // Write result
        if(hipThreadIdx_x < GROUPS)
        {
            group_size[hipBlockIdx_x * GROUPS + hipThreadIdx_x] = sdata[hipThreadIdx_x];
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t GROUPS, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgemm_group_reduce_part3(I* __restrict__ group_size)
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
        bsrgemm_group_reduce<BLOCKSIZE, GROUPS>(hipThreadIdx_x, sdata);

        // Write result back to global memory
        if(hipThreadIdx_x < GROUPS)
        {
            group_size[hipThreadIdx_x] = sdata[hipThreadIdx_x];
        }
    }

    // Copy an array
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgemm_copy(I size,
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
    ROCSPARSE_DEVICE_ILF void bsrgemm_copy_scale_device(I size, T beta, const T* in, T* out)
    {
        I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(idx >= size)
        {
            return;
        }

        out[idx] = beta * in[idx];
    }

    // Hash operation to insert pair into hash table
    template <uint32_t HASHVAL, uint32_t HASHSIZE, uint32_t BLOCKDIM, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void insert_pair_rxc(
        I key, T val, int row, int col, I* __restrict__ table, T* __restrict__ data, I empty)
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
                rocsparse::atomic_add(&data[BLOCKDIM * BLOCKDIM * hash + BLOCKDIM * row + col],
                                      val);
                break;
            }
            else if(temp == empty)
            {
                // If empty, add element with atomic
                if(rocsparse::atomic_cas(&table[hash], empty, key) == empty)
                {
                    // Add value
                    rocsparse::atomic_add(&data[BLOCKDIM * BLOCKDIM * hash + BLOCKDIM * row + col],
                                          val);
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
        bsrgemm_fill_wf_per_row_2x2_device(rocsparse_direction dir,
                                           J                   mb,
                                           J                   nkb,
                                           const J* __restrict__ offset,
                                           const J* __restrict__ perm,
                                           T alpha,
                                           const I* __restrict__ bsr_row_ptr_A,
                                           const J* __restrict__ bsr_col_ind_A,
                                           const T* __restrict__ bsr_val_A,
                                           const I* __restrict__ bsr_row_ptr_B,
                                           const J* __restrict__ bsr_col_ind_B,
                                           const T* __restrict__ bsr_val_B,
                                           T beta,
                                           const I* __restrict__ bsr_row_ptr_D,
                                           const J* __restrict__ bsr_col_ind_D,
                                           const T* __restrict__ bsr_val_D,
                                           const I* __restrict__ bsr_row_ptr_C,
                                           J* __restrict__ bsr_col_ind_C,
                                           T* __restrict__ bsr_val_C,
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
        J row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

        // Hash table in shared memory
        __shared__ J stable[(BLOCKSIZE / WFSIZE) * HASHSIZE];
        __shared__ T sdata[(BLOCKSIZE / WFSIZE) * 4 * HASHSIZE];

        // Local hash table
        J* table = &stable[wid * HASHSIZE];
        T* data  = &sdata[wid * 4 * HASHSIZE];

        // Initialize hash table
        for(uint32_t i = lid; i < HASHSIZE; i += WFSIZE)
        {
            table[i] = nkb;
        }

        for(uint32_t i = lid; i < 4 * HASHSIZE; i += WFSIZE)
        {
            data[i] = static_cast<T>(0);
        }

        // __threadfence_block();
        __syncthreads();

        // Bounds check
        if(row >= mb)
        {
            return;
        }

        // Apply permutation, if available
        row = perm ? perm[row + *offset] : row;

        // alpha * A * B part
        if(mul == true)
        {
            // Get row boundaries of the current row in A
            I row_begin_A = bsr_row_ptr_A[row] - idx_base_A;
            I row_end_A   = bsr_row_ptr_A[row + 1] - idx_base_A;

            // Loop over columns of A in current row
            for(I j = row_begin_A + lid; j < row_end_A; j += WFSIZE)
            {
                // Column of A in current row
                J col_A = bsr_col_ind_A[j] - idx_base_A;
                // Value of A in current row
                T val_A[4];
                if(dir == rocsparse_direction_row)
                {
                    val_A[0] = alpha * bsr_val_A[4 * j + 0]; // val[0][0]
                    val_A[1] = alpha * bsr_val_A[4 * j + 1]; // val[0][1]
                    val_A[2] = alpha * bsr_val_A[4 * j + 2]; // val[1][0]
                    val_A[3] = alpha * bsr_val_A[4 * j + 3]; // val[1][1]
                }
                else
                {
                    val_A[0] = alpha * bsr_val_A[4 * j + 0]; // val[0][0]
                    val_A[1] = alpha * bsr_val_A[4 * j + 2]; // val[0][1]
                    val_A[2] = alpha * bsr_val_A[4 * j + 1]; // val[1][0]
                    val_A[3] = alpha * bsr_val_A[4 * j + 3]; // val[1][1]
                }

                // Loop over columns of B in row col_A
                I row_begin_B = bsr_row_ptr_B[col_A] - idx_base_B;
                I row_end_B   = bsr_row_ptr_B[col_A + 1] - idx_base_B;

                // Insert all columns of B into hash table
                for(I k = row_begin_B; k < row_end_B; ++k)
                {
                    J col_B = bsr_col_ind_B[k] - idx_base_B;

                    T val_B[4];
                    if(dir == rocsparse_direction_row)
                    {
                        val_B[0] = bsr_val_B[4 * k + 0]; // val[0][0]
                        val_B[1] = bsr_val_B[4 * k + 1]; // val[0][1]
                        val_B[2] = bsr_val_B[4 * k + 2]; // val[1][0]
                        val_B[3] = bsr_val_B[4 * k + 3]; // val[1][1]
                    }
                    else
                    {
                        val_B[0] = bsr_val_B[4 * k + 0]; // val[0][0]
                        val_B[1] = bsr_val_B[4 * k + 2]; // val[0][1]
                        val_B[2] = bsr_val_B[4 * k + 1]; // val[1][0]
                        val_B[3] = bsr_val_B[4 * k + 3]; // val[1][1]
                    }

                    // Insert key value pair into hash table
                    insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                        col_B,
                        rocsparse::fma(val_A[0], val_B[0], val_A[1] * val_B[2]),
                        0,
                        0,
                        table,
                        data,
                        nkb);
                    insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                        col_B,
                        rocsparse::fma(val_A[0], val_B[1], val_A[1] * val_B[3]),
                        0,
                        1,
                        table,
                        data,
                        nkb);
                    insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                        col_B,
                        rocsparse::fma(val_A[2], val_B[0], val_A[3] * val_B[2]),
                        1,
                        0,
                        table,
                        data,
                        nkb);
                    insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                        col_B,
                        rocsparse::fma(val_A[2], val_B[1], val_A[3] * val_B[3]),
                        1,
                        1,
                        table,
                        data,
                        nkb);
                }
            }
        }

        __threadfence_block();

        // beta * D part
        if(add == true)
        {
            // Get row boundaries of the current row in D
            I row_begin_D = bsr_row_ptr_D[row] - idx_base_D;
            I row_end_D   = bsr_row_ptr_D[row + 1] - idx_base_D;

            // Loop over columns of D in current row and insert all columns of D into hash table
            for(I j = row_begin_D + lid; j < row_end_D; j += WFSIZE)
            {
                T val_D[4];
                if(dir == rocsparse_direction_row)
                {
                    val_D[0] = beta * bsr_val_D[4 * j + 0]; // val[0][0]
                    val_D[1] = beta * bsr_val_D[4 * j + 1]; // val[0][1]
                    val_D[2] = beta * bsr_val_D[4 * j + 2]; // val[1][0]
                    val_D[3] = beta * bsr_val_D[4 * j + 3]; // val[1][1]
                }
                else
                {
                    val_D[0] = beta * bsr_val_D[4 * j + 0]; // val[0][0]
                    val_D[1] = beta * bsr_val_D[4 * j + 2]; // val[0][1]
                    val_D[2] = beta * bsr_val_D[4 * j + 1]; // val[1][0]
                    val_D[3] = beta * bsr_val_D[4 * j + 3]; // val[1][1]
                }

                // Insert key value pair into hash table
                insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                    bsr_col_ind_D[j] - idx_base_D, val_D[0], 0, 0, table, data, nkb);
                insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                    bsr_col_ind_D[j] - idx_base_D, val_D[1], 0, 1, table, data, nkb);
                insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                    bsr_col_ind_D[j] - idx_base_D, val_D[2], 1, 0, table, data, nkb);
                insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                    bsr_col_ind_D[j] - idx_base_D, val_D[3], 1, 1, table, data, nkb);
            }
        }

        __threadfence_block();

        // Entry point of current row into C
        I row_begin_C = bsr_row_ptr_C[row] - idx_base_C;

        // Loop over hash table
        for(uint32_t i = lid; i < HASHSIZE; i += WFSIZE)
        {
            // Get column from hash table to fill it into C
            J col_C = table[i];

            // Skip hash table entry if not present
            if(col_C >= nkb)
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
            bsr_col_ind_C[idx_C] = col_C + idx_base_C;

            if(dir == rocsparse_direction_row)
            {
                bsr_val_C[4 * idx_C + 2 * 0 + 0] = data[4 * i + 2 * 0 + 0];
                bsr_val_C[4 * idx_C + 2 * 0 + 1] = data[4 * i + 2 * 0 + 1];
                bsr_val_C[4 * idx_C + 2 * 1 + 0] = data[4 * i + 2 * 1 + 0];
                bsr_val_C[4 * idx_C + 2 * 1 + 1] = data[4 * i + 2 * 1 + 1];
            }
            else
            {
                bsr_val_C[4 * idx_C + 2 * 0 + 0] = data[4 * i + 2 * 0 + 0];
                bsr_val_C[4 * idx_C + 2 * 0 + 1] = data[4 * i + 2 * 1 + 0];
                bsr_val_C[4 * idx_C + 2 * 1 + 0] = data[4 * i + 2 * 0 + 1];
                bsr_val_C[4 * idx_C + 2 * 1 + 1] = data[4 * i + 2 * 1 + 1];
            }
        }
    }

    // Compute column entries and accumulate values, where each row is processed by a single block
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrgemm_fill_block_per_row_2x2_device(rocsparse_direction dir,
                                              J                   mb,
                                              J                   nkb,
                                              const J* __restrict__ offset,
                                              const J* __restrict__ perm,
                                              T alpha,
                                              const I* __restrict__ bsr_row_ptr_A,
                                              const J* __restrict__ bsr_col_ind_A,
                                              const T* __restrict__ bsr_val_A,
                                              const I* __restrict__ bsr_row_ptr_B,
                                              const J* __restrict__ bsr_col_ind_B,
                                              const T* __restrict__ bsr_val_B,
                                              T beta,
                                              const I* __restrict__ bsr_row_ptr_D,
                                              const J* __restrict__ bsr_col_ind_D,
                                              const T* __restrict__ bsr_val_D,
                                              const I* __restrict__ bsr_row_ptr_C,
                                              J* __restrict__ bsr_col_ind_C,
                                              T* __restrict__ bsr_val_C,
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
            table[i] = nkb;
        }

        for(uint32_t i = hipThreadIdx_x; i < 4 * HASHSIZE; i += BLOCKSIZE)
        {
            data[i] = static_cast<T>(0);
        }

        __syncthreads();

        // Each block processes a row (apply permutation)
        J row = perm ? perm[hipBlockIdx_x + *offset] : hipBlockIdx_x;

        // alpha * A * B part
        if(mul == true)
        {
            // Get row boundaries of the current row in A
            I row_begin_A = bsr_row_ptr_A[row] - idx_base_A;
            I row_end_A   = bsr_row_ptr_A[row + 1] - idx_base_A;

            // Loop over columns of A in current row
            for(I j = row_begin_A + wid; j < row_end_A; j += (BLOCKSIZE / WFSIZE))
            {
                // Column of A in current row
                J col_A = bsr_col_ind_A[j] - idx_base_A;
                // Value of A in current row
                T val_A[4];
                if(dir == rocsparse_direction_row)
                {
                    val_A[0] = alpha * bsr_val_A[4 * j + 0]; // val[0][0]
                    val_A[1] = alpha * bsr_val_A[4 * j + 1]; // val[0][1]
                    val_A[2] = alpha * bsr_val_A[4 * j + 2]; // val[1][0]
                    val_A[3] = alpha * bsr_val_A[4 * j + 3]; // val[1][1]
                }
                else
                {
                    val_A[0] = alpha * bsr_val_A[4 * j + 0]; // val[0][0]
                    val_A[1] = alpha * bsr_val_A[4 * j + 2]; // val[0][1]
                    val_A[2] = alpha * bsr_val_A[4 * j + 1]; // val[1][0]
                    val_A[3] = alpha * bsr_val_A[4 * j + 3]; // val[1][1]
                }

                // Loop over columns of B in row col_A
                I row_begin_B = bsr_row_ptr_B[col_A] - idx_base_B;
                I row_end_B   = bsr_row_ptr_B[col_A + 1] - idx_base_B;

                // Insert all columns of B into hash table
                for(I k = row_begin_B + lid; k < row_end_B; k += WFSIZE)
                {
                    // Insert key value pair into hash table
                    J col_B = bsr_col_ind_B[k] - idx_base_B;

                    T val_B[4];
                    if(dir == rocsparse_direction_row)
                    {
                        val_B[0] = bsr_val_B[4 * k + 0]; // val[0][0]
                        val_B[1] = bsr_val_B[4 * k + 1]; // val[0][1]
                        val_B[2] = bsr_val_B[4 * k + 2]; // val[1][0]
                        val_B[3] = bsr_val_B[4 * k + 3]; // val[1][1]
                    }
                    else
                    {
                        val_B[0] = bsr_val_B[4 * k + 0]; // val[0][0]
                        val_B[1] = bsr_val_B[4 * k + 2]; // val[0][1]
                        val_B[2] = bsr_val_B[4 * k + 1]; // val[1][0]
                        val_B[3] = bsr_val_B[4 * k + 3]; // val[1][1]
                    }

                    // Insert key value pair into hash table
                    insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                        col_B,
                        rocsparse::fma(val_A[0], val_B[0], val_A[1] * val_B[2]),
                        0,
                        0,
                        table,
                        data,
                        nkb);
                    insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                        col_B,
                        rocsparse::fma(val_A[0], val_B[1], val_A[1] * val_B[3]),
                        0,
                        1,
                        table,
                        data,
                        nkb);
                    insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                        col_B,
                        rocsparse::fma(val_A[2], val_B[0], val_A[3] * val_B[2]),
                        1,
                        0,
                        table,
                        data,
                        nkb);
                    insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                        col_B,
                        rocsparse::fma(val_A[2], val_B[1], val_A[3] * val_B[3]),
                        1,
                        1,
                        table,
                        data,
                        nkb);
                }
            }
        }

        __syncthreads();

        // beta * D part
        if(add == true)
        {
            // Get row boundaries of the current row in D
            I row_begin_D = bsr_row_ptr_D[row] - idx_base_D;
            I row_end_D   = bsr_row_ptr_D[row + 1] - idx_base_D;

            // Loop over columns of D in current row and insert all columns of D into hash table
            for(I j = row_begin_D + hipThreadIdx_x; j < row_end_D; j += BLOCKSIZE)
            {
                T val_D[4];
                if(dir == rocsparse_direction_row)
                {
                    val_D[0] = beta * bsr_val_D[4 * j + 0]; // val[0][0]
                    val_D[1] = beta * bsr_val_D[4 * j + 1]; // val[0][1]
                    val_D[2] = beta * bsr_val_D[4 * j + 2]; // val[1][0]
                    val_D[3] = beta * bsr_val_D[4 * j + 3]; // val[1][1]
                }
                else
                {
                    val_D[0] = beta * bsr_val_D[4 * j + 0]; // val[0][0]
                    val_D[1] = beta * bsr_val_D[4 * j + 2]; // val[0][1]
                    val_D[2] = beta * bsr_val_D[4 * j + 1]; // val[1][0]
                    val_D[3] = beta * bsr_val_D[4 * j + 3]; // val[1][1]
                }

                // Insert key value pair into hash table
                insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                    bsr_col_ind_D[j] - idx_base_D, val_D[0], 0, 0, table, data, nkb);
                insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                    bsr_col_ind_D[j] - idx_base_D, val_D[1], 0, 1, table, data, nkb);
                insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                    bsr_col_ind_D[j] - idx_base_D, val_D[2], 1, 0, table, data, nkb);
                insert_pair_rxc<HASHVAL, HASHSIZE, 2>(
                    bsr_col_ind_D[j] - idx_base_D, val_D[3], 1, 1, table, data, nkb);
            }
        }

        __syncthreads();

        // Entry point of current row into C
        I row_begin_C = bsr_row_ptr_C[row] - idx_base_C;

        // Loop over hash table
        for(uint32_t i = hipThreadIdx_x; i < HASHSIZE; i += BLOCKSIZE)
        {
            // Get column from hash table to fill it into C
            J col_C = table[i];

            // Skip hash table entry if not present
            if(col_C >= nkb)
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
            bsr_col_ind_C[idx_C] = col_C + idx_base_C;
            if(dir == rocsparse_direction_row)
            {
                bsr_val_C[4 * idx_C + 0] = data[4 * i + 0];
                bsr_val_C[4 * idx_C + 1] = data[4 * i + 1];
                bsr_val_C[4 * idx_C + 2] = data[4 * i + 2];
                bsr_val_C[4 * idx_C + 3] = data[4 * i + 3];
            }
            else
            {
                bsr_val_C[4 * idx_C + 0] = data[4 * i + 0];
                bsr_val_C[4 * idx_C + 1] = data[4 * i + 2];
                bsr_val_C[4 * idx_C + 2] = data[4 * i + 1];
                bsr_val_C[4 * idx_C + 3] = data[4 * i + 3];
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              uint32_t BLOCKDIM,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void bsrgemm_fill_wf_per_row_device(rocsparse_direction dir,
                                                             J                   mb,
                                                             J                   nkb,
                                                             J                   block_dim,
                                                             const J* __restrict__ offset,
                                                             const J* __restrict__ perm,
                                                             T alpha,
                                                             const I* __restrict__ bsr_row_ptr_A,
                                                             const J* __restrict__ bsr_col_ind_A,
                                                             const T* __restrict__ bsr_val_A,
                                                             const I* __restrict__ bsr_row_ptr_B,
                                                             const J* __restrict__ bsr_col_ind_B,
                                                             const T* __restrict__ bsr_val_B,
                                                             T beta,
                                                             const I* __restrict__ bsr_row_ptr_D,
                                                             const J* __restrict__ bsr_col_ind_D,
                                                             const T* __restrict__ bsr_val_D,
                                                             const I* __restrict__ bsr_row_ptr_C,
                                                             J* __restrict__ bsr_col_ind_C,
                                                             T* __restrict__ bsr_val_C,
                                                             rocsparse_index_base idx_base_A,
                                                             rocsparse_index_base idx_base_B,
                                                             rocsparse_index_base idx_base_C,
                                                             rocsparse_index_base idx_base_D,
                                                             bool                 mul,
                                                             bool                 add)
    {
        int tid = hipThreadIdx_x;

        // Lane id
        int lid = tid & (WFSIZE - 1);
        // Wavefront id
        int wid = tid / WFSIZE;

        int slid = lid & (BLOCKDIM * BLOCKDIM - 1);
        int swid = lid / (BLOCKDIM * BLOCKDIM);

        int c = slid & (BLOCKDIM - 1);
        int r = slid / BLOCKDIM;

        // Each (sub)wavefront processes a row
        J row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

        // Hash table in shared memory
        __shared__ J stable[(BLOCKSIZE / WFSIZE) * HASHSIZE];
        __shared__ T sdata[(BLOCKSIZE / WFSIZE) * BLOCKDIM * BLOCKDIM * HASHSIZE];

        // Local hash table
        J* table = &stable[wid * HASHSIZE];
        T* data  = &sdata[wid * BLOCKDIM * BLOCKDIM * HASHSIZE];

        // Initialize hash table
        for(uint32_t i = lid; i < HASHSIZE; i += WFSIZE)
        {
            table[i] = nkb;
        }

        for(uint32_t i = lid; i < BLOCKDIM * BLOCKDIM * HASHSIZE; i += WFSIZE)
        {
            data[i] = static_cast<T>(0);
        }

        __syncthreads();

        // Bounds check
        if(row >= mb)
        {
            return;
        }

        // Apply permutation, if available
        row = perm ? perm[row + *offset] : row;

        // alpha * A * B part
        if(mul == true)
        {
            // Get row boundaries of the current row in A
            I row_begin_A = bsr_row_ptr_A[row] - idx_base_A;
            I row_end_A   = bsr_row_ptr_A[row + 1] - idx_base_A;

            // Loop over columns of A in current row
            for(I j = row_begin_A + swid; j < row_end_A; j += WFSIZE / (BLOCKDIM * BLOCKDIM))
            {
                // Column of A in current row
                J col_A = bsr_col_ind_A[j] - idx_base_A;

                // Loop over columns of B in row col_A
                I row_begin_B = bsr_row_ptr_B[col_A] - idx_base_B;
                I row_end_B   = bsr_row_ptr_B[col_A + 1] - idx_base_B;

                // Insert all columns of B into hash table
                for(I k = row_begin_B; k < row_end_B; ++k)
                {
                    J col_B = bsr_col_ind_B[k] - idx_base_B;

                    if(c < block_dim && r < block_dim)
                    {
                        T val_AB = static_cast<T>(0);
                        if(dir == rocsparse_direction_row)
                        {
                            for(int s = 0; s < block_dim; s++)
                            {
                                val_AB = rocsparse::fma(
                                    bsr_val_A[block_dim * block_dim * j + block_dim * r + s],
                                    bsr_val_B[block_dim * block_dim * k + block_dim * s + c],
                                    val_AB);
                            }
                        }
                        else
                        {
                            for(int s = 0; s < block_dim; s++)
                            {
                                val_AB = rocsparse::fma(
                                    bsr_val_A[block_dim * block_dim * j + block_dim * s + r],
                                    bsr_val_B[block_dim * block_dim * k + block_dim * c + s],
                                    val_AB);
                            }
                        }

                        insert_pair_rxc<HASHVAL, HASHSIZE, BLOCKDIM>(
                            col_B, alpha * val_AB, r, c, table, data, nkb);
                    }
                }
            }
        }

        __syncthreads();

        // beta * D part
        if(add == true)
        {
            // Get row boundaries of the current row in D
            I row_begin_D = bsr_row_ptr_D[row] - idx_base_D;
            I row_end_D   = bsr_row_ptr_D[row + 1] - idx_base_D;

            // Loop over columns of D in current row and insert all columns of D into hash table
            for(I j = row_begin_D + swid; j < row_end_D; j += WFSIZE / (BLOCKDIM * BLOCKDIM))
            {
                // Column of D in current row
                J col_D = bsr_col_ind_D[j] - idx_base_D;

                if(c < block_dim && r < block_dim)
                {
                    T val_D = static_cast<T>(0);
                    if(dir == rocsparse_direction_row)
                    {
                        val_D = bsr_val_D[block_dim * block_dim * j + block_dim * r + c];
                    }
                    else
                    {
                        val_D = bsr_val_D[block_dim * block_dim * j + block_dim * c + r];
                    }

                    // Insert key value pair into hash table
                    insert_pair_rxc<HASHVAL, HASHSIZE, BLOCKDIM>(
                        col_D, beta * val_D, r, c, table, data, nkb);
                }
            }
        }

        __syncthreads();

        // Entry point of current row into C
        I row_begin_C = bsr_row_ptr_C[row] - idx_base_C;

        // Loop over hash table
        for(uint32_t i = swid; i < HASHSIZE; i += WFSIZE / (BLOCKDIM * BLOCKDIM))
        {
            // Get column from hash table to fill it into C
            J col_C = table[i];

            // Skip hash table entry if not present
            if(col_C >= nkb)
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
            bsr_col_ind_C[idx_C] = col_C + idx_base_C;

            if(c < block_dim && r < block_dim)
            {
                if(dir == rocsparse_direction_row)
                {
                    bsr_val_C[block_dim * block_dim * idx_C + block_dim * r + c]
                        = data[BLOCKDIM * BLOCKDIM * i + BLOCKDIM * r + c];
                }
                else
                {
                    bsr_val_C[block_dim * block_dim * idx_C + block_dim * r + c]
                        = data[BLOCKDIM * BLOCKDIM * i + BLOCKDIM * c + r];
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              uint32_t BLOCKDIM,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void bsrgemm_fill_block_per_row_device(rocsparse_direction dir,
                                                                J                   mb,
                                                                J                   nkb,
                                                                J                   block_dim,
                                                                const J* __restrict__ offset,
                                                                const J* __restrict__ perm,
                                                                T alpha,
                                                                const I* __restrict__ bsr_row_ptr_A,
                                                                const J* __restrict__ bsr_col_ind_A,
                                                                const T* __restrict__ bsr_val_A,
                                                                const I* __restrict__ bsr_row_ptr_B,
                                                                const J* __restrict__ bsr_col_ind_B,
                                                                const T* __restrict__ bsr_val_B,
                                                                T beta,
                                                                const I* __restrict__ bsr_row_ptr_D,
                                                                const J* __restrict__ bsr_col_ind_D,
                                                                const T* __restrict__ bsr_val_D,
                                                                const I* __restrict__ bsr_row_ptr_C,
                                                                J* __restrict__ bsr_col_ind_C,
                                                                T* __restrict__ bsr_val_C,
                                                                rocsparse_index_base idx_base_A,
                                                                rocsparse_index_base idx_base_B,
                                                                rocsparse_index_base idx_base_C,
                                                                rocsparse_index_base idx_base_D,
                                                                bool                 mul,
                                                                bool                 add)
    {
        // Lane id
        int lid = hipThreadIdx_x & (BLOCKDIM * BLOCKDIM - 1);
        // Wavefront id
        int wid = hipThreadIdx_x / (BLOCKDIM * BLOCKDIM);

        int c = lid & (BLOCKDIM - 1);
        int r = lid / BLOCKDIM;

        // Each block processes a row (apply permutation)
        J row = perm ? perm[hipBlockIdx_x + *offset] : hipBlockIdx_x;

        // Hash table in shared memory
        __shared__ J table[HASHSIZE];
        __shared__ T data[BLOCKDIM * BLOCKDIM * HASHSIZE];

        // Initialize hash table
        for(uint32_t i = hipThreadIdx_x; i < HASHSIZE; i += BLOCKSIZE)
        {
            table[i] = nkb;
        }

        for(uint32_t i = hipThreadIdx_x; i < BLOCKDIM * BLOCKDIM * HASHSIZE; i += BLOCKSIZE)
        {
            data[i] = static_cast<T>(0);
        }

        __syncthreads();

        // alpha * A * B part
        if(mul == true)
        {
            // Get row boundaries of the current row in A
            I row_begin_A = bsr_row_ptr_A[row] - idx_base_A;
            I row_end_A   = bsr_row_ptr_A[row + 1] - idx_base_A;

            // Loop over columns of A in current row
            for(I j = row_begin_A + wid; j < row_end_A; j += (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
            {
                // Column of A in current row
                J col_A = bsr_col_ind_A[j] - idx_base_A;

                // Loop over columns of B in row col_A
                I row_begin_B = bsr_row_ptr_B[col_A] - idx_base_B;
                I row_end_B   = bsr_row_ptr_B[col_A + 1] - idx_base_B;

                // Insert all columns of B into hash table
                for(I k = row_begin_B; k < row_end_B; k++)
                {
                    // Insert key value pair into hash table
                    J col_B = bsr_col_ind_B[k] - idx_base_B;

                    T val_AB = static_cast<T>(0);
                    if(c < block_dim && r < block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            for(int i = 0; i < block_dim; i++)
                            {
                                val_AB = rocsparse::fma(
                                    bsr_val_A[block_dim * block_dim * j + block_dim * r + i],
                                    bsr_val_B[block_dim * block_dim * k + block_dim * i + c],
                                    val_AB);
                            }
                        }
                        else
                        {
                            for(int i = 0; i < block_dim; i++)
                            {
                                val_AB = rocsparse::fma(
                                    bsr_val_A[block_dim * block_dim * j + block_dim * i + r],
                                    bsr_val_B[block_dim * block_dim * k + block_dim * c + i],
                                    val_AB);
                            }
                        }

                        // Insert key value pair into hash table
                        insert_pair_rxc<HASHVAL, HASHSIZE, BLOCKDIM>(
                            col_B, alpha * val_AB, r, c, table, data, nkb);
                    }
                }
            }
        }

        __syncthreads();

        // beta * D part
        if(add == true)
        {
            // Get row boundaries of the current row in D
            I row_begin_D = bsr_row_ptr_D[row] - idx_base_D;
            I row_end_D   = bsr_row_ptr_D[row + 1] - idx_base_D;

            // Loop over columns of D in current row and insert all columns of D into hash table
            for(I j = row_begin_D + wid; j < row_end_D; j += (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
            {
                // Insert key value pair into hash table
                J col_D = bsr_col_ind_D[j] - idx_base_D;

                T val_D = static_cast<T>(0);
                if(c < block_dim && r < block_dim)
                {
                    if(dir == rocsparse_direction_row)
                    {
                        val_D = beta * bsr_val_D[block_dim * block_dim * j + block_dim * r + c];
                    }
                    else
                    {
                        val_D = beta * bsr_val_D[block_dim * block_dim * j + block_dim * c + r];
                    }

                    // Insert key value pair into hash table
                    insert_pair_rxc<HASHVAL, HASHSIZE, BLOCKDIM>(
                        col_D, val_D, r, c, table, data, nkb);
                }
            }
        }

        __syncthreads();

        // Entry point of current row into C
        I row_begin_C = bsr_row_ptr_C[row] - idx_base_C;

        // Loop over hash table
        for(uint32_t i = wid; i < HASHSIZE; i += (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
        {
            // Get column from hash table to fill it into C
            J col_C = table[i];

            // Skip hash table entry if not present
            if(col_C >= nkb)
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
            bsr_col_ind_C[idx_C] = col_C + idx_base_C;
            if(c < block_dim && r < block_dim)
            {
                if(dir == rocsparse_direction_row)
                {
                    bsr_val_C[block_dim * block_dim * idx_C + block_dim * r + c]
                        = data[BLOCKDIM * BLOCKDIM * i + BLOCKDIM * r + c];
                }
                else
                {
                    bsr_val_C[block_dim * block_dim * idx_C + block_dim * c + r]
                        = data[BLOCKDIM * BLOCKDIM * i + BLOCKDIM * r + c];
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t CHUNKSIZE,
              uint32_t BLOCKDIM,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrgemm_block_per_row_atomic_multipass_device(rocsparse_direction dir,
                                                      J                   nb,
                                                      J                   block_dim,
                                                      const J* __restrict__ offset_,
                                                      const J* __restrict__ perm,
                                                      T alpha,
                                                      const I* __restrict__ bsr_row_ptr_A,
                                                      const J* __restrict__ bsr_col_ind_A,
                                                      const T* __restrict__ bsr_val_A,
                                                      const I* __restrict__ bsr_row_ptr_B,
                                                      const J* __restrict__ bsr_col_ind_B,
                                                      const T* __restrict__ bsr_val_B,
                                                      T beta,
                                                      const I* __restrict__ bsr_row_ptr_D,
                                                      const J* __restrict__ bsr_col_ind_D,
                                                      const T* __restrict__ bsr_val_D,
                                                      const I* __restrict__ bsr_row_ptr_C,
                                                      J* __restrict__ bsr_col_ind_C,
                                                      T* __restrict__ bsr_val_C,
                                                      I* __restrict__ workspace_B,
                                                      rocsparse_index_base idx_base_A,
                                                      rocsparse_index_base idx_base_B,
                                                      rocsparse_index_base idx_base_C,
                                                      rocsparse_index_base idx_base_D,
                                                      bool                 mul,
                                                      bool                 add)
    {
        // Lane id
        int lid = hipThreadIdx_x & (BLOCKDIM * BLOCKDIM - 1);
        // Wavefront id
        int wid = hipThreadIdx_x / (BLOCKDIM * BLOCKDIM);

        int c = lid & (BLOCKDIM - 1);
        int r = lid / BLOCKDIM;

        // Each block processes a row (apply permutation)
        J row = perm ? perm[hipBlockIdx_x + *offset_] : hipBlockIdx_x;

        // Row entry marker and value accumulator
        __shared__ int table[CHUNKSIZE];
        __shared__ T   data[BLOCKDIM * BLOCKDIM * CHUNKSIZE];
        __shared__ T   shared_A[BLOCKSIZE];

        __shared__ J next_chunk;

        // Begin of the current row chunk (this is the column index of the current row)
        J chunk_begin = 0;
        J chunk_end   = CHUNKSIZE;

        // Get row boundaries of the current row in A
        I row_begin_A = (mul == true) ? bsr_row_ptr_A[row] - idx_base_A : 0;
        I row_end_A   = (mul == true) ? bsr_row_ptr_A[row + 1] - idx_base_A : 0;

        // Entry point into columns of C
        I row_begin_C = bsr_row_ptr_C[row] - idx_base_C;

        // Loop over the row chunks until the end of the row has been reached (which is
        // the number of total columns)
        while(chunk_begin < nb)
        {
            // Initialize row nnz table and accumulator
            for(uint32_t i = hipThreadIdx_x; i < CHUNKSIZE; i += BLOCKSIZE)
            {
                table[i] = 0;
            }

            for(uint32_t i = hipThreadIdx_x; i < BLOCKDIM * BLOCKDIM * CHUNKSIZE; i += BLOCKSIZE)
            {
                data[i] = static_cast<T>(0);
            }

            // Initialize next chunk column index
            if(hipThreadIdx_x == 0)
            {
                next_chunk = nb;
            }

            // Wait for all threads to finish initialization
            __syncthreads();

            // Initialize the beginning of the next chunk
            J min_col = nb;

            // alpha * A * B part
            if(mul == true)
            {
                // Loop over columns of A in current row
                for(I jj = row_begin_A; jj < row_end_A; jj += (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
                {
                    I j = jj + wid;

                    __syncthreads();

                    if(j < row_end_A)
                    {
                        // Load values into shared memory
                        if(c < block_dim && r < block_dim)
                        {
                            shared_A[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * r + c]
                                = bsr_val_A[block_dim * block_dim * j + block_dim * r + c];
                        }
                        else
                        {
                            shared_A[BLOCKDIM * BLOCKDIM * wid + BLOCKDIM * r + c]
                                = static_cast<T>(0);
                        }
                    }

                    __syncthreads();

                    I next_k = 0;
                    if(j < row_end_A)
                    {
                        // Column of A in current row
                        J col_A = bsr_col_ind_A[j] - idx_base_A;

                        // Loop over columns of B in row col_A
                        I row_begin_B = (chunk_begin == 0) ? bsr_row_ptr_B[col_A] - idx_base_B
                                                           : workspace_B[j];
                        I row_end_B   = bsr_row_ptr_B[col_A + 1] - idx_base_B;

                        // Keep track of the first k where the column index of B is exceeding
                        // the current chunks end point
                        next_k = row_begin_B;

                        // Loop over columns of B in row col_A
                        for(I k = row_begin_B; k < row_end_B; k++)
                        {
                            // Column of B in row col_A
                            J col_B = bsr_col_ind_B[k] - idx_base_B;

                            if(col_B >= chunk_begin && col_B < chunk_end)
                            {
                                // Mark nnz table if entry at col_B
                                table[col_B - chunk_begin] = 1;

                                // Atomically accumulate the intermediate products
                                T val_AB = static_cast<T>(0);
                                if(c < block_dim && r < block_dim)
                                {
                                    if(dir == rocsparse_direction_row)
                                    {
                                        for(int i = 0; i < block_dim; i++)
                                        {
                                            val_AB
                                                = rocsparse::fma(shared_A[BLOCKDIM * BLOCKDIM * wid
                                                                          + BLOCKDIM * r + i],
                                                                 bsr_val_B[block_dim * block_dim * k
                                                                           + block_dim * i + c],
                                                                 val_AB);
                                        }
                                    }
                                    else
                                    {
                                        for(int i = 0; i < block_dim; i++)
                                        {
                                            val_AB
                                                = rocsparse::fma(shared_A[BLOCKDIM * BLOCKDIM * wid
                                                                          + BLOCKDIM * i + r],
                                                                 bsr_val_B[block_dim * block_dim * k
                                                                           + block_dim * c + i],
                                                                 val_AB);
                                        }
                                    }

                                    rocsparse::atomic_add(
                                        &data[BLOCKDIM * BLOCKDIM * (col_B - chunk_begin)
                                              + BLOCKDIM * r + c],
                                        alpha * val_AB);
                                }
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
                    }

                    __syncthreads();
                    if(j < row_end_A)
                    {
                        workspace_B[j] = next_k;
                    }
                }
            }

            // beta * D part
            if(add == true)
            {
                // Get row boundaries of the current row in D
                I row_begin_D = bsr_row_ptr_D[row] - idx_base_D;
                I row_end_D   = bsr_row_ptr_D[row + 1] - idx_base_D;

                // Loop over columns of D in current row
                for(I j = row_begin_D + wid; j < row_end_D;
                    j += (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
                {
                    // Column of D in row col_A
                    J col_D = bsr_col_ind_D[j] - idx_base_D;

                    if(col_D >= chunk_begin && col_D < chunk_end)
                    {
                        // Mark nnz table if entry at col_D
                        table[col_D - chunk_begin] = 1;

                        // Accumulate the entry of D
                        if(c < block_dim && r < block_dim)
                        {
                            T val_D = static_cast<T>(0);
                            if(dir == rocsparse_direction_row)
                            {
                                val_D = beta
                                        * bsr_val_D[block_dim * block_dim * j + block_dim * r + c];
                            }
                            else
                            {
                                val_D = beta
                                        * bsr_val_D[block_dim * block_dim * j + block_dim * c + r];
                            }

                            rocsparse::atomic_add(&data[BLOCKDIM * BLOCKDIM * (col_D - chunk_begin)
                                                        + BLOCKDIM * r + c],
                                                  val_D);
                        }
                    }
                    else if(col_D >= chunk_end)
                    {
                        // Store the first column index of D that exceeds the current chunk
                        min_col = rocsparse::min(min_col, col_D);
                        break;
                    }
                }
            }

            if(lid == (BLOCKDIM * BLOCKDIM - 1))
            {
                // Atomically determine the new chunks beginning (minimum column index of B
                // that is larger than the current chunks end point)
                rocsparse::atomic_min(&next_chunk, min_col);
            }

            // Wait for all threads to finish
            __syncthreads();

            int tid = hipThreadIdx_x & (CHUNKSIZE - 1);

            int temp = table[tid];
            __syncthreads();

            // Segmented wavefront reduction
            for(uint32_t j = 1; j < CHUNKSIZE; j <<= 1)
            {
                if(tid >= j)
                {
                    temp = temp + table[tid - j];
                }
                __syncthreads();
                table[tid] = temp;
                __syncthreads();
            }

            int total_offset = table[CHUNKSIZE - 1];
            int prev         = (tid >= 1) ? table[tid - 1] : 0;

            __syncthreads();

            if(tid >= 1)
            {
                if(temp == prev)
                {
                    table[tid] = 0;
                }
            }

            __syncthreads();

            for(uint32_t i = wid; i < CHUNKSIZE; i += (BLOCKSIZE / (BLOCKDIM * BLOCKDIM)))
            {
                if(table[i])
                {
                    I idx = row_begin_C + table[i] - 1;

                    bsr_col_ind_C[idx] = i + chunk_begin + idx_base_C;

                    if(c < block_dim && r < block_dim)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            bsr_val_C[block_dim * block_dim * idx + block_dim * r + c]
                                = data[BLOCKDIM * BLOCKDIM * i + BLOCKDIM * r + c];
                        }
                        else
                        {
                            bsr_val_C[block_dim * block_dim * idx + block_dim * c + r]
                                = data[BLOCKDIM * BLOCKDIM * i + BLOCKDIM * r + c];
                        }
                    }
                }
            }

            __syncthreads();

            row_begin_C += total_offset;

            // Each thread loads the new chunk beginning and end point
            chunk_begin = next_chunk;
            chunk_end   = chunk_begin + CHUNKSIZE;

            // Wait for all threads to finish load from shared memory
            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t CHUNKSIZE,
              uint32_t BLOCKDIM,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void
        bsrgemm_block_per_row_multipass_device(rocsparse_direction dir,
                                               J                   nb,
                                               J                   block_dim,
                                               const J* __restrict__ offset_,
                                               const J* __restrict__ perm,
                                               T alpha,
                                               const I* __restrict__ bsr_row_ptr_A,
                                               const J* __restrict__ bsr_col_ind_A,
                                               const T* __restrict__ bsr_val_A,
                                               const I* __restrict__ bsr_row_ptr_B,
                                               const J* __restrict__ bsr_col_ind_B,
                                               const T* __restrict__ bsr_val_B,
                                               T beta,
                                               const I* __restrict__ bsr_row_ptr_D,
                                               const J* __restrict__ bsr_col_ind_D,
                                               const T* __restrict__ bsr_val_D,
                                               const I* __restrict__ bsr_row_ptr_C,
                                               J* __restrict__ bsr_col_ind_C,
                                               T* __restrict__ bsr_val_C,
                                               I* __restrict__ workspace_B,
                                               rocsparse_index_base idx_base_A,
                                               rocsparse_index_base idx_base_B,
                                               rocsparse_index_base idx_base_C,
                                               rocsparse_index_base idx_base_D,
                                               bool                 mul,
                                               bool                 add)
    {
        // Lane id
        int lid = hipThreadIdx_x & ((BLOCKSIZE / BLOCKDIM) - 1);
        // Wavefront id
        int wid = hipThreadIdx_x / (BLOCKSIZE / BLOCKDIM);

        // Each block processes a row (apply permutation)
        J row = perm ? perm[hipBlockIdx_x + *offset_] : hipBlockIdx_x;

        // Row entry marker and value accumulator
        __shared__ bool table[CHUNKSIZE];
        __shared__ T    data[BLOCKDIM * BLOCKDIM * CHUNKSIZE];

        // Begin of the current row chunk (this is the column index of the current row)
        J chunk_begin = 0;
        J chunk_end   = CHUNKSIZE;

        // Get row boundaries of the current row in A
        I row_begin_A = (mul == true) ? bsr_row_ptr_A[row] - idx_base_A : 0;
        I row_end_A   = (mul == true) ? bsr_row_ptr_A[row + 1] - idx_base_A : 0;

        // Entry point into columns of C
        I row_begin_C = bsr_row_ptr_C[row] - idx_base_C;

        // Loop over the row chunks until the end of the row has been reached (which is
        // the number of total columns)
        while(chunk_begin < nb)
        {
            // Initialize row nnz table and accumulator
            for(uint32_t i = hipThreadIdx_x; i < CHUNKSIZE; i += BLOCKSIZE)
            {
                table[i] = 0;
            }

            for(uint32_t i = hipThreadIdx_x; i < BLOCKDIM * BLOCKDIM * CHUNKSIZE; i += BLOCKSIZE)
            {
                data[i] = static_cast<T>(0);
            }

            // Initialize next chunk column index
            J next_chunk = nb;

            // Wait for all threads to finish initialization
            __syncthreads();

            // Initialize the beginning of the next chunk
            J min_col = nb;

            // alpha * A * B part
            if(mul == true)
            {
                // Loop over columns of A in current row
                for(I j = row_begin_A; j < row_end_A; j++)
                {
                    // Column of A in current row
                    J col_A = bsr_col_ind_A[j] - idx_base_A;

                    // Loop over columns of B in row col_A
                    I row_begin_B
                        = (chunk_begin == 0) ? bsr_row_ptr_B[col_A] - idx_base_B : workspace_B[j];
                    I row_end_B = bsr_row_ptr_B[col_A + 1] - idx_base_B;

                    // Keep track of the first k where the column index of B is exceeding
                    // the current chunks end point
                    I next_k = row_begin_B;

                    // Loop over columns of B in row col_A
                    for(I k = next_k; k < row_end_B; k++)
                    {
                        // Column of B in row col_A
                        J col_B = bsr_col_ind_B[k] - idx_base_B;

                        if(col_B >= chunk_begin && col_B < chunk_end)
                        {
                            // Mark nnz table if entry at col_B
                            table[col_B - chunk_begin] = 1;

                            // Accumulate the intermediate products
                            for(uint32_t i = 0; i < BLOCKDIM; i += BLOCKSIZE / BLOCKDIM)
                            {
                                if((i + lid) < block_dim && wid < block_dim)
                                {
                                    T val_AB = static_cast<T>(0);
                                    if(dir == rocsparse_direction_row)
                                    {
                                        for(int s = 0; s < block_dim; s++)
                                        {
                                            val_AB = rocsparse::fma(
                                                bsr_val_A[block_dim * block_dim * j
                                                          + block_dim * wid + s],
                                                bsr_val_B[block_dim * block_dim * k + block_dim * s
                                                          + (i + lid)],
                                                val_AB);
                                        }
                                    }
                                    else
                                    {
                                        for(int s = 0; s < block_dim; s++)
                                        {
                                            val_AB = rocsparse::fma(
                                                bsr_val_A[block_dim * block_dim * j + block_dim * s
                                                          + wid],
                                                bsr_val_B[block_dim * block_dim * k
                                                          + block_dim * (i + lid) + s],
                                                val_AB);
                                        }
                                    }

                                    data[BLOCKDIM * BLOCKDIM * (col_B - chunk_begin)
                                         + BLOCKDIM * wid + (i + lid)]
                                        = rocsparse::fma(
                                            alpha,
                                            val_AB,
                                            data[BLOCKDIM * BLOCKDIM * (col_B - chunk_begin)
                                                 + BLOCKDIM * wid + (i + lid)]);
                                }
                            }

                            __syncthreads();
                        }
                        //else if(col_B > chunk_begin)
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

                    workspace_B[j] = next_k;
                }
            }

            // beta * D part
            if(add == true)
            {
                // Get row boundaries of the current row in D
                I row_begin_D = bsr_row_ptr_D[row] - idx_base_D;
                I row_end_D   = bsr_row_ptr_D[row + 1] - idx_base_D;

                // Loop over columns of D in current row
                for(I j = row_begin_D; j < row_end_D; j++)
                {
                    // Column of D in row col_A
                    J col_D = bsr_col_ind_D[j] - idx_base_D;

                    if(col_D >= chunk_begin && col_D < chunk_end)
                    {
                        // Mark nnz table if entry at col_D
                        table[col_D - chunk_begin] = 1;

                        // Accumulate the entry of D
                        for(uint32_t i = 0; i < BLOCKDIM; i += BLOCKSIZE / BLOCKDIM)
                        {
                            if((i + lid) < block_dim && wid < block_dim)
                            {
                                T val_D = static_cast<T>(0);
                                if(dir == rocsparse_direction_row)
                                {
                                    val_D = bsr_val_D[block_dim * block_dim * j + block_dim * wid
                                                      + (i + lid)];
                                }
                                else
                                {
                                    val_D = bsr_val_D[block_dim * block_dim * j
                                                      + block_dim * (i + lid) + wid];
                                }

                                data[BLOCKDIM * BLOCKDIM * (col_D - chunk_begin) + BLOCKDIM * wid
                                     + (i + lid)]
                                    = rocsparse::fma(
                                        beta,
                                        val_D,
                                        data[BLOCKDIM * BLOCKDIM * (col_D - chunk_begin)
                                             + BLOCKDIM * wid + (i + lid)]);
                            }
                        }

                        __syncthreads();
                    }
                    // else if(col_D > chunk_begin)
                    else if(col_D >= chunk_end)
                    {
                        // Store the first column index of D that exceeds the current chunk
                        min_col = rocsparse::min(min_col, col_D);
                        break;
                    }
                }
            }

            next_chunk = rocsparse::min(next_chunk, min_col);

            // Wait for all threads to finish
            __syncthreads();

            int offset = 0;
            for(int j = 0; j < CHUNKSIZE; j++)
            {
                if(table[j])
                {
                    offset++;

                    I idx = row_begin_C + offset - 1;

                    bsr_col_ind_C[idx] = j + chunk_begin + idx_base_C;

                    for(uint32_t i = 0; i < BLOCKDIM; i += BLOCKSIZE / BLOCKDIM)
                    {
                        if((i + lid) < block_dim && wid < block_dim)
                        {
                            if(dir == rocsparse_direction_row)
                            {
                                bsr_val_C[block_dim * block_dim * idx + block_dim * wid + (i + lid)]
                                    = data[BLOCKDIM * BLOCKDIM * j + BLOCKDIM * wid + (i + lid)];
                            }
                            else
                            {
                                bsr_val_C[block_dim * block_dim * idx + block_dim * (i + lid) + wid]
                                    = data[BLOCKDIM * BLOCKDIM * j + BLOCKDIM * wid + (i + lid)];
                            }
                        }
                    }
                }
            }

            __syncthreads();

            // Broadcast the update of the start_C to all threads in the seegment. Choose the last
            // segment lane since that it contains the number of entries in the compressed sparse
            // row (even if its predicate is false).
            row_begin_C += offset;

            // Each thread loads the new chunk beginning and end point
            chunk_begin = next_chunk;
            chunk_end   = chunk_begin + CHUNKSIZE;

            // Wait for all threads to finish load from shared memory
            __syncthreads();
        }
    }
}
