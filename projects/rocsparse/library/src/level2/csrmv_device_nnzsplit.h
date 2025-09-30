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

#include "rocsparse_common.h"
#include "rocsparse_dichotomic_search.hpp"
namespace rocsparse
{
    template <typename I, typename J>
    ROCSPARSE_DEVICE_ILF J csrmv_nnzsplit_get_row_index(J              left,
                                                        J              right,
                                                        const J        offset,
                                                        const uint32_t local_nnz_index,
                                                        const I        nnz,
                                                        const I* __restrict__ csr_row_ptr_begin,
                                                        rocsparse_index_base idx_base)
    {
        return rocsparse::dichotomic_search<I, J>(
            left, right, offset + local_nnz_index + idx_base, nnz + idx_base, csr_row_ptr_begin);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t NNZ_PER_THREAD,
              bool     USE_STARTING_BLOCK_IDS,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmv_determine_block_starts(J m,
                                      const I* __restrict__ offsets,
                                      J* __restrict__ startingIds,
                                      J* __restrict__ starting_block_ids,
                                      rocsparse_index_base idx_base)
    {
        static constexpr uint32_t NNZ_PER_BLOCK = NNZ_PER_THREAD * BLOCKSIZE;

        const J id = blockIdx.x * blockDim.x + threadIdx.x;

        if(id > m)
            return;

        const I a = offsets[id] - idx_base;
        const I b = offsets[rocsparse::min(id + 1, m)] - idx_base;

        I       a_block = (a + NNZ_PER_BLOCK - 1) / NNZ_PER_BLOCK;
        const I b_block = (b - 1) / NNZ_PER_BLOCK;

        if(USE_STARTING_BLOCK_IDS)
        {
            if(a != b)
            {
                if(a_block == b_block)
                {
                    startingIds[a_block]        = id;
                    starting_block_ids[a_block] = 1;
                }
                else if(a_block < b_block)
                {
                    I       n_thread_blocks = b_block - a_block;
                    const I max_rep = (n_thread_blocks == 1) ? 1 : rocsparse::log2(n_thread_blocks);

                    for(; a_block < b_block; ++a_block)
                    {
                        startingIds[a_block]        = id;
                        starting_block_ids[a_block] = rocsparse::min(max_rep, n_thread_blocks);
                        n_thread_blocks             = rocsparse::max(
                            static_cast<I>(0), n_thread_blocks - starting_block_ids[a_block]);
                    }

                    startingIds[b_block]        = id;
                    starting_block_ids[b_block] = 1;
                }
            }

            if(id == m)
            {
                startingIds[(b + NNZ_PER_BLOCK - 1) / NNZ_PER_BLOCK]        = id - 1;
                starting_block_ids[(b + NNZ_PER_BLOCK - 1) / NNZ_PER_BLOCK] = 1;
            }
        }
        else
        {
            if(a != b)
            {
                for(; a_block <= b_block; ++a_block)
                {
                    startingIds[a_block] = id;
                }
            }

            if(id == m)
            {
                startingIds[(b + NNZ_PER_BLOCK - 1) / NNZ_PER_BLOCK] = id - 1;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t NNZ_PER_THREAD, typename T1, typename T2, typename I>
    ROCSPARSE_DEVICE_ILF void
        load_values(T1 (&values)[NNZ_PER_THREAD], const T2* __restrict__ csr_val, I nnz, bool conj)
    {

        const uint64_t offset_block  = blockIdx.x * (BLOCKSIZE * NNZ_PER_THREAD);
        const uint64_t offset_thread = threadIdx.x * NNZ_PER_THREAD;
        const uint64_t offset        = offset_block + offset_thread;

        if(offset < nnz)
        {
            for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
            {
                values[i] = (offset + i < nnz) ? static_cast<T1>(
                                rocsparse::conj_val(nontemporal_load(csr_val + offset + i), conj))
                                               : static_cast<T1>(0);
            }
        }
        else
        {
            for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
                values[i] = static_cast<T1>(0);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t NNZ_PER_THREAD, typename T, typename I>
    ROCSPARSE_DEVICE_ILF void load_indices(T (&values)[NNZ_PER_THREAD],
                                           const T*             indices,
                                           I                    nnz,
                                           rocsparse_index_base idx_base)
    {

        const uint64_t offset_block  = blockIdx.x * (BLOCKSIZE * NNZ_PER_THREAD);
        const uint64_t offset_thread = threadIdx.x * NNZ_PER_THREAD;
        const uint64_t offset        = offset_block + offset_thread;

        if(offset < nnz)
        {
            for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
            {
                values[i] = ((offset + i) < nnz) ? nontemporal_load(indices + offset + i) - idx_base
                                                 : static_cast<T>(0);
            }
        }
        else
        {
            for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
                values[i] = static_cast<T>(0);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t NNZ_PER_THREAD, typename T, typename I>
    ROCSPARSE_DEVICE_ILF void coalesced_load_indices(T (&values)[NNZ_PER_THREAD],
                                                     const T*             indices,
                                                     I                    nnz,
                                                     rocsparse_index_base idx_base)
    {
        const uint64_t offset = static_cast<uint64_t>(blockIdx.x * (BLOCKSIZE * NNZ_PER_THREAD));
        uint64_t       idx    = offset + threadIdx.x;

        if(offset < nnz)
        {
            for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i, idx += BLOCKSIZE)
            {
                values[i]
                    = (idx < nnz) ? nontemporal_load(indices + idx) - idx_base : static_cast<T>(0);
            }
        }
        else
        {
            for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
                values[i] = static_cast<T>(0);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t NNZ_PER_THREAD, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void
        csrmv_nnzsplit_get_row_indices(J*      row_indices,
                                       J       left,
                                       J       right,
                                       const J offset,
                                       const I nnz,
                                       const I* __restrict__ csr_row_ptr_begin,
                                       rocsparse_index_base idx_base)
    {
        for(uint32_t local_nnz_index = 0; local_nnz_index < NNZ_PER_THREAD; ++local_nnz_index)
        {
            row_indices[local_nnz_index] = csrmv_nnzsplit_get_row_index(
                left, right, offset, local_nnz_index, nnz, csr_row_ptr_begin, idx_base);

            left = row_indices[local_nnz_index];
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t NNZ_PER_THREAD, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void get_coalesced_row_indices(J* row_indices,
                                                        J  left_init,
                                                        J  right_init,
                                                        J  offset,
                                                        I  nnz,
                                                        const I* __restrict__ csr_row_ptr_begin,
                                                        rocsparse_index_base idx_base)
    {
        J left = left_init;

        const uint32_t wid      = rocsparse::read_first_lane(threadIdx.x / WFSIZE);
        const uint32_t lid      = threadIdx.x & (WFSIZE - 1);
        const J        w_offset = offset + wid * (WFSIZE * NNZ_PER_THREAD);

        J local_idx = lid;

        row_indices[NNZ_PER_THREAD - 1] = csrmv_nnzsplit_get_row_index(
            left,
            right_init,
            w_offset,
            static_cast<J>(local_idx + (NNZ_PER_THREAD - 1) * WFSIZE),
            nnz,
            csr_row_ptr_begin,
            idx_base);

        const J right
            = (row_indices[NNZ_PER_THREAD - 1] != 0) ? row_indices[NNZ_PER_THREAD - 1] : right_init;

        for(uint32_t local_nnz_index = 0; local_nnz_index < NNZ_PER_THREAD - 1; ++local_nnz_index)
        {
            row_indices[local_nnz_index] = csrmv_nnzsplit_get_row_index(
                left, right, w_offset, local_idx, nnz, csr_row_ptr_begin, idx_base);

            local_idx += WFSIZE;

            left = (row_indices[local_nnz_index] != 0) ? row_indices[local_nnz_index] : left_init;
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t NNZ_PER_THREAD,
              uint32_t WFSIZE,
              bool     USE_STARTING_BLOCK_IDS,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmv_nnzsplit_device(bool conj,
                                                    I    nnz,
                                                    J    m,
                                                    J    n,
                                                    T    alpha,
                                                    const I* __restrict__ csr_row_ptr_begin,
                                                    const J* __restrict__ startingIds,
                                                    const J* __restrict__ csr_col_ind,
                                                    const A* __restrict__ csr_val,
                                                    const X* __restrict__ x,
                                                    Y* __restrict__ y,
                                                    const J* __restrict__ starting_block_ids,
                                                    rocsparse_index_base idx_base)
    {
        static constexpr uint32_t NNZ_PER_BLOCK = NNZ_PER_THREAD * BLOCKSIZE;

        if(USE_STARTING_BLOCK_IDS)
        {
            if(starting_block_ids[blockIdx.x] * NNZ_PER_BLOCK > nnz)
                return;
            if(starting_block_ids[blockIdx.x] == starting_block_ids[blockIdx.x + 1])
                return;
        }

        const J startingId0 = USE_STARTING_BLOCK_IDS
                                  ? rocsparse::read_first_lane(starting_block_ids[blockIdx.x])
                                  : rocsparse::read_first_lane(blockIdx.x);

        const J startingId1 = USE_STARTING_BLOCK_IDS
                                  ? rocsparse::read_first_lane(starting_block_ids[blockIdx.x + 1])
                                  : rocsparse::read_first_lane(blockIdx.x + 1);

        I start_nnz_index = rocsparse::read_first_lane(I(startingId0 * NNZ_PER_BLOCK));

        const J bStart = rocsparse::read_first_lane(startingIds[startingId0]);
        const J bNum   = rocsparse::read_first_lane(startingIds[startingId1] - bStart + 1);

        // n_loops != 1 only for cases where bNum == 1
        const uint32_t n_loops = USE_STARTING_BLOCK_IDS
                                     ? rocsparse::read_first_lane(startingId1 - startingId0)
                                     : rocsparse::read_first_lane(1);

        const uint32_t tid = hipThreadIdx_x;
        const uint32_t lid = tid & (WFSIZE - 1);
        const uint32_t wid = rocsparse::read_first_lane(uint32_t(threadIdx.x / WFSIZE));

        T values[NNZ_PER_THREAD];

        // Each lane prefetches and computes its products
        {
            I idx = start_nnz_index + wid * NNZ_PER_THREAD * WFSIZE + lid;
            for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i, idx += WFSIZE)
            {
                values[i] = (idx < nnz)
                                ? static_cast<T>(
                                      rocsparse::conj_val(nontemporal_load(&csr_val[idx]), conj))
                                      * static_cast<T>(rocsparse::ldg(
                                          x + nontemporal_load(&csr_col_ind[idx]) - idx_base))
                                : static_cast<T>(0);
            }
        }

        // special cases for single row
        if(bNum == 1)
        {
            T value = static_cast<T>(0);

            for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
            {
                value += values[i];
            }

            for(uint32_t i_loops = 1; i_loops < n_loops; ++i_loops)
            {
                start_nnz_index += NNZ_PER_BLOCK;
                I idx = start_nnz_index + wid * NNZ_PER_THREAD * WFSIZE + lid;
                for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i, idx += WFSIZE)
                {
                    value += (idx < nnz)
                                 ? static_cast<T>(
                                       rocsparse::conj_val(nontemporal_load(&csr_val[idx]), conj))
                                       * static_cast<T>(rocsparse::ldg(
                                           x + nontemporal_load(&csr_col_ind[idx]) - idx_base))
                                 : static_cast<T>(0);
                }
            }

            __shared__ T shared_val[BLOCKSIZE];
            shared_val[tid] = value;
            __syncthreads();

            rocsparse::blockreduce_sum<BLOCKSIZE>(tid, shared_val);

            if(tid == 0)
            {
                rocsparse::atomic_add_check(&y[bStart], alpha * shared_val[0]);
            }

            return;
        }

        // compute row ids
        J row_indices[NNZ_PER_THREAD];

        const J offset = start_nnz_index;

        get_coalesced_row_indices<BLOCKSIZE, WFSIZE, NNZ_PER_THREAD>(
            row_indices, bStart, bStart + bNum, offset, nnz, csr_row_ptr_begin, idx_base);

        T sum = static_cast<T>(0);

        I prevrow = -1;

        for(uint32_t i = 0; i < NNZ_PER_THREAD - 1; ++i)
        {
            const J current_row = row_indices[i];

            // Determine if segmented wavefront scan is needed
            const J left_row  = rocsparse::shfl_up(current_row, 1);
            const J right_row = rocsparse::shfl_down(current_row, 1);

            const bool predicate
                = (left_row != current_row || right_row != current_row || current_row == -1
                   || (prevrow >= 0 && prevrow != current_row));

            if(rocsparse::any(predicate))
            {
                // Write out old values first
                sum = rocsparse::wfreduce_sum<WFSIZE>(sum);

                if(lid == WFSIZE - 1)
                {
                    if(prevrow >= 0)
                    {
                        rocsparse::atomic_add_check(&y[prevrow], alpha * sum);
                    }
                }

                sum = values[i];

                // Wavefront segmented reduction
                sum = rocsparse::wfsegmented_reduce<WFSIZE>(current_row, sum);

                if(lid < WFSIZE - 1)
                {
                    if(current_row != right_row && current_row >= 0)
                    {
                        rocsparse::atomic_add_check(&y[current_row], alpha * sum);
                    }
                }
                else
                {
                    if(current_row >= 0)
                    {
                        rocsparse::atomic_add_check(&y[current_row], alpha * sum);
                    }
                }

                // Reset sum to zero
                sum     = static_cast<T>(0);
                prevrow = -1;
            }
            else
            {
                sum += values[i];
                prevrow = current_row;
            }
        }

        // Determine if segmented wavefront scan is needed
        const I right_row = rocsparse::shfl_down(row_indices[NNZ_PER_THREAD - 1], 1);

        // Write out old values first
        sum = rocsparse::wfreduce_sum<WFSIZE>(sum);

        if(lid == WFSIZE - 1)
        {
            if(prevrow >= 0)
            {
                rocsparse::atomic_add_check(&y[prevrow], alpha * sum);
            }
        }

        sum = values[NNZ_PER_THREAD - 1];

        // Wavefront segmented reduction
        sum = rocsparse::wfsegmented_reduce<WFSIZE>(row_indices[NNZ_PER_THREAD - 1], sum);

        if(lid < WFSIZE - 1)
        {
            if(row_indices[NNZ_PER_THREAD - 1] != right_row && row_indices[NNZ_PER_THREAD - 1] >= 0)
            {
                rocsparse::atomic_add_check(&y[row_indices[NNZ_PER_THREAD - 1]], alpha * sum);
            }
        }
        else
        {
            if(row_indices[NNZ_PER_THREAD - 1] >= 0)
            {
                rocsparse::atomic_add_check(&y[row_indices[NNZ_PER_THREAD - 1]], alpha * sum);
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t NNZ_PER_THREAD,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmvt_nnzsplit_device(bool skip_diag,
                                                     bool conj,
                                                     I    nnz,
                                                     J    m,
                                                     J    n,
                                                     T    alpha,
                                                     const I* __restrict__ csr_row_ptr_begin,
                                                     const J* __restrict__ startingIds,
                                                     const J* __restrict__ csr_col_ind,
                                                     const A* __restrict__ csr_val,
                                                     const X* __restrict__ x,
                                                     Y* __restrict__ y,
                                                     rocsparse_index_base idx_base)
    {
        static constexpr uint32_t NNZ_PER_BLOCK = NNZ_PER_THREAD * BLOCKSIZE;

        const J bStart = rocsparse::read_first_lane(startingIds[blockIdx.x]);
        const J bNum   = rocsparse::read_first_lane(startingIds[blockIdx.x + 1] - bStart + 1);

        // special cases for single row
        // multiply with row value and atomic out... no overlap for sure
        if(bNum == 1)
        {
            const T mul = alpha * rocsparse::ldg(x + bStart);

            if(mul == static_cast<T>(0))
                return;

            const uint32_t startOffset = blockIdx.x * NNZ_PER_BLOCK + threadIdx.x;

            if(skip_diag)
            {
                for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
                {
                    const uint32_t toffset = i * BLOCKSIZE + startOffset;
                    if(toffset < nnz)
                    {
                        const J col_index = csr_col_ind[toffset] - idx_base;
                        if(col_index != bStart)
                        {
                            const T val = rocsparse::conj_val(csr_val[toffset], conj);
                            rocsparse::atomic_add_check(&y[col_index], val * mul);
                        }
                    }
                }
            }
            else
            {
                for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
                {
                    const uint32_t toffset = i * BLOCKSIZE + startOffset;
                    if(toffset < nnz)
                    {
                        const J col_index = csr_col_ind[toffset] - idx_base;
                        const T val       = rocsparse::conj_val(csr_val[toffset], conj);
                        rocsparse::atomic_add_check(&y[col_index], val * mul);
                    }
                }
            }

            return;
        }

        //load matrix
        T values[NNZ_PER_THREAD];

        load_values<BLOCKSIZE, NNZ_PER_THREAD>(values, csr_val, nnz, conj);

        // compute row ids
        J row_indices[NNZ_PER_THREAD];

        const J offset = blockIdx.x * NNZ_PER_BLOCK + threadIdx.x * NNZ_PER_THREAD;

        csrmv_nnzsplit_get_row_indices<BLOCKSIZE, NNZ_PER_THREAD>(
            row_indices, bStart, bStart + bNum, offset, nnz, csr_row_ptr_begin, idx_base);

        for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
        {
            T mul     = alpha * rocsparse::ldg(x + row_indices[i]);
            values[i] = values[i] * mul;
        }

        // load offset column ids
        J col_indices[NNZ_PER_THREAD];

        load_indices<BLOCKSIZE, NNZ_PER_THREAD>(col_indices, csr_col_ind, nnz, idx_base);

        if(skip_diag)
        {
            for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
            {
                if(col_indices[i] != row_indices[i])
                {
                    rocsparse::atomic_add_check(&y[col_indices[i]], values[i]);
                }
            }
        }
        else
        {
            for(uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
            {
                rocsparse::atomic_add_check(&y[col_indices[i]], values[i]);
            }
        }
        return;
    }
}
