/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

// Kernels of the bitmap path of csrgemm (rocsparse_csrgemm_bitmap.hpp).

#pragma once

#include "rocsparse_common.hpp"
#include "rocsparse_dichotomic_search.hpp"
#include "rocsparse_scalar.hpp"

#include <limits>

namespace rocsparse
{
    // How many entries of A each row of the group has, and the columns [span_lo, span_hi) the
    // row of R holds. A row without columns starts from the empty span [n, 0). The thread past
    // the last row zeroes the extra entry that the exclusive scans of the layout turn into the
    // totals.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_row_setup_kernel(J nslot,
                                         J n,
                                         const J* __restrict__ offset,
                                         const J* __restrict__ perm,
                                         const I* __restrict__ csr_row_ptr_A,
                                         const I* __restrict__ csr_row_ptr_R,
                                         const J* __restrict__ csr_col_ind_R,
                                         rocsparse_index_base idx_base_R,
                                         I* __restrict__ entry_count,
                                         J* __restrict__ span_lo,
                                         int64_t* __restrict__ span_hi)
    {
        for(int64_t i = static_cast<int64_t>(blockIdx.x) * BLOCKSIZE + threadIdx.x; i <= nslot;
            i += static_cast<int64_t>(gridDim.x) * BLOCKSIZE)
        {
            if(i == nslot)
            {
                entry_count[i] = 0;
                span_hi[i]     = 0;
                continue;
            }

            const J row = perm[i + *offset];

            entry_count[i] = csr_row_ptr_A[row + 1] - csr_row_ptr_A[row];

            J       lo = n;
            int64_t hi = 0;

            if(csr_row_ptr_R != nullptr)
            {
                const I row_begin_R = csr_row_ptr_R[row] - idx_base_R;
                const I row_end_R   = csr_row_ptr_R[row + 1] - idx_base_R;

                if(row_begin_R < row_end_R)
                {
                    lo = csr_col_ind_R[row_begin_R] - idx_base_R;
                    hi = static_cast<int64_t>(csr_col_ind_R[row_end_R - 1] - idx_base_R) + 1;
                }
            }

            span_lo[i] = lo;
            span_hi[i] = hi;
        }
    }

    // Widens the span of every row by the columns its products reach. B is sorted, so the first
    // and last column of the row of B an entry of A selects bound what that entry contributes.
    // The entries of A of a row are split over gridDim.y blocks that each reduce their share.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_span_kernel(J slot_base,
                                    const J* __restrict__ offset,
                                    const J* __restrict__ perm,
                                    const I* __restrict__ csr_row_ptr_A,
                                    const J* __restrict__ csr_col_ind_A,
                                    rocsparse_index_base idx_base_A,
                                    const I* __restrict__ csr_row_ptr_B,
                                    const J* __restrict__ csr_col_ind_B,
                                    rocsparse_index_base idx_base_B,
                                    J* __restrict__ span_lo,
                                    int64_t* __restrict__ span_hi)
    {
        const J       slot   = slot_base + static_cast<J>(blockIdx.x);
        const int64_t piece  = blockIdx.y;
        const int64_t npiece = gridDim.y;

        const J row = perm[slot + *offset];

        const I row_begin_A = csr_row_ptr_A[row] - idx_base_A;
        const I row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

        __shared__ J       slo[BLOCKSIZE];
        __shared__ int64_t shi[BLOCKSIZE];

        J       lo = std::numeric_limits<J>::max();
        int64_t hi = 0;

        for(I a = row_begin_A + static_cast<I>(piece * BLOCKSIZE + threadIdx.x); a < row_end_A;
            a += static_cast<I>(npiece * BLOCKSIZE))
        {
            const J col_A = csr_col_ind_A[a] - idx_base_A;

            const I row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
            const I row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

            if(row_begin_B < row_end_B)
            {
                lo = rocsparse::min(lo, static_cast<J>(csr_col_ind_B[row_begin_B] - idx_base_B));
                hi = rocsparse::max(
                    hi, static_cast<int64_t>(csr_col_ind_B[row_end_B - 1] - idx_base_B) + 1);
            }
        }

        slo[threadIdx.x] = lo;
        shi[threadIdx.x] = hi;
        __syncthreads();

        rocsparse::blockreduce_min<BLOCKSIZE>(threadIdx.x, slo);
        rocsparse::blockreduce_max<BLOCKSIZE>(threadIdx.x, shi);

        if(threadIdx.x == 0 && shi[0] > 0)
        {
            rocsparse::atomic_min(&span_lo[slot], slo[0]);
            rocsparse::atomic_max(&span_hi[slot], shi[0]);
        }
    }

    // Turns the span of every row into the words its bitmap covers: the first one in word_first,
    // and in word_offset the count, which a scan turns into where the row starts in the packing.
    template <uint32_t BLOCKSIZE, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_span_words_kernel(J nslot,
                                          J* __restrict__ word_first,
                                          int64_t* __restrict__ word_offset)
    {
        for(int64_t i = static_cast<int64_t>(blockIdx.x) * BLOCKSIZE + threadIdx.x; i < nslot;
            i += static_cast<int64_t>(gridDim.x) * BLOCKSIZE)
        {
            const J       lo = word_first[i];
            const int64_t hi = word_offset[i];

            if(hi > lo)
            {
                word_first[i]  = lo >> 5;
                word_offset[i] = ((hi - 1) >> 5) - (lo >> 5) + 1;
            }
            else
            {
                word_first[i]  = 0;
                word_offset[i] = 0;
            }
        }
    }

    // Points the bitmap of a row walked in tiles at one tile: its first word of the span, and its
    // end in the packing. The end of a row is where the next one starts, so this moves the next
    // row, and the walk sets the row back the same way once its tiles are done.
    template <typename J>
    ROCSPARSE_KERNEL(1)
    void csrgemm_bitmap_set_tile_kernel(J       slot,
                                        J       lo_word,
                                        int64_t word_end,
                                        J* __restrict__ word_first,
                                        int64_t* __restrict__ word_offset)
    {
        word_first[slot]      = lo_word;
        word_offset[slot + 1] = word_end;
    }

    // The row of the group that owns a flat entry index, and where that entry sits in A. The
    // entries are numbered row after row, so the owner is the row of the pass whose entries
    // [entry_offset[slot], entry_offset[slot + 1]) contain e; an empty row never does.
    template <typename I, typename J>
    ROCSPARSE_DEVICE_ILF void csrgemm_bitmap_entry(int64_t e,
                                                   int64_t entry_last,
                                                   J       slot_base,
                                                   J       count,
                                                   const I* __restrict__ entry_offset,
                                                   const J* __restrict__ offset,
                                                   const J* __restrict__ perm,
                                                   const I* __restrict__ csr_row_ptr_A,
                                                   rocsparse_index_base idx_base_A,
                                                   J&                   slot,
                                                   I&                   index)
    {
        slot = rocsparse::dichotomic_search(slot_base,
                                            static_cast<J>(slot_base + count - 1),
                                            static_cast<I>(e),
                                            static_cast<I>(entry_last),
                                            entry_offset);

        const J row = perm[slot + *offset];

        index = csr_row_ptr_A[row] - idx_base_A + static_cast<I>(e - entry_offset[slot]);
    }

    // Most words of a row whose bitmap is built in LDS: 16 KB, 131072 columns.
    inline constexpr int64_t csrgemm_bitmap_lds_words = 4096;

    // Whether the products of a row whose bitmap takes nword words are marked in LDS, or in the
    // pass directly. A row without words is marked in neither.
    __device__ __host__ inline constexpr bool csrgemm_bitmap_marked_in_lds(int64_t nword)
    {
        return nword > 0 && nword <= rocsparse::csrgemm_bitmap_lds_words;
    }

    __device__ __host__ inline constexpr bool csrgemm_bitmap_marked_in_pass(int64_t nword)
    {
        return nword > rocsparse::csrgemm_bitmap_lds_words;
    }

    // Marks the columns of the entries [begin, end) of a row. The bits of a word are gathered in a
    // register and issued as one atomic per word the entries touch, so consecutive columns do not
    // all hit the same word. Words outside the row's bitmap are dropped: the span assumes the row
    // is sorted, and a row that is not must not write past it.
    template <typename I, typename J>
    ROCSPARSE_DEVICE_ILF void csrgemm_bitmap_mark_chunk(I begin,
                                                        I end,
                                                        const J* __restrict__ csr_col_ind,
                                                        rocsparse_index_base idx_base,
                                                        J                    lo_word,
                                                        int64_t              row_nword,
                                                        uint32_t* __restrict__ row_bits)
    {
        J        word = -1;
        uint32_t bits = 0u;

        for(I k = begin; k < end; ++k)
        {
            const J col = csr_col_ind[k] - idx_base;
            const J w   = (col >> 5) - lo_word;

            if(w != word)
            {
                if(bits != 0u && word >= 0 && word < row_nword)
                {
                    atomicOr(&row_bits[word], bits);
                }

                word = w;
                bits = 0u;
            }

            bits |= 1u << (col & 31);
        }

        if(bits != 0u && word >= 0 && word < row_nword)
        {
            atomicOr(&row_bits[word], bits);
        }
    }

    // Marks the columns of a row of B, split over the WFSIZE lanes of a subgroup in contiguous
    // chunks: lanes striding over a dense row would all hit the same word.
    template <uint32_t WFSIZE, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void csrgemm_bitmap_mark_row_B(int lid,
                                                        I   row_begin_B,
                                                        I   row_end_B,
                                                        const J* __restrict__ csr_col_ind_B,
                                                        rocsparse_index_base idx_base_B,
                                                        J                    lo_word,
                                                        int64_t              row_nword,
                                                        uint32_t* __restrict__ row_bits)
    {
        const int64_t len = static_cast<int64_t>(row_end_B) - row_begin_B;

        if(len <= 0)
        {
            return;
        }

        const int64_t chunk = (len - 1) / WFSIZE + 1;
        const int64_t begin = lid * chunk;

        if(begin >= len)
        {
            return;
        }

        rocsparse::csrgemm_bitmap_mark_chunk(
            static_cast<I>(row_begin_B + begin),
            static_cast<I>(row_begin_B + rocsparse::min(begin + chunk, len)),
            csr_col_ind_B,
            idx_base_B,
            lo_word,
            row_nword,
            row_bits);
    }

    // Marks the products of the rows whose span fits in LDS. The bitmap of the row is built in
    // LDS, where the atomics are cheap, and merged into the pass in one sweep. The entries of a
    // row are split over gridDim.y blocks, a subgroup of WFSIZE lanes per entry.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_mark_lds_kernel(J       slot_base,
                                        int64_t word_base,
                                        const int64_t* __restrict__ word_offset,
                                        const J* __restrict__ word_first,
                                        const J* __restrict__ offset,
                                        const J* __restrict__ perm,
                                        const I* __restrict__ csr_row_ptr_A,
                                        const J* __restrict__ csr_col_ind_A,
                                        rocsparse_index_base idx_base_A,
                                        const I* __restrict__ csr_row_ptr_B,
                                        const J* __restrict__ csr_col_ind_B,
                                        rocsparse_index_base idx_base_B,
                                        uint32_t* __restrict__ bitmap)
    {
        const J       slot   = slot_base + static_cast<J>(blockIdx.x);
        const int64_t piece  = blockIdx.y;
        const int64_t npiece = gridDim.y;

        const int64_t row_nword = word_offset[slot + 1] - word_offset[slot];

        if(!rocsparse::csrgemm_bitmap_marked_in_lds(row_nword))
        {
            return;
        }

        __shared__ uint32_t sbits[rocsparse::csrgemm_bitmap_lds_words];

        for(int64_t w = threadIdx.x; w < row_nword; w += BLOCKSIZE)
        {
            sbits[w] = 0u;
        }
        __syncthreads();

        const J row = perm[slot + *offset];

        const I row_begin_A = csr_row_ptr_A[row] - idx_base_A;
        const I row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;
        const J lo_word     = word_first[slot];

        const int     lid = threadIdx.x & (WFSIZE - 1);
        const int64_t nwf = BLOCKSIZE / WFSIZE;
        const int64_t wf  = threadIdx.x / WFSIZE;

        for(I a = row_begin_A + static_cast<I>(piece * nwf + wf); a < row_end_A;
            a += static_cast<I>(npiece * nwf))
        {
            const J col_A = csr_col_ind_A[a] - idx_base_A;

            rocsparse::csrgemm_bitmap_mark_row_B<WFSIZE>(lid,
                                                         csr_row_ptr_B[col_A] - idx_base_B,
                                                         csr_row_ptr_B[col_A + 1] - idx_base_B,
                                                         csr_col_ind_B,
                                                         idx_base_B,
                                                         lo_word,
                                                         row_nword,
                                                         sbits);
        }
        __syncthreads();

        uint32_t* row_bits = bitmap + (word_offset[slot] - word_base);

        for(int64_t w = threadIdx.x; w < row_nword; w += BLOCKSIZE)
        {
            const uint32_t bits = sbits[w];

            if(bits != 0u)
            {
                atomicOr(&row_bits[w], bits);
            }
        }
    }

    // Marks the column of every intermediate product of the rows too wide for LDS. A subgroup of
    // WFSIZE lanes takes an entry of A and splits its row of B.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_mark_kernel(int64_t entry_first,
                                    int64_t entry_last,
                                    J       slot_base,
                                    J       count,
                                    int64_t word_base,
                                    const I* __restrict__ entry_offset,
                                    const int64_t* __restrict__ word_offset,
                                    const J* __restrict__ word_first,
                                    const J* __restrict__ offset,
                                    const J* __restrict__ perm,
                                    const I* __restrict__ csr_row_ptr_A,
                                    const J* __restrict__ csr_col_ind_A,
                                    rocsparse_index_base idx_base_A,
                                    const I* __restrict__ csr_row_ptr_B,
                                    const J* __restrict__ csr_col_ind_B,
                                    rocsparse_index_base idx_base_B,
                                    uint32_t* __restrict__ bitmap)
    {
        const int     lid    = threadIdx.x & (WFSIZE - 1);
        const int64_t nwf    = BLOCKSIZE / WFSIZE;
        const int64_t wf     = static_cast<int64_t>(blockIdx.x) * nwf + threadIdx.x / WFSIZE;
        const int64_t stride = static_cast<int64_t>(gridDim.x) * nwf;

        for(int64_t e = entry_first + wf; e < entry_last; e += stride)
        {
            J slot;
            I index;

            rocsparse::csrgemm_bitmap_entry(e,
                                            entry_last,
                                            slot_base,
                                            count,
                                            entry_offset,
                                            offset,
                                            perm,
                                            csr_row_ptr_A,
                                            idx_base_A,
                                            slot,
                                            index);

            const int64_t row_nword = word_offset[slot + 1] - word_offset[slot];

            if(!rocsparse::csrgemm_bitmap_marked_in_pass(row_nword))
            {
                continue;
            }

            const J col_A = csr_col_ind_A[index] - idx_base_A;

            rocsparse::csrgemm_bitmap_mark_row_B<WFSIZE>(lid,
                                                         csr_row_ptr_B[col_A] - idx_base_B,
                                                         csr_row_ptr_B[col_A + 1] - idx_base_B,
                                                         csr_col_ind_B,
                                                         idx_base_B,
                                                         word_first[slot],
                                                         row_nword,
                                                         bitmap + (word_offset[slot] - word_base));
        }
    }

    // Unions a row of some other matrix into the bitmap. With D this makes the occupancy describe
    // both terms of D + alpha * A * B. With a C whose pattern is already known it makes the rank
    // index address that pattern directly. A column addresses its own bit, so the threads may mark
    // in any order; the row itself must be sorted, since its span was taken from its ends.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_mark_rows_kernel(J       slot_base,
                                         int64_t word_base,
                                         const int64_t* __restrict__ word_offset,
                                         const J* __restrict__ word_first,
                                         const J* __restrict__ offset,
                                         const J* __restrict__ perm,
                                         const I* __restrict__ csr_row_ptr_R,
                                         const J* __restrict__ csr_col_ind_R,
                                         rocsparse_index_base idx_base_R,
                                         uint32_t* __restrict__ bitmap)
    {
        const J       slot   = slot_base + static_cast<J>(blockIdx.x);
        const int64_t piece  = blockIdx.y;
        const int64_t npiece = gridDim.y;

        const J row = perm[slot + *offset];

        const I row_begin_R = csr_row_ptr_R[row] - idx_base_R;
        const I row_end_R   = csr_row_ptr_R[row + 1] - idx_base_R;

        const int64_t len = row_end_R - row_begin_R;

        if(len <= 0)
        {
            return;
        }

        // Every thread of the row's blocks takes a contiguous chunk of the row
        const int64_t nthread = npiece * BLOCKSIZE;
        const int64_t chunk   = (len - 1) / nthread + 1;
        const int64_t begin   = (piece * BLOCKSIZE + threadIdx.x) * chunk;

        if(begin >= len)
        {
            return;
        }

        rocsparse::csrgemm_bitmap_mark_chunk(
            static_cast<I>(row_begin_R + begin),
            static_cast<I>(row_begin_R + rocsparse::min(begin + chunk, len)),
            csr_col_ind_R,
            idx_base_R,
            word_first[slot],
            word_offset[slot + 1] - word_offset[slot],
            bitmap + (word_offset[slot] - word_base));
    }

    // Clears the counts the sweep below accumulates into.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_clear_nnz_kernel(J nslot,
                                         const J* __restrict__ offset,
                                         const J* __restrict__ perm,
                                         I* __restrict__ row_nnz)
    {
        for(int64_t slot = static_cast<int64_t>(blockIdx.x) * BLOCKSIZE + threadIdx.x; slot < nslot;
            slot += static_cast<int64_t>(gridDim.x) * BLOCKSIZE)
        {
            row_nnz[perm[slot + *offset]] = 0;
        }
    }

    // Distinct column count of each row of the group, which is what the nnz stage owes its caller.
    // A group of a few rows would leave the device idle if a row were swept by one block, so the
    // words of a row are split over gridDim.y blocks that each contribute a partial count.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_count_kernel(J       slot_base,
                                     int64_t word_base,
                                     const int64_t* __restrict__ word_offset,
                                     const J* __restrict__ offset,
                                     const J* __restrict__ perm,
                                     const uint32_t* __restrict__ bitmap,
                                     I* __restrict__ row_nnz)
    {
        const J       slot   = slot_base + static_cast<J>(blockIdx.x);
        const int64_t piece  = blockIdx.y;
        const int64_t npiece = gridDim.y;

        const int64_t row_nword = word_offset[slot + 1] - word_offset[slot];

        if(row_nword == 0)
        {
            return;
        }

        const uint32_t* row_bits = bitmap + (word_offset[slot] - word_base);

        __shared__ I sdata[BLOCKSIZE];

        I sum = 0;
        for(int64_t w = piece * BLOCKSIZE + threadIdx.x; w < row_nword; w += npiece * BLOCKSIZE)
        {
            sum += static_cast<I>(__popc(row_bits[w]));
        }

        sdata[threadIdx.x] = sum;
        __syncthreads();

        rocsparse::blockreduce_sum<BLOCKSIZE>(threadIdx.x, sdata);

        if(threadIdx.x == 0)
        {
            rocsparse::atomic_add(&row_nnz[perm[slot + *offset]], sdata[0]);
        }
    }

    // Rank of a column within its row: how many marked columns precede it, which is exactly the
    // offset of that column inside the row of C. Monotonic in the column, so the row comes out in
    // ascending order without anything being sorted. A column the row does not hold, outside its
    // span or unmarked, has no rank.
    template <typename I, typename J>
    ROCSPARSE_DEVICE_ILF bool csrgemm_bitmap_rank(const uint32_t* __restrict__ row_bits,
                                                  const I* __restrict__ row_rank,
                                                  I       rank_origin,
                                                  J       lo_word,
                                                  int64_t row_nword,
                                                  J       col,
                                                  I&      rank)
    {
        const int64_t word = static_cast<int64_t>(col >> 5) - lo_word;

        if(word < 0 || word >= row_nword)
        {
            return false;
        }

        const uint32_t bits = row_bits[word];
        const uint32_t bit  = 1u << (col & 31);

        if((bits & bit) == 0u)
        {
            return false;
        }

        rank = row_rank[word] - rank_origin + static_cast<I>(__popc(bits & (bit - 1u)));

        return true;
    }

    // Population count of every word, which an exclusive scan turns into the rank index.
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_popcount_kernel(int64_t nword,
                                        const uint32_t* __restrict__ bitmap,
                                        I* __restrict__ rank)
    {
        const int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCKSIZE;

        for(int64_t w = static_cast<int64_t>(blockIdx.x) * BLOCKSIZE + threadIdx.x; w < nword;
            w += stride)
        {
            rank[w] = static_cast<I>(__popc(bitmap[w]));
        }
    }

    // Writes the column indices of C. carry is how many columns of the row the tiles before this
    // one took, zero for a row that is not walked in tiles.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_emit_kernel(J       slot_base,
                                    I       carry,
                                    int64_t word_base,
                                    const int64_t* __restrict__ word_offset,
                                    const J* __restrict__ word_first,
                                    const J* __restrict__ offset,
                                    const J* __restrict__ perm,
                                    const uint32_t* __restrict__ bitmap,
                                    const I* __restrict__ rank,
                                    const I* __restrict__ csr_row_ptr_C,
                                    rocsparse_index_base idx_base_C,
                                    J* __restrict__ csr_col_ind_C)
    {
        const J       slot   = slot_base + static_cast<J>(blockIdx.x);
        const int64_t piece  = blockIdx.y;
        const int64_t npiece = gridDim.y;

        const int64_t row_nword = word_offset[slot + 1] - word_offset[slot];

        if(row_nword == 0)
        {
            return;
        }

        const J row         = perm[slot + *offset];
        const I row_begin_C = csr_row_ptr_C[row] - idx_base_C;
        const I row_nnz_C   = csr_row_ptr_C[row + 1] - idx_base_C - row_begin_C;

        const uint32_t* row_bits       = bitmap + (word_offset[slot] - word_base);
        const I* __restrict__ row_rank = rank + (word_offset[slot] - word_base);
        const I       rank_origin      = row_rank[0] - carry;
        const int64_t lo_col           = static_cast<int64_t>(word_first[slot]) * 32;

        for(int64_t w = piece * BLOCKSIZE + threadIdx.x; w < row_nword; w += npiece * BLOCKSIZE)
        {
            uint32_t bits = row_bits[w];

            I pos = row_rank[w] - rank_origin;

            // A row pointer of C that does not match the products must not write past its row
            while(bits != 0u && pos < row_nnz_C)
            {
                const int bit = __ffs(bits) - 1;
                const J   col = static_cast<J>(lo_col + w * 32 + bit);

                csr_col_ind_C[row_begin_C + pos] = col + idx_base_C;

                ++pos;
                bits &= bits - 1u;
            }
        }
    }

    // Zeroes the accumulators of the group's rows of C.
    template <uint32_t BLOCKSIZE, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_clear_val_kernel(J slot_base,
                                         const J* __restrict__ offset,
                                         const J* __restrict__ perm,
                                         const I* __restrict__ csr_row_ptr_C,
                                         rocsparse_index_base idx_base_C,
                                         T* __restrict__ csr_val_C)
    {
        const J slot = slot_base + static_cast<J>(blockIdx.x);
        const J row  = perm[slot + *offset];

        const I row_begin_C = csr_row_ptr_C[row] - idx_base_C;
        const I row_end_C   = csr_row_ptr_C[row + 1] - idx_base_C;

        for(I k = row_begin_C + blockIdx.y * BLOCKSIZE + threadIdx.x; k < row_end_C;
            k += static_cast<I>(gridDim.y) * BLOCKSIZE)
        {
            csr_val_C[k] = static_cast<T>(0);
        }
    }

    // Adds every intermediate product straight into its entry of C. The rank index turns a column
    // into an offset in constant time, so nothing has to be expanded, sorted or reduced: the
    // products are consumed where they are produced. carry is as for the emit kernel.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_accumulate_kernel(J       m,
                                          int64_t entry_first,
                                          int64_t entry_last,
                                          J       slot_base,
                                          J       count,
                                          I       carry,
                                          int64_t word_base,
                                          const I* __restrict__ entry_offset,
                                          const int64_t* __restrict__ word_offset,
                                          const J* __restrict__ word_first,
                                          const J* __restrict__ offset,
                                          const J* __restrict__ perm,
                                          ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                          const I* __restrict__ csr_row_ptr_A,
                                          const J* __restrict__ csr_col_ind_A,
                                          const T* __restrict__ csr_val_A,
                                          rocsparse_index_base idx_base_A,
                                          const I* __restrict__ csr_row_ptr_B,
                                          const J* __restrict__ csr_col_ind_B,
                                          const T* __restrict__ csr_val_B,
                                          rocsparse_index_base idx_base_B,
                                          const uint32_t* __restrict__ bitmap,
                                          const I* __restrict__ rank,
                                          const I* __restrict__ csr_row_ptr_C,
                                          rocsparse_index_base idx_base_C,
                                          T* __restrict__ csr_val_C,
                                          bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        const int64_t nnz_C = csr_row_ptr_C[m] - idx_base_C;

        const int     lid    = threadIdx.x & (WFSIZE - 1);
        const int64_t nwf    = BLOCKSIZE / WFSIZE;
        const int64_t wf     = static_cast<int64_t>(blockIdx.x) * nwf + threadIdx.x / WFSIZE;
        const int64_t stride = static_cast<int64_t>(gridDim.x) * nwf;

        for(int64_t e = entry_first + wf; e < entry_last; e += stride)
        {
            J slot;
            I index;

            rocsparse::csrgemm_bitmap_entry(e,
                                            entry_last,
                                            slot_base,
                                            count,
                                            entry_offset,
                                            offset,
                                            perm,
                                            csr_row_ptr_A,
                                            idx_base_A,
                                            slot,
                                            index);

            const int64_t row_nword = word_offset[slot + 1] - word_offset[slot];

            if(row_nword == 0)
            {
                continue;
            }

            const J col_A = csr_col_ind_A[index] - idx_base_A;

            const J row         = perm[slot + *offset];
            const I row_begin_C = csr_row_ptr_C[row] - idx_base_C;
            const I row_nnz_C   = csr_row_ptr_C[row + 1] - idx_base_C - row_begin_C;

            const uint32_t* row_bits       = bitmap + (word_offset[slot] - word_base);
            const I* __restrict__ row_rank = rank + (word_offset[slot] - word_base);
            const I rank_origin            = row_rank[0] - carry;
            const J lo_word                = word_first[slot];

            const T val_A = alpha * csr_val_A[index];

            const I row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
            const I row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

            for(I k = row_begin_B + lid; k < row_end_B; k += WFSIZE)
            {
                const J col_B = csr_col_ind_B[k] - idx_base_B;

                I pos;
                if(rocsparse::csrgemm_bitmap_rank<I, J>(
                       row_bits, row_rank, rank_origin, lo_word, row_nword, col_B, pos)
                   && pos < row_nnz_C)
                {
                    rocsparse::atomic_add(
                        csr_val_C, row_begin_C + pos, nnz_C, val_A * csr_val_B[k]);
                }
            }
        }
    }

    // Adds the row of D into its places in C. A column of D is located by the same rank query that
    // places a product, so the two terms are accumulated through one index. carry is as for the
    // emit kernel.
    template <uint32_t BLOCKSIZE, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_bitmap_accumulate_D_kernel(J       m,
                                            J       slot_base,
                                            I       carry,
                                            int64_t word_base,
                                            const int64_t* __restrict__ word_offset,
                                            const J* __restrict__ word_first,
                                            const J* __restrict__ offset,
                                            const J* __restrict__ perm,
                                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                            const I* __restrict__ csr_row_ptr_D,
                                            const J* __restrict__ csr_col_ind_D,
                                            const T* __restrict__ csr_val_D,
                                            rocsparse_index_base idx_base_D,
                                            const uint32_t* __restrict__ bitmap,
                                            const I* __restrict__ rank,
                                            const I* __restrict__ csr_row_ptr_C,
                                            rocsparse_index_base idx_base_C,
                                            T* __restrict__ csr_val_C,
                                            bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        const int64_t nnz_C = csr_row_ptr_C[m] - idx_base_C;

        const J       slot   = slot_base + static_cast<J>(blockIdx.x);
        const int64_t piece  = blockIdx.y;
        const int64_t npiece = gridDim.y;

        const int64_t row_nword = word_offset[slot + 1] - word_offset[slot];

        if(row_nword == 0)
        {
            return;
        }

        const J row         = perm[slot + *offset];
        const I row_begin_C = csr_row_ptr_C[row] - idx_base_C;
        const I row_nnz_C   = csr_row_ptr_C[row + 1] - idx_base_C - row_begin_C;

        const uint32_t* row_bits       = bitmap + (word_offset[slot] - word_base);
        const I* __restrict__ row_rank = rank + (word_offset[slot] - word_base);
        const I rank_origin            = row_rank[0] - carry;
        const J lo_word                = word_first[slot];

        const I row_begin_D = csr_row_ptr_D[row] - idx_base_D;
        const I row_end_D   = csr_row_ptr_D[row + 1] - idx_base_D;

        for(I k = row_begin_D + static_cast<I>(piece * BLOCKSIZE + threadIdx.x); k < row_end_D;
            k += static_cast<I>(npiece * BLOCKSIZE))
        {
            const J col_D = csr_col_ind_D[k] - idx_base_D;

            I pos;
            if(rocsparse::csrgemm_bitmap_rank<I, J>(
                   row_bits, row_rank, rank_origin, lo_word, row_nword, col_D, pos)
               && pos < row_nnz_C)
            {
                rocsparse::atomic_add(csr_val_C, row_begin_C + pos, nnz_C, beta * csr_val_D[k]);
            }
        }
    }
}
