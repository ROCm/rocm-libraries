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

#include "rocsparse_csrgemm_bitmap.hpp"
#include "csrgemm_bitmap_device.h"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

#include <algorithm>
#include <vector>

namespace rocsparse
{
    static_assert(rocsparse::csrgemm_bitmap_tile_words_min <= rocsparse::csrgemm_bitmap_lds_words,
                  "The smallest tile of a row is marked in LDS.");

    // Grid-stride kernels launch at most this many blocks.
    inline constexpr int64_t csrgemm_bitmap_max_blocks = 65535;

    // Blocks per compute unit that the kernels splitting rows into pieces aim for.
    inline constexpr int64_t csrgemm_bitmap_blocks_per_cu = 4;

    // Most pieces the entries of A of one row are split into.
    inline constexpr int64_t csrgemm_bitmap_max_entry_pieces = 64;

    // Alignment of every block of the workspace.
    inline constexpr size_t csrgemm_bitmap_block_align = 256;

    // Bytes a block of the workspace takes, so that the next one starts aligned.
    inline size_t csrgemm_bitmap_block_bytes(size_t bytes)
    {
        return ((bytes + csrgemm_bitmap_block_align - 1) / csrgemm_bitmap_block_align)
               * csrgemm_bitmap_block_align;
    }

    template <typename I, typename J>
    csrgemm_bitmap_workspace<I, J>::csrgemm_bitmap_workspace(rocsparse_handle handle,
                                                             size_t           pass_bytes_max,
                                                             size_t           alloc_bytes_max)
        : handle_(handle)
        , pass_bytes_max_(pass_bytes_max)
        , alloc_bytes_max_(alloc_bytes_max)
    {
    }

    template <typename I, typename J>
    csrgemm_bitmap_workspace<I, J>::~csrgemm_bitmap_workspace()
    {
        // Stream ordered: the memory is released once the kernels already queued are done with it
        if(pass_memory_ != nullptr)
        {
            WARNING_IF_HIP_ERROR(rocsparse_hipFreeAsync(pass_memory_, handle_->stream));
        }

        if(layout_memory_ != nullptr)
        {
            WARNING_IF_HIP_ERROR(rocsparse_hipFreeAsync(layout_memory_, handle_->stream));
        }
    }

    template <typename I, typename J>
    rocsparse_status csrgemm_bitmap_workspace<I, J>::reserve_layout(J nslot)
    {
        if(nslot <= layout_rows_)
        {
            return rocsparse_status_success;
        }

        const size_t noffset = static_cast<size_t>(nslot) + 1;

        size_t entry_scan_size = 0;
        size_t word_scan_size  = 0;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<I, I>(
            handle_, static_cast<I>(0), noffset, &entry_scan_size)));
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::exclusive_scan_buffer_size<int64_t, int64_t>(
                handle_, int64_t{0}, noffset, &word_scan_size)));

        const size_t scan_size   = std::max(entry_scan_size, word_scan_size);
        const size_t entry_bytes = csrgemm_bitmap_block_bytes(sizeof(I) * noffset);
        const size_t word_bytes  = csrgemm_bitmap_block_bytes(sizeof(int64_t) * noffset);
        const size_t first_bytes
            = csrgemm_bitmap_block_bytes(sizeof(J) * static_cast<size_t>(nslot));

        if(layout_memory_ != nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(layout_memory_, handle_->stream));
            layout_memory_   = nullptr;
            layout_rows_     = 0;
            entry_offset     = nullptr;
            word_offset      = nullptr;
            word_first       = nullptr;
            offset_scan      = nullptr;
            offset_scan_size = 0;
        }

        void* memory = nullptr;
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            &memory, entry_bytes + word_bytes + first_bytes + scan_size, handle_->stream));

        char* const base = static_cast<char*>(memory);

        layout_memory_   = memory;
        layout_rows_     = nslot;
        entry_offset     = reinterpret_cast<I*>(base);
        word_offset      = reinterpret_cast<int64_t*>(base + entry_bytes);
        word_first       = reinterpret_cast<J*>(base + entry_bytes + word_bytes);
        offset_scan      = base + entry_bytes + word_bytes + first_bytes;
        offset_scan_size = scan_size;

        return rocsparse_status_success;
    }

    template <typename I, typename J>
    rocsparse_status csrgemm_bitmap_workspace<I, J>::reserve_pass(int64_t row_words,
                                                                  int64_t total_words,
                                                                  bool    with_rank)
    {
        const size_t  word_bytes = sizeof(uint32_t) + (with_rank ? sizeof(I) : 0);
        const int64_t cap_words  = static_cast<int64_t>(pass_bytes_max_ / word_bytes);

        int64_t words = std::max(row_words, std::min(total_words, cap_words));

        // Below row_words the widest row is walked in tiles, which only happens once it failed
        // whole.
        const int64_t floor_words = std::min(row_words, rocsparse::csrgemm_bitmap_tile_words_min);

        if(words == 0 || (words <= pass_words && (!with_rank || rank != nullptr)))
        {
            return rocsparse_status_success;
        }

        if(pass_memory_ != nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(pass_memory_, handle_->stream));
            pass_memory_   = nullptr;
            pass_words     = 0;
            bitmap         = nullptr;
            rank           = nullptr;
            rank_scan      = nullptr;
            rank_scan_size = 0;
        }

        for(;;)
        {
            if(words < row_words
               && (row_words - 1) / words + 1 > rocsparse::csrgemm_bitmap_tiles_max)
            {
                return rocsparse_status_memory_error;
            }

            const size_t nword = static_cast<size_t>(words);

            size_t scan_size = 0;
            if(with_rank)
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<I, I>(
                    handle_, static_cast<I>(0), nword, &scan_size)));
            }

            const size_t bitmap_bytes = csrgemm_bitmap_block_bytes(sizeof(uint32_t) * nword);
            const size_t rank_bytes = with_rank ? csrgemm_bitmap_block_bytes(sizeof(I) * nword) : 0;

            const size_t bytes = bitmap_bytes + rank_bytes + scan_size;

            void*            memory    = nullptr;
            const bool       simulated = bytes > alloc_bytes_max_;
            const hipError_t status
                = simulated ? hipErrorOutOfMemory
                            : rocsparse_hipMallocAsync(&memory, bytes, handle_->stream);

            if(status == hipErrorOutOfMemory)
            {
                // The failure stays the last error otherwise, and a later launch check reads it
                if(!simulated)
                {
                    (void)hipGetLastError();
                }

                if(words <= floor_words)
                {
                    return rocsparse_status_memory_error;
                }

                words = std::max(words > row_words ? row_words : floor_words, words / 2);
                continue;
            }

            RETURN_IF_HIP_ERROR(status);

            char* const base = static_cast<char*>(memory);

            pass_memory_   = memory;
            pass_words     = words;
            bitmap         = reinterpret_cast<uint32_t*>(base);
            rank           = with_rank ? reinterpret_cast<I*>(base + bitmap_bytes) : nullptr;
            rank_scan      = with_rank ? base + bitmap_bytes + rank_bytes : nullptr;
            rank_scan_size = scan_size;

            return rocsparse_status_success;
        }
    }

    // Where the entries of A of every row of the group start in the flat entry numbering, and
    // where its bitmap starts in the packing and which word it starts from; on the device for the
    // kernels and the offsets on the host to cut the group into passes.
    template <typename I, typename J>
    struct csrgemm_bitmap_layout
    {
        I*                   entry_offset = nullptr;
        int64_t*             word_offset  = nullptr;
        J*                   word_first   = nullptr;
        std::vector<I>       h_entry_offset;
        std::vector<int64_t> h_word_offset;
    };

    // Blocks per row for the kernels that split the entries of A of a row: enough to fill the
    // device.
    inline int64_t csrgemm_bitmap_entry_pieces(rocsparse_handle handle, int64_t count)
    {
        const int64_t target
            = rocsparse::csrgemm_bitmap_blocks_per_cu * handle->properties.multiProcessorCount;

        return std::clamp(
            (target - 1) / count + 1, int64_t{1}, rocsparse::csrgemm_bitmap_max_entry_pieces);
    }

    // Blocks for a kernel with a thread per item, capped at csrgemm_bitmap_max_blocks; the kernels
    // stride beyond it.
    template <uint32_t BLOCKSIZE>
    inline int64_t csrgemm_bitmap_item_blocks(int64_t nitem)
    {
        return rocsparse::min((nitem - 1) / BLOCKSIZE + 1, rocsparse::csrgemm_bitmap_max_blocks);
    }

    // Rows one launch of a kernel with a block per row can take: a grid dimension holds fewer
    // than 2^32 threads.
    template <uint32_t BLOCKSIZE>
    inline constexpr int64_t csrgemm_bitmap_launch_rows = (int64_t{1} << 32) / BLOCKSIZE - 1;

    // The rows of a group: nslot of them, found through the permutation from the group's offset.
    template <typename J>
    struct csrgemm_bitmap_group
    {
        J        nslot;
        const J* offset;
        const J* perm;
    };

    // The sparsity pattern of a matrix; a null row_ptr leaves the matrix out.
    template <typename I, typename J>
    struct csrgemm_bitmap_csr
    {
        const I*             row_ptr;
        const J*             col_ind;
        rocsparse_index_base base;
    };

    // The scalars and values of the terms alpha * A * B and beta * D a pass adds into C.
    template <typename T>
    struct csrgemm_bitmap_values
    {
        const T* alpha;
        const T* val_A;
        const T* val_B;
        const T* beta;
        const T* val_D;
    };

    // The spans come from R, the rows of D or of C that join the occupancy, and, when
    // mark_products is set, from the products of A * B. Once they are known on the host the
    // workspace is given a pass for them, with a rank index when with_rank is set.
    template <typename I, typename J>
    rocsparse_status csrgemm_bitmap_build_layout(rocsparse_handle                handle,
                                                 csrgemm_bitmap_workspace<I, J>& workspace,
                                                 J                               n,
                                                 const csrgemm_bitmap_group<J>&  group,
                                                 bool                            mark_products,
                                                 bool                            with_rank,
                                                 const csrgemm_bitmap_csr<I, J>& A,
                                                 const csrgemm_bitmap_csr<I, J>& B,
                                                 const csrgemm_bitmap_csr<I, J>& R,
                                                 csrgemm_bitmap_layout<I, J>&    layout)
    {
        hipStream_t stream = handle->stream;

        const J nslot = group.nslot;

        static constexpr uint32_t BLOCKSIZE = 256;

        const size_t noffset = static_cast<size_t>(nslot) + 1;

        RETURN_IF_ROCSPARSE_ERROR(workspace.reserve_layout(nslot));

        layout.entry_offset = workspace.entry_offset;
        layout.word_offset  = workspace.word_offset;
        layout.word_first   = workspace.word_first;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_bitmap_row_setup_kernel<BLOCKSIZE, I, J>),
            dim3(rocsparse::csrgemm_bitmap_item_blocks<BLOCKSIZE>(static_cast<int64_t>(nslot) + 1)),
            dim3(BLOCKSIZE),
            0,
            stream,
            nslot,
            n,
            group.offset,
            group.perm,
            A.row_ptr,
            R.row_ptr,
            R.col_ind,
            R.base,
            layout.entry_offset,
            layout.word_first,
            layout.word_offset);

        if(mark_products)
        {
            for(J base = 0; base < nslot;)
            {
                const J count = static_cast<J>(
                    rocsparse::min(static_cast<int64_t>(nslot - base),
                                   rocsparse::csrgemm_bitmap_launch_rows<BLOCKSIZE>));

                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrgemm_bitmap_span_kernel<BLOCKSIZE, I, J>),
                    dim3(count, rocsparse::csrgemm_bitmap_entry_pieces(handle, count)),
                    dim3(BLOCKSIZE),
                    0,
                    stream,
                    base,
                    group.offset,
                    group.perm,
                    A.row_ptr,
                    A.col_ind,
                    A.base,
                    B.row_ptr,
                    B.col_ind,
                    B.base,
                    layout.word_first,
                    layout.word_offset);

                base += count;
            }
        }

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_bitmap_span_words_kernel<BLOCKSIZE, J>),
            dim3(rocsparse::csrgemm_bitmap_item_blocks<BLOCKSIZE>(nslot)),
            dim3(BLOCKSIZE),
            0,
            stream,
            nslot,
            layout.word_first,
            layout.word_offset);

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan(handle,
                                                                         layout.entry_offset,
                                                                         layout.entry_offset,
                                                                         static_cast<I>(0),
                                                                         noffset,
                                                                         workspace.offset_scan_size,
                                                                         workspace.offset_scan)));
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan(handle,
                                                                         layout.word_offset,
                                                                         layout.word_offset,
                                                                         int64_t{0},
                                                                         noffset,
                                                                         workspace.offset_scan_size,
                                                                         workspace.offset_scan)));

        layout.h_entry_offset.resize(noffset);
        layout.h_word_offset.resize(noffset);

        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(layout.h_entry_offset.data(),
                                                     layout.entry_offset,
                                                     sizeof(I) * noffset,
                                                     hipMemcpyDeviceToHost,
                                                     stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(layout.h_word_offset.data(),
                                                     layout.word_offset,
                                                     sizeof(int64_t) * noffset,
                                                     hipMemcpyDeviceToHost,
                                                     stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        int64_t row_words = 0;
        for(J slot = 0; slot < nslot; ++slot)
        {
            row_words
                = std::max(row_words, layout.h_word_offset[slot + 1] - layout.h_word_offset[slot]);
        }

        RETURN_IF_ROCSPARSE_ERROR(
            workspace.reserve_pass(row_words, layout.h_word_offset[nslot], with_rank));

        return rocsparse_status_success;
    }

    // Blocks per row for the kernels that split the words of a row: enough to fill the device,
    // never so many that a block sweeps less than one block's worth of words.
    template <uint32_t BLOCKSIZE>
    inline int64_t csrgemm_bitmap_word_pieces(rocsparse_handle handle, int64_t count, int64_t nword)
    {
        const int64_t target
            = rocsparse::csrgemm_bitmap_blocks_per_cu * handle->properties.multiProcessorCount;
        const int64_t word_blocks = (nword - 1) / BLOCKSIZE + 1;

        return std::clamp((target - 1) / count + 1, int64_t{1}, word_blocks);
    }

    // A subgroup of WFSIZE lanes per entry of A, capped at csrgemm_bitmap_max_blocks; the kernels
    // stride beyond it.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE>
    inline int64_t csrgemm_bitmap_entry_blocks(int64_t nentry)
    {
        return rocsparse::min((nentry - 1) / (BLOCKSIZE / WFSIZE) + 1,
                              rocsparse::csrgemm_bitmap_max_blocks);
    }

    // The rows [base, base + count) of the group one pass takes: as many as fit in pass_words and
    // in one launch with a block per row, at least one. Their bitmaps are words
    // [word_base, word_base + nword) of the packing, the widest is row_words, and npiece
    // blocks sweep it; the kernels that walk the entries of a row of R, C or D split them over as
    // many blocks. lds_rows of them are marked in LDS. A row wider than the pass is alone in
    // it and walked in tiles, a pass each: carry is how many of its columns the tiles before
    // found, first_tile is set on its first tile and last_tile on its last. A pass that is not a
    // tile has no carry and counts as both.
    template <typename J>
    struct csrgemm_bitmap_pass
    {
        J       base       = 0;
        J       count      = 0;
        int64_t word_base  = 0;
        int64_t nword      = 0;
        int64_t row_words  = 0;
        J       lds_rows   = 0;
        int64_t npiece     = 0;
        int64_t carry      = 0;
        bool    first_tile = true;
        bool    last_tile  = true;
    };

    template <uint32_t BLOCKSIZE, typename J>
    csrgemm_bitmap_pass<J> csrgemm_bitmap_next_pass(rocsparse_handle            handle,
                                                    const std::vector<int64_t>& h_word_offset,
                                                    J                           base,
                                                    J                           nslot,
                                                    int64_t                     pass_words)
    {
        const auto first = h_word_offset.begin();
        const auto limit = std::upper_bound(
            first + base + 1, first + nslot + 1, h_word_offset[base] + pass_words);
        const J end = static_cast<J>(
            std::min(std::max<int64_t>(limit - first - 1, int64_t{base} + 1),
                     int64_t{base} + rocsparse::csrgemm_bitmap_launch_rows<BLOCKSIZE>));

        int64_t row_words = 0;
        J       lds_rows  = 0;
        for(J slot = base; slot < end; ++slot)
        {
            const int64_t words = h_word_offset[slot + 1] - h_word_offset[slot];

            row_words = std::max(row_words, words);
            lds_rows += rocsparse::csrgemm_bitmap_marked_in_lds(words);
        }

        csrgemm_bitmap_pass<J> pass;

        pass.base      = base;
        pass.count     = end - base;
        pass.word_base = h_word_offset[base];
        pass.nword     = h_word_offset[end] - h_word_offset[base];
        pass.row_words = row_words;
        pass.lds_rows  = lds_rows;
        pass.npiece
            = rocsparse::csrgemm_bitmap_word_pieces<BLOCKSIZE>(handle, pass.count, row_words);

        return pass;
    }

    // Walks the row of a pass wider than pass_words in tiles of it. The layout is pointed at one
    // tile at a time; the kernels drop the columns outside the tile when marking and give them no
    // rank, so each tile holds exactly its own columns. Pointing it at a tile moves the start of
    // the next row, so the layout is restored before returning, whatever the tiles returned. tile
    // reports how many columns it found, which the tiles after it carry; the last tile need not.
    template <uint32_t BLOCKSIZE, typename I, typename J, typename Tile>
    rocsparse_status csrgemm_bitmap_walk_tiles(rocsparse_handle              handle,
                                               csrgemm_bitmap_layout<I, J>&  layout,
                                               const csrgemm_bitmap_pass<J>& pass,
                                               int64_t                       pass_words,
                                               Tile&&                        tile)
    {
        hipStream_t stream = handle->stream;

        const J       slot      = pass.base;
        const int64_t row_begin = layout.h_word_offset[slot];
        const int64_t row_end   = layout.h_word_offset[slot + 1];

        J lo_word = 0;
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
            &lo_word, layout.word_first + slot, sizeof(J), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        const auto point = [&](J first, int64_t end) -> rocsparse_status {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_bitmap_set_tile_kernel<J>),
                                               dim3(1),
                                               dim3(1),
                                               0,
                                               stream,
                                               slot,
                                               first,
                                               end,
                                               layout.word_first,
                                               layout.word_offset);
            return rocsparse_status_success;
        };

        const rocsparse_status status = [&]() -> rocsparse_status {
            rocsparse::csrgemm_bitmap_pass<J> part = pass;

            for(int64_t done = 0; done < row_end - row_begin; done += pass_words)
            {
                const int64_t nword = std::min(pass_words, row_end - row_begin - done);

                RETURN_IF_ROCSPARSE_ERROR(point(static_cast<J>(lo_word + done), row_begin + nword));

                part.nword     = nword;
                part.row_words = nword;
                part.lds_rows  = rocsparse::csrgemm_bitmap_marked_in_lds(nword);
                part.npiece    = rocsparse::csrgemm_bitmap_word_pieces<BLOCKSIZE>(handle, 1, nword);
                part.first_tile = (done == 0);
                part.last_tile  = (done + nword == row_end - row_begin);

                int64_t found = 0;
                RETURN_IF_ROCSPARSE_ERROR(tile(part, found));

                part.carry += found;
            }

            return rocsparse_status_success;
        }();

        const rocsparse_status restored = point(lo_word, row_end);

        return status != rocsparse_status_success ? status : restored;
    }

    // Walks the group a pass at a time and hands every pass to step, a row wider than pass_words
    // one tile at a time. step reports how many columns a tile found for the tiles after it; a
    // pass that is not a tile need not.
    template <uint32_t BLOCKSIZE, typename I, typename J, typename Step>
    rocsparse_status csrgemm_bitmap_walk(rocsparse_handle             handle,
                                         csrgemm_bitmap_layout<I, J>& layout,
                                         J                            nslot,
                                         int64_t                      pass_words,
                                         Step&&                       step)
    {
        for(J base = 0; base < nslot;)
        {
            const rocsparse::csrgemm_bitmap_pass<J> pass
                = rocsparse::csrgemm_bitmap_next_pass<BLOCKSIZE>(
                    handle, layout.h_word_offset, base, nslot, pass_words);

            if(pass.row_words > pass_words)
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrgemm_bitmap_walk_tiles<BLOCKSIZE>(
                    handle, layout, pass, pass_words, step)));
            }
            else
            {
                int64_t found = 0;
                RETURN_IF_ROCSPARSE_ERROR(step(pass, found));
            }

            base += pass.count;
        }

        return rocsparse_status_success;
    }

    // Marks the products of a pass: in LDS for the rows whose span fits there, in the pass
    // directly for the others.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    rocsparse_status csrgemm_bitmap_mark_products(rocsparse_handle                   handle,
                                                  const csrgemm_bitmap_layout<I, J>& layout,
                                                  const csrgemm_bitmap_pass<J>&      pass,
                                                  const csrgemm_bitmap_group<J>&     group,
                                                  const csrgemm_bitmap_csr<I, J>&    A,
                                                  const csrgemm_bitmap_csr<I, J>&    B,
                                                  uint32_t*                          bitmap)
    {
        hipStream_t stream = handle->stream;

        const int64_t entry_first = layout.h_entry_offset[pass.base];
        const int64_t entry_last  = layout.h_entry_offset[pass.base + pass.count];

        if(entry_last <= entry_first)
        {
            return rocsparse_status_success;
        }

        if(pass.lds_rows > 0)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgemm_bitmap_mark_lds_kernel<BLOCKSIZE, WFSIZE, I, J>),
                dim3(pass.count, rocsparse::csrgemm_bitmap_entry_pieces(handle, pass.count)),
                dim3(BLOCKSIZE),
                0,
                stream,
                pass.base,
                pass.word_base,
                layout.word_offset,
                layout.word_first,
                group.offset,
                group.perm,
                A.row_ptr,
                A.col_ind,
                A.base,
                B.row_ptr,
                B.col_ind,
                B.base,
                bitmap);
        }

        if(rocsparse::csrgemm_bitmap_marked_in_pass(pass.row_words))
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgemm_bitmap_mark_kernel<BLOCKSIZE, WFSIZE, I, J>),
                dim3(rocsparse::csrgemm_bitmap_entry_blocks<BLOCKSIZE, WFSIZE>(entry_last
                                                                               - entry_first)),
                dim3(BLOCKSIZE),
                0,
                stream,
                entry_first,
                entry_last,
                pass.base,
                pass.count,
                pass.word_base,
                layout.entry_offset,
                layout.word_offset,
                layout.word_first,
                group.offset,
                group.perm,
                A.row_ptr,
                A.col_ind,
                A.base,
                B.row_ptr,
                B.col_ind,
                B.base,
                bitmap);
        }

        return rocsparse_status_success;
    }

    // Clears the bitmap of a pass and marks its occupancy: the products of A * B when
    // mark_products is set, and the rows of R when R is given.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    rocsparse_status csrgemm_bitmap_mark_pass(rocsparse_handle                   handle,
                                              const csrgemm_bitmap_layout<I, J>& layout,
                                              const csrgemm_bitmap_pass<J>&      pass,
                                              const csrgemm_bitmap_group<J>&     group,
                                              bool                               mark_products,
                                              const csrgemm_bitmap_csr<I, J>&    A,
                                              const csrgemm_bitmap_csr<I, J>&    B,
                                              const csrgemm_bitmap_csr<I, J>&    R,
                                              uint32_t*                          bitmap)
    {
        hipStream_t stream = handle->stream;

        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(
            bitmap, 0, sizeof(uint32_t) * static_cast<size_t>(pass.nword), stream));

        if(mark_products)
        {
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrgemm_bitmap_mark_products<BLOCKSIZE, WFSIZE>(
                handle, layout, pass, group, A, B, bitmap)));
        }

        if(R.row_ptr != nullptr)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgemm_bitmap_mark_rows_kernel<BLOCKSIZE, I, J>),
                dim3(pass.count, pass.npiece),
                dim3(BLOCKSIZE),
                0,
                stream,
                pass.base,
                pass.word_base,
                layout.word_offset,
                layout.word_first,
                group.offset,
                group.perm,
                R.row_ptr,
                R.col_ind,
                R.base,
                bitmap);
        }

        return rocsparse_status_success;
    }

    // Group 10 of the nnz stage: the distinct column count of every row, from the occupancy of its
    // products and, with D, of its row of D.
    template <typename I, typename J>
    rocsparse_status csrgemm_nnz_bitmap(rocsparse_handle                handle,
                                        csrgemm_bitmap_workspace<I, J>& workspace,
                                        J                               n,
                                        J                               nslot,
                                        const J*                        d_group_offset,
                                        const J*                        d_perm,
                                        const I*                        csr_row_ptr_A,
                                        const J*                        csr_col_ind_A,
                                        rocsparse_index_base            idx_base_A,
                                        const I*                        csr_row_ptr_B,
                                        const J*                        csr_col_ind_B,
                                        rocsparse_index_base            idx_base_B,
                                        bool                            add,
                                        const I*                        csr_row_ptr_D,
                                        const J*                        csr_col_ind_D,
                                        rocsparse_index_base            idx_base_D,
                                        I*                              row_nnz)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hipStream_t stream = handle->stream;

        static constexpr uint32_t BLOCKSIZE = 256;
        static constexpr uint32_t WFSIZE    = 16;

        if(nslot <= 0 || n <= 0)
        {
            return rocsparse_status_success;
        }

        const csrgemm_bitmap_group<J>  group{nslot, d_group_offset, d_perm};
        const csrgemm_bitmap_csr<I, J> A{csr_row_ptr_A, csr_col_ind_A, idx_base_A};
        const csrgemm_bitmap_csr<I, J> B{csr_row_ptr_B, csr_col_ind_B, idx_base_B};
        const csrgemm_bitmap_csr<I, J> D{add ? csr_row_ptr_D : nullptr, csr_col_ind_D, idx_base_D};

        // A group whose rows of A are all empty still has D to count, so it is walked too. The
        // count needs no rank index, so a pass holds more words.
        rocsparse::csrgemm_bitmap_layout<I, J> layout;

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrgemm_bitmap_build_layout(
            handle, workspace, n, group, true, false, A, B, D, layout)));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_bitmap_clear_nnz_kernel<BLOCKSIZE, I, J>),
            dim3(rocsparse::csrgemm_bitmap_item_blocks<BLOCKSIZE>(nslot)),
            dim3(BLOCKSIZE),
            0,
            stream,
            nslot,
            group.offset,
            group.perm,
            row_nnz);

        // The count kernel adds into row_nnz, so the tiles of a row need nothing carried
        return rocsparse::csrgemm_bitmap_walk<BLOCKSIZE>(
            handle,
            layout,
            nslot,
            workspace.pass_words,
            [&](const rocsparse::csrgemm_bitmap_pass<J>& pass, int64_t&) -> rocsparse_status {
                if(pass.nword == 0)
                {
                    return rocsparse_status_success;
                }

                RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrgemm_bitmap_mark_pass<BLOCKSIZE, WFSIZE>(
                    handle, layout, pass, group, true, A, B, D, workspace.bitmap)));

                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrgemm_bitmap_count_kernel<BLOCKSIZE, I, J>),
                    dim3(pass.count, pass.npiece),
                    dim3(BLOCKSIZE),
                    0,
                    stream,
                    pass.base,
                    pass.word_base,
                    layout.word_offset,
                    group.offset,
                    group.perm,
                    workspace.bitmap,
                    row_nnz);

                return rocsparse_status_success;
            });
    }

    // Builds the bitmap of one pass and scans it into the rank index. The calc and symbolic
    // stages take both terms of D + alpha * A * B, the numeric stage takes C alone.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    rocsparse_status csrgemm_bitmap_rank_pass(rocsparse_handle                      handle,
                                              const csrgemm_bitmap_workspace<I, J>& workspace,
                                              const csrgemm_bitmap_layout<I, J>&    layout,
                                              const csrgemm_bitmap_pass<J>&         pass,
                                              const csrgemm_bitmap_group<J>&        group,
                                              bool                                  mark_products,
                                              const csrgemm_bitmap_csr<I, J>&       A,
                                              const csrgemm_bitmap_csr<I, J>&       B,
                                              const csrgemm_bitmap_csr<I, J>&       R)
    {
        hipStream_t stream = handle->stream;

        const int64_t nword = pass.nword;

        if(nword == 0)
        {
            return rocsparse_status_success;
        }

        // R joins the occupancy before it is counted, so the rank index spans it too
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrgemm_bitmap_mark_pass<BLOCKSIZE, WFSIZE>(
            handle, layout, pass, group, mark_products, A, B, R, workspace.bitmap)));

        const int64_t word_blocks = rocsparse::csrgemm_bitmap_item_blocks<BLOCKSIZE>(nword);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_bitmap_popcount_kernel<BLOCKSIZE, I>),
            dim3(word_blocks),
            dim3(BLOCKSIZE),
            0,
            stream,
            nword,
            workspace.bitmap,
            workspace.rank);

        // Every row subtracts its own rank_origin from the rank, so scanning a pass at a time
        // gives the same offsets a scan over the whole group would
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan(handle,
                                                                         workspace.rank,
                                                                         workspace.rank,
                                                                         static_cast<I>(0),
                                                                         nword,
                                                                         workspace.rank_scan_size,
                                                                         workspace.rank_scan)));

        return rocsparse_status_success;
    }

    // How many columns the pass just scanned holds: the rank of its last word plus that word's own.
    template <typename I, typename J>
    rocsparse_status csrgemm_bitmap_pass_columns(rocsparse_handle                      handle,
                                                 const csrgemm_bitmap_workspace<I, J>& workspace,
                                                 const csrgemm_bitmap_pass<J>&         pass,
                                                 int64_t&                              found)
    {
        const size_t last      = static_cast<size_t>(pass.nword - 1);
        I            last_rank = 0;
        uint32_t     last_bits = 0;

        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
            &last_rank, workspace.rank + last, sizeof(I), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(&last_bits,
                                                     workspace.bitmap + last,
                                                     sizeof(uint32_t),
                                                     hipMemcpyDeviceToHost,
                                                     handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));

        found = static_cast<int64_t>(last_rank) + __builtin_popcount(last_bits);

        return rocsparse_status_success;
    }

    // Walks the group a pass at a time: builds the bitmap and rank index of every pass and hands
    // them to consume(layout, pass). The occupancy comes from the products of A * B when
    // mark_products is set and from the rows of R when R is given, and so do the spans. A row
    // walked in tiles is handed over one tile at a time, so consume places the columns it finds
    // after the pass's carry, and clears a row of C only on its first tile.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J, typename Consume>
    rocsparse_status csrgemm_bitmap_rank_walk(rocsparse_handle                handle,
                                              csrgemm_bitmap_workspace<I, J>& workspace,
                                              J                               n,
                                              const csrgemm_bitmap_group<J>&  group,
                                              bool                            mark_products,
                                              const csrgemm_bitmap_csr<I, J>& A,
                                              const csrgemm_bitmap_csr<I, J>& B,
                                              const csrgemm_bitmap_csr<I, J>& R,
                                              Consume&&                       consume)
    {
        rocsparse::csrgemm_bitmap_layout<I, J> layout;

        // A group whose rows of A are all empty still has D or C to place, so it is walked too
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrgemm_bitmap_build_layout(
            handle, workspace, n, group, mark_products, true, A, B, R, layout)));

        return rocsparse::csrgemm_bitmap_walk<BLOCKSIZE>(
            handle,
            layout,
            group.nslot,
            workspace.pass_words,
            [&](const rocsparse::csrgemm_bitmap_pass<J>& pass, int64_t& found) -> rocsparse_status {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrgemm_bitmap_rank_pass<BLOCKSIZE, WFSIZE>(
                    handle, workspace, layout, pass, group, mark_products, A, B, R)));
                RETURN_IF_ROCSPARSE_ERROR(consume(layout, pass));

                if(!pass.last_tile)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::csrgemm_bitmap_pass_columns(handle, workspace, pass, found));
                }

                return rocsparse_status_success;
            });
    }

    // Writes the columns of the pass's rows of C through its rank index. A row walked in tiles
    // takes the columns of a tile after the carry of the tiles before it.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    rocsparse_status csrgemm_bitmap_emit_pass(rocsparse_handle                      handle,
                                              const csrgemm_bitmap_workspace<I, J>& workspace,
                                              const csrgemm_bitmap_layout<I, J>&    layout,
                                              const csrgemm_bitmap_pass<J>&         pass,
                                              const csrgemm_bitmap_group<J>&        group,
                                              const I*                              csr_row_ptr_C,
                                              J*                                    csr_col_ind_C,
                                              rocsparse_index_base                  idx_base_C)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_bitmap_emit_kernel<BLOCKSIZE, I, J>),
                                           dim3(pass.count, pass.npiece),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           pass.base,
                                           static_cast<I>(pass.carry),
                                           pass.word_base,
                                           layout.word_offset,
                                           layout.word_first,
                                           group.offset,
                                           group.perm,
                                           workspace.bitmap,
                                           workspace.rank,
                                           csr_row_ptr_C,
                                           idx_base_C,
                                           csr_col_ind_C);

        return rocsparse_status_success;
    }

    // Clears the pass's rows of C and adds both terms into them through its rank index. A row
    // walked in tiles is cleared by its first tile only.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J, typename T>
    rocsparse_status csrgemm_bitmap_values_pass(rocsparse_handle                      handle,
                                                const csrgemm_bitmap_workspace<I, J>& workspace,
                                                const csrgemm_bitmap_layout<I, J>&    layout,
                                                const csrgemm_bitmap_pass<J>&         pass,
                                                const csrgemm_bitmap_group<J>&        group,
                                                const csrgemm_bitmap_csr<I, J>&       A,
                                                const csrgemm_bitmap_csr<I, J>&       B,
                                                const csrgemm_bitmap_csr<I, J>&       D,
                                                const csrgemm_bitmap_values<T>&       values,
                                                J                                     m,
                                                const I*                              csr_row_ptr_C,
                                                T*                                    csr_val_C,
                                                rocsparse_index_base                  idx_base_C)
    {
        hipStream_t stream = handle->stream;

        const bool is_host_mode = handle->pointer_mode == rocsparse_pointer_mode_host;

        const int64_t entry_first = layout.h_entry_offset[pass.base];
        const int64_t entry_last  = layout.h_entry_offset[pass.base + pass.count];

        if(pass.first_tile)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgemm_bitmap_clear_val_kernel<BLOCKSIZE, I, J, T>),
                dim3(pass.count, pass.npiece),
                dim3(BLOCKSIZE),
                0,
                stream,
                pass.base,
                group.offset,
                group.perm,
                csr_row_ptr_C,
                idx_base_C,
                csr_val_C);
        }

        if(pass.nword == 0)
        {
            return rocsparse_status_success;
        }

        if(entry_last > entry_first)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgemm_bitmap_accumulate_kernel<BLOCKSIZE, WFSIZE, I, J, T>),
                dim3(rocsparse::csrgemm_bitmap_entry_blocks<BLOCKSIZE, WFSIZE>(entry_last
                                                                               - entry_first)),
                dim3(BLOCKSIZE),
                0,
                stream,
                m,
                entry_first,
                entry_last,
                pass.base,
                pass.count,
                static_cast<I>(pass.carry),
                pass.word_base,
                layout.entry_offset,
                layout.word_offset,
                layout.word_first,
                group.offset,
                group.perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, values.alpha),
                A.row_ptr,
                A.col_ind,
                values.val_A,
                A.base,
                B.row_ptr,
                B.col_ind,
                values.val_B,
                B.base,
                workspace.bitmap,
                workspace.rank,
                csr_row_ptr_C,
                idx_base_C,
                csr_val_C,
                is_host_mode);
        }

        if(D.row_ptr != nullptr)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgemm_bitmap_accumulate_D_kernel<BLOCKSIZE, I, J, T>),
                dim3(pass.count, pass.npiece),
                dim3(BLOCKSIZE),
                0,
                stream,
                m,
                pass.base,
                static_cast<I>(pass.carry),
                pass.word_base,
                layout.word_offset,
                layout.word_first,
                group.offset,
                group.perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, values.beta),
                D.row_ptr,
                D.col_ind,
                values.val_D,
                D.base,
                workspace.bitmap,
                workspace.rank,
                csr_row_ptr_C,
                idx_base_C,
                csr_val_C,
                is_host_mode);
        }

        return rocsparse_status_success;
    }

    // Group 10 of the calc stage, which owes both the columns and the values of C.
    template <typename I, typename J, typename T>
    rocsparse_status csrgemm_calc_bitmap(rocsparse_handle                handle,
                                         csrgemm_bitmap_workspace<I, J>& workspace,
                                         J                               m,
                                         J                               n,
                                         J                               nslot,
                                         const J*                        d_group_offset,
                                         const J*                        d_perm,
                                         const T*                        alpha_device_host,
                                         const I*                        csr_row_ptr_A,
                                         const J*                        csr_col_ind_A,
                                         const T*                        csr_val_A,
                                         rocsparse_index_base            idx_base_A,
                                         const I*                        csr_row_ptr_B,
                                         const J*                        csr_col_ind_B,
                                         const T*                        csr_val_B,
                                         rocsparse_index_base            idx_base_B,
                                         bool                            add,
                                         const T*                        beta_device_host,
                                         const I*                        csr_row_ptr_D,
                                         const J*                        csr_col_ind_D,
                                         const T*                        csr_val_D,
                                         rocsparse_index_base            idx_base_D,
                                         const I*                        csr_row_ptr_C,
                                         J*                              csr_col_ind_C,
                                         T*                              csr_val_C,
                                         rocsparse_index_base            idx_base_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        static constexpr uint32_t BLOCKSIZE = 256;
        static constexpr uint32_t WFSIZE    = 16;

        if(nslot <= 0 || n <= 0)
        {
            return rocsparse_status_success;
        }

        const csrgemm_bitmap_group<J>  group{nslot, d_group_offset, d_perm};
        const csrgemm_bitmap_csr<I, J> A{csr_row_ptr_A, csr_col_ind_A, idx_base_A};
        const csrgemm_bitmap_csr<I, J> B{csr_row_ptr_B, csr_col_ind_B, idx_base_B};
        const csrgemm_bitmap_csr<I, J> D{add ? csr_row_ptr_D : nullptr, csr_col_ind_D, idx_base_D};
        const csrgemm_bitmap_values<T> values{
            alpha_device_host, csr_val_A, csr_val_B, beta_device_host, csr_val_D};

        return rocsparse::csrgemm_bitmap_rank_walk<BLOCKSIZE, WFSIZE>(
            handle,
            workspace,
            n,
            group,
            true,
            A,
            B,
            D,
            [&](const auto& layout, const auto& pass) -> rocsparse_status {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::csrgemm_bitmap_emit_pass<BLOCKSIZE>(handle,
                                                                    workspace,
                                                                    layout,
                                                                    pass,
                                                                    group,
                                                                    csr_row_ptr_C,
                                                                    csr_col_ind_C,
                                                                    idx_base_C)));

                return rocsparse::csrgemm_bitmap_values_pass<BLOCKSIZE, WFSIZE>(handle,
                                                                                workspace,
                                                                                layout,
                                                                                pass,
                                                                                group,
                                                                                A,
                                                                                B,
                                                                                D,
                                                                                values,
                                                                                m,
                                                                                csr_row_ptr_C,
                                                                                csr_val_C,
                                                                                idx_base_C);
            });
    }

    // Group 10 of the symbolic stage. The row pointers of C are known, so only its columns are
    // owed: the calc stage without the values.
    template <typename I, typename J>
    rocsparse_status csrgemm_symbolic_bitmap(rocsparse_handle                handle,
                                             csrgemm_bitmap_workspace<I, J>& workspace,
                                             J                               n,
                                             J                               nslot,
                                             const J*                        d_group_offset,
                                             const J*                        d_perm,
                                             const I*                        csr_row_ptr_A,
                                             const J*                        csr_col_ind_A,
                                             rocsparse_index_base            idx_base_A,
                                             const I*                        csr_row_ptr_B,
                                             const J*                        csr_col_ind_B,
                                             rocsparse_index_base            idx_base_B,
                                             bool                            add,
                                             const I*                        csr_row_ptr_D,
                                             const J*                        csr_col_ind_D,
                                             rocsparse_index_base            idx_base_D,
                                             const I*                        csr_row_ptr_C,
                                             J*                              csr_col_ind_C,
                                             rocsparse_index_base            idx_base_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        static constexpr uint32_t BLOCKSIZE = 256;
        static constexpr uint32_t WFSIZE    = 16;

        if(nslot <= 0 || n <= 0)
        {
            return rocsparse_status_success;
        }

        const csrgemm_bitmap_group<J>  group{nslot, d_group_offset, d_perm};
        const csrgemm_bitmap_csr<I, J> A{csr_row_ptr_A, csr_col_ind_A, idx_base_A};
        const csrgemm_bitmap_csr<I, J> B{csr_row_ptr_B, csr_col_ind_B, idx_base_B};
        const csrgemm_bitmap_csr<I, J> D{add ? csr_row_ptr_D : nullptr, csr_col_ind_D, idx_base_D};

        return rocsparse::csrgemm_bitmap_rank_walk<BLOCKSIZE, WFSIZE>(
            handle, workspace, n, group, true, A, B, D, [&](const auto& layout, const auto& pass) {
                return rocsparse::csrgemm_bitmap_emit_pass<BLOCKSIZE>(handle,
                                                                      workspace,
                                                                      layout,
                                                                      pass,
                                                                      group,
                                                                      csr_row_ptr_C,
                                                                      csr_col_ind_C,
                                                                      idx_base_C);
            });
    }

    // Rows of the numeric stage that do not fit an LDS hash table: group 10, and the rows of groups
    // 6 to 9 whose table outgrows LDS for a wide value type. The pattern of C is given, so it is
    // marked instead of being rebuilt from the products: the rank of a column is then its place in
    // C by construction, and the products are walked once, to be accumulated. This relies on C
    // holding the sorted pattern the symbolic stage produced; a product whose column C does not
    // hold is dropped.
    template <typename I, typename J, typename T>
    rocsparse_status csrgemm_numeric_bitmap(rocsparse_handle                handle,
                                            csrgemm_bitmap_workspace<I, J>& workspace,
                                            J                               m,
                                            J                               n,
                                            J                               nslot,
                                            const J*                        d_group_offset,
                                            const J*                        d_perm,
                                            const T*                        alpha_device_host,
                                            const I*                        csr_row_ptr_A,
                                            const J*                        csr_col_ind_A,
                                            const T*                        csr_val_A,
                                            rocsparse_index_base            idx_base_A,
                                            const I*                        csr_row_ptr_B,
                                            const J*                        csr_col_ind_B,
                                            const T*                        csr_val_B,
                                            rocsparse_index_base            idx_base_B,
                                            bool                            add,
                                            const T*                        beta_device_host,
                                            const I*                        csr_row_ptr_D,
                                            const J*                        csr_col_ind_D,
                                            const T*                        csr_val_D,
                                            rocsparse_index_base            idx_base_D,
                                            const I*                        csr_row_ptr_C,
                                            const J*                        csr_col_ind_C,
                                            T*                              csr_val_C,
                                            rocsparse_index_base            idx_base_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        static constexpr uint32_t BLOCKSIZE = 256;
        static constexpr uint32_t WFSIZE    = 16;

        if(nslot <= 0 || n <= 0)
        {
            return rocsparse_status_success;
        }

        const csrgemm_bitmap_group<J>  group{nslot, d_group_offset, d_perm};
        const csrgemm_bitmap_csr<I, J> A{csr_row_ptr_A, csr_col_ind_A, idx_base_A};
        const csrgemm_bitmap_csr<I, J> B{csr_row_ptr_B, csr_col_ind_B, idx_base_B};
        const csrgemm_bitmap_csr<I, J> D{add ? csr_row_ptr_D : nullptr, csr_col_ind_D, idx_base_D};
        const csrgemm_bitmap_csr<I, J> C{csr_row_ptr_C, csr_col_ind_C, idx_base_C};
        const csrgemm_bitmap_values<T> values{
            alpha_device_host, csr_val_A, csr_val_B, beta_device_host, csr_val_D};

        return rocsparse::csrgemm_bitmap_rank_walk<BLOCKSIZE, WFSIZE>(
            handle, workspace, n, group, false, A, B, C, [&](const auto& layout, const auto& pass) {
                return rocsparse::csrgemm_bitmap_values_pass<BLOCKSIZE, WFSIZE>(handle,
                                                                                workspace,
                                                                                layout,
                                                                                pass,
                                                                                group,
                                                                                A,
                                                                                B,
                                                                                D,
                                                                                values,
                                                                                m,
                                                                                csr_row_ptr_C,
                                                                                csr_val_C,
                                                                                idx_base_C);
            });
    }
}

template class rocsparse::csrgemm_bitmap_workspace<int32_t, int32_t>;
template class rocsparse::csrgemm_bitmap_workspace<int32_t, int64_t>;
template class rocsparse::csrgemm_bitmap_workspace<int64_t, int32_t>;
template class rocsparse::csrgemm_bitmap_workspace<int64_t, int64_t>;

#define INSTANTIATE_NNZ(I, J)                                      \
    template rocsparse_status rocsparse::csrgemm_nnz_bitmap(       \
        rocsparse_handle                           handle,         \
        rocsparse::csrgemm_bitmap_workspace<I, J>& workspace,      \
        J                                          n,              \
        J                                          nslot,          \
        const J*                                   d_group_offset, \
        const J*                                   d_perm,         \
        const I*                                   csr_row_ptr_A,  \
        const J*                                   csr_col_ind_A,  \
        rocsparse_index_base                       idx_base_A,     \
        const I*                                   csr_row_ptr_B,  \
        const J*                                   csr_col_ind_B,  \
        rocsparse_index_base                       idx_base_B,     \
        bool                                       add,            \
        const I*                                   csr_row_ptr_D,  \
        const J*                                   csr_col_ind_D,  \
        rocsparse_index_base                       idx_base_D,     \
        I*                                         row_nnz)

INSTANTIATE_NNZ(int32_t, int32_t);
INSTANTIATE_NNZ(int32_t, int64_t);
INSTANTIATE_NNZ(int64_t, int32_t);
INSTANTIATE_NNZ(int64_t, int64_t);

#undef INSTANTIATE_NNZ

#define INSTANTIATE_SYMBOLIC(I, J)                                 \
    template rocsparse_status rocsparse::csrgemm_symbolic_bitmap(  \
        rocsparse_handle                           handle,         \
        rocsparse::csrgemm_bitmap_workspace<I, J>& workspace,      \
        J                                          n,              \
        J                                          nslot,          \
        const J*                                   d_group_offset, \
        const J*                                   d_perm,         \
        const I*                                   csr_row_ptr_A,  \
        const J*                                   csr_col_ind_A,  \
        rocsparse_index_base                       idx_base_A,     \
        const I*                                   csr_row_ptr_B,  \
        const J*                                   csr_col_ind_B,  \
        rocsparse_index_base                       idx_base_B,     \
        bool                                       add,            \
        const I*                                   csr_row_ptr_D,  \
        const J*                                   csr_col_ind_D,  \
        rocsparse_index_base                       idx_base_D,     \
        const I*                                   csr_row_ptr_C,  \
        J*                                         csr_col_ind_C,  \
        rocsparse_index_base                       idx_base_C)

INSTANTIATE_SYMBOLIC(int32_t, int32_t);
INSTANTIATE_SYMBOLIC(int64_t, int32_t);
INSTANTIATE_SYMBOLIC(int64_t, int64_t);

#undef INSTANTIATE_SYMBOLIC

#define INSTANTIATE_CALC(I, J, T)                                     \
    template rocsparse_status rocsparse::csrgemm_calc_bitmap(         \
        rocsparse_handle                           handle,            \
        rocsparse::csrgemm_bitmap_workspace<I, J>& workspace,         \
        J                                          m,                 \
        J                                          n,                 \
        J                                          nslot,             \
        const J*                                   d_group_offset,    \
        const J*                                   d_perm,            \
        const T*                                   alpha_device_host, \
        const I*                                   csr_row_ptr_A,     \
        const J*                                   csr_col_ind_A,     \
        const T*                                   csr_val_A,         \
        rocsparse_index_base                       idx_base_A,        \
        const I*                                   csr_row_ptr_B,     \
        const J*                                   csr_col_ind_B,     \
        const T*                                   csr_val_B,         \
        rocsparse_index_base                       idx_base_B,        \
        bool                                       add,               \
        const T*                                   beta_device_host,  \
        const I*                                   csr_row_ptr_D,     \
        const J*                                   csr_col_ind_D,     \
        const T*                                   csr_val_D,         \
        rocsparse_index_base                       idx_base_D,        \
        const I*                                   csr_row_ptr_C,     \
        J*                                         csr_col_ind_C,     \
        T*                                         csr_val_C,         \
        rocsparse_index_base                       idx_base_C)

INSTANTIATE_CALC(int32_t, int32_t, float);
INSTANTIATE_CALC(int32_t, int32_t, double);
INSTANTIATE_CALC(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_CALC(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_CALC(int32_t, int32_t, _Float16);
INSTANTIATE_CALC(int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_CALC(int64_t, int32_t, float);
INSTANTIATE_CALC(int64_t, int32_t, double);
INSTANTIATE_CALC(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_CALC(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_CALC(int64_t, int32_t, _Float16);
INSTANTIATE_CALC(int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_CALC(int64_t, int64_t, float);
INSTANTIATE_CALC(int64_t, int64_t, double);
INSTANTIATE_CALC(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_CALC(int64_t, int64_t, rocsparse_double_complex);
INSTANTIATE_CALC(int64_t, int64_t, _Float16);
INSTANTIATE_CALC(int64_t, int64_t, rocsparse_bfloat16);

#undef INSTANTIATE_CALC

#define INSTANTIATE_NUMERIC(I, J, T)                                  \
    template rocsparse_status rocsparse::csrgemm_numeric_bitmap(      \
        rocsparse_handle                           handle,            \
        rocsparse::csrgemm_bitmap_workspace<I, J>& workspace,         \
        J                                          m,                 \
        J                                          n,                 \
        J                                          nslot,             \
        const J*                                   d_group_offset,    \
        const J*                                   d_perm,            \
        const T*                                   alpha_device_host, \
        const I*                                   csr_row_ptr_A,     \
        const J*                                   csr_col_ind_A,     \
        const T*                                   csr_val_A,         \
        rocsparse_index_base                       idx_base_A,        \
        const I*                                   csr_row_ptr_B,     \
        const J*                                   csr_col_ind_B,     \
        const T*                                   csr_val_B,         \
        rocsparse_index_base                       idx_base_B,        \
        bool                                       add,               \
        const T*                                   beta_device_host,  \
        const I*                                   csr_row_ptr_D,     \
        const J*                                   csr_col_ind_D,     \
        const T*                                   csr_val_D,         \
        rocsparse_index_base                       idx_base_D,        \
        const I*                                   csr_row_ptr_C,     \
        const J*                                   csr_col_ind_C,     \
        T*                                         csr_val_C,         \
        rocsparse_index_base                       idx_base_C)

INSTANTIATE_NUMERIC(int32_t, int32_t, float);
INSTANTIATE_NUMERIC(int32_t, int32_t, double);
INSTANTIATE_NUMERIC(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_NUMERIC(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_NUMERIC(int32_t, int32_t, _Float16);
INSTANTIATE_NUMERIC(int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_NUMERIC(int64_t, int32_t, float);
INSTANTIATE_NUMERIC(int64_t, int32_t, double);
INSTANTIATE_NUMERIC(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_NUMERIC(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_NUMERIC(int64_t, int32_t, _Float16);
INSTANTIATE_NUMERIC(int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_NUMERIC(int64_t, int64_t, float);
INSTANTIATE_NUMERIC(int64_t, int64_t, double);
INSTANTIATE_NUMERIC(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_NUMERIC(int64_t, int64_t, rocsparse_double_complex);
INSTANTIATE_NUMERIC(int64_t, int64_t, _Float16);
INSTANTIATE_NUMERIC(int64_t, int64_t, rocsparse_bfloat16);

#undef INSTANTIATE_NUMERIC
