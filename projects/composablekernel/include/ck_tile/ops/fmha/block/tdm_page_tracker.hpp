// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Tracks which physical page a K or V tile window currently sits in, and
// produces a base pointer that lets the window keep GLOBAL seqlen coordinates.
//
// The pipeline this serves (qr_tdm_sched) reads the window origin as an
// absolute seqlen_k coordinate for masking and tail predication, so the origin
// must not be rewritten to a page-local one. Instead the paging is folded into
// the base pointer:
//
//   base(p) = pool + block_table[p]*block_stride + fixed_offset
//                  - p*page_block_size*stride_kv
//
// Indexing base(p) with a global origin then lands inside page p. This is only
// valid when a tile never straddles two pages, i.e. page_block_size is a
// multiple of TileRows; the host dispatch enforces that.
//
// Group mode (gappy KV cache) can start a sequence partway into its own page
// pool: l2p_offset shifts the logical-to-physical mapping so the page index
// and the reverse offset are both taken on (l2p_offset + global_seq) instead
// of on global_seq alone. Batch mode passes l2p_offset = 0, which reduces the
// formula above back to exactly the previous behavior. l2p_offset must be a
// multiple of TileRows so the first token position inside a page stays tile
// aligned (the tile-advance countdown below assumes that).
//
// Page advance is a countdown, so the steady-state cost is one decrement plus a
// predicted-not-taken branch. The division only happens in seek(), which the
// sink path uses at most once per kernel.
template <typename DataType_, index_t TileRows_>
struct TdmPageTracker
{
    using DataType                    = remove_cvref_t<DataType_>;
    static constexpr index_t TileRows = TileRows_;

    CK_TILE_HOST_DEVICE constexpr TdmPageTracker(const DataType* physical_blocks,
                                                 long_index_t block_stride,
                                                 long_index_t fixed_offset,
                                                 const int32_t* block_indices,
                                                 index_t page_block_size,
                                                 index_t stride_kv,
                                                 index_t num_pages,
                                                 index_t global_seq_start,
                                                 index_t l2p_offset)
        : physical_blocks_{physical_blocks},
          block_stride_{block_stride},
          fixed_offset_{fixed_offset},
          block_indices_{block_indices},
          page_block_size_{page_block_size},
          stride_kv_{stride_kv},
          num_pages_{num_pages},
          l2p_offset_{l2p_offset},
          tiles_per_page_{page_block_size / TileRows},
          cur_page_{0},
          tiles_left_{0},
          cached_base_{nullptr}
    {
        seek(global_seq_start);
    }

    // Base pointer to index with a GLOBAL seqlen coordinate.
    CK_TILE_HOST_DEVICE const DataType* base_ptr() const { return cached_base_; }

    CK_TILE_HOST_DEVICE index_t current_page() const { return cur_page_; }

    // Step forward by exactly one tile of TileRows tokens.
    CK_TILE_HOST_DEVICE void advance()
    {
        --tiles_left_;
        if(tiles_left_ <= 0)
        {
            // Saturate: K runs two tiles ahead of the computed tile, so this
            // legitimately walks past the end. Reading block_indices_ there
            // would be out of bounds, and the loaded data is discarded anyway
            // (TDM clamps it to zero and the score predicate masks it).
            if(cur_page_ + 1 < num_pages_)
            {
                ++cur_page_;
                refresh();
            }
            tiles_left_ = tiles_per_page_;
        }
    }

    // Absolute jump to a global token position. Used by the sink path, which
    // skips from the sink prefix to the real window start.
    CK_TILE_HOST_DEVICE void seek(index_t global_seq)
    {
        if(global_seq < 0)
            global_seq = 0;

        // Logical-to-physical: in group mode the sequence starts partway into
        // the page pool, so the page index is taken on the shifted coordinate.
        const index_t phys_seq = l2p_offset_ + global_seq;

        index_t page = phys_seq / page_block_size_;
        if(page >= num_pages_)
            page = num_pages_ - 1;
        if(page < 0)
            page = 0;

        cur_page_ = page;
        refresh();

        const index_t tok_in_page = phys_seq - page * page_block_size_;
        tiles_left_               = tiles_per_page_ - tok_in_page / TileRows;
        if(tiles_left_ <= 0)
            tiles_left_ = tiles_per_page_;
    }

    private:
    // On the device the block table is read through the constant address space so the load
    // is a scalar load into an SGPR. A generic load inside the pipeline loop is emitted as a
    // VMEM load, because the loop's TDM and barrier intrinsics may clobber memory.
    CK_TILE_HOST_DEVICE index_t load_block_index(index_t page) const
    {
#if defined(__HIP_DEVICE_COMPILE__)
        return cast_pointer_to_constant_address_space(block_indices_)[page];
#else
        return block_indices_[page];
#endif
    }

    CK_TILE_HOST_DEVICE void refresh()
    {
        const long_index_t phys = static_cast<long_index_t>(load_block_index(cur_page_));
        // All of this is 64-bit: the reverse term can push the result far below
        // physical_blocks_, and only adding the global offset brings it back.
        // The l2p_offset_ shift is folded back in here so that indexing this
        // base pointer with a GLOBAL (l2p-unaware) window origin still lands on
        // the correct physical token.
        cached_base_ =
            physical_blocks_ + phys * block_stride_ + fixed_offset_ -
            (static_cast<long_index_t>(cur_page_) * static_cast<long_index_t>(page_block_size_) -
             static_cast<long_index_t>(l2p_offset_)) *
                static_cast<long_index_t>(stride_kv_);
    }

    const DataType* physical_blocks_;
    long_index_t block_stride_;
    long_index_t fixed_offset_;
    const int32_t* block_indices_;
    index_t page_block_size_;
    index_t stride_kv_;
    index_t num_pages_;
    index_t l2p_offset_;
    index_t tiles_per_page_;
    index_t cur_page_;
    index_t tiles_left_;
    const DataType* cached_base_;
};

} // namespace ck_tile
