// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// K/V DRAM windows of dense attention advance linearly along the sequence.
struct BlockFmhaDenseKVRows
{
    static constexpr bool kIsPaged = false;
};

// Paged K/V (batch_prefill): every warp reads its rows of a K/V tile from the page holding
// them, through a view of that page cut at the sequence end, so TDM zero-fills the rows past
// seqlen_k instead of reading them. A warp's rows must not straddle two pages.
// MakeKView/MakeVView(physical_page, rows) return the view of the first `rows` rows of a
// page, with the type of the K/V views the pipeline was called with.
template <index_t kPageSize, typename MakeKView, typename MakeVView>
struct BlockFmhaPagedKVRows
{
    static constexpr bool kIsPaged      = true;
    static constexpr index_t kPageSize_ = kPageSize;

    const index_t* page_idx; // this sequence's physical page ids
    index_t max_page;        // last valid index into page_idx
    index_t seqlen_k;
    MakeKView make_k_view;
    MakeVView make_v_view;

    // Consecutive K/V tiles mostly stay in one page: reload only when the page changes.
    mutable index_t cached_page_id = -1;
    mutable index_t cached_page    = 0;

    // Point `window` at sequence row `row`.
    template <bool kIsK, typename Window>
    CK_TILE_DEVICE void move_to(Window& window, index_t row) const
    {
        // Window row of this warp's rows (the K/V TDM distributions split rows by warp).
        const auto dstr        = window.get_tile_distribution();
        using Dstr             = remove_cvref_t<decltype(dstr)>;
        const index_t warp_row = __builtin_amdgcn_readfirstlane(
            make_tensor_adaptor_coordinate(
                dstr.get_ps_ys_to_xs_adaptor(),
                container_concat(Dstr::get_partition_index(),
                                 array<index_t, Dstr::get_num_of_dimension_y()>{0}))
                .get_bottom_index()[number<0>{}]);
        const index_t warp_seq_row = row + warp_row;

        // A wave-uniform page index keeps the page table read on the scalar path.
        const index_t page_id =
            __builtin_amdgcn_readfirstlane(min(warp_seq_row / kPageSize, max_page));
        if(page_id != cached_page_id)
        {
            cached_page    = __builtin_amdgcn_readfirstlane(page_idx[page_id]);
            cached_page_id = page_id;
        }
        const index_t page_start = page_id * kPageSize;
        const index_t rows       = max(index_t{0}, min(kPageSize, seqlen_k - page_start));

        auto origin         = window.get_window_origin();
        origin(number<0>{}) = warp_seq_row - page_start - warp_row;
        if constexpr(kIsK)
            window = make_tile_window(make_k_view(cached_page, rows),
                                      window.get_window_lengths(),
                                      origin,
                                      window.get_tile_distribution());
        else
            window = make_tile_window(make_v_view(cached_page, rows),
                                      window.get_window_lengths(),
                                      origin,
                                      window.get_tile_distribution());
    }
};

} // namespace ck_tile
