// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_tile_index.hpp"

namespace ck_tile {

// Merged GQA rows: query position row % seqlen_q, head row / seqlen_q of num_groups.
struct FmhaTdmSchedFp8MergedHeads
{
    index_t seqlen_q      = 1;
    index_t num_groups    = 1;
    const float* sink_ptr = nullptr;
    // Per-head q descales of the group, folded as (scale_s * q_descale) * k_descale.
    const float* q_descale_ptr    = nullptr;
    index_t q_descale_head_stride = 0;
    float scale_s                 = 1.0f;
    float k_descale               = 1.0f;
};

// Per-row query position, softmax scale and sink of merged GQA rows.
struct FmhaTdmSchedFp8MergedRows
{
    // Query position of each lane row (row tensor layout); merged rows of one head share one.
    template <typename RowTensor>
    CK_TILE_DEVICE static auto MergedRowPositions(const FmhaTdmSchedFp8MergedHeads& merged,
                                                  index_t row0)
    {
        auto pos =
            make_static_distributed_tensor<index_t>(typename RowTensor::StaticTileDistribution{});
        sweep_tile_span(decltype(pos)::get_distributed_spans()[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const index_t row    = row0 + FmhaTdmSchedFp8TileIndex::LaneRowOf<RowTensor>(i_idx);
            pos(i_idx)           = merged.seqlen_q == 1 ? 0 : row % merged.seqlen_q;
        });
        return pos;
    }

    // Softmax scale of a merged row's head, folded as the kernel folds it for one head.
    template <bool kBlockScale>
    CK_TILE_DEVICE static float MergedRowScale(const FmhaTdmSchedFp8MergedHeads& merged,
                                               index_t row)
    {
        const index_t g       = min(row / merged.seqlen_q, merged.num_groups - 1);
        const float q_descale = merged.q_descale_ptr[g * merged.q_descale_head_stride];
        if constexpr(kBlockScale)
            return merged.scale_s * q_descale;
        else
            return merged.scale_s * q_descale * merged.k_descale;
    }

    // Sink logit of a merged row's head, in the units of the unmerged per-head sink_v * scale_s.
    CK_TILE_DEVICE static float
    MergedSinkLogit(const FmhaTdmSchedFp8MergedHeads& merged, index_t row, float scale_s)
    {
        // row is tile-local: merged query heads always fit one M tile, so it is the query row.
        const index_t g = min(row / merged.seqlen_q, merged.num_groups - 1);
        // Rounds like the unmerged path, which passes sink / scale_s and multiplies by scale_s.
        return (merged.sink_ptr[g] / scale_s) * scale_s;
    }
};

} // namespace ck_tile
