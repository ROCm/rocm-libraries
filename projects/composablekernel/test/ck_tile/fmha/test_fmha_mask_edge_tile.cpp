// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host tests of SimplifiedGenericAttentionMask::IsEdgeTileForValidRows and
// SimplifiedGenericAttentionMask::IsOutOfBoundUnsignedStoredRow.

#include <algorithm>
#include <cstddef>

#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "gtest/gtest.h"

namespace {

// IsEdgeTileForValidRows may report "no per-pixel mask needed" only for tiles in which no valid
// row (row < y_total) has a masked pixel, must agree with IsEdgeTile on full query tiles, and
// must never report an edge where IsEdgeTile does not. Brute force over small tiles, causal and
// windowed masks, both alignments.
TEST(FmhaMask, EdgeTileForValidRows)
{
    using Mask = ck_tile::SimplifiedGenericAttentionMask<true>;
    using ck_tile::index_t;
    constexpr index_t kH = 8;
    constexpr index_t kW = 8;
    std::size_t narrowed = 0;
    for(index_t y_total = 1; y_total <= 40; y_total += 3)
        for(index_t x_total = 1; x_total <= 48; x_total += 5)
            for(index_t left : {-1, 0, 3, 17})
                for(index_t right : {-1, 0, 5})
                    for(bool top_left : {true, false})
                    {
                        const auto mask = ck_tile::make_generic_attention_mask_from_lr_window<Mask>(
                            left, right, 0, y_total, x_total, top_left);
                        for(index_t i_y = 0; i_y < y_total; i_y += kH)
                        {
                            // Like the pipeline, visit only the key tiles in the mask's range.
                            const auto [x_start, x_end] = mask.GetTileRangeAlongX(
                                i_y, ck_tile::number<kH>{}, ck_tile::number<kW>{});
                            for(index_t i_x = x_start; i_x < x_end; i_x += kW)
                            {
                                const bool edge = mask.IsEdgeTileForValidRows(
                                    i_y, i_x, ck_tile::number<kH>{}, ck_tile::number<kW>{});
                                const bool old = mask.IsEdgeTile(
                                    i_y, i_x, ck_tile::number<kH>{}, ck_tile::number<kW>{});
                                ASSERT_TRUE(!edge || old) << "new edge where IsEdgeTile had none";
                                if(i_y + kH <= y_total)
                                    ASSERT_EQ(edge, old) << "full query tile differs";
                                narrowed += old && !edge;
                                if(edge)
                                    continue;
                                for(index_t r = i_y; r < std::min(i_y + kH, y_total); ++r)
                                    for(index_t c = i_x; c < i_x + kW; ++c)
                                        ASSERT_FALSE(mask.IsOutOfBound(r, c))
                                            << "skipped tile has a masked valid pixel y_total="
                                            << y_total << " x_total=" << x_total << " left=" << left
                                            << " right=" << right << " r=" << r << " c=" << c;
                            }
                        }
                    }
    EXPECT_GT(narrowed, 0u) << "the partial-tile case was not exercised";
}

// IsOutOfBoundUnsignedStoredRow must equal IsOutOfBound on every row that is written out
// (row < y_total; other rows are unspecified). The columns extend past both ends of the key
// range: the unsigned compare also has to reject the columns left of the window.
TEST(FmhaMask, OutOfBoundUnsignedStoredRow)
{
    using Mask = ck_tile::SimplifiedGenericAttentionMask<true>;
    using ck_tile::index_t;
    for(index_t y_total = 1; y_total <= 40; y_total += 3)
        for(index_t x_total = 1; x_total <= 48; x_total += 5)
            for(index_t left : {-1, 0, 3, 17})
                for(index_t right : {-1, 0, 5})
                    for(bool top_left : {true, false})
                    {
                        const auto mask = ck_tile::make_generic_attention_mask_from_lr_window<Mask>(
                            left, right, 0, y_total, x_total, top_left);
                        for(index_t r = 0; r < y_total; ++r)
                            for(index_t c = -8; c < x_total + 8; ++c)
                                ASSERT_EQ(mask.IsOutOfBoundUnsignedStoredRow(r, c),
                                          mask.IsOutOfBound(r, c))
                                    << "y_total=" << y_total << " x_total=" << x_total
                                    << " left=" << left << " right=" << right << " r=" << r
                                    << " c=" << c;
                    }
}

} // namespace
