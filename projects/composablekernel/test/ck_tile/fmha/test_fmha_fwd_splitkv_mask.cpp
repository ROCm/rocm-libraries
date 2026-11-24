// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "gtest/gtest.h"

namespace {

using namespace ck_tile;

constexpr index_t kTileWidth        = 64; // kN0 of the gfx1250 split-KV tile
constexpr index_t kMaxNumTiles      = 2048;
constexpr index_t kMaxNumSplits     = 128;
constexpr index_t kWideTileWidth    = 128;
constexpr index_t kMaxWideNumTiles  = 256;
constexpr index_t kMaxWideNumSplits = 64;

constexpr index_t kDecodeQueryLength = 1;
constexpr index_t kDecodeRow         = 0;
constexpr auto kDecodeRowTile        = number<1>{};

// kM0 of the gfx1250 fp8 split-KV tile, the row tile of rows that are not merged Q heads
constexpr index_t kUnmergedRowTile      = 16;
constexpr index_t kMaxUnmergedNumTiles  = 64;
constexpr index_t kMaxUnmergedNumSplits = 16;

constexpr index_t kLastTileShortfall = 17;
constexpr index_t kLeadingTiles      = 3;
constexpr index_t kWindowStartOffset = 5;

using CausalMask = SimplifiedGenericAttentionMask<true>;
using NoMask     = SimplifiedGenericAttentionMask<false>;

// Key length of num_tiles tiles whose last tile is short by last_tile_shortfall keys.
index_t KeyLength(index_t num_tiles, index_t tile_width, index_t last_tile_shortfall)
{
    if(num_tiles == 0)
        return 0;
    return num_tiles * tile_width - last_tile_shortfall;
}

// The 5D paged KV pipeline reads the page of a split's first tile from where the split starts:
// through the split-KV mask, every non-empty split starts on a multiple of the tile width.
template <typename Mask, index_t TileHeight, index_t TileWidth>
::testing::AssertionResult NonEmptySplitsStartOnTiles(const Mask& mask,
                                                      index_t i_y,
                                                      number<TileHeight> height,
                                                      number<TileWidth> width,
                                                      index_t num_splits)
{
    for(index_t i_split = 0; i_split < num_splits; ++i_split)
    {
        const SplitKVAttentionMask<Mask, false> split_mask{mask, num_splits, i_split};
        const auto [start, end] = split_mask.GetTileRangeAlongX(i_y, height, width);
        if(start < end && start % TileWidth != 0)
            return ::testing::AssertionFailure() << "split " << i_split << " of " << num_splits
                                                 << " starts at " << start << ", i_y " << i_y;
    }
    return ::testing::AssertionSuccess();
}

// make_mask(num_tiles) returns a decode mask whose range starts at expected_start and spans
// num_tiles tiles. Stops at the first failing case.
template <index_t TileWidth, typename MakeMask>
void SweepTileAndSplitCounts(number<TileWidth> width,
                             index_t min_num_tiles,
                             index_t max_num_tiles,
                             index_t max_num_splits,
                             index_t expected_start,
                             MakeMask make_mask)
{
    for(index_t num_tiles = min_num_tiles; num_tiles <= max_num_tiles; ++num_tiles)
    {
        const auto mask           = make_mask(num_tiles);
        const auto [start, end]   = mask.GetTileRangeAlongX(kDecodeRow, kDecodeRowTile, width);
        const index_t range_tiles = integer_divide_ceil(end - start, TileWidth);
        ASSERT_EQ(start, expected_start) << "num_tiles " << num_tiles;
        ASSERT_EQ(range_tiles, num_tiles);

        for(index_t num_splits = 1; num_splits <= max_num_splits; ++num_splits)
            ASSERT_TRUE(
                NonEmptySplitsStartOnTiles(mask, kDecodeRow, kDecodeRowTile, width, num_splits))
                << "num_tiles " << num_tiles << ", x_total " << mask.GetXTotal();
    }
}

template <index_t TileWidth>
void SweepNoMask(number<TileWidth> width, index_t max_num_tiles, index_t max_num_splits)
{
    for(index_t shortfall : {index_t{0}, kLastTileShortfall})
        SweepTileAndSplitCounts(width, 0, max_num_tiles, max_num_splits, 0, [&](index_t num_tiles) {
            return NoMask{kDecodeQueryLength, KeyLength(num_tiles, TileWidth, shortfall)};
        });
}

template <index_t TileWidth>
void SweepCausalDecode(number<TileWidth> width, index_t max_num_tiles, index_t max_num_splits)
{
    for(index_t shortfall : {index_t{0}, kLastTileShortfall})
        SweepTileAndSplitCounts(width, 0, max_num_tiles, max_num_splits, 0, [&](index_t num_tiles) {
            const index_t x_total = KeyLength(num_tiles, TileWidth, shortfall);
            return make_generic_attention_mask_from_lr_window<CausalMask>(
                -1, 0, kDecodeQueryLength, x_total, false);
        });
}

// The window starts kWindowStartOffset keys into tile kLeadingTiles, so the range starts there.
template <index_t TileWidth>
void SweepSlidingWindowDecode(number<TileWidth> width,
                              index_t max_num_tiles,
                              index_t max_num_splits)
{
    const index_t range_start   = kLeadingTiles * TileWidth;
    const index_t window_start  = range_start + kWindowStartOffset;
    const index_t min_num_tiles = 1; // the window always holds the query's own key
    for(index_t shortfall : {index_t{0}, kLastTileShortfall})
    {
        const auto make_mask = [&](index_t num_tiles) {
            const index_t x_total     = KeyLength(kLeadingTiles + num_tiles, TileWidth, shortfall);
            const index_t window_left = x_total - 1 - window_start;
            return make_generic_attention_mask_from_lr_window<CausalMask>(
                window_left, 0, kDecodeQueryLength, x_total, false);
        };
        SweepTileAndSplitCounts(
            width, min_num_tiles, max_num_tiles, max_num_splits, range_start, make_mask);
    }
}

TEST(FmhaFwdSplitKVMask, NoMaskSplitsStartOnTiles)
{
    SweepNoMask(number<kTileWidth>{}, kMaxNumTiles, kMaxNumSplits);
    SweepNoMask(number<kWideTileWidth>{}, kMaxWideNumTiles, kMaxWideNumSplits);
}

TEST(FmhaFwdSplitKVMask, CausalDecodeSplitsStartOnTiles)
{
    SweepCausalDecode(number<kTileWidth>{}, kMaxNumTiles, kMaxNumSplits);
    SweepCausalDecode(number<kWideTileWidth>{}, kMaxWideNumTiles, kMaxWideNumSplits);
}

TEST(FmhaFwdSplitKVMask, SlidingWindowDecodeSplitsStartOnTiles)
{
    SweepSlidingWindowDecode(number<kTileWidth>{}, kMaxNumTiles, kMaxNumSplits);
    SweepSlidingWindowDecode(number<kWideTileWidth>{}, kMaxWideNumTiles, kMaxWideNumSplits);
}

// The kernel's one tile of unmerged rows (i_y 0) over queries of 1, 5 and 16 rows, whose rows
// reach different keys under a bottom-right causal or sliding window mask.
TEST(FmhaFwdSplitKVMask, UnmergedRowTileSplitsStartOnTiles)
{
    constexpr auto row_tile   = number<kUnmergedRowTile>{};
    constexpr auto key_tile   = number<kTileWidth>{};
    constexpr index_t i_y     = 0;
    const index_t window_left = kLeadingTiles * kTileWidth + kWindowStartOffset;
    // every key length: the last row ends up to query_length - 1 keys past the first
    for(const index_t query_length : {1, 5, kUnmergedRowTile})
        for(index_t x_total = 0; x_total <= kMaxUnmergedNumTiles * kTileWidth; ++x_total)
            for(const index_t left : {index_t{-1}, window_left})
            {
                const auto mask = make_generic_attention_mask_from_lr_window<CausalMask>(
                    left, 0, query_length, x_total, false);
                for(index_t num_splits = 1; num_splits <= kMaxUnmergedNumSplits; ++num_splits)
                    ASSERT_TRUE(
                        NonEmptySplitsStartOnTiles(mask, i_y, row_tile, key_tile, num_splits))
                        << "query_length " << query_length << ", x_total " << x_total
                        << ", window_left " << left;
            }
}

} // namespace
