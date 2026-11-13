// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/tdm_page_tracker.hpp"

namespace {

using ck_tile::index_t;
using ck_tile::long_index_t;

// Fake element type; only pointer arithmetic is exercised.
using Elem = std::uint16_t;

constexpr index_t kTileRows      = 128;
constexpr index_t kPageBlockSize = 256;
constexpr index_t kStrideKv      = 128; // elements per token
constexpr index_t kNumPages      = 8;

// block_stride is the element distance between two physical pages.
constexpr long_index_t kBlockStride = static_cast<long_index_t>(kPageBlockSize) * kStrideKv;

using Tracker = ck_tile::TdmPageTracker<Elem, kTileRows>;

// Reference: where should a global token index land?
// block_stride defaults to kBlockStride (the single-kv-head packing every
// other test in this file uses); pass a distinct value to exercise layouts
// where block_stride is not page_block_size * stride_kv (see
// BlockStrideDiffersFromPageBlockSizeTimesStrideKv below).
const Elem* expected_addr(const Elem* pool,
                          const std::vector<std::int32_t>& table,
                          index_t global_seq,
                          long_index_t fixed_offset,
                          long_index_t block_stride = kBlockStride)
{
    const index_t page = global_seq / kPageBlockSize;
    const index_t tok  = global_seq % kPageBlockSize;
    return pool + static_cast<long_index_t>(table[page]) * block_stride + fixed_offset +
           static_cast<long_index_t>(tok) * kStrideKv;
}

Tracker make_tracker(const Elem* pool,
                     const std::vector<std::int32_t>& table,
                     long_index_t fixed_offset,
                     index_t global_seq_start,
                     index_t l2p_offset = 0)
{
    return Tracker{pool,
                   kBlockStride,
                   fixed_offset,
                   table.data(),
                   kPageBlockSize,
                   kStrideKv,
                   kNumPages,
                   global_seq_start,
                   l2p_offset};
}

TEST(TdmPageTracker, IdentityTableMatchesDenseAddressing)
{
    const Elem* pool = reinterpret_cast<const Elem*>(0x100000000ULL);
    std::vector<std::int32_t> table(kNumPages);
    for(index_t p = 0; p < kNumPages; ++p)
        table[p] = p;

    auto tracker = make_tracker(pool, table, 0, 0);

    // Identity table: the reverse offset cancels, so base_ptr must equal pool.
    for(index_t i = 0; i < kNumPages * (kPageBlockSize / kTileRows); ++i)
    {
        EXPECT_EQ(tracker.base_ptr(), pool) << "tile " << i;
        tracker.advance();
    }
}

TEST(TdmPageTracker, ReversedTableAddressesCorrectToken)
{
    const Elem* pool = reinterpret_cast<const Elem*>(0x100000000ULL);
    std::vector<std::int32_t> table(kNumPages);
    for(index_t p = 0; p < kNumPages; ++p)
        table[p] = kNumPages - 1 - p;

    auto tracker = make_tracker(pool, table, 0, 0);

    const index_t num_tiles = kNumPages * (kPageBlockSize / kTileRows);
    for(index_t i = 0; i < num_tiles; ++i)
    {
        const index_t global_seq = i * kTileRows;
        // base_ptr() + global_seq * stride must land on the right physical token.
        const Elem* got = tracker.base_ptr() + static_cast<long_index_t>(global_seq) * kStrideKv;
        EXPECT_EQ(got, expected_addr(pool, table, global_seq, 0)) << "tile " << i;
        tracker.advance();
    }
}

TEST(TdmPageTracker, FixedOffsetIsApplied)
{
    const Elem* pool                   = reinterpret_cast<const Elem*>(0x100000000ULL);
    constexpr long_index_t kHeadOffset = 4096;
    std::vector<std::int32_t> table(kNumPages);
    for(index_t p = 0; p < kNumPages; ++p)
        table[p] = (p * 3 + 1) % kNumPages;

    auto tracker = make_tracker(pool, table, kHeadOffset, 0);

    for(index_t i = 0; i < kNumPages * (kPageBlockSize / kTileRows); ++i)
    {
        const index_t global_seq = i * kTileRows;
        const Elem* got = tracker.base_ptr() + static_cast<long_index_t>(global_seq) * kStrideKv;
        EXPECT_EQ(got, expected_addr(pool, table, global_seq, kHeadOffset)) << "tile " << i;
        tracker.advance();
    }
}

TEST(TdmPageTracker, SeekJumpsToArbitraryGlobalPosition)
{
    const Elem* pool = reinterpret_cast<const Elem*>(0x100000000ULL);
    std::vector<std::int32_t> table(kNumPages);
    for(index_t p = 0; p < kNumPages; ++p)
        table[p] = (p * 5 + 2) % kNumPages;

    auto tracker = make_tracker(pool, table, 0, 0);

    // Jump straight into the middle of page 5, then keep walking.
    const index_t jump_to = 5 * kPageBlockSize + kTileRows;
    tracker.seek(jump_to);
    EXPECT_EQ(tracker.current_page(), 5);

    for(index_t i = 0; i < 4; ++i)
    {
        const index_t global_seq = jump_to + i * kTileRows;
        const Elem* got = tracker.base_ptr() + static_cast<long_index_t>(global_seq) * kStrideKv;
        EXPECT_EQ(got, expected_addr(pool, table, global_seq, 0)) << "step " << i;
        tracker.advance();
    }
}

TEST(TdmPageTracker, StartOffsetIsHonoured)
{
    const Elem* pool = reinterpret_cast<const Elem*>(0x100000000ULL);
    std::vector<std::int32_t> table(kNumPages);
    for(index_t p = 0; p < kNumPages; ++p)
        table[p] = kNumPages - 1 - p;

    // Construct already positioned at page 3.
    auto tracker = make_tracker(pool, table, 0, 3 * kPageBlockSize);
    EXPECT_EQ(tracker.current_page(), 3);

    const index_t global_seq = 3 * kPageBlockSize;
    const Elem* got = tracker.base_ptr() + static_cast<long_index_t>(global_seq) * kStrideKv;
    EXPECT_EQ(got, expected_addr(pool, table, global_seq, 0));
}

// K runs two kN0 tiles ahead of the tile being computed, so the tracker WILL be
// advanced past the last page. It must not read block_indices out of bounds.
TEST(TdmPageTracker, AdvancingPastLastPageDoesNotReadTableOutOfBounds)
{
    const Elem* pool = reinterpret_cast<const Elem*>(0x100000000ULL);
    std::vector<std::int32_t> table(kNumPages);
    for(index_t p = 0; p < kNumPages; ++p)
        table[p] = kNumPages - 1 - p;

    auto tracker = make_tracker(pool, table, 0, (kNumPages - 1) * kPageBlockSize);
    ASSERT_EQ(tracker.current_page(), kNumPages - 1);

    const Elem* last_valid_base = tracker.base_ptr();

    // Walk well past the end. The page index must saturate and the base pointer
    // must stay at the last valid page's value.
    for(int i = 0; i < 8; ++i)
    {
        tracker.advance();
        EXPECT_LE(tracker.current_page(), kNumPages - 1);
        EXPECT_EQ(tracker.base_ptr(), last_valid_base) << "after " << (i + 1) << " advances";
    }
}

TEST(TdmPageTracker, L2pOffsetShiftsPageSelection)
{
    const Elem* pool = reinterpret_cast<const Elem*>(0x100000000ULL);
    std::vector<std::int32_t> table(kNumPages);
    for(index_t p = 0; p < kNumPages; ++p)
        table[p] = kNumPages - 1 - p;

    // Sequence starts 1.5 pages into the pool.
    const index_t l2p = kPageBlockSize + kTileRows;

    Tracker tracker{pool,
                    kBlockStride,
                    0,
                    table.data(),
                    kPageBlockSize,
                    kStrideKv,
                    kNumPages,
                    /*global_seq_start=*/0,
                    l2p};

    EXPECT_EQ(tracker.current_page(), 1);

    for(index_t i = 0; i < 6; ++i)
    {
        const index_t global_seq = i * kTileRows;
        const Elem* got = tracker.base_ptr() + static_cast<long_index_t>(global_seq) * kStrideKv;
        EXPECT_EQ(got, expected_addr(pool, table, l2p + global_seq, 0)) << "tile " << i;
        tracker.advance();
    }
}

TEST(TdmPageTracker, SeekPastLastPageSaturates)
{
    const Elem* pool = reinterpret_cast<const Elem*>(0x100000000ULL);
    std::vector<std::int32_t> table(kNumPages);
    for(index_t p = 0; p < kNumPages; ++p)
        table[p] = p;

    auto tracker = make_tracker(pool, table, 0, 0);
    tracker.seek(kNumPages * kPageBlockSize + 999);
    // EXPECT_LE also passes if current_page() clamped to 0 (or anything below
    // the last page) instead of saturating at kNumPages - 1; that would still
    // be a clamp-to-zero-style bug, just one this assertion could not catch.
    EXPECT_EQ(tracker.current_page(), kNumPages - 1);
}

// nhead > 1 shape: block_stride is NOT page_block_size * stride_kv -- for a
// [page, kv_head, token, dim] physical layout, advancing one page also has to
// skip over the other kv heads' slices of that page, so
// block_stride == num_kv_heads * page_block_size * stride_kv. Every other
// test in this file uses make_tracker(), which hardcodes
// kBlockStride == kPageBlockSize * kStrideKv (see the constant above); none
// of them would catch a refresh() regression that recomputed the reverse
// offset from page_block_size and stride_kv instead of using the
// caller-supplied block_stride, since for them the two expressions are
// numerically identical. This test is the one that distinguishes them.
TEST(TdmPageTracker, BlockStrideDiffersFromPageBlockSizeTimesStrideKv)
{
    const Elem* pool              = reinterpret_cast<const Elem*>(0x100000000ULL);
    constexpr index_t kNumKvHeads = 3;
    constexpr long_index_t kDistinctBlockStride =
        static_cast<long_index_t>(kNumKvHeads) * kPageBlockSize * kStrideKv;
    static_assert(kDistinctBlockStride != kBlockStride,
                  "this test requires a block_stride distinct from page_block_size * stride_kv");

    std::vector<std::int32_t> table(kNumPages);
    for(index_t p = 0; p < kNumPages; ++p)
        table[p] = kNumPages - 1 - p;

    Tracker tracker{pool,
                    kDistinctBlockStride,
                    0,
                    table.data(),
                    kPageBlockSize,
                    kStrideKv,
                    kNumPages,
                    /*global_seq_start=*/0,
                    /*l2p_offset=*/0};

    const index_t num_tiles = kNumPages * (kPageBlockSize / kTileRows);
    for(index_t i = 0; i < num_tiles; ++i)
    {
        const index_t global_seq = i * kTileRows;
        const Elem* got = tracker.base_ptr() + static_cast<long_index_t>(global_seq) * kStrideKv;
        EXPECT_EQ(got, expected_addr(pool, table, global_seq, 0, kDistinctBlockStride))
            << "tile " << i;
        tracker.advance();
    }
}

} // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
