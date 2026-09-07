// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/// \file
/// Geometry tests for the paged-KV gather.
///
/// These cover the pure derivation -- shape checks, page arithmetic, dense
/// layout -- and need no device. The copy itself is exercised by the paged
/// bundles once the plan consumes this adapter.
///
/// The cases below are the ones where a wrong answer would still LOOK right:
/// a missing last-page trim, an out-of-range block id, and a page table too
/// small for the declared length. Each of those yields plausible numbers rather
/// than an error if it is not checked, which is why they are pinned here.

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "../src/harness/gpu-graph-executor/detail/PagedKvGather.hpp"

namespace
{

using hipdnn_test_sdk::detail::denseKvDims;
using hipdnn_test_sdk::detail::denseKvStrides;
using hipdnn_test_sdk::detail::derivePagedKvGeometry;

// The committed paged bundle's geometry: hd128, page 64, GQA 4, 2 sequences.
constexpr int64_t kNumBlocks       = 128;
constexpr int64_t kNumKvHeads      = 4;
constexpr int64_t kPageSize        = 64;
constexpr int64_t kHeadSize        = 128;
constexpr int64_t kNumSeqs         = 2;
constexpr int64_t kMaxBlocksPerSeq = 32;

std::vector<int64_t> bundleKDims()
{
    return {kNumBlocks, kNumKvHeads, kPageSize, kHeadSize};
}

std::vector<int64_t> bundlePageTableDims()
{
    return {kNumSeqs, kMaxBlocksPerSeq};
}

TEST(PagedKvGather, DerivesTheBundleGeometry)
{
    const auto geometry
        = derivePagedKvGeometry(bundleKDims(), bundlePageTableDims(), {1024, 2048});

    EXPECT_EQ(geometry.numBlocks, kNumBlocks);
    EXPECT_EQ(geometry.numKvHeads, kNumKvHeads);
    EXPECT_EQ(geometry.pageSize, kPageSize);
    EXPECT_EQ(geometry.headSize, kHeadSize);
    EXPECT_EQ(geometry.numSeqs, kNumSeqs);
    EXPECT_EQ(geometry.maxBlocksPerSeq, kMaxBlocksPerSeq);
    // The dense buffer is sized by the LONGEST sequence, so every sequence
    // occupies one rectangular slot the reference can address with strides.
    EXPECT_EQ(geometry.denseSeqLen, 2048);
}

TEST(PagedKvGather, DenseLayoutIsPackedBhsd)
{
    const auto geometry
        = derivePagedKvGeometry(bundleKDims(), bundlePageTableDims(), {1024, 2048});

    EXPECT_EQ(denseKvDims(geometry),
              (std::vector<int64_t>{kNumSeqs, kNumKvHeads, 2048, kHeadSize}));
    // Packed BHSD: each stride is the product of the extents below it.
    EXPECT_EQ(denseKvStrides(geometry),
              (std::vector<int64_t>{kNumKvHeads * 2048 * kHeadSize, 2048 * kHeadSize, kHeadSize, 1}));
}

TEST(PagedKvGather, AcceptsALengthThatDoesNotFillItsLastPage)
{
    // 100 tokens at page 64 spans two pages with 28 live tokens in the second.
    // The geometry must accept it; the copy is what trims it.
    const auto geometry = derivePagedKvGeometry(bundleKDims(), bundlePageTableDims(), {100, 64});
    EXPECT_EQ(geometry.denseSeqLen, 100);
}

TEST(PagedKvGather, RejectsALengthThePageTableCannotAddress)
{
    // 32 blocks x 64 tokens = 2048 addressable; 2049 needs a 33rd block.
    EXPECT_THROW(derivePagedKvGeometry(bundleKDims(), bundlePageTableDims(), {2049, 64}),
                 std::invalid_argument);
}

TEST(PagedKvGather, RejectsANonPositiveLength)
{
    // An empty sequence has no live KV, so there is nothing correct to gather.
    EXPECT_THROW(derivePagedKvGeometry(bundleKDims(), bundlePageTableDims(), {0, 64}),
                 std::invalid_argument);
}

TEST(PagedKvGather, RejectsASequenceCountMismatch)
{
    // The page table describes 2 sequences; three lengths means the caller and
    // the graph disagree about the batch, which would silently gather the wrong
    // rows rather than fail.
    EXPECT_THROW(derivePagedKvGeometry(bundleKDims(), bundlePageTableDims(), {1024, 2048, 512}),
                 std::invalid_argument);
}

TEST(PagedKvGather, RejectsAKvCacheOfTheWrongRank)
{
    EXPECT_THROW(derivePagedKvGeometry({kNumBlocks, kPageSize, kHeadSize},
                                       bundlePageTableDims(),
                                       {1024, 2048}),
                 std::invalid_argument);
}

TEST(PagedKvGather, RejectsAPageTableOfTheWrongRank)
{
    EXPECT_THROW(derivePagedKvGeometry(bundleKDims(), {kNumSeqs}, {1024, 2048}),
                 std::invalid_argument);
}

} // namespace
