// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of checked-in kernel traits against ranked tiles.
// The relative include avoids shadowing the clients' utility.hpp with the
// library's internal header of the same name.

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocroller_kernel_match.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace
{
    constexpr RocRollerKernelTraits kernel(int m, int n, int k, bool sk, bool wgm)
    {
        RocRollerKernelTraits traits{};
        traits.m                = m;
        traits.n                = n;
        traits.k                = k;
        traits.streamK          = sk;
        traits.workgroupMapping = wgm;
        return traits;
    }

    constexpr RocRollerSelectedTile
        tile(int m, int n, int k, bool wgm, bool sk, bool ntA = false, bool ntB = false)
    {
        RocRollerSelectedTile t{};
        t.m                = m;
        t.n                = n;
        t.k                = k;
        t.workgroupMapping = wgm;
        t.streamK          = sk;
        t.nonTemporalA     = ntA;
        t.nonTemporalB     = ntB;
        return t;
    }

    constexpr RocRollerKernelTraits plain16x16 = kernel(16, 16, 128, false, false);
    constexpr RocRollerKernelTraits sk16x16    = kernel(16, 16, 128, true, false);
    constexpr RocRollerKernelTraits sk64x32    = kernel(64, 32, 128, true, false);
    constexpr RocRollerKernelTraits wgm256x256 = kernel(256, 256, 128, false, true);
    constexpr RocRollerKernelTraits sk256x192  = kernel(256, 192, 128, true, false);

    struct MatchCase
    {
        RocRollerKernelTraits kernel;
        RocRollerSelectedTile tile;
        bool                  expected;
        const char*           why;
    };

    const std::vector<MatchCase>& matchCases()
    {
        static const std::vector<MatchCase> cases = {
            {sk16x16, tile(16, 16, 128, true, true), true, "SK kernel matches its SK tile"},
            {sk64x32, tile(64, 32, 128, true, true), true, "SK 64x32x128"},
            {sk256x192, tile(256, 192, 128, true, true), true, "SK 256x192x128"},
            {plain16x16,
             tile(16, 16, 128, false, false),
             true,
             "plain kernel matches its plain tile"},
            {wgm256x256, tile(256, 256, 128, true, false), true, "WGM kernel matches its WGM tile"},
            {plain16x16,
             tile(16, 16, 128, true, true),
             false,
             "an SK tile must not take the plain kernel of the same tile"},
            {sk16x16,
             tile(16, 16, 128, false, false),
             false,
             "a plain tile must not take the SK kernel of the same tile"},
            {sk64x32,
             tile(16, 16, 128, true, true),
             false,
             "a different tile shape must not match"},
            {plain16x16,
             tile(16, 16, 256, false, false),
             false,
             "a different depth must not match"},
            {wgm256x256,
             tile(256, 256, 128, false, false),
             false,
             "a non-WGM tile must not take a WGM kernel"},
            {plain16x16,
             tile(16, 16, 128, true, false),
             false,
             "a WGM tile must not take a kernel without workgroup mapping"},
            {plain16x16,
             tile(16, 16, 128, false, false, true, false),
             false,
             "no non-temporal-A kernel is checked in"},
            {plain16x16,
             tile(16, 16, 128, false, false, false, true),
             false,
             "no non-temporal-B kernel is checked in"},
        };
        return cases;
    }
}

TEST(HostUnitRocRollerKernelMatch, smoke_MatchesOnlyTheKernelImplementingThatTile)
{
    for(const auto& c : matchCases())
    {
        SCOPED_TRACE(::testing::Message()
                     << c.why << "\n  kernel: " << c.kernel.m << "x" << c.kernel.n << "x"
                     << c.kernel.k << " sk=" << c.kernel.streamK
                     << " wgm=" << c.kernel.workgroupMapping << "\n  tile:   " << c.tile.m << "x"
                     << c.tile.n << "x" << c.tile.k << " wgm=" << c.tile.workgroupMapping
                     << " sk=" << c.tile.streamK);
        EXPECT_EQ(rocRollerKernelMatchesTile(c.kernel, c.tile), c.expected);
    }
}

// A StreamK kernel never maps workgroups, so a StreamK tile with
// workgroupMapping set must still match it. A StreamK kernel that did declare
// the mapping kernarg, which the generator does not emit, matches no tile.
TEST(HostUnitRocRollerKernelMatch, smoke_StreamKIgnoresWorkgroupMappingOnTheTile)
{
    EXPECT_TRUE(rocRollerKernelMatchesTile(sk16x16, tile(16, 16, 128, true, true)));
    EXPECT_TRUE(rocRollerKernelMatchesTile(sk16x16, tile(16, 16, 128, false, true)));

    const RocRollerKernelTraits skPlusWgm = kernel(16, 16, 128, true, true);
    EXPECT_FALSE(rocRollerKernelMatchesTile(skPlusWgm, tile(16, 16, 128, true, true)));
    EXPECT_FALSE(rocRollerKernelMatchesTile(skPlusWgm, tile(16, 16, 128, false, true)));
}

// The fallback's order, first difference deciding: ranked position of the
// kernel's tile, then the larger M x N, then fewer WGM/StreamK differences
// from what the ranker gives that tile, then the lower solution index.
TEST(HostUnitRocRollerKernelMatch, smoke_FallbackPrefersTheEarliestRankedTile)
{
    const std::vector<RocRollerSelectedTile> ranked
        = {tile(64, 64, 128, true, false), tile(256, 256, 128, true, false)};

    const auto early
        = rocRollerFallbackRank(kernel(64, 64, 128, false, false), ranked, 1024, 1024, 1024, 9);
    const auto late = rocRollerFallbackRank(wgm256x256, ranked, 1024, 1024, 1024, 1);
    const auto none
        = rocRollerFallbackRank(kernel(512, 512, 128, false, true), ranked, 1024, 1024, 1024, 0);
    EXPECT_EQ(early.tilePosition, 0u);
    EXPECT_EQ(late.tilePosition, 1u);
    EXPECT_EQ(none.tilePosition, ranked.size());
    EXPECT_LT(early, late);
    EXPECT_LT(late, none);
}

TEST(HostUnitRocRollerKernelMatch, smoke_FallbackPrefersTheLargerTileAmongUnlisted)
{
    const std::vector<RocRollerSelectedTile> ranked = {tile(32, 32, 128, false, false)};

    const auto big
        = rocRollerFallbackRank(kernel(256, 128, 128, false, false), ranked, 1024, 1024, 1024, 7);
    const auto small
        = rocRollerFallbackRank(kernel(128, 128, 128, false, false), ranked, 1024, 1024, 1024, 2);
    EXPECT_EQ(big.area, 256 * 128);
    EXPECT_LT(big, small);
}

TEST(HostUnitRocRollerKernelMatch, smoke_FallbackCountsDifferencesFromTheRankedVariant)
{
    // The ranker asks for 128x128 with WGM: the WGM kernel differs in nothing,
    // the plain one in WGM, the StreamK one in both.
    const std::vector<RocRollerSelectedTile> ranked = {tile(128, 128, 128, true, false)};

    const auto wgm
        = rocRollerFallbackRank(kernel(128, 128, 128, false, true), ranked, 1024, 1024, 1024, 5);
    const auto plain
        = rocRollerFallbackRank(kernel(128, 128, 128, false, false), ranked, 1024, 1024, 1024, 3);
    const auto sk
        = rocRollerFallbackRank(kernel(128, 128, 128, true, false), ranked, 1024, 1024, 1024, 1);
    EXPECT_EQ(wgm.mismatches, 0);
    EXPECT_EQ(plain.mismatches, 1);
    EXPECT_EQ(sk.mismatches, 2);
    EXPECT_LT(wgm, plain);
    EXPECT_LT(plain, sk);

    // A StreamK tile wants no workgroup mapping even when the ranker set it.
    const std::vector<RocRollerSelectedTile> skRanked = {tile(128, 128, 128, true, true)};
    EXPECT_EQ(
        rocRollerFallbackRank(kernel(128, 128, 128, true, false), skRanked, 1024, 1024, 1024, 0)
            .mismatches,
        0);
}

// An unlisted tile wants WGM exactly when the ranker would give it WGM.
TEST(HostUnitRocRollerKernelMatch, smoke_FallbackWantsWgmOnAnUnlistedTileAsTheRankerWould)
{
    const std::vector<RocRollerSelectedTile> ranked = {tile(16, 16, 128, false, false)};
    const RocRollerKernelTraits              wgm    = kernel(128, 128, 128, false, true);
    const RocRollerKernelTraits              plain  = kernel(128, 128, 128, false, false);

    // 32 x 32 = 1024 tiles of 128x128: only K decides.
    EXPECT_EQ(rocRollerFallbackRank(wgm, ranked, 4096, 4096, 4095, 0).mismatches, 1);
    EXPECT_EQ(rocRollerFallbackRank(plain, ranked, 4096, 4096, 4095, 0).mismatches, 0);
    EXPECT_EQ(rocRollerFallbackRank(wgm, ranked, 4096, 4096, 4096, 0).mismatches, 0);
    EXPECT_EQ(rocRollerFallbackRank(plain, ranked, 4096, 4096, 4096, 0).mismatches, 1);

    // 8 x 8 = 64 tiles: too small a grid for WGM at any K.
    EXPECT_EQ(rocRollerFallbackRank(wgm, ranked, 1024, 1024, 16384, 0).mismatches, 1);
    EXPECT_EQ(rocRollerFallbackRank(plain, ranked, 1024, 1024, 16384, 0).mismatches, 0);
}

TEST(HostUnitRocRollerKernelMatch, smoke_FallbackBreaksAFullTieOnTheSolutionIndex)
{
    const std::vector<RocRollerSelectedTile> ranked = {tile(64, 64, 128, false, false)};

    std::vector<RocRollerFallbackRank> ranks
        = {rocRollerFallbackRank(plain16x16, ranked, 1024, 1024, 1024, 12),
           rocRollerFallbackRank(plain16x16, ranked, 1024, 1024, 1024, 4),
           rocRollerFallbackRank(plain16x16, ranked, 1024, 1024, 1024, 8)};
    std::sort(ranks.begin(), ranks.end());
    EXPECT_EQ(ranks[0].solutionIndex, 4);
    EXPECT_EQ(ranks[1].solutionIndex, 8);
    EXPECT_EQ(ranks[2].solutionIndex, 12);
}

TEST(HostUnitRocRollerKernelMatch, smoke_OutputTilesRoundUpEachDimension)
{
    EXPECT_EQ(rocRollerOutputTiles(256, 256, 128, 128), 4);
    EXPECT_EQ(rocRollerOutputTiles(257, 129, 128, 128), 6);
    EXPECT_EQ(rocRollerOutputTiles(0, 256, 128, 128), 0);
    EXPECT_EQ(rocRollerOutputTiles(256, 256, 0, 128), 0);
}

// WGM only for K >= 4096, MT0 >= 128 and at least 512 output tiles; each
// bound is checked on both sides with the other two satisfied.
TEST(HostUnitRocRollerKernelMatch, smoke_WorkgroupMappingNeedsLongKWideTilesAndALargeGrid)
{
    // 16 x 32 = 512 tiles of 128x128.
    EXPECT_TRUE(rocRollerWantsWorkgroupMapping(2048, 4096, 4096, 128, 128));
    EXPECT_FALSE(rocRollerWantsWorkgroupMapping(2048, 4096, 4095, 128, 128));

    // N = 3969 still cuts into 32 columns; 3968 into 31, 496 tiles.
    EXPECT_TRUE(rocRollerWantsWorkgroupMapping(2048, 3969, 4096, 128, 128));
    EXPECT_FALSE(rocRollerWantsWorkgroupMapping(2048, 3968, 4096, 128, 128));

    // Narrow tiles never, however large the grid.
    EXPECT_TRUE(rocRollerWantsWorkgroupMapping(8192, 8192, 8192, 128, 64));
    EXPECT_FALSE(rocRollerWantsWorkgroupMapping(8192, 8192, 8192, 64, 128));
    EXPECT_FALSE(rocRollerWantsWorkgroupMapping(8192, 8192, 8192, 16, 16));
}

TEST(HostUnitRocRollerKernelMatch, smoke_UsableCUsHonourOnlyASmallerRequest)
{
    EXPECT_EQ(rocRollerUsableCUs(0, 256), 256);
    EXPECT_EQ(rocRollerUsableCUs(-1, 256), 256);
    EXPECT_EQ(rocRollerUsableCUs(80, 256), 80);
    EXPECT_EQ(rocRollerUsableCUs(256, 256), 256);
    EXPECT_EQ(rocRollerUsableCUs(512, 256), 256);
}

TEST(HostUnitRocRollerKernelMatch, smoke_LastWaveFill)
{
    const RocRollerKernelTraits t128 = kernel(128, 128, 128, false, false);

    // 16 tiles on 256 CUs: one sixteenth busy.
    EXPECT_DOUBLE_EQ(rocRollerLastWaveFill(t128, 512, 512, 1, 256), 16.0 / 256);
    // 256 tiles: one full wave.
    EXPECT_DOUBLE_EQ(rocRollerLastWaveFill(t128, 2048, 2048, 1, 256), 1.0);
    // 260 tiles (N rounds up to 13 columns): a second wave of 4.
    EXPECT_DOUBLE_EQ(rocRollerLastWaveFill(t128, 2560, 1540, 1, 256), 260.0 / 512);
    // The batch multiplies the grid.
    EXPECT_DOUBLE_EQ(rocRollerLastWaveFill(t128, 512, 512, 16, 256), 1.0);
    // StreamK sizes its grid to the CUs.
    EXPECT_DOUBLE_EQ(rocRollerLastWaveFill(kernel(128, 128, 128, true, false), 512, 512, 1, 256),
                     1.0);
    // Nothing to fill, or nothing to fill it with.
    EXPECT_DOUBLE_EQ(rocRollerLastWaveFill(t128, 0, 512, 1, 256), 0.0);
    EXPECT_DOUBLE_EQ(rocRollerLastWaveFill(t128, 512, 512, 1, 0), 0.0);
}

TEST(HostUnitRocRollerKernelMatch, smoke_FirstFillingKeepsTheHeadUnlessAnEarlyEntryFills)
{
    const std::vector<double> fills = {0.25, 0.4, 0.6, 0.9, 1.0, 1.0};

    EXPECT_EQ(rocRollerFirstFilling(fills, rocRollerRankedFillWindow, rocRollerRankedMinFill), 2u);
    EXPECT_EQ(rocRollerFirstFilling(fills, fills.size(), rocRollerFallbackMinFill), 3u);
    // Outside the window does not count, and no hit keeps the head.
    EXPECT_EQ(rocRollerFirstFilling(fills, 2, rocRollerRankedMinFill), 0u);
    EXPECT_EQ(rocRollerFirstFilling({0.1, 0.2, 0.3}, 5, rocRollerRankedMinFill), 0u);
    EXPECT_EQ(rocRollerFirstFilling({}, 5, rocRollerRankedMinFill), 0u);
    // A head that already fills stays.
    EXPECT_EQ(rocRollerFirstFilling({0.5, 1.0}, 5, rocRollerRankedMinFill), 0u);
}
