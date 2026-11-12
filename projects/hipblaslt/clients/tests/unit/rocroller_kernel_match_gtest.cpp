// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of a checked-in kernel's declared fields against ranked tiles.
// The relative include avoids shadowing the clients' utility.hpp with the
// library's internal header of the same name.

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocroller_kernel_match.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace
{
    // {tileM, tileN, tileK, unroll, workgroupMapping, streamK}
    constexpr RocRollerKernelFields plain16x16{16, 16, 128, 2, false, false};
    constexpr RocRollerKernelFields sk16x16{16, 16, 128, 2, false, true};
    constexpr RocRollerKernelFields wgm256x128{256, 128, 128, 2, true, false};
    constexpr RocRollerKernelFields fp4Unroll4{256, 192, 128, 4, false, true};

    // {m, n, k, workgroupMapping, streamK, nonTemporalA, nonTemporalB, unroll}
    constexpr RocRollerSelectedTile plainTile16x16{16, 16, 128, false, false, false, false, 2};
    constexpr RocRollerSelectedTile skTile16x16{16, 16, 128, true, true, false, false, 2};
    constexpr RocRollerSelectedTile wgmTile256x128{256, 128, 128, true, false, false, false, 2};
    constexpr RocRollerSelectedTile skTile256x192{256, 192, 128, true, true, false, false, 4};

    struct MatchCase
    {
        RocRollerKernelFields kernel;
        RocRollerSelectedTile tile;
        bool                  expected;
        const char*           why;
    };

    const std::vector<MatchCase>& matchCases()
    {
        static const std::vector<MatchCase> cases = {
            {plain16x16, plainTile16x16, true, "plain kernel matches its plain tile"},
            {sk16x16, skTile16x16, true, "Stream-K kernel matches its Stream-K tile"},
            {wgm256x128, wgmTile256x128, true, "WGM kernel matches its WGM tile"},
            {fp4Unroll4, skTile256x192, true, "unroll 4 matches unroll 4"},
            {plain16x16, skTile16x16, false, "a Stream-K tile must not take the plain kernel"},
            {sk16x16, plainTile16x16, false, "a plain tile must not take the Stream-K kernel"},
            {wgm256x128,
             {256, 128, 128, false, false, false, false, 2},
             false,
             "a tile without WGM must not take a WGM kernel"},
            {plain16x16,
             {16, 16, 128, true, false, false, false, 2},
             false,
             "a WGM tile must not take a kernel without the WGM kernarg"},
            {plain16x16,
             {16, 16, 64, false, false, false, false, 2},
             false,
             "a different K extent must not match"},
            {plain16x16,
             {32, 16, 128, false, false, false, false, 2},
             false,
             "a different M extent must not match"},
            {plain16x16,
             {16, 16, 128, false, false, false, false, 4},
             false,
             "a different unroll must not match"},
            {{16, 16, 128, 0, false, false},
             {16, 16, 128, false, false, false, false, 0},
             false,
             "a kernel that declares no unroll never matches"},
        };
        return cases;
    }
}

TEST(HostUnitRocRollerKernelMatch, smoke_MatchesOnlyTheKernelDescribingThatTile)
{
    for(const auto& c : matchCases())
    {
        SCOPED_TRACE(::testing::Message()
                     << c.why << "\n  tile: " << c.tile.m << "x" << c.tile.n << "x" << c.tile.k
                     << " unroll=" << c.tile.unroll << " wgm=" << c.tile.workgroupMapping
                     << " sk=" << c.tile.streamK);
        EXPECT_EQ(rocRollerKernelMatchesTile(c.kernel, c.tile), c.expected);
    }
}

// rocRoller turns workgroup mapping off for Stream-K, so a Stream-K tile
// matches its kernel whether or not the tile requests it, and never a Stream-K
// kernel that declares the WGM kernarg.
TEST(HostUnitRocRollerKernelMatch, smoke_StreamKIgnoresWorkgroupMappingOnTheTile)
{
    constexpr RocRollerSelectedTile skNoWgm{16, 16, 128, false, true, false, false, 2};
    EXPECT_TRUE(rocRollerKernelMatchesTile(sk16x16, skTile16x16));
    EXPECT_TRUE(rocRollerKernelMatchesTile(sk16x16, skNoWgm));

    constexpr RocRollerKernelFields skWithWgm{16, 16, 128, 2, true, true};
    EXPECT_FALSE(rocRollerKernelMatchesTile(skWithWgm, skTile16x16));
}

// Non-temporal hints only set cache bits on the loads, so one kernel serves
// every combination of them.
TEST(HostUnitRocRollerKernelMatch, smoke_NonTemporalHintsDoNotMatter)
{
    for(const bool a : {false, true})
        for(const bool b : {false, true})
        {
            RocRollerSelectedTile tile = plainTile16x16;
            tile.nonTemporalA          = a;
            tile.nonTemporalB          = b;
            EXPECT_TRUE(rocRollerKernelMatchesTile(plain16x16, tile)) << a << b;
        }
}
