// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// RocRollerStreamKKernargs.hpp against a frozen table that
// tensilelite/tensilelite/RocRollerStreamK.py checks its Python oracle
// against. The oracle evaluates the .expression trees rocRoller serialises
// into each kernel's .amdgpu_metadata, so this table is the contract between
// the two implementations -- edit both together.

#include <array>
#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>
#include <Tensile/RocRollerStreamKKernargs.hpp>

using namespace TensileLite;

namespace
{
    struct StreamKCase
    {
        int64_t                  m;
        int64_t                  n;
        int64_t                  k;
        uint32_t                 tileM;
        uint32_t                 tileN;
        uint32_t                 tileK;
        uint32_t                 numWGs;
        std::array<uint32_t, 10> expected;
    };

    // expected is in RocRollerStreamKKernargs field order.
    const StreamKCase streamKCases[] = {
        // tiles (65536) > numWGs - 1: Stream-K remainder plus full DP rounds.
        {4096, 4096, 8192, 16, 16, 128, 224, {256, 224, 22528, 101, 4171776, 18624, 0, 5, 0, 7}},
        // tiles (256) just over numWGs: one SK round, fullRounds == 0.
        {256, 256, 32768, 16, 16, 128, 224, {16, 224, 65536, 293, 0, 0, 0, 7, 0, 3}},
        // tiles (128) <= numWGs - 1: all Stream-K, zero data-parallel.
        {32, 4096, 16384, 32, 32, 128, 304, {128, 304, 16384, 54, 0, 0, 0, 6, 0, 6}},
        // MT_N == 192 is not a power of two: ntN is a real ceil-divide (43),
        // and its libdivide magic is the non-trivial branch.
        {8192, 8192, 1024, 256, 192, 128, 224, {43, 224, 2048, 10, 8960, 40, 0, 2, 2097542168, 5}},
        {1024, 2048, 4096, 128, 128, 128, 256, {16, 256, 4096, 16, 0, 0, 0, 4, 0, 3}},
        // Degenerate: one tile, one iteration, one workgroup. iters == 1 takes
        // rocRollerMagicShiftsU32's 1u << 31 special case.
        {16, 16, 128, 16, 16, 128, 1, {1, 1, 1, 1, 0, 0, 0, 2147483648u, 0, 2147483648u}},
        // MT_N == 192 again, at the flag-region grid cap.
        {4096, 4096, 8192, 256, 192, 128, 2048, {22, 2048, 22528, 11, 0, 0, 0, 5, 1952257862, 4}},
    };

    std::array<uint32_t, 10> fields(RocRollerStreamKKernargs const& k)
    {
        return {k.numTiles1,
                k.numWGs,
                k.numSKTiles,
                k.numSKTilesPerWG,
                k.numDPTiles,
                k.numDPTilesPerWG,
                k.magicMultipleIters,
                k.magicShiftsIters,
                k.magicMultipleNumTiles1,
                k.magicShiftsNumTiles1};
    }
}

TEST(RocRollerStreamKKernargs, KernargsFollowTheHoistedFormula)
{
    for(auto const& c : streamKCases)
    {
        SCOPED_TRACE(::testing::Message()
                     << "m=" << c.m << " n=" << c.n << " k=" << c.k << " tile=" << c.tileM << "x"
                     << c.tileN << "x" << c.tileK << " numWGs=" << c.numWGs);
        EXPECT_EQ(fields(evaluateRocRollerStreamKKernargs(
                      c.m, c.n, c.k, c.tileM, c.tileN, c.tileK, c.numWGs)),
                  c.expected);
    }
}

TEST(RocRollerStreamKKernargs, NonPowerOfTwoMacroTileN)
{
    // 8192 / 192 is 42.67: ntN must be the ceil (43), not a shift.
    auto k = evaluateRocRollerStreamKKernargs(8192, 8192, 1024, 256, 192, 128, 224);
    EXPECT_EQ(k.numTiles1, 43u);
    EXPECT_EQ(k.magicMultipleNumTiles1, rocRollerMagicMultipleU32(43u));
    EXPECT_EQ(k.magicShiftsNumTiles1, rocRollerMagicShiftsU32(43u));
}

TEST(RocRollerStreamKKernargs, AllStreamKWhenTilesFitTheGrid)
{
    // tiles == 128, numWGs == 304: every tile is a Stream-K tile.
    auto k = evaluateRocRollerStreamKKernargs(32, 4096, 16384, 32, 32, 128, 304);
    EXPECT_EQ(k.numDPTiles, 0u);
    EXPECT_EQ(k.numDPTilesPerWG, 0u);
    EXPECT_EQ(k.numSKTiles, 128u * 128u);
}

TEST(RocRollerStreamKKernargs, KIsTruncatedNotRounded)
{
    // K = 200 with MT_K = 128 is one iteration, not two.
    auto k = evaluateRocRollerStreamKKernargs(16, 16, 200, 16, 16, 128, 1);
    EXPECT_EQ(k.numSKTiles, 1u);
}

TEST(RocRollerStreamKKernargs, RejectsZeroMacroTileOrGrid)
{
    EXPECT_THROW(evaluateRocRollerStreamKKernargs(4096, 4096, 8192, 16, 16, 128, 0),
                 std::runtime_error);
    EXPECT_THROW(evaluateRocRollerStreamKKernargs(4096, 4096, 8192, 0, 16, 128, 224),
                 std::runtime_error);
    EXPECT_THROW(evaluateRocRollerStreamKKernargs(4096, 4096, 8192, 16, 0, 128, 224),
                 std::runtime_error);
    EXPECT_THROW(evaluateRocRollerStreamKKernargs(4096, 4096, 8192, 16, 16, 0, 224),
                 std::runtime_error);
}
