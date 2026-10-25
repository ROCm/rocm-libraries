// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// RocRollerWgmKernargs.hpp against the frozen libdivide 5.4.0 tables that
// tensilelite/Tests/unit/test_rocroller_wgm.py checks the Python oracle
// against. Edit both tables together so the two implementations stay in step.

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>
#include <Tensile/RocRollerWgmKernargs.hpp>

using namespace TensileLite;

namespace
{
    enum class MagicKind
    {
        S32,
        U32,
        S64
    };

    // For U32, second is MagicShifts as int32 (divisor 1 gives INT32_MIN).
    // For S32 and S64, second is MagicShiftAndSign.
    struct MagicCase
    {
        MagicKind kind;
        int64_t   divisor;
        int64_t   magic;
        int64_t   second;
    };

    constexpr MagicCase libdivideCases[] = {
        {MagicKind::S32, 0, 1073741823, 0},
        {MagicKind::S32, 1, 0, 0},
        {MagicKind::S32, -1, 0, 128},
        {MagicKind::S32, 2, 0, 1},
        {MagicKind::S32, 3, -1431655765, 65},
        {MagicKind::S32, 5, -858993459, 66},
        {MagicKind::S32, 6, -1431655765, 66},
        {MagicKind::S32, 7, -1840700269, 66},
        {MagicKind::S32, -5, -858993459, 194},
        {MagicKind::S32, 15, -2004318071, 67},
        {MagicKind::S32, 16, 0, 4},
        {MagicKind::S32, 17, -252645135, 68},
        {MagicKind::S32, 31, -2078209981, 68},
        {MagicKind::S32, 32, 0, 5},
        {MagicKind::S32, 100, -1546188226, 70},
        {MagicKind::S32, 127, -2130574327, 70},
        {MagicKind::S32, 128, 0, 7},
        {MagicKind::S32, 255, -2139062143, 71},
        {MagicKind::S32, 256, 0, 8},
        {MagicKind::S32, 1000, -2095944040, 73},
        {MagicKind::S32, 1024, 0, 10},
        {MagicKind::S32, 2147483647, -2147483646, 94},
        {MagicKind::S32, -2147483648, 0, 159},
        {MagicKind::S32, -2147483647, -2147483646, 222},
        {MagicKind::S32, 9, -477218588, 67},
        {MagicKind::S32, 10, -858993459, 67},
        {MagicKind::U32, 0, 2147483647, 0},
        {MagicKind::U32, 1, 0, -2147483648},
        {MagicKind::U32, 2, 0, 0},
        {MagicKind::U32, 3, 1431655766, 1},
        {MagicKind::U32, 4, 0, 1},
        {MagicKind::U32, 5, 2576980378, 2},
        {MagicKind::U32, 6, 1431655766, 2},
        {MagicKind::U32, 7, 613566757, 2},
        {MagicKind::U32, 16, 0, 3},
        {MagicKind::U32, 17, 3789677026, 4},
        {MagicKind::U32, 100, 1202590843, 6},
        {MagicKind::U32, 2147483648, 0, 30},
        {MagicKind::U32, 4294967295, 2, 31},
        {MagicKind::S64, 0, 4611686018427387903, 0},
        {MagicKind::S64, 1, 0, 0},
        {MagicKind::S64, -1, 0, 128},
        {MagicKind::S64, 2, 0, 1},
        {MagicKind::S64, 3, -6148914691236517205, 65},
        {MagicKind::S64, 7, -7905747460161236406, 66},
        {MagicKind::S64, 16, 0, 4},
        {MagicKind::S64, 100, -6640827866535438581, 70},
        {MagicKind::S64, 4294967296, 0, 32},
        {MagicKind::S64, 4294967297, -4294967295, 96},
        {MagicKind::S64, -5, -3689348814741910323, 194},
        {MagicKind::S64, std::numeric_limits<int64_t>::min(), 0, 191},
        {MagicKind::S64, 9223372036854775807, -9223372036854775806, 126},
    };

    // {numerator, divisor, rocRollerMagicDivS32}. Divisor 0 is not a
    // truncating division; it follows MagicMultiple's special case.
    constexpr int32_t magicDivCases[][3] = {
        {0, 0, 0},
        {1, 0, 1},
        {1, 1, 1},
        {5, 3, 1},
        {32, 2, 16},
        {-20, 6, -3},
        {7, 0, 8},
    };

    struct WgmCase
    {
        int64_t                 m;
        int64_t                 n;
        uint32_t                tileM;
        uint32_t                tileN;
        int32_t                 wgm;
        std::array<int64_t, 11> expected;
    };

    // expected is in RocRollerWgmKernargs field order (WGM_ROLE_ORDER).
    const WgmCase wgmCases[] = {
        {4096, 4096, 128, 256, 2, {16, 0, 0, 2, 1073741823, 1, 0, 4, 0, 9, 16}},
        {0, 0, 128, 32, 2, {0, 1073741823, 0, 2, 1073741823, 1, 1073741823, 0, 0, 0, 0}},
        {1, 1, 16, 256, 1, {1, 0, 0, 1, 1073741823, 0, 0, 0, 0, 0, 1}},
        {2147483647, 1048576, 32, 64, 2, {0, 1073741823, 0, 2, 1073741823, 1, 0, 14, 0, 0, 33554432}},
        {17, 19, 256, 256, 6, {0, 1073741823, -1431655765, 6, 0, 66, 0, 0, 0, 0, 0}},
        {5, 0, 64, 64, 2, {0, 1073741823, 0, 2, 0, 1, 1073741823, 0, 0, 0, 0}},
    };

    char const* toString(MagicKind kind)
    {
        switch(kind)
        {
        case MagicKind::S32:
            return "s32";
        case MagicKind::U32:
            return "u32";
        case MagicKind::S64:
            return "s64";
        }
        return "?";
    }

    std::pair<int64_t, int64_t> magic(MagicCase const& c)
    {
        switch(c.kind)
        {
        case MagicKind::S32:
        {
            auto d = static_cast<int32_t>(c.divisor);
            return {rocRollerMagicMultipleS32(d), rocRollerMagicShiftAndSignS32(d)};
        }
        case MagicKind::U32:
        {
            auto d = static_cast<uint32_t>(c.divisor);
            return {rocRollerMagicMultipleU32(d), static_cast<int32_t>(rocRollerMagicShiftsU32(d))};
        }
        case MagicKind::S64:
            return {rocRollerMagicMultipleS64(c.divisor), rocRollerMagicShiftAndSignS64(c.divisor)};
        }
        throw std::logic_error("unknown MagicKind");
    }

    std::array<int64_t, 11> fields(RocRollerWgmKernargs const& k)
    {
        return {k.quotientTilesByBlock,
                k.magicMultipleWgmMainBlock,
                k.magicMultipleWgm,
                k.workgroupMapping,
                k.magicMultipleWgmTail,
                k.magicShiftAndSignWgm,
                k.magicMultipleNumTilesN,
                k.magicShiftAndSignNumTilesN,
                k.magicShiftAndSignWgmTail,
                k.magicShiftAndSignWgmMainBlock,
                k.quotientTilesMByWgm};
    }
}

TEST(RocRollerWgmKernargs, MagicMatchesLibdivideBranchfree)
{
    for(auto const& c : libdivideCases)
    {
        SCOPED_TRACE(::testing::Message() << toString(c.kind) << " divisor=" << c.divisor);
        EXPECT_EQ(magic(c), std::make_pair(c.magic, c.second));
    }
}

TEST(RocRollerWgmKernargs, MagicDivMatchesRocRollerTree)
{
    for(auto const& [numerator, divisor, expected] : magicDivCases)
    {
        SCOPED_TRACE(::testing::Message() << numerator << " / " << divisor);
        EXPECT_EQ(rocRollerMagicDivS32(numerator, divisor), expected);
    }
}

TEST(RocRollerWgmKernargs, KernargsFollowTheHoistedFormula)
{
    for(auto const& c : wgmCases)
    {
        SCOPED_TRACE(::testing::Message() << "m=" << c.m << " n=" << c.n << " tile=" << c.tileM
                                          << "x" << c.tileN << " wgm=" << c.wgm);
        EXPECT_EQ(fields(evaluateRocRollerWgmKernargs(c.m, c.n, c.tileM, c.tileN, c.wgm)),
                  c.expected);
    }
}

TEST(RocRollerWgmKernargs, RejectsNonPowerOfTwoMacroTile)
{
    EXPECT_THROW(evaluateRocRollerWgmKernargs(128, 128, 96, 128), std::runtime_error);
    EXPECT_THROW(evaluateRocRollerWgmKernargs(128, 128, 128, 0), std::runtime_error);
}
