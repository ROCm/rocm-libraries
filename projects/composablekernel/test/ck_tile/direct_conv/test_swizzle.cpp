// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/ops/direct_convolution/utils/swizzle.hpp"

#include <set>

using namespace ck_tile::direct_conv;

// ---------------------------------------------------------------------------
// Typed test infrastructure
//
// Each SwizzleParam bundles a concrete swizzle type with the row count used
// when exercising it.  The row count mimics a realistic BLOCK_W value so that
// x % C8 wraps at least once (BLOCK_W > C8 is the common case).
// ---------------------------------------------------------------------------

template <typename Swizzle_, int NRows_>
struct SwizzleParam
{
    using Swizzle              = Swizzle_;
    static constexpr int NRows = NRows_;
};

// Instantiation list: (swizzle type, #rows to test over)
using SwizzleTypes = ::testing::Types<
    SwizzleParam<SwizzleT<64>, 10>,    // C8=8,  NRows=10 > C8 → x%C8 wraps
    SwizzleParam<SwizzleT<128>, 20>,   // C8=16, NRows=20 > C8 → x%C8 wraps
    SwizzleParam<SwizzleXOR<64>, 10>,  // C8=8,  NRows=10 > C8 → x%C8 wraps
    SwizzleParam<SwizzleXOR<128>, 20>  // C8=16, NRows=20 > C8 → x%C8 wraps
    >;

template <typename P>
class SwizzleTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(SwizzleTest, SwizzleTypes);

// ---------------------------------------------------------------------------
// 1. offset_uint4 and offset_uint2 are consistent:
//    offset_uint4(x, c8) == offset_uint2(x, c8*2) / 2
// ---------------------------------------------------------------------------
TYPED_TEST(SwizzleTest, Uint4ConsistentWithUint2)
{
    using Sw = typename TypeParam::Swizzle;

    for(int x = 0; x < TypeParam::NRows; x++)
    {
        for(int c8 = 0; c8 < Sw::C8; c8++)
        {
            EXPECT_EQ(Sw::offset_uint4(x, c8), Sw::offset_uint2(x, c8 * 2) / 2)
                << "x=" << x << " c8=" << c8;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Forward and inverse are consistent:
//    x(offset_uint4(x, c8)) == x
//    c8(offset_uint4(x, c8)) == c8
// ---------------------------------------------------------------------------
TYPED_TEST(SwizzleTest, InverseRoundtrip)
{
    using Sw = typename TypeParam::Swizzle;

    for(int x = 0; x < TypeParam::NRows; x++)
    {
        for(int c8 = 0; c8 < Sw::C8; c8++)
        {
            const int off = Sw::offset_uint4(x, c8);
            EXPECT_EQ(Sw::x(off), x) << "x=" << x << " c8=" << c8;
            EXPECT_EQ(Sw::c8(off), c8) << "x=" << x << " c8=" << c8;
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Per-row bijection:
//    For each row x, the C8 offsets produced by offset_uint4(x, 0..C8-1)
//    are all distinct and lie within the row's slot range [x*C8, (x+1)*C8).
// ---------------------------------------------------------------------------
TYPED_TEST(SwizzleTest, PerRowBijection)
{
    using Sw = typename TypeParam::Swizzle;

    for(int x = 0; x < TypeParam::NRows; x++)
    {
        std::set<int> seen;
        for(int c8 = 0; c8 < Sw::C8; c8++)
        {
            const int off = Sw::offset_uint4(x, c8);

            // Must fall within this row's contiguous slot range.
            EXPECT_GE(off, x * Sw::C8) << "x=" << x << " c8=" << c8;
            EXPECT_LT(off, (x + 1) * Sw::C8) << "x=" << x << " c8=" << c8;

            // Must not collide with another c8 in the same row.
            EXPECT_TRUE(seen.insert(off).second)
                << "Duplicate offset " << off << " at x=" << x << " c8=" << c8;
        }
    }
}

// ---------------------------------------------------------------------------
// 4. No LDS bank conflicts within a row (uint2 / 64-bit access).
//
// LDS has 32 banks of 4 bytes each.  A uint2 (8 bytes) spans 2 banks.
// The bank of the first 4-byte word of a uint2 at offset O (in uint2 units):
//   bank = (O * 2) % 32
//
// For C <= 64 (C4 <= 16 <= 32) all c4 values within one row are checked.
// For C = 128 (C4 = 32 = 32 banks) a single wave uses one half (C8 = 16
// consecutive c4 values); check each half separately as in the original test.
// ---------------------------------------------------------------------------
TYPED_TEST(SwizzleTest, NoBankConflicts)
{
    using Sw = typename TypeParam::Swizzle;

    // Number of c4 values checked per bank-conflict group.
    // When C4 < 32 (fits within 32 LDS banks) we check the whole row at once;
    // otherwise we split into groups of C8 (one wave per group), matching
    // the hardware access pattern where each wave accesses C8 consecutive
    // c4 values.
    const int group_size = (Sw::C4 < 32) ? Sw::C4 : Sw::C8;
    const int num_groups = Sw::C4 / group_size;

    for(int x = 0; x < TypeParam::NRows; x++)
    {
        for(int g = 0; g < num_groups; g++)
        {
            std::set<int> banks;
            for(int i = 0; i < group_size; i++)
            {
                const int c4  = g * group_size + i;
                const int off = Sw::offset_uint2(x, c4);
                // bank of the first 4-byte word: (byte_offset / 4) % 32
                // byte_offset = off * 8  =>  bank = (off * 2) % 32
                const int bank = (off * 2) % 32;
                EXPECT_TRUE(banks.insert(bank).second)
                    << "Bank conflict: x=" << x << " group=" << g
                    << " c4=" << c4 << " bank=" << bank;
            }
        }
    }
}
