// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/ops/direct_convolution/utils/swizzle.hpp"

#include <set>

using namespace ck_tile::direct_conv;

using Sw64 = SwizzleT<64>;
using Sw128 = SwizzleT<128>;

TEST(Swizzle, Uint4ConsistentWithUint2_64)
{
    // offset_uint4(x, c8) == offset_uint2(x, c8*2) / 2
    for(int x = 0; x < 8; x++)
    {
        for(int c8 = 0; c8 < Sw64::C8; c8++)
        {
            EXPECT_EQ(Sw64::offset_uint4(x, c8), Sw64::offset_uint2(x, c8 * 2) / 2)
                << "x=" << x << " c8=" << c8;
        }
    }
}

TEST(Swizzle, Uint4ConsistentWithUint2_128)
{
    for(int x = 0; x < 8; x++)
    {
        for(int c8 = 0; c8 < Sw128::C8; c8++)
        {
            EXPECT_EQ(Sw128::offset_uint4(x, c8), Sw128::offset_uint2(x, c8 * 2) / 2)
                << "x=" << x << " c8=" << c8;
        }
    }
}

TEST(Swizzle, InverseRoundtrip_64)
{
    // x(offset_uint4(x, c8)) == x and c8(offset_uint4(x, c8)) == c8
    for(int x = 0; x < 8; x++)
    {
        for(int c8 = 0; c8 < Sw64::C8; c8++)
        {
            int off = Sw64::offset_uint4(x, c8);
            EXPECT_EQ(Sw64::x(off), x) << "x=" << x << " c8=" << c8;
            EXPECT_EQ(Sw64::c8(off), c8) << "x=" << x << " c8=" << c8;
        }
    }
}

TEST(Swizzle, InverseRoundtrip_128)
{
    for(int x = 0; x < 8; x++)
    {
        for(int c8 = 0; c8 < Sw128::C8; c8++)
        {
            int off = Sw128::offset_uint4(x, c8);
            EXPECT_EQ(Sw128::x(off), x) << "x=" << x << " c8=" << c8;
            EXPECT_EQ(Sw128::c8(off), c8) << "x=" << x << " c8=" << c8;
        }
    }
}

TEST(Swizzle, NoBankConflicts_64)
{
    // For any two threads accessing the same row (x), their uint2 offsets
    // should map to different 32-byte banks.
    // Bank = (byte_offset / 4) % 32, where byte_offset = offset_uint2 * 8
    for(int x = 0; x < 8; x++)
    {
        std::set<int> banks;
        bool conflict = false;
        for(int c4 = 0; c4 < Sw64::C4; c4++)
        {
            int off = Sw64::offset_uint2(x, c4);
            int bank = (off * 8 / 4) % 32;
            if(banks.count(bank))
            {
                conflict = true;
                break;
            }
            banks.insert(bank);
        }
        EXPECT_FALSE(conflict) << "Bank conflict at x=" << x;
    }
}

TEST(Swizzle, NoBankConflicts_128)
{
    // With C=128, C4=32 total accesses per row, but each wave only accesses
    // C8=16 consecutive c4 values. Check bank conflicts within each wave's group.
    for(int x = 0; x < 8; x++)
    {
        for(int wave = 0; wave < 2; wave++)
        {
            std::set<int> banks;
            bool conflict = false;
            int c4_start = wave * Sw128::C8;
            for(int i = 0; i < Sw128::C8; i++)
            {
                int off = Sw128::offset_uint2(x, c4_start + i);
                int bank = (off * 8 / 4) % 32;
                if(banks.count(bank))
                {
                    conflict = true;
                    break;
                }
                banks.insert(bank);
            }
            EXPECT_FALSE(conflict) << "Bank conflict at x=" << x << " wave=" << wave;
        }
    }
}
