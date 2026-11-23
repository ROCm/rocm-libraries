// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A 32-bit long (Windows) makes `1L << 33` undefined, so these inputs pin
// magicNumberAlg1's 33-bit shift (divisor 3) and 31-bit fallback (divisor 1).

#include <gtest/gtest.h>

#include <Tensile/MagicDivision.hpp>

#include <cstdint>

using namespace TensileLite;

TEST(MagicDivisionTest, Alg1Uses33BitShiftWhenTheMagicNumberFits)
{
    uint32_t       shift = 0;
    const uint32_t magic = magicNumberAlg1(3, &shift);

    EXPECT_EQ(shift, 33u);
    EXPECT_EQ(magic, 0xAAAAAAABu);
}

TEST(MagicDivisionTest, Alg1FallsBackTo31BitShiftWhenTheMagicNumberOverflows)
{
    uint32_t       shift = 0;
    const uint32_t magic = magicNumberAlg1(1, &shift);

    EXPECT_EQ(shift, 31u);
    EXPECT_EQ(magic, 0x80000001u);
}
