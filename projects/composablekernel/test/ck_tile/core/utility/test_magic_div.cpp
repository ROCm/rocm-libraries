// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/core/utility/magic_div.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"

using namespace ck_tile;
using M16 = magic_division16_bit_range;

static void check16(uint32_t divisor, uint32_t dividend)
{
    auto magic        = M16::calculate_magic_numbers(divisor);
    uint32_t mult     = magic[number<0>{}];
    uint32_t shift    = magic[number<1>{}];
    uint32_t got      = M16::do_magic_division(dividend, mult, shift);
    uint32_t expected = dividend / divisor;
    EXPECT_EQ(got, expected) << "divisor=" << divisor << " dividend=" << dividend
                             << " multiplier=" << mult << " shift=" << shift;
}

// Regression: divisor=1, dividend=65536 returned 65537 before the fix.
TEST(MagicDivision16BitRange, DivisorOne)
{
    for(uint32_t n : {0u, 1u, 65535u, 65536u})
        check16(1, n);
}

// All power-of-two divisors trigger the same zero-numerator path.
TEST(MagicDivision16BitRange, PowerOfTwoDivisors)
{
    for(uint32_t s = 0; s <= 16; ++s)
    {
        uint32_t d = 1u << s;
        for(uint32_t n : {0u, 1u, 65535u, 65536u})
            check16(d, n);
    }
}

// Spot-check a handful of non-power-of-two divisors at boundary dividends.
TEST(MagicDivision16BitRange, NonPowerOfTwoDivisors)
{
    for(uint32_t d : {3u, 5u, 7u, 32769u, 65535u})
    {
        for(uint32_t n : {0u, d - 1u, d, 65535u, 65536u})
            check16(d, n);
    }
}
