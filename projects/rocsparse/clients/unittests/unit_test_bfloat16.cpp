/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

//
// Unit tests for the rocsparse_bfloat16 type of
//   library/include/rocsparse_bfloat16.h
// (constructors, rounding modes, conversions, arithmetic, comparisons and the
// std:: helpers). The full struct is only defined when the header is compiled
// as HIP, so this TU is built with `-x hip --offload-host-only` (see
// cmake/unit_tests_host.cmake): host code only, no device code object, no GPU needed.
//
#include "rocsparse_bfloat16.h"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <sstream>

namespace
{
    float bits_to_float(uint32_t bits)
    {
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    rocsparse_bfloat16 from_data(uint16_t data)
    {
        rocsparse_bfloat16 b;
        b.data = data;
        return b;
    }

    // Opaque to the optimizer so the constexpr std:: helpers run at runtime.
    rocsparse_bfloat16 opaque(rocsparse_bfloat16 v)
    {
        volatile uint16_t data = v.data;
        return from_data(data);
    }
}

TEST(bfloat16, constructors)
{
    EXPECT_EQ(rocsparse_bfloat16(1.0f).data, 0x3F80);
    EXPECT_EQ(rocsparse_bfloat16(-2.0).data, 0xC000);
    EXPECT_EQ(rocsparse_bfloat16(int32_t(3)).data, 0x4040);
    EXPECT_EQ(rocsparse_bfloat16(int64_t(-4)).data, 0xC080);
    EXPECT_EQ(rocsparse_bfloat16(0.0f).data, 0x0000);
}

// Float bit patterns with the bfloat16 each rounding mode must give.
TEST(bfloat16, rounding_modes)
{
    using bf = rocsparse_bfloat16;
    const struct
    {
        uint32_t bits;
        uint16_t near_even, near_zero, truncate;
    } cases[] = {
        // Low 16 bits exactly 0x8000 is a tie: near-even rounds to even, near-zero toward zero.
        {0x3F808000u, 0x3F80, 0x3F80, 0x3F80}, // even mantissa
        {0x3F818000u, 0x3F82, 0x3F81, 0x3F81}, // odd mantissa
        // Just above the tie, and all low bits set.
        {0x3F808001u, 0x3F81, 0x3F81, 0x3F80},
        {0x3F80FFFFu, 0x3F81, 0x3F81, 0x3F80},
        {0xBF80FFFFu, 0xBF81, 0xBF81, 0xBF80},
        // Infinity and a quiet NaN keep their encoding.
        {0x7F800000u, 0x7F80, 0x7F80, 0x7F80},
        {0x7FC00000u, 0x7FC0, 0x7FC0, 0x7FC0},
        // A NaN with payload only in the low 16 bits must stay a NaN.
        {0x7F800001u, 0x7F81, 0x7F81, 0x7F81},
    };
    for(const auto& c : cases)
    {
        const float f = bits_to_float(c.bits);
        EXPECT_EQ(bf(f).data, c.near_even) << std::hex << c.bits;
        EXPECT_EQ(bf(f, bf::rocsparse_round_near_even).data, c.near_even) << std::hex << c.bits;
        EXPECT_EQ(bf(f, bf::rocsparse_round_near_zero).data, c.near_zero) << std::hex << c.bits;
        EXPECT_EQ(bf(f, bf::rocsparse_truncate).data, c.truncate) << std::hex << c.bits;
    }
}

TEST(bfloat16, assign_from_float_truncates)
{
    rocsparse_bfloat16  b(0.0f);
    rocsparse_bfloat16& r = (b = bits_to_float(0x3F80FFFFu));
    EXPECT_EQ(&r, &b);
    EXPECT_EQ(b.data, 0x3F80);
}

TEST(bfloat16, conversions)
{
    const rocsparse_bfloat16 b(-2.5f);
    EXPECT_EQ(static_cast<float>(b), -2.5f);
    EXPECT_EQ(static_cast<double>(b), -2.5);
    EXPECT_EQ(static_cast<int32_t>(b), -2);
    EXPECT_EQ(static_cast<int64_t>(b), -2);
    EXPECT_TRUE(static_cast<bool>(b));
    EXPECT_FALSE(static_cast<bool>(rocsparse_bfloat16(0.0f)));
    // Negative zero is still false.
    EXPECT_FALSE(static_cast<bool>(from_data(0x8000)));

    std::ostringstream os;
    os << rocsparse_bfloat16(1.5f);
    EXPECT_EQ(os.str(), "1.5");
}

TEST(bfloat16, unary_and_binary_arithmetic)
{
    using bf = rocsparse_bfloat16;
    const bf a(6.0f);
    const bf b(2.0f);

    EXPECT_EQ((+a).data, a.data);
    EXPECT_EQ((-a).data, 0xC0C0);
    EXPECT_EQ(float(a + b), 8.0f);
    EXPECT_EQ(float(a - b), 4.0f);
    EXPECT_EQ(float(a * b), 12.0f);
    EXPECT_EQ(float(a / b), 3.0f);
    EXPECT_EQ(0.5f * b, 1.0f);
}

TEST(bfloat16, compound_assignment)
{
    using bf = rocsparse_bfloat16;
    bf x(1.0f);

    EXPECT_EQ(float(x += bf(2.0f)), 3.0f);
    EXPECT_EQ(float(x += 1.0f), 4.0f);
    EXPECT_EQ(float(x -= bf(1.0f)), 3.0f);
    EXPECT_EQ(float(x -= 1.0f), 2.0f);
    EXPECT_EQ(float(x *= bf(3.0f)), 6.0f);
    EXPECT_EQ(float(x *= 2.0f), 12.0f);
    EXPECT_EQ(float(x /= bf(4.0f)), 3.0f);
    EXPECT_EQ(float(x /= 3.0f), 1.0f);

    float f = 1.0f;
    EXPECT_EQ(f += bf(2.0f), 3.0f);
    EXPECT_EQ(f -= bf(1.0f), 2.0f);
    EXPECT_EQ(f *= bf(4.0f), 8.0f);
    EXPECT_EQ(f /= bf(2.0f), 4.0f);
}

TEST(bfloat16, increment_decrement)
{
    rocsparse_bfloat16 x(1.0f);

    EXPECT_EQ(float(++x), 2.0f);
    EXPECT_EQ(float(--x), 1.0f);

    const rocsparse_bfloat16 post_inc = x++;
    EXPECT_EQ(float(post_inc), 1.0f);
    EXPECT_EQ(float(x), 2.0f);

    const rocsparse_bfloat16 post_dec = x--;
    EXPECT_EQ(float(post_dec), 2.0f);
    EXPECT_EQ(float(x), 1.0f);
}

TEST(bfloat16, comparisons)
{
    using bf = rocsparse_bfloat16;
    const bf one(1.0f);
    const bf two(2.0f);

    EXPECT_TRUE(one < two);
    EXPECT_FALSE(two < one);
    EXPECT_TRUE(two > one);
    EXPECT_FALSE(one > two);
    EXPECT_TRUE(one <= two);
    EXPECT_TRUE(one <= one);
    EXPECT_FALSE(two <= one);
    EXPECT_TRUE(two >= one);
    EXPECT_TRUE(one >= one);
    EXPECT_FALSE(one >= two);
    EXPECT_TRUE(one == bf(1.0f));
    EXPECT_FALSE(one == two);
    EXPECT_TRUE(one != two);
    EXPECT_FALSE(one != bf(1.0f));
    EXPECT_TRUE(one != 2);
    EXPECT_FALSE(one != 1);
    // +0 and -0 compare equal through float.
    EXPECT_TRUE(from_data(0x0000) == from_data(0x8000));
}

TEST(bfloat16, std_helpers)
{
    using bf      = rocsparse_bfloat16;
    const bf inf  = opaque(from_data(0x7F80));
    const bf ninf = opaque(from_data(0xFF80));
    const bf qnan = opaque(from_data(0x7FC0));
    const bf one  = opaque(bf(1.0f));
    const bf zero = opaque(from_data(0x0000));
    const bf nzer = opaque(from_data(0x8000));

    EXPECT_TRUE(std::isinf(inf));
    EXPECT_TRUE(std::isinf(ninf));
    EXPECT_FALSE(std::isinf(qnan));
    EXPECT_FALSE(std::isinf(one));

    EXPECT_TRUE(std::isnan(qnan));
    EXPECT_FALSE(std::isnan(inf));
    EXPECT_FALSE(std::isnan(one));

    EXPECT_TRUE(std::iszero(zero));
    EXPECT_TRUE(std::iszero(nzer));
    EXPECT_FALSE(std::iszero(one));

    EXPECT_EQ(float(std::abs(bf(-3.0f))), 3.0f);
    EXPECT_EQ(float(std::abs(bf(3.0f))), 3.0f);
    EXPECT_EQ(float(std::sin(zero)), 0.0f);
    EXPECT_EQ(float(std::cos(zero)), 1.0f);
    EXPECT_EQ(std::real(one).data, one.data);
}
