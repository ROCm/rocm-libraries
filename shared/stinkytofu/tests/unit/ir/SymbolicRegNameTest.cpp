/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#include <gtest/gtest.h>

#include <string>
#include <unordered_map>

#include "stinkytofu/ir/asm/SymbolicRegName.hpp"

using namespace stinkytofu;

namespace {

TEST(SymbolicRegNameTest, ParsesBareName) {
    const auto parsed = parseSymbolicRegName("sgprGSU");
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->start.base, "sgprGSU");
    EXPECT_TRUE(parsed->start.terms.empty());
    EXPECT_FALSE(parsed->rangeEnd.has_value());
}

TEST(SymbolicRegNameTest, ParsesSingleOffset) {
    const auto parsed = parseSymbolicRegName("vgprValuA_X0_I0+4");
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->start.base, "vgprValuA_X0_I0");
    ASSERT_EQ(parsed->start.terms.size(), 1u);
    EXPECT_EQ(parsed->start.terms[0], 4);
}

TEST(SymbolicRegNameTest, ParsesMultiOffset) {
    const auto parsed = parseSymbolicRegName("vgprFoo+1+2");
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->start.base, "vgprFoo");
    ASSERT_EQ(parsed->start.terms.size(), 2u);
    EXPECT_EQ(parsed->start.terms[0], 1);
    EXPECT_EQ(parsed->start.terms[1], 2);
}

TEST(SymbolicRegNameTest, ParsesNegativeOffset) {
    const auto parsed = parseSymbolicRegName("vgprSerial-512");
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->start.base, "vgprSerial");
    ASSERT_EQ(parsed->start.terms.size(), 1u);
    EXPECT_EQ(parsed->start.terms[0], -512);
}

TEST(SymbolicRegNameTest, ParsesExplicitRange) {
    const auto parsed = parseSymbolicRegName("vgprFoo+0:vgprFoo+3");
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->start.base, "vgprFoo");
    ASSERT_EQ(parsed->start.terms.size(), 1u);
    EXPECT_EQ(parsed->start.terms[0], 0);
    ASSERT_TRUE(parsed->rangeEnd.has_value());
    EXPECT_EQ(parsed->rangeEnd->base, "vgprFoo");
    ASSERT_EQ(parsed->rangeEnd->terms.size(), 1u);
    EXPECT_EQ(parsed->rangeEnd->terms[0], 3);
}

TEST(SymbolicRegNameTest, ResolveNamedIndexMatchesSetMap) {
    const std::unordered_map<std::string, int64_t> map{{"vgprFoo", 46}, {"sgprTmp", 7}};
    EXPECT_EQ(resolveNamedIndex("sgprTmp", map), 7);
    EXPECT_EQ(resolveNamedIndex("vgprFoo+4", map), 50);
    EXPECT_EQ(resolveNamedIndex("vgprFoo+1+2", map), 49);
    EXPECT_EQ(resolveNamedIndex("vgprSerial-512", {{"vgprSerial", 768}}), 256);
    EXPECT_EQ(resolveNamedIndex("vgprFoo+0:vgprFoo+3", map, /*regNum=*/4), 46);
    EXPECT_FALSE(resolveNamedIndex("vgprFoo+0:vgprFoo+3", map, /*regNum=*/2).has_value());
}

}  // namespace
