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

#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

using namespace stinkytofu;

// The numeric s_wait_alu decoder (the passes see these fields; the emitter
// prints the word as written while they are unchanged).
TEST(WaitAluDataTest, FromEncodingDecodesTheDepctrWord) {
    auto all = SWaitAluData::fromEncoding(0);
    ASSERT_TRUE(all.has_value());
    for (int f = 0; f < SWaitAluData::FIELD_COUNT; ++f) {
        EXPECT_TRUE(all->hasField(static_cast<SWaitAluData::Field>(f)));
        EXPECT_EQ(all->getField(static_cast<SWaitAluData::Field>(f)), 0);
    }

    auto sa = SWaitAluData::fromEncoding(0xff9e);
    ASSERT_TRUE(sa.has_value());
    EXPECT_TRUE(sa->hasField(SWaitAluData::SA_SDST));
    EXPECT_FALSE(sa->hasField(SWaitAluData::VA_VDST));
    EXPECT_EQ(sa->encode(), 0xff9e);

    auto two = SWaitAluData::fromEncoding(0x3f87);
    ASSERT_TRUE(two.has_value());
    EXPECT_EQ(two->getField(SWaitAluData::VA_VDST), 3);
    EXPECT_EQ(two->getField(SWaitAluData::VM_VSRC), 1);
    EXPECT_FALSE(two->hasField(SWaitAluData::SA_SDST));

    EXPECT_FALSE(SWaitAluData::fromEncoding(0xfffe).has_value());  // reserved bits
}

TEST(WaitAluDataTest, SpellingKeyTracksTheFields) {
    auto a = SWaitAluData::fromEncoding(0);
    auto b = SWaitAluData::fromEncoding(0);
    ASSERT_TRUE(a && b);
    EXPECT_EQ(a->spellingKey(), b->spellingKey());
    b->clearField(SWaitAluData::VA_VDST);
    EXPECT_NE(a->spellingKey(), b->spellingKey());
}
