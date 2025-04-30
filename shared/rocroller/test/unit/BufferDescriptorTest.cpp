/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/CodeGen/Buffer.hpp>

#include "SimpleFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

class BufferDescriptorTest : public SimpleFixture
{
};

TEST_F(BufferDescriptorTest, Default)
{
    GFX9BufferDescriptorOptions options;

    auto expected = R"(GFX9BufferDescriptorOptions: 0x00020000
        dst_sel_x: 0
        dst_sel_y: 0
        dst_sel_z: 0
        dst_sel_w: 0
        num_format: 0
        data_format: DF32
        user_vm_enable: 0
        user_vm_mode: 0
        index_stride: 0
        add_tid_enable: 0
        _unusedA: 0
        nv: 0
        _unusedB: 0
        type: 0
    )";

    EXPECT_EQ(NormalizedSource(expected), NormalizedSource(options.toString()));

    EXPECT_EQ(0x00020000, options.rawValue());
    EXPECT_EQ(0, options.dst_sel_x);
    EXPECT_EQ(0, options.dst_sel_y);
    EXPECT_EQ(0, options.dst_sel_z);
    EXPECT_EQ(0, options.dst_sel_w);
    EXPECT_EQ(0, options.num_format);
    EXPECT_EQ(GFX9BufferDescriptorOptions::DF32, options.data_format);
    EXPECT_EQ(0, options.user_vm_enable);
    EXPECT_EQ(0, options.user_vm_mode);

    auto literal = options.literal();
    ASSERT_NE(nullptr, literal);
    EXPECT_EQ(CommandArgumentValue(0x00020000u), literal->getLiteralValue());
}

TEST_F(BufferDescriptorTest, Values)
{
    GFX9BufferDescriptorOptions options;
    options.dst_sel_x = 1;
    EXPECT_EQ(0x00020001, options.rawValue());

    EXPECT_THAT(options.toString(), ::testing::HasSubstr("dst_sel_x: 1\n"));

    options._unusedA = 3;
    EXPECT_ANY_THROW(options.validate());
    options._unusedA = 0;
    EXPECT_NO_THROW(options.validate());

    options.nv = 1;
    EXPECT_EQ(0x08020001, options.rawValue());
}
