/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>

#include <Tensile/EmbeddedData.hpp>

namespace Tensile
{
    namespace Tests
    {
        struct Foo
        {
        };
        struct Bar
        {
        };

        EmbedData<Tests::Foo> a{1, 2, 3, 4, 5};

        EmbedData<Tests::Bar> b{0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x00};
        EmbedData<Tests::Bar> c{0x77, 0x6f, 0x72, 0x6c, 0x64, 0x00};

        EmbedData<Tests::Foo> d("asdf", {4, 3, 2, 1});
        EmbedData<Tests::Foo> e("asdf", {1, 2, 3, 4});

    } // namespace Tests
} // namespace Tensile

using namespace Tensile;

TEST(EmbeddedData, Simple)
{
    auto const& fooData = EmbeddedData<Tests::Foo>::Get();

    ASSERT_EQ(fooData.size(), 1);
    std::vector<uint8_t> fooRef{1, 2, 3, 4, 5};
    EXPECT_EQ(fooData[0], fooRef);

    auto const& barData = EmbeddedData<Tests::Bar>::Get();

    ASSERT_EQ(barData.size(), 2);

    std::string bar0((const char*)barData[0].data());
    EXPECT_EQ(bar0, "hello");

    std::string bar1((const char*)barData[1].data());
    EXPECT_EQ(bar1, "world");

    EXPECT_EQ(EmbeddedData<Tests::Foo>::Get("nope").size(), 0);

    auto const& fooKeyData = EmbeddedData<Tests::Foo>::Get("asdf");

    ASSERT_EQ(fooKeyData.size(), 2);

    EXPECT_EQ(fooKeyData[0], std::vector<uint8_t>({4, 3, 2, 1}));
    EXPECT_EQ(fooKeyData[1], std::vector<uint8_t>({1, 2, 3, 4}));
}
