// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../common_test_header.hpp"

using Params = ::testing::Types<
    char,
    unsigned char,
    int8_t,
    uint8_t,
    short,
    unsigned short,
    int16_t,
    uint16_t,
    rocprim::half,
    rocprim::bfloat16,
    int,
    unsigned int,
    int32_t,
    uint32_t,
    float,
    long long,
    unsigned long long,
    int64_t,
    uint64_t,
    double,
    rocprim::int128_t,
    rocprim::uint128_t>;

template <typename T>
class DoubleBufferTest : public ::testing::Test
{
protected:
    T value1{};
    T value2{};
    rocprim::double_buffer<T> db{&value1, &value2};
};

TYPED_TEST_SUITE(DoubleBufferTest, Params);

TYPED_TEST(DoubleBufferTest, CurrentBuffer)
{
    EXPECT_EQ(this->db.current(), &this->value1);
}

TYPED_TEST(DoubleBufferTest, AlternateBuffer)
{
    EXPECT_EQ(this->db.alternate(), &this->value2);
}

TYPED_TEST(DoubleBufferTest, SwapBuffers)
{
    this->db.swap();
    EXPECT_EQ(this->db.current(), &this->value2);
    EXPECT_EQ(this->db.alternate(), &this->value1);
}
