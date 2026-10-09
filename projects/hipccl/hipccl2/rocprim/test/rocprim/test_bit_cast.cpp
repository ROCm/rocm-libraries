// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PUrocprimOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../common_test_header.hpp"

namespace
{
// Non-trivial types for illegal combination tests
struct NonTrivial32
{
    int a;
    NonTrivial32() : a(0) {}
    NonTrivial32(const NonTrivial32& other) : a(other.a) {}
};

struct NonTrivial64
{
    long long a;
    NonTrivial64() : a(0) {}
    NonTrivial64(const NonTrivial64& other) : a(other.a) {}
};

struct NonTrivial128
{
    rocprim::int128_t a;
    NonTrivial128() : a(0) {}
    NonTrivial128(const NonTrivial128& other) : a(other.a) {}
};

// Self-defined groups for
struct Group32_I
{
    char    a;
    uint8_t b;
    short   c;
};

struct Group32_II
{
    unsigned short a;
    rocprim::half  b;
};

struct Group64_I
{
    uint8_t a;
    char    b;
    short   c;
    int     d;
};

struct Group64_II
{
    long long a;
};

static_assert(rocprim::detail::is_valid_bit_cast<int, float>,
              "input types (int, float) must be valid!");
static_assert(!rocprim::detail::is_valid_bit_cast<int, double>,
              "input types (int, double) must be invalid!");
static_assert(!rocprim::detail::is_valid_bit_cast<int, NonTrivial32>,
              "input types (int, NonTrivial32) must be invalid!");
static_assert(rocprim::detail::is_valid_bit_cast<Group32_I, int>,
              "input types (int, Group32_I) must be valid!");
} // namespace

template<typename Source, typename Destination>
void TestBitCastCombinationImpl(const Source& source)
{
    if constexpr(rocprim::detail::is_valid_bit_cast<Destination, Source>)
    {
        Destination dest_memcpy;
        std::memcpy(&dest_memcpy, &source, sizeof(Destination));
        Destination dest_bitcast = rocprim::detail::bit_cast<Destination, Source>(source);
        ASSERT_EQ(std::memcmp(&dest_memcpy, &dest_bitcast, sizeof(Destination)), 0);
    }
}

template<typename Source, typename Tuple, std::size_t... I>
void TestBitCastForDestinationsImpl(const Source& source, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{
        (TestBitCastCombinationImpl<Source, std::tuple_element_t<I, Tuple>>(source), 0)...};
}

template<typename Source, typename Tuple>
void TestBitCastForDestinations(const Source& source)
{
    TestBitCastForDestinationsImpl<Source, Tuple>(
        source,
        std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

// Types in google test arrray
using AllTypes = ::testing::Types<
    // 8 bit
    char,
    unsigned char,
    int8_t,
    uint8_t,
    // 16 bit
    short,
    unsigned short,
    int16_t,
    uint16_t,
    rocprim::half,
    rocprim::bfloat16,
    // 32 bit
    int,
    unsigned int,
    int32_t,
    uint32_t,
    float,
    // 64 bit
    long long,
    unsigned long long,
    int64_t,
    uint64_t,
    double,
    // 128 bit
    rocprim::int128_t,
    rocprim::uint128_t,
    // non trivial
    NonTrivial32,
    NonTrivial64,
    NonTrivial128,
    // self-defined structs
    Group32_I,
    Group32_II,
    Group64_I,
    Group64_II>;

// Types in tuple for interation
using AllTypesTuple = std::tuple<
    // 8 bit
    char,
    unsigned char,
    int8_t,
    uint8_t,
    // 16 bit
    short,
    unsigned short,
    int16_t,
    uint16_t,
    rocprim::half,
    rocprim::bfloat16,
    // 32 bit
    int,
    unsigned int,
    int32_t,
    uint32_t,
    float,
    // 64 bit
    long long,
    unsigned long long,
    int64_t,
    uint64_t,
    double,
    // 128 bit
    rocprim::int128_t,
    rocprim::uint128_t,
    // non trivial
    NonTrivial32,
    NonTrivial64,
    NonTrivial128,
    // self-defined structs
    Group32_I,
    Group32_II,
    Group64_I,
    Group64_II>;

template<typename Source>
class BitCastPairTest : public ::testing::Test
{
public:
    using SourceType = Source;
};

TYPED_TEST_SUITE(BitCastPairTest, AllTypes);

TYPED_TEST(BitCastPairTest, BitCastPairTest)
{
    using Source = typename TestFixture::SourceType;

    unsigned char buffer[sizeof(Source)];
    for(size_t i = 0; i < sizeof(Source); ++i)
    {
        buffer[i] = static_cast<unsigned char>(rand() & 0xFF);
    }
    Source source;
    std::memcpy(reinterpret_cast<void*>(&source), buffer, sizeof(Source));

    TestBitCastForDestinations<Source, AllTypesTuple>(source);
}
