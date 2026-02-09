// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_data_generation.hpp"

#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_sort_comparator.hpp"

#include <rocprim/device/detail/device_radix_sort.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>
#include <rocprim/types/tuple.hpp>

#include <cstddef>
#include <ios>
#include <ostream>
#include <sstream>
#include <stdint.h>
#include <vector>

struct extract_digit_params
{
    unsigned int start;
    unsigned int radix_bits;
    unsigned int expected_result;
};

std::ostream& operator<<(std::ostream& os, const extract_digit_params& params)
{
    std::stringstream sstream;
    sstream << "{ start: " << params.start << ", radix_bits: " << params.radix_bits
            << ", expected_result: 0x" << std::hex << params.expected_result << " }";
    return os << sstream.str();
}

class RadixKeyCodecTest : public ::testing::TestWithParam<extract_digit_params>
{};

INSTANTIATE_TEST_SUITE_P(RocprimBlockRadixSort,
                         RadixKeyCodecTest,
                         ::testing::Values(extract_digit_params{0, 8, 0x01},
                                           extract_digit_params{8, 16, 0xcdef},
                                           extract_digit_params{24, 8, 0xab},
                                           extract_digit_params{7, 11, 0b01'1110'1111'0},
                                           extract_digit_params{0, 1, 1},
                                           extract_digit_params{1, 1, 0},
                                           extract_digit_params{0, 32, 0xabcdef01},
                                           extract_digit_params{1, 31, 0xabcdef01 >> 1},
                                           extract_digit_params{8, 12, 0xdef},
                                           extract_digit_params{12, 12, 0xcde},
                                           extract_digit_params{12, 13, 0x1cde},
                                           extract_digit_params{12, 20, 0xabcde}));

struct custom_key
{
    uint8_t  a;
    uint16_t b;
    uint8_t  c;
};

struct custom_key_decomposer
{
    auto operator()(custom_key& value) const
    {
        return ::rocprim::tuple<uint8_t&, uint16_t&, uint8_t&>{value.a, value.b, value.c};
    }
};

TEST_P(RadixKeyCodecTest, ExtractDigit)
{
    using codec = decltype(rocprim::traits::get<custom_key>().radix_key_codec());

    const custom_key key{0xab, 0xcdef, 0x01};
    const auto       digit = codec::extract_digit(key,
                                            GetParam().start,
                                            GetParam().radix_bits,
                                            custom_key_decomposer{});

    ASSERT_EQ(digit, GetParam().expected_result);
}

class RadixKeyCodecUnusedTest : public ::testing::TestWithParam<extract_digit_params>
{};

INSTANTIATE_TEST_SUITE_P(RocprimBlockRadixSort,
                         RadixKeyCodecUnusedTest,
                         ::testing::Values(extract_digit_params{0, 16, 0xab01},
                                           extract_digit_params{0, 8, 0x01},
                                           extract_digit_params{8, 8, 0xab},
                                           extract_digit_params{1, 14, 0b010'1011'0000'000},
                                           extract_digit_params{14, 2, 0b10}));

struct custom_key_decomposer_with_unused
{
    auto operator()(custom_key& value) const
    {
        return ::rocprim::tuple<uint8_t&, uint8_t&>{value.a, value.c};
    }
};

TEST_P(RadixKeyCodecUnusedTest, ExtractDigitUnused)
{
    using codec = decltype(rocprim::traits::get<custom_key>().radix_key_codec());

    const custom_key key{0xab, 0xcdef, 0x01};
    const auto       digit = codec::extract_digit(key,
                                            GetParam().start,
                                            GetParam().radix_bits,
                                            custom_key_decomposer_with_unused{});

    ASSERT_EQ(digit, GetParam().expected_result);
}

TEST(RadixKeyCodecTest, ExtractCustomTestType)
{
    using T       = common::custom_type<int, int, true>;
    using codec_t = decltype(rocprim::traits::get<T>().radix_key_codec<true>());

    T value{12, 34};

    test_utils::custom_test_type_decomposer<T> decomposer;
    codec_t::encode_inplace(value, decomposer);

    ASSERT_EQ(0x7FFFFFDD, codec_t::extract_digit(value, 0, 32, decomposer));
    ASSERT_EQ(0x7FFFFFF3, codec_t::extract_digit(value, 32, 32, decomposer));
}

template<class Params>
struct RadixMergeCompareTest : public ::testing::Test
{
    using params = Params;
};

template<bool Descending>
struct RadixMergeCompareTestParams
{
    static constexpr bool descending = Descending;
};

using RadixMergeCompareTestTypes
    = ::testing::Types<RadixMergeCompareTestParams<false>, RadixMergeCompareTestParams<true>>;
TYPED_TEST_SUITE(RadixMergeCompareTest, RadixMergeCompareTestTypes);

struct custom_large_key
{
    uint16_t a;
    int64_t  b;
    uint8_t  c;
    double   d;

    static constexpr size_t bits = 8 * (sizeof(a) + sizeof(b) + sizeof(c) + sizeof(d));
};

struct custom_large_key_decomposer
{
    auto operator()(custom_large_key& value) const
    {
        return ::rocprim::tuple<uint16_t&, int64_t&, uint8_t&, double&>{value.a,
                                                                        value.b,
                                                                        value.c,
                                                                        value.d};
    }
};

TYPED_TEST(RadixMergeCompareTest, FullRange)
{
    using params              = typename TestFixture::params;
    constexpr bool descending = params::descending;
    using merge_compare       = rocprim::detail::
        radix_merge_compare<descending, true, custom_large_key, custom_large_key_decomposer>;

    const merge_compare comparator(0, custom_large_key::bits, custom_large_key_decomposer{});

    {
        const custom_large_key lhs{1, 2, 3, 4};
        const custom_large_key rhs{3, 2, 1, 11};
        EXPECT_TRUE(descending != comparator(lhs, rhs));
    }
    {
        const custom_large_key lhs{1, 3, 3, 4};
        const custom_large_key rhs{1, 2, 1, 11};
        EXPECT_FALSE(descending != comparator(lhs, rhs));
    }
    {
        const custom_large_key lhs{1, 2, 3, 4};
        const custom_large_key rhs{1, 2, 1, 11};
        EXPECT_FALSE(descending != comparator(lhs, rhs));
    }
    {
        const custom_large_key lhs{1, 2, 3, 4};
        const custom_large_key rhs{1, 2, 3, 11};
        EXPECT_TRUE(descending != comparator(lhs, rhs));
    }
    {
        const custom_large_key lhs{1, 2, 3, 11};
        const custom_large_key rhs{1, 2, 3, 11};
        EXPECT_FALSE(comparator(lhs, rhs));
    }
}

TYPED_TEST(RadixMergeCompareTest, NotNullStartBit)
{
    using params              = typename TestFixture::params;
    constexpr bool descending = params::descending;
    using merge_compare       = rocprim::detail::
        radix_merge_compare<descending, true, custom_large_key, custom_large_key_decomposer>;

    constexpr unsigned int start_bit = 64;
    const merge_compare    comparator(start_bit,
                                   custom_large_key::bits - start_bit,
                                   custom_large_key_decomposer{});

    {
        const custom_large_key lhs{3, 2, 3, 4};
        const custom_large_key rhs{3, 2, 1, 11};
        EXPECT_FALSE(descending != comparator(lhs, rhs));
    }
    {
        const custom_large_key lhs{3, 2, 1, 4};
        const custom_large_key rhs{3, 2, 3, 11};
        EXPECT_TRUE(descending != comparator(lhs, rhs));
    }
    {
        const custom_large_key lhs{3, 2, 1, 4};
        const custom_large_key rhs{3, 2, 1, 11};
        EXPECT_FALSE(comparator(lhs, rhs));
    }
}

TYPED_TEST(RadixMergeCompareTest, MidRange)
{
    using params              = typename TestFixture::params;
    constexpr bool descending = params::descending;
    using merge_compare       = rocprim::detail::
        radix_merge_compare<descending, true, custom_large_key, custom_large_key_decomposer>;

    constexpr unsigned int start_bit     = 64;
    constexpr unsigned int excluded_bits = 16;
    const merge_compare    comparator(start_bit,
                                   custom_large_key::bits - start_bit - excluded_bits,
                                   custom_large_key_decomposer{});

    {
        const custom_large_key lhs{3, 2, 3, 4};
        const custom_large_key rhs{4, 2, 1, 11};
        EXPECT_FALSE(descending != comparator(lhs, rhs));
    }
    {
        const custom_large_key lhs{3, 2, 1, 4};
        const custom_large_key rhs{4, 2, 3, 11};
        EXPECT_TRUE(descending != comparator(lhs, rhs));
    }
    {
        const custom_large_key lhs{3, 2, 3, 4};
        const custom_large_key rhs{4, 2, 3, 11};
        EXPECT_FALSE(comparator(lhs, rhs));
    }
}

template<class Params>
struct TypedRadixKeyCodecTest : public ::testing::Test
{
    using params = Params;
};

template<class KeyType, unsigned int StartBit = 0, unsigned int RadixBits = 8>
struct TypedRadixKeyCodecTestParams
{
    using Key                                = KeyType;
    static constexpr unsigned int start_bit  = StartBit;
    static constexpr unsigned int radix_bits = RadixBits;
};

template<class T>
struct custom_test_type_decomposer
{
    auto operator()(common::custom_type<T, T, true>& value) const
    {
        return ::rocprim::tuple<T&, T&>{value.x, value.y};
    }
};

using TypedRadixKeyCodecTestTypes
    = ::testing::Types<TypedRadixKeyCodecTestParams<int8_t>,
                       TypedRadixKeyCodecTestParams<uint8_t>,
                       TypedRadixKeyCodecTestParams<char>,
                       TypedRadixKeyCodecTestParams<signed char>,
                       TypedRadixKeyCodecTestParams<int16_t>,
                       TypedRadixKeyCodecTestParams<uint16_t>,
                       TypedRadixKeyCodecTestParams<short>,
                       TypedRadixKeyCodecTestParams<unsigned short>,
                       TypedRadixKeyCodecTestParams<int>,
                       TypedRadixKeyCodecTestParams<unsigned int>,
                       TypedRadixKeyCodecTestParams<long long>,
                       TypedRadixKeyCodecTestParams<rocprim::half>,
                       TypedRadixKeyCodecTestParams<rocprim::bfloat16>,
                       TypedRadixKeyCodecTestParams<float>,
                       TypedRadixKeyCodecTestParams<double>>;

TYPED_TEST_SUITE(TypedRadixKeyCodecTest, TypedRadixKeyCodecTestTypes);

template<bool Descending, class Key, class Decomposer>
void encode_then_decode_test(Key key, Decomposer decomposer)
{
    constexpr auto input_traits = ::rocprim::traits::get<Key>();
    constexpr auto codec        = input_traits.template radix_key_codec<Descending>();
    using codec_t               = decltype(codec);
    using BitKey  = typename codec_t::bit_key_type;

    BitKey bit_key = codec_t::encode(key, decomposer);
    codec_t::encode_inplace(key, decomposer);

    Key decoded_key = codec_t::decode(bit_key, decomposer);
    codec_t::decode_inplace(key, decomposer);

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(decoded_key, key));
}

template<class Key, class Decomposer = ::rocprim::identity_decomposer>
void encode_then_decode_test(Key key, Decomposer decomposer = {})
{
    encode_then_decode_test<true>(key, decomposer); /*decreasing sort*/
    encode_then_decode_test<false>(key, decomposer); /*increasing sort*/
}

template<bool Descending, class Key, class Decomposer>
void encode_then_extract_test(Key                key,
                              const unsigned int start_bit,
                              const unsigned int radix_bits,
                              Decomposer         decomposer)
{
    constexpr auto input_traits = ::rocprim::traits::get<Key>();
    constexpr auto codec        = input_traits.template radix_key_codec<Descending>();
    using codec_t               = decltype(codec);
    using BitKey  = typename codec_t::bit_key_type;

    BitKey bit_key = codec_t::encode(key, decomposer);
    codec_t::encode_inplace(key, decomposer);

    const unsigned int bits = codec_t::extract_digit(bit_key, start_bit, radix_bits);
    const unsigned int inplace_bits
        = codec_t::extract_digit(key, start_bit, radix_bits, decomposer);

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(bits, inplace_bits));
}

template<class Key, class Decomposer = ::rocprim::identity_decomposer>
void encode_then_extract_test(Key                key,
                              const unsigned int start_bit,
                              const unsigned int radix_bits,
                              Decomposer         decomposer = {})
{
    encode_then_extract_test<true>(key, start_bit, radix_bits, decomposer); /*decreasing sort*/
    encode_then_extract_test<false>(key, start_bit, radix_bits, decomposer); /*increasing sort*/
}

template<bool Descending, class Key, class Decomposer>
void encode_then_extract_test_custom(Key                key,
                                     const unsigned int start_bit,
                                     const unsigned int radix_bits,
                                     Decomposer         decomposer)
{
    constexpr auto input_traits = ::rocprim::traits::get<Key>();
    constexpr auto codec        = input_traits.template radix_key_codec<Descending>();
    using codec_t               = decltype(codec);
    using BitKey  = typename codec_t::bit_key_type;

    BitKey bit_key = codec_t::encode(key, decomposer);
    codec_t::encode_inplace(key, decomposer);

    const unsigned int bits = codec_t::extract_digit(bit_key, start_bit, radix_bits, decomposer);
    const unsigned int inplace_bits
        = codec_t::extract_digit(key, start_bit, radix_bits, decomposer);

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(bits, inplace_bits));
}

template<class Key, class Decomposer = ::rocprim::identity_decomposer>
void encode_then_extract_test_custom(Key                key,
                                     const unsigned int start_bit,
                                     const unsigned int radix_bits,
                                     Decomposer         decomposer = {})
{
    encode_then_extract_test_custom<true>(key,
                                          start_bit,
                                          radix_bits,
                                          decomposer); /*decreasing sort*/
    encode_then_extract_test_custom<false>(key,
                                           start_bit,
                                           radix_bits,
                                           decomposer); /*increasing sort*/
}

TYPED_TEST(TypedRadixKeyCodecTest, EncodeDecodeExtract)
{
    using params                      = typename TestFixture::params;
    using Key                         = typename params::Key;
    using CustomKey                   = typename common::custom_type<Key, Key, true>;
    using CustomDecomposer            = custom_test_type_decomposer<Key>;
    constexpr unsigned int start_bit  = params::start_bit;
    constexpr unsigned int radix_bits = params::radix_bits;

    CustomDecomposer custom_decomposer{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const size_t     size = (1 << 20) + 123;
        std::vector<Key> input_keys
            = test_utils::get_random_data_wrapped<Key>(size,
                                                       common::generate_limits<Key>::min(),
                                                       common::generate_limits<Key>::max(),
                                                       seed_value);

        for(size_t i = 0; i < size; ++i)
        {
            SCOPED_TRACE(testing::Message() << "with index = " << i);

            encode_then_decode_test(input_keys[i]);

            encode_then_extract_test(input_keys[i], start_bit, radix_bits);

            // With custom types
            encode_then_decode_test(CustomKey(input_keys[i]), custom_decomposer);

            encode_then_extract_test_custom(CustomKey(input_keys[i]),
                                            start_bit,
                                            radix_bits,
                                            custom_decomposer);
        }
    }
}
