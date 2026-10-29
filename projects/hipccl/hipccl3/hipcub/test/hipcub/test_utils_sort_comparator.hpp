// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_TEST_TEST_UTILS_SORT_COMPARATOR_HPP_
#define HIPCUB_TEST_TEST_UTILS_SORT_COMPARATOR_HPP_

#include <cstdint>
#include <type_traits>
#ifdef __HIP_PLATFORM_AMD__
#include <rocprim/type_traits.hpp>
#endif

#include "test_utils_bfloat16.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_half.hpp"

#include <hipcub/libcxx.hpp>
#include <hipcub/tuple.hpp>

#include <cstring>

namespace test_utils
{
namespace detail
{

template<class Key>
constexpr bool is_extended_int
    = std::is_same_v<Key, __int128> || std::is_same_v<Key, unsigned __int128>;
template<unsigned int StartBit,
         unsigned int EndBit,
         class Key,
         std::enable_if_t<
             // Catch integral types and extended integral types.
             // The std::is_same_v<...> clauses can be removed once
             // libhipcxx is a hard depedency and test types half_t
             // and bfloat_t are removed.
             _HIPCUB_STD::is_integral_v<Key> || is_extended_int<Key>,
             int>
         = 0>
Key to_bits(const Key key)
{
    using Bits                             = typename hipcub::Traits<Key>::UnsignedBits;
    static constexpr Key radix_mask_upper  = EndBit == 8 * sizeof(Key)
                                                 ? static_cast<Key>(~Bits(0))
                                                 : static_cast<Key>((Bits(1) << EndBit) - 1);
    static constexpr Key radix_mask_bottom = static_cast<Key>((Key(1) << StartBit) - 1);
    static constexpr Key radix_mask        = radix_mask_upper ^ radix_mask_bottom;

    return key & radix_mask;
}

template<class Key>
constexpr bool is_extended_fp
    = std::is_same_v<Key, __half> || std::is_same_v<Key, hip_bfloat16>
      || std::is_same_v<Key, native_half> || std::is_same_v<Key, native_bfloat16>
      || std::is_same_v<Key, test_utils::bfloat16>;

template<unsigned int StartBit,
         unsigned int EndBit,
         class Key,
         std::enable_if_t<
             // Catch floating types and extended floating types.
             // The std::is_same_v<...> clauses can be removed once
             // libhipcxx is a hard depedency and test types half_t
             // and bfloat_t are removed.
             _HIPCUB_STD::is_floating_point_v<Key> || is_extended_fp<Key>,
             int>
         = 0>
auto to_bits(const Key key)
{
    using unsigned_bits_type = typename hipcub::NumericTraits<Key>::UnsignedBits;
    static_assert(sizeof(unsigned_bits_type) == sizeof(Key));

    unsigned_bits_type bit_key;
    std::memcpy(&bit_key, &key, sizeof(unsigned_bits_type));

    // Remove signed zero, this case is supposed to be treated the same as
    // unsigned zero in hipcub sorting algorithms.
    constexpr unsigned_bits_type minus_zero = unsigned_bits_type{1}
                                              << (8 * sizeof(unsigned_bits_type) - 1);

    // Positive and negative zero should compare the same.
    if(bit_key == minus_zero)
    {
        bit_key = 0;
    }
    // Flip bits mantissa and exponent if the key is negative, so as to make
    // 'more negative' values compare before 'less negative'.
    if(bit_key & minus_zero)
    {
        bit_key ^= ~minus_zero;
    }
    // Make negatives compare before positives.
    bit_key ^= minus_zero;

    return to_bits<StartBit, EndBit>(bit_key);
}

template<unsigned int StartBit,
         unsigned int EndBit,
         class Key,
         std::enable_if_t<is_custom_test_type<std::decay_t<Key>>::value, int> = 0>
auto to_bits(const Key& key)
{
    using inner_t            = typename inner_type<Key>::type;
    using unsigned_bits_type = typename hipcub::NumericTraits<inner_t>::UnsignedBits;
    using result_bits_type   = std::conditional_t<
        sizeof(inner_t) == 1,
        uint16_t,
        std::conditional_t<sizeof(inner_t) == 2,
                           uint32_t,
                           std::conditional_t<sizeof(inner_t) == 4, uint64_t, void>>>;

    auto bit_key_upper = static_cast<unsigned_bits_type>(to_bits<0, sizeof(inner_t) * 8>(key.x));
    auto bit_key_lower = static_cast<unsigned_bits_type>(to_bits<0, sizeof(inner_t) * 8>(key.y));

    // Flip sign bit to properly order signed types
    if(std::is_signed<inner_t>::value)
    {
        constexpr auto sign_bit = static_cast<unsigned_bits_type>(1) << (sizeof(inner_t) * 8 - 1);
        bit_key_upper ^= sign_bit;
        bit_key_lower ^= sign_bit;
    }

    // Create the result containing both parts
    const auto bit_key
        = (static_cast<result_bits_type>(bit_key_upper) << (8 * sizeof(unsigned_bits_type)))
          | bit_key_lower;

    // The last call to to_bits mask the result to the specified bit range
    return to_bits<StartBit, EndBit>(bit_key);
}

} // namespace detail

template<class T>
constexpr auto is_floating_nan_host(const T& a) ->
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type
{
    return (a != a);
}

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator
{
    bool operator()(const Key lhs, const Key rhs) const
    {
        const auto l = detail::to_bits<StartBit, EndBit>(lhs);
        const auto r = detail::to_bits<StartBit, EndBit>(rhs);
        return Descending ? (r < l) : (l < r);
    }
};

template<class Key, class Value, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_value_comparator
{
    bool operator()(const std::pair<Key, Value>& lhs, const std::pair<Key, Value>& rhs) const
    {
        return key_comparator<Key, Descending, StartBit, EndBit>()(lhs.first, rhs.first);
    }
};

template<class CustomTestType>
struct custom_test_type_decomposer
{
    static_assert(is_custom_test_type<CustomTestType>::value,
                  "custom_test_type_decomposer can only be used with custom_test_type<T>");
    using inner_t = typename inner_type<CustomTestType>::type;

    __host__ __device__ auto operator()(CustomTestType& key) const
    {
        return ::hipcub::tuple<inner_t&, inner_t&>{key.x, key.y};
    }
};
}
#endif // TEST_UTILS_SORT_COMPARATOR_HPP_
