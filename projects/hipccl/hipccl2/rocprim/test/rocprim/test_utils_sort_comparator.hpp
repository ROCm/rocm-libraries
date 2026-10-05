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

#ifndef TEST_UTILS_SORT_COMPARATOR_HPP_
#define TEST_UTILS_SORT_COMPARATOR_HPP_

#include "../../common/utils_custom_type.hpp"

#include "test_utils_custom_float_type.hpp"
#include "test_utils_custom_test_types.hpp"

#include <rocprim/config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>
#include <rocprim/types/tuple.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace test_utils
{

namespace detail
{

template<unsigned int StartBit,
         unsigned int EndBit,
         class Key,
         std::enable_if_t<(rocprim::is_integral<Key>::value && !std::is_same<Key, bool>::value)
                              || std::is_same<Key, rocprim::uint128_t>::value
                              || std::is_same<Key, rocprim::int128_t>::value,
                          int>
         = 0>
ROCPRIM_HOST_DEVICE
auto to_bits(const Key key) -> typename rocprim::get_unsigned_bits_type<Key>::unsigned_type
{
    using unsigned_bits_type = typename rocprim::get_unsigned_bits_type<Key>::unsigned_type;

    static constexpr unsigned_bits_type radix_mask_upper
        = EndBit == 8 * sizeof(Key)
              ? ~unsigned_bits_type(0)
              : static_cast<unsigned_bits_type>((unsigned_bits_type(1) << EndBit) - 1);
    static constexpr unsigned_bits_type radix_mask_bottom
        = static_cast<unsigned_bits_type>((unsigned_bits_type(1) << StartBit) - 1);
    static constexpr unsigned_bits_type radix_mask = radix_mask_upper ^ radix_mask_bottom;

    auto bit_key = static_cast<unsigned_bits_type>(key);
    // Flip sign bit to properly order signed types
    if(::rocprim::is_signed<Key>::value)
    {
        constexpr auto sign_bit = static_cast<unsigned_bits_type>(1) << (sizeof(Key) * 8 - 1);
        bit_key ^= sign_bit;
    }

    return bit_key & radix_mask;
}

template<unsigned int StartBit,
         unsigned int EndBit,
         class Key,
         std::enable_if_t<std::is_same<Key, bool>::value, int> = 0>
ROCPRIM_HOST_DEVICE
auto to_bits(const Key key) -> typename rocprim::get_unsigned_bits_type<Key>::unsigned_type
{
    using unsigned_bits_type = typename rocprim::get_unsigned_bits_type<Key>::unsigned_type;
    unsigned_bits_type bit_key;
    memcpy(&bit_key, &key, sizeof(bit_key));
    return to_bits<StartBit, EndBit>(bit_key);
}

template<unsigned int StartBit,
         unsigned int EndBit,
         class Key,
         std::enable_if_t<rocprim::is_floating_point<Key>::value
                              // custom_float_type is used in testing a hacky way of
                              // radix sorting custom types. A part of this workaround
                              // is to specialize rocprim::is_floating_point<custom_float_type>
                              // that we must counter here.
                              && !std::is_same<Key, custom_float_type>::value,
                          int>
         = 0>
ROCPRIM_HOST_DEVICE
auto to_bits(const Key key) -> typename rocprim::get_unsigned_bits_type<Key>::unsigned_type
{
    using unsigned_bits_type = typename rocprim::get_unsigned_bits_type<Key>::unsigned_type;

    unsigned_bits_type bit_key;
    memcpy(&bit_key, &key, sizeof(Key));

    // Remove signed zero, this case is supposed to be treated the same as
    // unsigned zero in rocprim sorting algorithms.
    constexpr unsigned_bits_type minus_zero = unsigned_bits_type{1} << (8 * sizeof(Key) - 1);
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
         std::enable_if_t<common::is_custom_type<Key>::value
                              // custom_float_type is used in testing a hacky way of
                              // radix sorting custom types. A part of this workaround
                              // is to specialize common::is_custom_type<custom_float_type>
                              // that we must counter here.
                              && !std::is_same<Key, custom_float_type>::value,
                          int>
         = 0>
ROCPRIM_HOST_DEVICE
auto to_bits(const Key& key) -> typename rocprim::get_unsigned_bits_type<Key>::unsigned_type
{
    using inner_t            = typename inner_type<Key>::type;
    using unsigned_bits_type = typename ::rocprim::get_unsigned_bits_type<inner_t>::unsigned_type;
    // For two doubles, we need uint128, but that is not part of rocprim::get_unsigned_bits_type
    using result_bits_type = std::conditional_t<
        sizeof(inner_t) == 8,
        rocprim::uint128_t,
        typename rocprim::get_unsigned_bits_type<void,
                                                 rocprim::min(static_cast<size_t>(8),
                                                              sizeof(inner_t) * 2)>::unsigned_type>;

    auto bit_key_upper = static_cast<unsigned_bits_type>(to_bits<0, sizeof(key.x) * 8>(key.x));
    auto bit_key_lower = static_cast<unsigned_bits_type>(to_bits<0, sizeof(key.y) * 8>(key.y));

    // Create the result containing both parts
    const auto bit_key
        = (static_cast<result_bits_type>(bit_key_upper) << (8 * sizeof(unsigned_bits_type)))
          | bit_key_lower;

    // The last call to to_bits mask the result to the specified bit range
    return to_bits<StartBit, EndBit>(bit_key);
}

template<unsigned int StartBit,
         unsigned int EndBit,
         class Key,
         std::enable_if_t<std::is_same<Key, custom_float_type>::value, int> = 0>
ROCPRIM_HOST_DEVICE
auto to_bits(const Key key) -> typename rocprim::get_unsigned_bits_type<Key>::unsigned_type
{
    return to_bits<StartBit, EndBit>(key.x);
}
} // namespace detail

template<class T>
constexpr bool is_floating_nan_host(const T& a)
{
    return (a != a);
}

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator
{
    ROCPRIM_HOST_DEVICE
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
    static_assert(
        common::is_custom_type<CustomTestType>::value,
        "custom_test_type_decomposer can only be used with common::custom_type<T, T, true>");
    using inner_t = typename inner_type<CustomTestType>::type;

    __host__ __device__
    auto operator()(CustomTestType& key) const
    {
        return ::rocprim::tuple<inner_t&, inner_t&>{key.x, key.y};
    }
};

template<class Key>
struct select_decomposer
{
    using type = ::rocprim::identity_decomposer;
};

template<class InnerType>
struct select_decomposer<common::custom_type<InnerType, InnerType, true>>
{
    using type = custom_test_type_decomposer<common::custom_type<InnerType, InnerType, true>>;
};

template<class Key>
using select_decomposer_t = typename select_decomposer<Key>::type;

} // namespace test_utils

#endif // TEST_UTILS_SORT_COMPARATOR_HPP_
