// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_TEST_UTILS_ASSERTIONS_HPP
#define ROCPRIM_TEST_UTILS_ASSERTIONS_HPP

#include "test_utils_half.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_custom_test_types.hpp"

#include <rocprim/type_traits.hpp>

#include <gtest/gtest.h>

// Std::memcpy and std::memcmp
#include <cstring>
#include <iterator>
#include <ostream>
#include <sstream>
#include <type_traits>
#include <vector>

namespace test_utils {

// begin assert_eq
template<class T>
bool inline bit_equal(const T& a, const T& b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/// Checks if `vector<T> result` matches `vector<T> expected`.
/// If max_length is given, equality of `result.size()` and `expected.size()`
/// is ignored and checks only the first max_length elements.
/// \tparam T
/// \param result
/// \param expected
/// \param max_length
template<class T>
void assert_eq(const std::vector<T>& result,
               const std::vector<T>& expected,
               const size_t          max_length = SIZE_MAX)
{
    if(max_length == SIZE_MAX || max_length > expected.size())
        ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < std::min(result.size(), max_length); i++)
    {
        if(bit_equal(result[i], expected[i]))
            continue; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
        ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
    }
}

template<class T>
void assert_eq(const std::vector<custom_test_type<T>>& result,
               const std::vector<custom_test_type<T>>& expected,
               const size_t                            max_length = SIZE_MAX)
{
    if(max_length == SIZE_MAX || max_length > expected.size())
    {
        ASSERT_EQ(result.size(), expected.size());
    }
    for(size_t i = 0; i < std::min(result.size(), max_length); i++)
    {
        if(bit_equal(result[i].x, expected[i].x) && bit_equal(result[i].y, expected[i].y))
            continue; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
        ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
    }
}

template<>
inline void assert_eq<rocprim::half>(const std::vector<rocprim::half>& result,
                                     const std::vector<rocprim::half>& expected,
                                     const size_t                      max_length)
{
    if(max_length == SIZE_MAX || max_length > expected.size())
        ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < std::min(result.size(), max_length); i++)
    {
        if(bit_equal(result[i], expected[i]))
            continue; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
        ASSERT_EQ(half_to_native(result[i]), half_to_native(expected[i])) << "where index = " << i;
    }
}

template<>
inline void assert_eq<rocprim::bfloat16>(const std::vector<rocprim::bfloat16>& result,
                                         const std::vector<rocprim::bfloat16>& expected,
                                         const size_t                          max_length)
{
    if(max_length == SIZE_MAX || max_length > expected.size()) ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < std::min(result.size(), max_length); i++)
    {
        if(bit_equal(result[i], expected[i]))
            continue; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
        ASSERT_EQ(bfloat16_to_native(result[i]), bfloat16_to_native(expected[i]))
            << "where index = " << i;
    }
}

template<class T>
void assert_eq(const T& result, const T& expected)
{
    if(bit_equal(result, expected))
        return; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
    ASSERT_EQ(result, expected);
}

template<class T>
void assert_eq(const custom_test_type<T>& result, const custom_test_type<T>& expected)
{
    if(bit_equal(result.x, expected.x) && bit_equal(result.y, expected.y))
        return; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
    ASSERT_EQ(result, expected);
}

template<>
inline void assert_eq<rocprim::half>(const rocprim::half& result, const rocprim::half& expected)
{
    if(bit_equal(result, expected))
        return; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
    ASSERT_EQ(half_to_native(result), half_to_native(expected));
}

template<>
inline void assert_eq<rocprim::bfloat16>(const rocprim::bfloat16& result,
                                         const rocprim::bfloat16& expected)
{
    if(bit_equal(result, expected))
        return; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
    ASSERT_EQ(bfloat16_to_native(result), bfloat16_to_native(expected));
}

template<class ResultIt, class ExpectedIt>
void assert_eq(ResultIt   result_begin,
               ResultIt   result_end,
               ExpectedIt expected_begin,
               ExpectedIt expected_end)
{
    ASSERT_EQ(std::distance(result_begin, result_end), std::distance(expected_begin, expected_end));
    auto result_it   = result_begin;
    auto expected_it = expected_begin;
    for(; result_it != result_end; ++result_it, ++expected_it)
    {
        assert_eq(static_cast<typename std::iterator_traits<ResultIt>::value_type>(*result_it),
                  static_cast<typename std::iterator_traits<ExpectedIt>::value_type>(*expected_it));
    }
}
// end assert_eq

// begin assert_near
template<class T>
auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        if(bit_equal(result[i], expected[i])) continue; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
        auto diff = std::abs(percent * std::max(result[i], expected[i]));
        ASSERT_NEAR(result[i], expected[i], diff) << "where index = " << i;
    }
}

template<class T>
auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float)
    -> typename std::enable_if<!rocprim::is_floating_point<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
    }
}

template<class T, std::enable_if_t<std::is_same<T, rocprim::bfloat16>::value ||
                                        std::is_same<T, rocprim::half>::value, bool> = true>
void assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        if(bit_equal(result[i], expected[i])) continue; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
        auto diff = std::abs(percent * static_cast<float>(expected[i]));
        ASSERT_NEAR(static_cast<float>(result[i]), static_cast<float>(expected[i]), diff) << "where index = " << i;
    }
}

template<class T>
auto assert_near(const std::vector<custom_test_type<T>>& result, const std::vector<custom_test_type<T>>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff1 = std::abs(percent * expected[i].x);
        auto diff2 = std::abs(percent * expected[i].y);
        if(!bit_equal(result[i].x, expected[i].x)) ASSERT_NEAR(result[i].x, expected[i].x, diff1) << "where index = " << i;
        if(!bit_equal(result[i].y, expected[i].y)) ASSERT_NEAR(result[i].y, expected[i].y, diff2) << "where index = " << i;
    }
}

template<class T>
auto assert_near(const std::vector<custom_test_type<T>>& result, const std::vector<custom_test_type<T>>& expected, const float)
    -> typename std::enable_if<std::is_integral<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(result[i].x, expected[i].x) << "where index = " << i;
        ASSERT_EQ(result[i].y, expected[i].y) << "where index = " << i;
    }
}

template<class T, std::enable_if_t<std::is_same<T, rocprim::bfloat16>::value ||
                                        std::is_same<T, rocprim::half>::value, bool> = true>
void assert_near(const std::vector<custom_test_type<T>>& result, const std::vector<custom_test_type<T>>& expected, const float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff1 = std::abs(percent * static_cast<float>(expected[i].x));
        auto diff2 = std::abs(percent * static_cast<float>(expected[i].y));
        // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
        if(!bit_equal(result[i].x, expected[i].x))
            ASSERT_NEAR(static_cast<float>(result[i].x), static_cast<float>(expected[i].x), diff1) << "where index = " << i;
        if(!bit_equal(result[i].y, expected[i].y))
            ASSERT_NEAR(static_cast<float>(result[i].y), static_cast<float>(expected[i].y), diff2) << "where index = " << i;
    }
}

template<class T>
auto assert_near(const T& result, const T& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    if(bit_equal(result, expected)) return; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
    auto diff = std::abs(percent * expected);
    ASSERT_NEAR(result, expected, diff);
}

template<class T>
auto assert_near(const T& result, const T& expected, const float)
    -> typename std::enable_if<std::is_integral<T>::value>::type
{
    ASSERT_EQ(result, expected);
}

template<class T, std::enable_if_t<std::is_same<T, rocprim::bfloat16>::value ||
                                        std::is_same<T, rocprim::half>::value, bool> = true>
void assert_near(const T& result, const T& expected, const float percent)
{
    if(bit_equal(result, expected)) return; // Check bitwise equality for +NaN, -NaN, +0.0, -0.0, +inf, -inf.
    auto diff = std::abs(percent * static_cast<float>(expected));
    ASSERT_NEAR(static_cast<float>(result), static_cast<float>(expected), diff);
}

template<class T>
auto assert_near(const custom_test_type<T>& result, const custom_test_type<T>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    auto diff1 = std::abs(percent * expected.x);
    auto diff2 = std::abs(percent * expected.y);
    if(!bit_equal(result.x, expected.x)) ASSERT_NEAR(result.x, expected.x, diff1);
    if(!bit_equal(result.x, expected.x)) ASSERT_NEAR(result.y, expected.y, diff2);
}

template<class T>
auto assert_near(const custom_test_type<T>& result, const custom_test_type<T>& expected, const float)
    -> typename std::enable_if<std::is_integral<T>::value>::type
{
    ASSERT_EQ(result.x,expected.x);
    ASSERT_EQ(result.y,expected.y);
}

// End assert_near

#if ROCPRIM_HAS_INT128_SUPPORT
template<class T>
auto operator<<(std::ostream& os, const T& value)
    -> std::enable_if_t<std::is_same<T, rocprim::int128_t>::value
                            || std::is_same<T, rocprim::uint128_t>::value,
                        std::ostream&>
{
    static const char* charmap = "0123456789";

    std::string result;
    result.reserve(41); // max. 40 digits possible ( uint64_t has 20) plus sign
    rocprim::uint128_t helper = (value < 0) ? -value : value;

    do
    {
        result += charmap[helper % 10];
        helper /= 10;
    }
    while(helper);
    if(value < 0)
    {
        result += "-";
    }
    std::reverse(result.begin(), result.end());

    os << result;
    return os;
}

#endif

template<typename T>
testing::AssertionResult
    bitwise_equal(const char* a_expr, const char* b_expr, const T& a, const T& b)
{
    if(bit_equal(a, b))
    {
        return testing::AssertionSuccess();
    }

    // googletest's operator<< doesn't see the above overload for int128_t
    std::stringstream a_str;
    std::stringstream b_str;
    a_str << std::hexfloat << a;
    b_str << std::hexfloat << b;

    return testing::AssertionFailure()
           << "Expected strict/bitwise equality of these values: " << std::endl
           << "     " << a_expr << ": " << std::hexfloat << a_str.str() << std::endl
           << "     " << b_expr << ": " << std::hexfloat << b_str.str() << std::endl;
}

#define ASSERT_BITWISE_EQ(a, b) ASSERT_PRED_FORMAT2(bitwise_equal, a, b)

template<typename IterA, typename IterB>
void assert_bit_eq(IterA result_begin, IterA result_end, IterB expected_begin, IterB expected_end)
{
    using value_a_t = typename std::iterator_traits<IterA>::value_type;
    using value_b_t = typename std::iterator_traits<IterB>::value_type;

    ASSERT_EQ(std::distance(result_begin, result_end), std::distance(expected_begin, expected_end));
    auto result_it   = result_begin;
    auto expected_it = expected_begin;
    for(size_t index = 0; result_it != result_end; ++result_it, ++expected_it, ++index)
    {
        // The cast is needed, because the argument can be an std::vector<bool> iterator, which's operator*
        // returns a proxy object that must be converted to bool
        const auto result   = static_cast<value_a_t>(*result_it);
        const auto expected = static_cast<value_b_t>(*expected_it);

        ASSERT_BITWISE_EQ(result, expected) << "where index = " << index;
    }
}

template<typename T>
void assert_bit_eq(const std::vector<T>& result, const std::vector<T>& expected)
{
    assert_bit_eq(result.begin(), result.end(), expected.begin(), expected.end());
}
}

#endif //ROCPRIM_TEST_UTILS_ASSERTIONS_HPP
