// MIT License
//
// Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_TEST_HIPCUB_TEST_UTILS_ASSERTIONS_HPP_
#define HIPCUB_TEST_HIPCUB_TEST_UTILS_ASSERTIONS_HPP_

// Std::memcpy and std::memcmp
#include <cstring>

#include "test_utils_half.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_custom_test_types.hpp"

namespace test_utils{

template<class T>
bool inline bit_equal(const T a, const T b){
    return std::memcmp(&a,  &b, sizeof(T))==0;
}

/// Checks if `vector<T> result` matches `vector<T> expected`.
/// If max_length is given, equality of `result.size()` and `expected.size()`
/// is ignored and checks only the first max_length elements.
/// \tparam T
/// \param result
/// \param expected
/// \param max_length
template<class T>
inline void assert_eq(const std::vector<T>& result, const std::vector<T>& expected, const size_t max_length = SIZE_MAX)
{
    if(max_length == SIZE_MAX || max_length > expected.size()) ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < std::min(result.size(), max_length); i++)
    {
        if(bit_equal(result[i], expected[i])) continue; // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.

#if defined(_WIN32)
        // GTest's ASSERT_EQ prints the values if the test fails. On Windows, the version of GTest provided by vcpkg doesn't
        // provide overloads for printing 128 bit types, resulting in linker errors.
        // Check if we're testing with 128 bit types. If so, test using bools so GTest doesn't try to print them on failure.
        if (test_utils::is_int128<T>::value || test_utils::is_uint128<T>::value)
        {
            const bool values_equal = (result[i] == expected[i]);
            ASSERT_EQ(values_equal, true) << "where index = " << i;
        }
        else
        {
            ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
        }
#else
        ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
#endif
    }
}

inline void assert_eq(const std::vector<test_utils::half>& result, const std::vector<test_utils::half>& expected, const size_t max_length = SIZE_MAX)
{
    if(max_length == SIZE_MAX || max_length > expected.size()) ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < std::min(result.size(), max_length); i++)
    {
        if(bit_equal(result[i], expected[i])) continue; // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.
        ASSERT_EQ(test_utils::native_half(result[i]), test_utils::native_half(expected[i])) << "where index = " << i;
    }
}

inline void assert_eq(const std::vector<test_utils::bfloat16>& result, const std::vector<test_utils::bfloat16>& expected, const size_t max_length = SIZE_MAX)
{
    if(max_length == SIZE_MAX || max_length > expected.size()) ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < std::min(result.size(), max_length); i++)
    {
        if(bit_equal(result[i], expected[i])) continue; // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.
        ASSERT_EQ(test_utils::native_bfloat16(result[i]), test_utils::native_bfloat16(expected[i])) << "where index = " << i;
    }
}

template<class T>
inline void assert_eq(const T& result, const T& expected)
{
    if(bit_equal(result, expected)) return; // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.
    ASSERT_EQ(result, expected);
}

inline void assert_eq(const test_utils::half& result, const test_utils::half& expected)
{
    if(bit_equal(result, expected)) return; // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.
    ASSERT_EQ(test_utils::native_half(result), test_utils::native_half(expected));
}

inline void assert_eq(const test_utils::bfloat16& result, const test_utils::bfloat16& expected)
{
    if(bit_equal(result, expected)) return; // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.
    ASSERT_EQ(test_utils::native_bfloat16(result), test_utils::native_bfloat16(expected));
}
// end assert_eq

// begin assert_near
template<class T>
inline auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        if(bit_equal(result[i], expected[i])) continue; // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.
        auto diff = std::abs(percent * expected[i]);
        ASSERT_NEAR(result[i], expected[i], diff) << "where index = " << i;
    }
}

template<class T>
inline auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float percent)
    -> typename std::enable_if<std::is_integral<T>::value>::type
{
    (void)percent;
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
    }
}

template<class T, std::enable_if_t<std::is_same<T, test_utils::bfloat16>::value ||
                                       std::is_same<T, test_utils::half>::value, bool> = true>
inline void assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        if(bit_equal(result[i], expected[i])) continue; // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.
        auto diff = std::abs(percent * static_cast<float>(expected[i]));
        ASSERT_NEAR(static_cast<float>(result[i]), static_cast<float>(expected[i]), diff) << "where index = " << i;
    }
}

template<class T>
inline auto assert_near(const std::vector<custom_test_type<T>>& result, const std::vector<custom_test_type<T>>& expected, const float percent)
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
inline auto assert_near(const std::vector<custom_test_type<T>>& result, const std::vector<custom_test_type<T>>& expected, const float)
    -> typename std::enable_if<std::is_integral<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(result[i].x, expected[i].x) << "where index = " << i;
        ASSERT_EQ(result[i].y, expected[i].y) << "where index = " << i;
    }
}

template<class T, std::enable_if_t<std::is_same<T, test_utils::bfloat16>::value ||
                                       std::is_same<T, test_utils::half>::value, bool> = true>
inline void assert_near(const std::vector<custom_test_type<T>>& result, const std::vector<custom_test_type<T>>& expected, const float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff1 = std::abs(percent * static_cast<float>(expected[i].x));
        auto diff2 = std::abs(percent * static_cast<float>(expected[i].y));
        // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.
        if(!bit_equal(result[i].x, expected[i].x))
            ASSERT_NEAR(static_cast<float>(result[i].x), static_cast<float>(expected[i].x), diff1) << "where index = " << i;
        if(!bit_equal(result[i].y, expected[i].y))
            ASSERT_NEAR(static_cast<float>(result[i].y), static_cast<float>(expected[i].y), diff2) << "where index = " << i;
    }
}

template<class T>
inline auto assert_near(const T& result, const T& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    if(bit_equal(result, expected)) return; // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.
    auto diff = std::abs(percent * expected);
    ASSERT_NEAR(result, expected, diff);
}

template<class T>
inline auto assert_near(const T& result, const T& expected, const float)
    -> typename std::enable_if<std::is_integral<T>::value>::type
{
    ASSERT_EQ(result, expected);
}

template<class T, std::enable_if_t<std::is_same<T, test_utils::bfloat16>::value ||
                                       std::is_same<T, test_utils::half>::value, bool> = true>
inline void assert_near(const T& result, const T& expected, const float percent)
{
    if(bit_equal(result, expected)) return; // Check to also regard equality of NaN's, -NaN, +inf, -inf as correct.
    auto diff = std::abs(percent * static_cast<float>(expected));
    ASSERT_NEAR(static_cast<float>(result), static_cast<float>(expected), diff);
}

template<class T>
inline auto assert_near(const custom_test_type<T>& result, const custom_test_type<T>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    auto diff1 = std::abs(percent * expected.x);
    auto diff2 = std::abs(percent * expected.y);
    if(!bit_equal(result.x, expected.x)) ASSERT_NEAR(result.x, expected.x, diff1);
    if(!bit_equal(result.x, expected.x)) ASSERT_NEAR(result.y, expected.y, diff2);
}

template<class T>
inline auto assert_near(const custom_test_type<T>& result, const custom_test_type<T>& expected, const float)
    -> typename std::enable_if<std::is_integral<T>::value>::type
{
    ASSERT_EQ(result.x,expected.x);
    ASSERT_EQ(result.y,expected.y);
}

// End assert_near

template<class T>
inline void assert_bit_eq(const std::vector<T>& result, const std::vector<T>& expected)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        if(!bit_equal(result[i], expected[i]))
        {
            FAIL() << "Expected strict/bitwise equality of these values: " << std::endl
                   << "     result[i]: " << result[i] << std::endl
                   << "     expected[i]: " << expected[i] << std::endl
                   << "where index = " << i;
        }
    }
}

#if HIPCUB_IS_INT128_ENABLED
inline void assert_bit_eq(const std::vector<__int128_t>& result,
                          const std::vector<__int128_t>& expected)
{
    ASSERT_EQ(result.size(), expected.size());

    auto to_string = [](__int128_t value)
    {
        static const char* charmap = "0123456789";

        std::string result;
        result.reserve(41); // max. 40 digits possible ( uint64_t has 20) plus sign
        __uint128_t helper = (value < 0) ? -value : value;

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
        return result;
    };

    for(size_t i = 0; i < result.size(); i++)
    {
        if(!bit_equal(result[i], expected[i]))
        {
            FAIL() << "Expected strict/bitwise equality of these values: " << std::endl
                   << "     result[i]: " << to_string(result[i]) << std::endl
                   << "     expected[i]: " << to_string(expected[i]) << std::endl
                   << "where index = " << i;
        }
    }
}

inline void assert_bit_eq(const std::vector<__uint128_t>& result,
                          const std::vector<__uint128_t>& expected)
{
    ASSERT_EQ(result.size(), expected.size());

    auto to_string = [](__uint128_t value)
    {
        static const char* charmap = "0123456789";

        std::string result;
        result.reserve(40); // max. 40 digits possible ( uint64_t has 20)
        __uint128_t helper = value;

        do
        {
            result += charmap[helper % 10];
            helper /= 10;
        }
        while(helper);
        std::reverse(result.begin(), result.end());
        return result;
    };

    for(size_t i = 0; i < result.size(); i++)
    {
        if(!bit_equal(result[i], expected[i]))
        {
            FAIL() << "Expected strict/bitwise equality of these values: " << std::endl
                   << "     result[i]: " << to_string(result[i]) << std::endl
                   << "     expected[i]: " << to_string(expected[i]) << std::endl
                   << "where index = " << i;
        }
    }
}
#endif //HIPCUB_IS_INT128_ENABLED

/// Compile-time assertion for type equality of two objects.
template<class ExpectedT, class ActualT>
inline void assert_type(ExpectedT /*obj1*/, ActualT /*obj2*/)
{
    testing::StaticAssertTypeEq<ExpectedT, ActualT>();
}
}
#endif  // HIPCUB_TEST_HIPCUB_TEST_UTILS_ASSERTIONS_HPP_
