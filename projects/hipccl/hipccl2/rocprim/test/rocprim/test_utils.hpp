// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_TEST_UTILS_HPP_
#define TEST_TEST_UTILS_HPP_

#include <algorithm>
#include <vector>
#include <random>
#include <type_traits>
#include <cstdlib>

// Google Test
#include <gtest/gtest.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

// Identity iterator
#include "identity_iterator.hpp"
// Bounds checking iterator
#include "bounds_checking_iterator.hpp"

// For better Google Test reporting and debug output of half values
inline
std::ostream& operator<<(std::ostream& stream, const rocprim::half& value)
{
    stream << static_cast<float>(value);
    return stream;
}

namespace test_utils
{

template<class T>
inline auto get_random_data(size_t size, T min, T max)
    -> typename std::enable_if<rocprim::is_integral<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<T> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max)
    -> typename std::enable_if<rocprim::is_floating_point<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    // Generate floats when T is half
    using dis_type = typename std::conditional<std::is_same<rocprim::half, T>::value, float, T>::type;
    std::uniform_real_distribution<dis_type> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template<class T>
inline std::vector<T> get_random_data01(size_t size, float p)
{
    const size_t max_random_size = 1024 * 1024;
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::bernoulli_distribution distribution(p);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return distribution(gen); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline auto get_random_value(T min, T max)
    -> typename std::enable_if<rocprim::is_arithmetic<T>::value, T>::type
{
    return get_random_data(1, min, max)[0];
}

// Can't use std::prefix_sum for inclusive/exclusive scan, because
// it does not handle short[] -> int(int a, int b) { a + b; } -> int[]
// they way we expect. That's because sum in std::prefix_sum's implementation
// is of type typename std::iterator_traits<InputIt>::value_type (short)
template<class InputIt, class OutputIt, class BinaryOperation>
OutputIt host_inclusive_scan(InputIt first, InputIt last,
                             OutputIt d_first, BinaryOperation op)
{
    using input_type = typename std::iterator_traits<InputIt>::value_type;
    using output_type = typename std::iterator_traits<OutputIt>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, output_type, BinaryOperation
    >::type;

    if (first == last) return d_first;

    result_type sum = *first;
    *d_first = sum;

    while (++first != last) {
       sum = op(sum, *first);
       *++d_first = sum;
    }
    return ++d_first;
}

template<class InputIt, class T, class OutputIt, class BinaryOperation>
OutputIt host_exclusive_scan(InputIt first, InputIt last,
                             T initial_value, OutputIt d_first,
                             BinaryOperation op)
{
    using input_type = typename std::iterator_traits<InputIt>::value_type;
    using output_type = typename std::iterator_traits<OutputIt>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, output_type, BinaryOperation
    >::type;

    if (first == last) return d_first;

    result_type sum = initial_value;
    *d_first = initial_value;

    while ((first+1) != last)
    {
       sum = op(sum, *first);
       *++d_first = sum;
       first++;
    }
    return ++d_first;
}

template<class InputIt, class KeyIt, class T, class OutputIt, class BinaryOperation, class KeyCompare>
OutputIt host_exclusive_scan_by_key(InputIt first, InputIt last, KeyIt k_first,
                                    T initial_value, OutputIt d_first,
                                    BinaryOperation op, KeyCompare key_compare_op)
{
    using input_type = typename std::iterator_traits<InputIt>::value_type;
    using output_type = typename std::iterator_traits<OutputIt>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, output_type, BinaryOperation
    >::type;

    if (first == last) return d_first;

    result_type sum = initial_value;
    *d_first = initial_value;

    while ((first+1) != last)
    {
        if(key_compare_op(*k_first, *++k_first))
        {
            sum = op(sum, *first);
        }
        else
        {
            sum = initial_value;
        }
        *++d_first = sum;
        first++;
    }
    return ++d_first;
}

#ifdef ROCPRIM_HC_API
inline
size_t get_max_tile_size(hc::accelerator acc = hc::accelerator())
{
    return acc.get_max_tile_static_size();
}
#endif

#ifdef ROCPRIM_HIP_API
inline
size_t get_max_block_size()
{
    hipDeviceProp_t device_properties;
    hipError_t error = hipGetDeviceProperties(&device_properties, 0);
    if(error != hipSuccess)
    {
        std::cout << "HIP error: " << error
                  << " file: " << __FILE__
                  << " line: " << __LINE__
                  << std::endl;
        std::exit(error);
    }
    return device_properties.maxThreadsPerBlock;
}
#endif

template<class T>
struct is_custom_test_type : std::false_type
{
};

template<class T>
struct inner_type
{
    using type = T;
};

#if defined(ROCPRIM_HC_API) || defined(ROCPRIM_HIP_API)
// Custom type used in tests
template<class T>
struct custom_test_type
{
    using value_type = T;

    T x;
    T y;

    ROCPRIM_HOST_DEVICE inline
    custom_test_type() {}

    ROCPRIM_HOST_DEVICE inline
    custom_test_type(T x, T y) : x(x), y(y) {}

    ROCPRIM_HOST_DEVICE inline
    custom_test_type(T xy) : x(xy), y(xy) {}

    template<class U>
    ROCPRIM_HOST_DEVICE inline
    custom_test_type(const custom_test_type<U>& other)
    {
        x = other.x;
        y = other.y;
    }

    ROCPRIM_HOST_DEVICE inline
    custom_test_type(const custom_test_type& other) : x(other.x), y(other.y) {}

    ROCPRIM_HOST_DEVICE inline
    ~custom_test_type() {}

    ROCPRIM_HOST_DEVICE inline
    custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(x + other.x, y + other.y);
    }

    ROCPRIM_HOST_DEVICE inline
    custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(x - other.x, y - other.y);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(const custom_test_type& other) const
    {
        return (x < other.x && y < other.y);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(const custom_test_type& other) const
    {
        return (x > other.x && y > other.y);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(const custom_test_type& other) const
    {
        return (x == other.x && y == other.y);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
};

template<class T> inline
std::ostream& operator<<(std::ostream& stream,
                         const custom_test_type<T>& value)
{
    stream << "[" << value.x << "; " << value.y << "]";
    return stream;
}

template<class T>
struct is_custom_test_type<custom_test_type<T>> : std::true_type
{
};

template<class T>
struct inner_type<custom_test_type<T>>
{
    using type = T;
};

namespace detail
{
    template<class T>
    struct numeric_limits_custom_test_type : public std::numeric_limits<typename T::value_type>
    {
    };
}

// Numeric limits which also supports custom_test_type<U> classes
template<class T>
struct numeric_limits : public std::conditional<
        is_custom_test_type<T>::value,
        detail::numeric_limits_custom_test_type<T>,
        std::numeric_limits<T>
    >::type
{
};

template<class T>
inline auto get_random_data(size_t size, typename T::value_type min, typename T::value_type max)
    -> typename std::enable_if<
           is_custom_test_type<T>::value && std::is_integral<typename T::value_type>::value,
           std::vector<T>
       >::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<typename T::value_type> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template<class T>
inline auto get_random_data(size_t size, typename T::value_type min, typename T::value_type max)
    -> typename std::enable_if<
           is_custom_test_type<T>::value && std::is_floating_point<typename T::value_type>::value,
           std::vector<T>
       >::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<typename T::value_type> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen)); });
    return data;
}

template<class T>
inline auto get_random_value(typename T::value_type min, typename T::value_type max)
    -> typename std::enable_if<is_custom_test_type<T>::value, T>::type
{
    return get_random_data(1, min, max)[0];
}
#endif

template<class T>
auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float percent)
    -> typename std::enable_if<!is_custom_test_type<T>::value && std::is_arithmetic<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff = std::max<T>(std::abs(percent * expected[i]), T(percent));
        if(std::is_integral<T>::value) diff = 0;
        ASSERT_NEAR(result[i], expected[i], diff) << "where index = " << i;
    }
}

template<class T>
auto assert_near(const T& result, const T& expected, const float percent)
    -> typename std::enable_if<!is_custom_test_type<T>::value && std::is_arithmetic<T>::value>::type
{
    auto diff = std::max<T>(std::abs(percent * expected), T(percent));
    if(std::is_integral<T>::value) diff = 0;
    ASSERT_NEAR(result, expected, diff);
}


template<class T>
auto assert_near(const T& result, const T& expected, const float percent)
    -> typename std::enable_if<is_custom_test_type<T>::value>::type
{
    using value_type = typename T::value_type;
    auto diff1 = std::max<value_type>(std::abs(percent * expected.x), value_type(percent));
    auto diff2 = std::max<value_type>(std::abs(percent * expected.y), value_type(percent));
    if(std::is_integral<value_type>::value)
    {
        diff1 = 0;
        diff2 = 0;
    }
    ASSERT_NEAR(result.x, expected.x, diff1);
    ASSERT_NEAR(result.y, expected.y, diff2);
}

template<class T>
auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float percent)
    -> typename std::enable_if<is_custom_test_type<T>::value>::type
{
    using value_type = typename T::value_type;
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff1 = std::max<value_type>(std::abs(percent * expected[i].x), value_type(percent));
        auto diff2 = std::max<value_type>(std::abs(percent * expected[i].y), value_type(percent));
        if(std::is_integral<value_type>::value)
        {
            diff1 = 0;
            diff2 = 0;
        }
        ASSERT_NEAR(result[i].x, expected[i].x, diff1) << "where index = " << i;
        ASSERT_NEAR(result[i].y, expected[i].y, diff2) << "where index = " << i;
    }
}

template<class T>
auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float)
    -> typename std::enable_if<!is_custom_test_type<T>::value && !std::is_arithmetic<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
    }
}

template<class T>
void assert_eq(const std::vector<T>& result, const std::vector<T>& expected)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
    }
}

void assert_eq(const std::vector<rocprim::half>& result, const std::vector<rocprim::half>& expected)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(static_cast<float>(result[i]), static_cast<float>(expected[i])) << "where index = " << i;
    }
}

} // end test_utils namespace

#endif // TEST_TEST_UTILS_HPP_
