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

// Support half operators on host side

ROCPRIM_HOST inline
_Float16 half_to_native(const rocprim::half& x)
{
    return *reinterpret_cast<const _Float16 *>(&x);
}

ROCPRIM_HOST inline
rocprim::half native_to_half(const _Float16& x)
{
    return *reinterpret_cast<const rocprim::half *>(&x);
}

struct half_less
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b;
        #else
        return half_to_native(a) < half_to_native(b);
        #endif
    }
};

struct half_less_equal
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a <= b;
        #else
        return half_to_native(a) <= half_to_native(b);
        #endif
    }
};

struct half_greater
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a > b;
        #else
        return half_to_native(a) > half_to_native(b);
        #endif
    }
};

struct half_greater_equal
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a >= b;
        #else
        return half_to_native(a) >= half_to_native(b);
        #endif
    }
};

struct half_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a == b;
        #else
        return half_to_native(a) == half_to_native(b);
        #endif
    }
};

struct half_not_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a != b;
        #else
        return half_to_native(a) != half_to_native(b);
        #endif
    }
};

struct half_plus
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::half operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a + b;
        #else
        return native_to_half(half_to_native(a) + half_to_native(b));
        #endif
    }
};

struct half_minus
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::half operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a - b;
        #else
        return native_to_half(half_to_native(a) - half_to_native(b));
        #endif
    }
};

struct half_multiplies
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::half operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a * b;
        #else
        return native_to_half(half_to_native(a) * half_to_native(b));
        #endif
    }
};

struct half_maximum
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::half operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b ? b : a;
        #else
        return half_to_native(a) < half_to_native(b) ? b : a;
        #endif
    }
};

struct half_minimum
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::half operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b ? a : b;
        #else
        return half_to_native(a) < half_to_native(b) ? a : b;
        #endif
    }
};

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
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, BinaryOperation
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
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, BinaryOperation
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
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, BinaryOperation
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

template<class T>
struct is_custom_test_type : std::false_type
{
};

template<class T>
struct is_custom_test_array_type : std::false_type
{
};

template<class T>
struct inner_type
{
    using type = T;
};

// Custom type used in tests
template<class T>
struct custom_test_type
{
    using value_type = T;

    T x;
    T y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_test_type(), value) != value
    ROCPRIM_HOST_DEVICE inline
    custom_test_type() : x(12), y(34) {}

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
        return (x < other.x || (x == other.x && y < other.y));
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(const custom_test_type& other) const
    {
        return (x > other.x || (x == other.x && y > other.y));
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

//Overload for rocprim::half
template<>
struct custom_test_type<rocprim::half>
{
    using value_type = rocprim::half;

    rocprim::half x;
    rocprim::half y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_test_type(), value) != value
    ROCPRIM_HOST_DEVICE inline
    custom_test_type() : x(12), y(34) {}

    ROCPRIM_HOST_DEVICE inline
    custom_test_type(rocprim::half x, rocprim::half y) : x(x), y(y) {}

    ROCPRIM_HOST_DEVICE inline
    custom_test_type(rocprim::half xy) : x(xy), y(xy) {}

    template<class U>
    ROCPRIM_HOST_DEVICE inline
    custom_test_type(const custom_test_type<U>& other)
    {
        x = other.x;
        y = other.y;
    }

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
        return custom_test_type(half_plus()(x, other.x), half_plus()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
    custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(half_minus()(x, other.x), half_minus()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(const custom_test_type& other) const
    {
        return (half_less()(x, other.x) || (half_equal_to()(x, other.x) && half_less()(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(const custom_test_type& other) const
    {
        return (half_greater()(x, other.x) || (half_equal_to()(x, other.x) && half_greater()(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(const custom_test_type& other) const
    {
        return (half_equal_to()(x, other.x) && half_equal_to()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
};

// Custom type used in tests
template<class T, size_t N>
struct custom_test_array_type
{
    using value_type = T;
    static constexpr size_t size = N;

    T values[N];

    ROCPRIM_HOST_DEVICE inline
    custom_test_array_type()
    {
        for(size_t i = 0; i < N; i++)
        {
            values[i] = T(i + 1);
        }
    }

    ROCPRIM_HOST_DEVICE inline
    custom_test_array_type(T v)
    {
        for(size_t i = 0; i < N; i++)
        {
            values[i] = v;
        }
    }

    template<class U>
    ROCPRIM_HOST_DEVICE inline
    custom_test_array_type(const custom_test_array_type<U, N>& other)
    {
        for(size_t i = 0; i < N; i++)
        {
            values[i] = other.values[i];
        }
    }

    ROCPRIM_HOST_DEVICE inline
    ~custom_test_array_type() {}

    ROCPRIM_HOST_DEVICE inline
    custom_test_array_type& operator=(const custom_test_array_type& other)
    {
        for(size_t i = 0; i < N; i++)
        {
            values[i] = other.values[i];
        }
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    custom_test_array_type operator+(const custom_test_array_type& other) const
    {
        custom_test_array_type result;
        for(size_t i = 0; i < N; i++)
        {
            result.values[i] = values[i] + other.values[i];
        }
        return result;
    }

    ROCPRIM_HOST_DEVICE inline
    custom_test_array_type operator-(const custom_test_array_type& other) const
    {
        custom_test_array_type result;
        for(size_t i = 0; i < N; i++)
        {
            result.values[i] = values[i] - other.values[i];
        }
        return result;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(const custom_test_array_type& other) const
    {
        for(size_t i = 0; i < N; i++)
        {
            if(values[i] >= other.values[i])
            {
                return false;
            }
        }
        return true;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(const custom_test_array_type& other) const
    {
        for(size_t i = 0; i < N; i++)
        {
            if(values[i] <= other.values[i])
            {
                return false;
            }
        }
        return true;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(const custom_test_array_type& other) const
    {
        for(size_t i = 0; i < N; i++)
        {
            if(values[i] != other.values[i])
            {
                return false;
            }
        }
        return true;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(const custom_test_array_type& other) const
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

template<class T, size_t N> inline
std::ostream& operator<<(std::ostream& stream,
                         const custom_test_array_type<T, N>& value)
{
    stream << "[";
    for(size_t i = 0; i < N; i++)
    {
        stream << value.values[i];
        if(i != N - 1)
        {
            stream << "; ";
        }
    }
    stream << "]";
    return stream;
}

template<class T>
struct is_custom_test_type<custom_test_type<T>> : std::true_type
{
};

template<class T, size_t N>
struct is_custom_test_array_type<custom_test_array_type<T, N>> : std::true_type
{
};


template<class T>
struct inner_type<custom_test_type<T>>
{
    using type = T;
};

template<class T, size_t N>
struct inner_type<custom_test_array_type<T, N>>
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
    std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen), distribution(gen)); });
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
    std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen), distribution(gen)); });
    return data;
}

template<class T>
inline auto get_random_data(size_t size, typename T::value_type min, typename T::value_type max)
    -> typename std::enable_if<
           is_custom_test_array_type<T>::value && std::is_integral<typename T::value_type>::value,
           std::vector<T>
       >::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<typename T::value_type> distribution(min, max);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.end(),
        [&]()
        {
            T result;
            for(size_t i = 0; i < T::size; i++)
            {
                result.values[i] = distribution(gen);
            }
            return result;
        }
    );
    return data;
}

template<class T>
inline auto get_random_value(typename T::value_type min, typename T::value_type max)
    -> typename std::enable_if<is_custom_test_type<T>::value || is_custom_test_array_type<T>::value, T>::type
{
    return get_random_data(1, min, max)[0];
}

template<class T>
auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff = std::max<T>(std::abs(percent * expected[i]), T(percent));
        ASSERT_NEAR(result[i], expected[i], diff) << "where index = " << i;
    }
}

template<class T>
auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float percent)
    -> typename std::enable_if<!std::is_floating_point<T>::value>::type
{
    (void)percent;
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
    }
}

void assert_near(const std::vector<rocprim::half>& result, const std::vector<rocprim::half>& expected, float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff = std::max<float>(std::abs(percent * static_cast<float>(expected[i])), percent);
        ASSERT_NEAR(static_cast<float>(result[i]), static_cast<float>(expected[i]), diff) << "where index = " << i;
    }
}

void assert_near(const std::vector<custom_test_type<rocprim::half>>& result, const std::vector<custom_test_type<rocprim::half>>& expected, const float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff1 = std::max<float>(std::abs(percent * static_cast<float>(expected[i].x)), percent);
        auto diff2 = std::max<float>(std::abs(percent * static_cast<float>(expected[i].y)), percent);
        ASSERT_NEAR(static_cast<float>(result[i].x), static_cast<float>(expected[i].x), diff1) << "where index = " << i;
        ASSERT_NEAR(static_cast<float>(result[i].y), static_cast<float>(expected[i].y), diff2) << "where index = " << i;
    }
}

template<class T>
auto assert_near(const std::vector<custom_test_type<T>>& result, const std::vector<custom_test_type<T>>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff1 = std::max<T>(std::abs(percent * expected[i].x), T(percent));
        auto diff2 = std::max<T>(std::abs(percent * expected[i].y), T(percent));
        ASSERT_NEAR(result[i].x, expected[i].x, diff1) << "where index = " << i;
        ASSERT_NEAR(result[i].y, expected[i].y, diff2) << "where index = " << i;
    }
}

template<class T>
auto assert_near(const T& result, const T& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    auto diff = std::max<T>(std::abs(percent * expected), T(percent));
    ASSERT_NEAR(result, expected, diff);
}

template<class T>
auto assert_near(const T& result, const T& expected, const float percent)
    -> typename std::enable_if<!std::is_floating_point<T>::value>::type
{
    (void)percent;
    ASSERT_EQ(result, expected);
}

void assert_near(const rocprim::half& result, const rocprim::half& expected, float percent)
{
    auto diff = std::max<float>(std::abs(percent * static_cast<float>(expected)), percent);
    ASSERT_NEAR(static_cast<float>(result), static_cast<float>(expected), diff);
}

template<class T>
auto assert_near(const custom_test_type<T>& result, const custom_test_type<T>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    auto diff1 = std::max<T>(std::abs(percent * expected.x), T(percent));
    auto diff2 = std::max<T>(std::abs(percent * expected.y), T(percent));
    ASSERT_NEAR(result.x, expected.x, diff1);
    ASSERT_NEAR(result.y, expected.y, diff2);
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
        ASSERT_EQ(half_to_native(result[i]), half_to_native(expected[i])) << "where index = " << i;
    }
}

template<class T>
void custom_assert_eq(const std::vector<T>& result, const std::vector<T>& expected, size_t size)
{
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
    }
}

void custom_assert_eq(const std::vector<rocprim::half>& result, const std::vector<rocprim::half>& expected, size_t size)
{
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(half_to_native(result[i]), half_to_native(expected[i])) << "where index = " << i;
    }
}

template<class T>
void assert_eq(const T& result, const T& expected)
{
    ASSERT_EQ(result, expected);
}

void assert_eq(const rocprim::half& result, const rocprim::half& expected)
{
    ASSERT_EQ(half_to_native(result), half_to_native(expected));
}


} // end test_utils namespace

#endif // TEST_TEST_UTILS_HPP_
