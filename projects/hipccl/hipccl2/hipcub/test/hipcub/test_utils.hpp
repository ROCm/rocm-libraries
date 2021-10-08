// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_TEST_TEST_UTILS_HPP_
#define HIPCUB_TEST_TEST_UTILS_HPP_

#ifndef TEST_UTILS_INCLUDE_GAURD
    #error test_utils.hpp must ONLY be included by common_test_header.hpp. Please include common_test_header.hpp instead.
#endif

// hipCUB API
#ifdef __HIP_PLATFORM_AMD__
    #include "hipcub/backend/rocprim/util_ptx.hpp"
#elif defined(__HIP_PLATFORM_NVIDIA__)
    #include "hipcub/config.hpp"
    #include <cub/util_ptx.cuh>
#endif

#include "test_utils_sort_comparator.hpp"
#include "test_utils_half.hpp"
#include "test_utils_bfloat16.hpp"

// Seed values
#include "test_seed.hpp"

inline
std::ostream& operator<<(std::ostream& stream, const test_utils::half& value)
{
    stream << static_cast<float>(value);
    return stream;
}

namespace test_utils
{

template<class T>
struct precision_threshold
{
    static constexpr float percentage = 0.01f;
};

template<>
struct precision_threshold<test_utils::half>
{
    static constexpr float percentage = 0.075f;
};

template<>
struct precision_threshold<test_utils::bfloat16>
{
    static constexpr float percentage = 0.075f;
};

template<class T>
inline auto get_random_data(size_t size, T min, T max, int seed_value)
    -> typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<T> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max, int seed_value)
    -> typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_real_distribution<T> distribution(min, max);
    std::vector<T> data(size);
    std::generate(
        data.begin(),
        data.end(),
        [&]() { return static_cast<T>(distribution(gen)); }
    );
    return data;
}

#if defined(_WIN32) && defined(__clang__)
template<>
inline std::vector<unsigned char> get_random_data(size_t size, unsigned char min, unsigned char max, int seed_value)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<int> distribution(static_cast<int>(min), static_cast<int>(max));
    std::vector<unsigned char> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<unsigned char>(distribution(gen)); });
    return data;
}

template<>
inline std::vector<signed char> get_random_data(size_t size, signed char min, signed char max, int seed_value)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<int> distribution(static_cast<int>(min), static_cast<int>(max));
    std::vector<signed char> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<signed char>(distribution(gen)); });
    return data;
}

template<>
inline std::vector<char> get_random_data(size_t size, char min, char max, int seed_value)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<int> distribution(static_cast<int>(min), static_cast<int>(max));
    std::vector<char> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<char>(distribution(gen)); });
    return data;
}
#endif

template<class T>
inline auto get_random_data(size_t size, T min, T max, int seed_value)
    -> typename std::enable_if<std::is_same<test_utils::half, T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_real_distribution<float> distribution(min, max);
    std::vector<T> data(size);
    std::generate(
        data.begin(),
        data.end(),
        [&]() { return static_cast<T>(distribution(gen)); }
    );
    return data;
}

template<class T>
inline auto get_random_data(size_t size, float min, float max, int seed_value)
    -> typename std::enable_if<std::is_same<test_utils::bfloat16, T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_real_distribution<float> distribution(min, max);
    std::vector<T> data(size);
    std::generate(
        data.begin(),
        data.end(),
        [&]() { return static_cast<T>(distribution(gen)); }
        );
    return data;
}

template<class T>
inline std::vector<T> get_random_data01(size_t size, float p, int seed_value)
{
    const size_t max_random_size = 1024 * 1024;
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
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
inline auto get_random_value(T min, T max, int seed_value)
    -> typename std::enable_if<std::is_arithmetic<T>::value, T>::type
{
    return get_random_data(1, min, max, seed_value)[0];
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
    using result_type =
        typename std::conditional<
            std::is_void<output_type>::value, input_type, output_type
        >::type;

    if (first == last) return d_first;

    result_type sum = *first;
    *d_first = sum;

    while (++first != last) {
       sum = op(sum, static_cast<result_type>(*first));
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
    using result_type =
        typename std::conditional<
            std::is_void<output_type>::value, input_type, output_type
        >::type;

    if (first == last) return d_first;

    result_type sum = initial_value;
    *d_first = initial_value;

    while ((first+1) != last)
    {
       sum = op(sum, static_cast<result_type>(*first));
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
    using result_type =
        typename std::conditional<
            std::is_void<output_type>::value, input_type, output_type
        >::type;

    if (first == last) return d_first;

    result_type sum = initial_value;
    *d_first = initial_value;

    while ((first+1) != last)
    {
        if(key_compare_op(*k_first, *++k_first))
        {
            sum = op(sum, static_cast<result_type>(*first));
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

template<class T>
HIPCUB_HOST_DEVICE inline
constexpr T max(const T& a, const T& b)
{
    return a < b ? b : a;
}

template<class T>
HIPCUB_HOST_DEVICE inline
constexpr T min(const T& a, const T& b)
{
    return a < b ? a : b;
}

template<class T>
HIPCUB_HOST_DEVICE inline
constexpr bool is_power_of_two(const T x)
{
    static_assert(std::is_integral<T>::value, "T must be integer type");
    return (x > 0) && ((x & (x - 1)) == 0);
}

template<class T>
HIPCUB_HOST_DEVICE inline
constexpr T next_power_of_two(const T x, const T acc = 1)
{
    static_assert(std::is_unsigned<T>::value, "T must be unsigned type");
    return acc >= x ? acc : next_power_of_two(x, 2 * acc);
}

// Return id of "logical warp" in a block
template<unsigned int LogicalWarpSize = HIPCUB_DEVICE_WARP_THREADS>
HIPCUB_DEVICE inline
unsigned int logical_warp_id()
{
    return hipcub::RowMajorTid(1, 1, 1)/LogicalWarpSize;
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

// Select the minimal warp size for block of size block_size, it's
// useful for blocks smaller than maximal warp size.
template<class T>
HIPCUB_HOST_DEVICE inline
constexpr T get_min_warp_size(const T block_size, const T max_warp_size)
{
    static_assert(std::is_unsigned<T>::value, "T must be unsigned type");
    return block_size >= max_warp_size ? max_warp_size : next_power_of_two(block_size);
}

template<class T>
struct custom_test_type
{
    using value_type = T;

    T x;
    T y;

    HIPCUB_HOST_DEVICE inline
    constexpr custom_test_type() {}

    HIPCUB_HOST_DEVICE inline
    constexpr custom_test_type(T x, T y) : x(x), y(y) {}

    HIPCUB_HOST_DEVICE inline
    constexpr custom_test_type(T xy) : x(xy), y(xy) {}

    template<class U>
    HIPCUB_HOST_DEVICE inline
    custom_test_type(const custom_test_type<U>& other)
    {
        x = other.x;
        y = other.y;
    }

    #ifndef HIPCUB_CUB_API
    HIPCUB_HOST_DEVICE inline
    ~custom_test_type() = default;
    #endif

    HIPCUB_HOST_DEVICE inline
    custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    HIPCUB_HOST_DEVICE inline
    custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(x + other.x, y + other.y);
    }

    HIPCUB_HOST_DEVICE inline
    custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(x - other.x, y - other.y);
    }

    HIPCUB_HOST_DEVICE inline
    bool operator<(const custom_test_type& other) const
    {
        return (x < other.x || (x == other.x && y < other.y));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator>(const custom_test_type& other) const
    {
        return (x > other.x || (x == other.x && y > other.y));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator==(const custom_test_type& other) const
    {
        return (x == other.x && y == other.y);
    }

    HIPCUB_HOST_DEVICE inline
    bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
};

//Overload for test_utils::half
template<>
struct custom_test_type<test_utils::half>
{
    using value_type = test_utils::half;

    test_utils::half x;
    test_utils::half y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_test_type(), value) != value
    HIPCUB_HOST_DEVICE inline
    custom_test_type() : x(12), y(34) {}

    HIPCUB_HOST_DEVICE inline
    custom_test_type(test_utils::half x, test_utils::half y) : x(x), y(y) {}

    HIPCUB_HOST_DEVICE inline
    custom_test_type(test_utils::half xy) : x(xy), y(xy) {}

    template<class U>
    HIPCUB_HOST_DEVICE inline
    custom_test_type(const custom_test_type<U>& other)
    {
        x = other.x;
        y = other.y;
    }

    HIPCUB_HOST_DEVICE inline
    ~custom_test_type() {}

    HIPCUB_HOST_DEVICE inline
    custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    HIPCUB_HOST_DEVICE inline
    custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(half_plus()(x, other.x), half_plus()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
    custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(half_minus()(x, other.x), half_minus()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator<(const custom_test_type& other) const
    {
        return (test_utils::less()(x, other.x) || (half_equal_to()(x, other.x) && test_utils::less()(y, other.y)));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator>(const custom_test_type& other) const
    {
        return (greater()(x, other.x) || (half_equal_to()(x, other.x) && greater()(y, other.y)));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator==(const custom_test_type& other) const
    {
        return (half_equal_to()(x, other.x) && half_equal_to()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
};

//Overload for test_utils::bfloat16
template<>
struct custom_test_type<test_utils::bfloat16>
{
    using value_type = test_utils::bfloat16;

    test_utils::bfloat16 x;
    test_utils::bfloat16 y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_test_type(), value) != value
    HIPCUB_HOST_DEVICE inline
    custom_test_type() : x(float(12)), y(float(34)) {}

    HIPCUB_HOST_DEVICE inline
    custom_test_type(test_utils::bfloat16 x, test_utils::bfloat16 y) : x(x), y(y) {}

    HIPCUB_HOST_DEVICE inline
    custom_test_type(test_utils::bfloat16 xy) : x(xy), y(xy) {}

    template<class U>
    HIPCUB_HOST_DEVICE inline
    custom_test_type(const custom_test_type<U>& other)
    {
        x = other.x;
        y = other.y;
    }

    HIPCUB_HOST_DEVICE inline
    ~custom_test_type() {}

    HIPCUB_HOST_DEVICE inline
    custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    HIPCUB_HOST_DEVICE inline
    custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(bfloat16_plus()(x, other.x), bfloat16_plus()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
    custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(bfloat16_minus()(x, other.x), bfloat16_minus()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator<(const custom_test_type& other) const
    {
        return (test_utils::less()(x, other.x) || (bfloat16_equal_to()(x, other.x) && test_utils::less()(y, other.y)));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator>(const custom_test_type& other) const
    {
        return (greater()(x, other.x) || (bfloat16_equal_to()(x, other.x) && greater()(y, other.y)));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator==(const custom_test_type& other) const
    {
        return (bfloat16_equal_to()(x, other.x) && bfloat16_equal_to()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
};

template<class T>
struct is_custom_test_type : std::false_type
{
};

template<class T>
struct is_custom_test_type<custom_test_type<T>> : std::true_type
{
};

template<class T>
inline auto get_random_data(size_t size, typename T::value_type min, typename T::value_type max, int seed_value)
    -> typename std::enable_if<
           is_custom_test_type<T>::value && std::is_integral<typename T::value_type>::value,
           std::vector<T>
       >::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<typename T::value_type> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen), distribution(gen)); });
    return data;
}

template<class T>
inline auto get_random_data(size_t size, typename T::value_type min, typename T::value_type max, int seed_value)
    -> typename std::enable_if<
           is_custom_test_type<T>::value && std::is_floating_point<typename T::value_type>::value,
           std::vector<T>
       >::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    gen.seed(seed_value);
    std::uniform_real_distribution<typename T::value_type> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen), distribution(gen)); });
    return data;
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

void assert_near(const std::vector<test_utils::half>& result, const std::vector<test_utils::half>& expected, float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff = std::max<float>(std::abs(percent * static_cast<float>(expected[i])), percent);
        ASSERT_NEAR(static_cast<float>(result[i]), static_cast<float>(expected[i]), diff) << "where index = " << i;
    }
}

void assert_near(const std::vector<custom_test_type<test_utils::half>>& result, const std::vector<custom_test_type<test_utils::half>>& expected, const float percent)
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

void assert_near(const std::vector<test_utils::bfloat16>& result, const std::vector<test_utils::bfloat16>& expected, float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff = std::max<float>(std::abs(percent * static_cast<float>(expected[i])), percent);
        ASSERT_NEAR(static_cast<float>(result[i]), static_cast<float>(expected[i]), diff) << "where index = " << i;
    }
}

void assert_near(const std::vector<custom_test_type<test_utils::bfloat16>>& result, const std::vector<custom_test_type<test_utils::bfloat16>>& expected, const float percent)
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

void assert_near(const test_utils::half& result, const test_utils::half& expected, float percent)
{
    auto diff = std::max<float>(std::abs(percent * static_cast<float>(expected)), percent);
    ASSERT_NEAR(static_cast<float>(result), static_cast<float>(expected), diff);
}

void assert_near(const test_utils::bfloat16& result, const test_utils::bfloat16& expected, float percent)
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

void assert_eq(const std::vector<test_utils::half>& result, const std::vector<test_utils::half>& expected)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(test_utils::native_half(result[i]), test_utils::native_half(expected[i])) << "where index = " << i;
    }
}

void assert_eq(const std::vector<test_utils::bfloat16>& result, const std::vector<test_utils::bfloat16>& expected)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(test_utils::native_bfloat16(result[i]), test_utils::native_bfloat16(expected[i])) << "where index = " << i;
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

void custom_assert_eq(const std::vector<test_utils::half>& result, const std::vector<test_utils::half>& expected, size_t size)
{
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(test_utils::native_half(result[i]), test_utils::native_half(expected[i])) << "where index = " << i;
    }
}

void custom_assert_eq(const std::vector<test_utils::bfloat16>& result, const std::vector<test_utils::bfloat16>& expected, size_t size)
{
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(test_utils::native_bfloat16(result[i]), test_utils::native_bfloat16(expected[i])) << "where index = " << i;
    }
}


template<class T>
void assert_eq(const T& result, const T& expected)
{
    ASSERT_EQ(result, expected);
}

void assert_eq(const test_utils::half& result, const test_utils::half& expected)
{
    ASSERT_EQ(test_utils::native_half(result), test_utils::native_half(expected));
}

void assert_eq(const test_utils::bfloat16& result, const test_utils::bfloat16& expected)
{
    ASSERT_EQ(test_utils::native_bfloat16(result), test_utils::native_bfloat16(expected));
}


} // end test_util namespace

// Need for hipcub::DeviceReduce::Min/Max etc.
namespace std
{
    template<>
    class numeric_limits<test_utils::custom_test_type<int>>
    {
        using T = typename test_utils::custom_test_type<int>;

        public:

        static constexpr inline T max()
        {
            return std::numeric_limits<typename T::value_type>::max();
        }

        static constexpr inline T lowest()
        {
            return std::numeric_limits<typename T::value_type>::lowest();
        }
    };

    template<>
    class numeric_limits<test_utils::custom_test_type<float>>
    {
        using T = typename test_utils::custom_test_type<float>;

        public:

        static constexpr inline T max()
        {
            return std::numeric_limits<typename T::value_type>::max();
        }

        static constexpr inline T min()
        {
            return std::numeric_limits<typename T::value_type>::min();
        }

        static constexpr inline T lowest()
        {
            return std::numeric_limits<typename T::value_type>::lowest();
        }
    };

    template<>
    class numeric_limits<test_utils::half> {
        using T = typename test_utils::half;

    public:

        static inline T max() {
            return test_utils::half(65504.0f);
        }

        static inline T min() {
            return test_utils::half(0.0f);
        }

        static inline T lowest() {
            return test_utils::half(-65504.0f);
        }
    };

    template<>
    class numeric_limits<test_utils::bfloat16> {
        using T = typename test_utils::bfloat16;

    public:

        static inline T max() {
            return test_utils::bfloat16(std::numeric_limits<float>::max()*0.998);
        }

        static inline T min() {
            return test_utils::bfloat16(std::numeric_limits<float>::min());
        }

        static inline T lowest() {
            return test_utils::bfloat16(std::numeric_limits<float>::lowest()*0.998);
        }
    };
}

#endif // HIPCUB_TEST_HIPCUB_TEST_UTILS_HPP_
