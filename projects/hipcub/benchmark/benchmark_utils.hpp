// MIT License
//
// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "primbench.hpp"

#include <hip/hip_runtime.h>

#ifdef __HIP_PLATFORM_AMD__
    #include <hipcub/backend/rocprim/util_ptx.hpp>
#elif defined(__HIP_PLATFORM_NVIDIA__)
    #include <cub/util_ptx.cuh>
    #include <hipcub/config.hpp>
#endif

#include <hipcub/tuple.hpp>

#ifndef HIPCUB_CUB_API
    #define HIPCUB_WARP_THREADS_MACRO warpSize
#else
    #define HIPCUB_WARP_THREADS_MACRO CUB_PTX_WARP_THREADS
#endif

#define HIP_CHECK(condition)                                                           \
    {                                                                                  \
        hipError_t error = condition;                                                  \
        if(error != hipSuccess)                                                        \
        {                                                                              \
            std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
            exit(error);                                                               \
        }                                                                              \
    }

namespace benchmark_utils
{

const size_t default_max_random_size = 1024 * 1024;

// get_random_data() generates only part of sequence and replicates it,
// because benchmarks usually do not need "true" random sequence.
template<class T>
inline auto
    get_random_data(size_t items, T min, T max, size_t max_random_size = default_max_random_size) ->
    typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
{
    std::random_device         rd;
    std::default_random_engine gen(rd());
    using distribution_type = typename std::conditional<(sizeof(T) == 1), short, T>::type;
    std::uniform_int_distribution<distribution_type> distribution(min, max);
    std::vector<T>                                   data(items);
    std::generate(data.begin(),
                  data.begin() + std::min(items, max_random_size),
                  [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < items; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(items - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline auto
    get_random_data(size_t items, T min, T max, size_t max_random_size = default_max_random_size) ->
    typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
{
    std::random_device                rd;
    std::default_random_engine        gen(rd());
    std::uniform_real_distribution<T> distribution(min, max);
    std::vector<T>                    data(items);
    std::generate(data.begin(),
                  data.begin() + std::min(items, max_random_size),
                  [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < items; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(items - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline std::vector<T>
    get_random_data01(size_t items, float p, size_t max_random_size = default_max_random_size)
{
    std::random_device          rd;
    std::default_random_engine  gen(rd());
    std::bernoulli_distribution distribution(p);
    std::vector<T>              data(items);
    std::generate(data.begin(),
                  data.begin() + std::min(items, max_random_size),
                  [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < items; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(items - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline T get_random_value(T min, T max)
{
    return get_random_data(1, min, max)[0];
}

// Can't use std::prefix_sum for inclusive/exclusive scan, because
// it does not handle short[] -> int(int a, int b) { a + b; } -> int[]
// they way we expect. That's because sum in std::prefix_sum's implementation
// is of type typename std::iterator_traits<InputIt>::value_type (short)
template<class InputIt, class OutputIt, class BinaryOperation>
OutputIt host_inclusive_scan(InputIt first, InputIt last, OutputIt d_first, BinaryOperation op)
{
    using input_type  = typename std::iterator_traits<InputIt>::value_type;
    using output_type = typename std::iterator_traits<OutputIt>::value_type;
    using result_type =
        typename std::conditional<std::is_void<output_type>::value, input_type, output_type>::type;

    if(first == last)
        return d_first;

    result_type sum = *first;
    *d_first        = sum;

    while(++first != last)
    {
        sum        = op(sum, static_cast<result_type>(*first));
        *++d_first = sum;
    }
    return ++d_first;
}

template<class InputIt, class T, class OutputIt, class BinaryOperation>
OutputIt host_exclusive_scan(
    InputIt first, InputIt last, T initial_value, OutputIt d_first, BinaryOperation op)
{
    using input_type  = typename std::iterator_traits<InputIt>::value_type;
    using output_type = typename std::iterator_traits<OutputIt>::value_type;
    using result_type =
        typename std::conditional<std::is_void<output_type>::value, input_type, output_type>::type;

    if(first == last)
        return d_first;

    result_type sum = initial_value;
    *d_first        = initial_value;

    while((first + 1) != last)
    {
        sum        = op(sum, static_cast<result_type>(*first));
        *++d_first = sum;
        first++;
    }
    return ++d_first;
}

template<class InputIt,
         class KeyIt,
         class T,
         class OutputIt,
         class BinaryOperation,
         class KeyCompare>
OutputIt host_exclusive_scan_by_key(InputIt         first,
                                    InputIt         last,
                                    KeyIt           k_first,
                                    T               initial_value,
                                    OutputIt        d_first,
                                    BinaryOperation op,
                                    KeyCompare      key_compare_op)
{
    using input_type  = typename std::iterator_traits<InputIt>::value_type;
    using output_type = typename std::iterator_traits<OutputIt>::value_type;
    using result_type =
        typename std::conditional<std::is_void<output_type>::value, input_type, output_type>::type;

    if(first == last)
        return d_first;

    result_type sum = initial_value;
    *d_first        = initial_value;

    while((first + 1) != last)
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

template<class T, class U = T>
struct custom_type
{
    using first_type  = T;
    using second_type = U;

    T x;
    U y;

    HIPCUB_HOST_DEVICE inline constexpr custom_type() : x(T()), y(U()) {}

    HIPCUB_HOST_DEVICE inline constexpr custom_type(T xx, U yy) : x(xx), y(yy) {}

    HIPCUB_HOST_DEVICE inline constexpr custom_type(T xy) : x(xy), y(xy) {}

    template<class V, class W = V>
    HIPCUB_HOST_DEVICE inline custom_type(const custom_type<V, W>& other) : x(other.x), y(other.y)
    {}

#ifndef HIPCUB_CUB_API
    HIPCUB_HOST_DEVICE inline ~custom_type() = default;
#endif

    HIPCUB_HOST_DEVICE
    inline custom_type& operator=(const custom_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    HIPCUB_HOST_DEVICE
    inline custom_type operator+(const custom_type& rhs) const
    {
        return custom_type(x + rhs.x, y + rhs.y);
    }

    HIPCUB_HOST_DEVICE
    inline custom_type operator-(const custom_type& other) const
    {
        return custom_type(x - other.x, y - other.y);
    }

    HIPCUB_HOST_DEVICE
    inline bool operator<(const custom_type& rhs) const
    {
        // intentionally suboptimal choice for short-circuting,
        // required to generate more performant device code
        return ((x == rhs.x && y < rhs.y) || x < rhs.x);
    }

    HIPCUB_HOST_DEVICE
    inline bool operator>(const custom_type& other) const
    {
        return (x > other.x || (x == other.x && y > other.y));
    }

    HIPCUB_HOST_DEVICE
    inline bool operator==(const custom_type& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }

    HIPCUB_HOST_DEVICE
    inline bool operator!=(const custom_type& other) const
    {
        return !(*this == other);
    }

    HIPCUB_HOST_DEVICE
    custom_type& operator+=(const custom_type& rhs)
    {
        this->x += rhs.x;
        this->y += rhs.y;
        return *this;
    }
};

inline size_t validation_size(size_t items)
{
    return std::min(items, size_t{1024});
}

template<class T>
T read_output(const T* d_output)
{
    T output{};
    HIP_CHECK(hipMemcpy(&output, d_output, sizeof(T), hipMemcpyDeviceToHost));
    return output;
}

template<class T>
std::vector<T> read_output(const T* d_output, size_t count)
{
    std::vector<T> output(count);
    if(count != 0)
    {
        HIP_CHECK(hipMemcpy(output.data(), d_output, count * sizeof(T), hipMemcpyDeviceToHost));
    }
    return output;
}

template<class T, class Predicate>
size_t get_selected_prefix(const std::vector<T>& input, std::vector<T>& expected, Predicate select)
{
    const auto count    = validation_size(input.size());
    size_t     selected = 0;
    for(size_t i = 0; i < input.size(); ++i)
    {
        if(select(i))
        {
            if(expected.size() < count)
                expected.push_back(input[i]);
            ++selected;
        }
    }
    return selected;
}

template<class T>
void check_value(const T& output, const T& expected)
{
    if constexpr(std::is_integral_v<T>)
    {
        PRIMBENCH_ASSERT(output, expected);
    }
    else
    {
        PRIMBENCH_ASSERT(output == expected, true);
    }
}

template<class T>
void check_output(const std::vector<T>& output, const std::vector<T>& expected)
{
    if constexpr(std::is_integral_v<T>)
    {
        PRIMBENCH_ASSERT(output, expected);
    }
    else
    {
        if(output.size() != expected.size())
            PRIMBENCH_ASSERT(output.size(), expected.size());
        const auto mismatch
            = std::mismatch(output.begin(), output.end(), expected.begin(), expected.end());
        PRIMBENCH_ASSERT(static_cast<size_t>(mismatch.first - output.begin()), expected.size());
    }
}

template<class T>
void check_output(const T* d_output, const std::vector<T>& expected)
{
    check_output(read_output(d_output, expected.size()), expected);
}

template<class T>
T add(T a, T b)
{
    if constexpr(std::is_integral_v<T>)
    {
        using unsigned_type = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<unsigned_type>(a) + static_cast<unsigned_type>(b));
    }
    else
    {
        return a + b;
    }
}

template<class T>
void check_near(const T& output, const T& expected)
{
    if constexpr(std::is_floating_point_v<T>)
    {
        const auto tolerance = std::abs(static_cast<double>(expected)) * 1e-4;
        if(!(std::abs(static_cast<double>(output) - expected) <= tolerance))
        {
            PRIMBENCH_ASSERT(std::isfinite(output), true);
            PRIMBENCH_ASSERT(output, expected, tolerance);
        }
    }
    else if constexpr(std::is_integral_v<T>)
    {
        check_value(output, expected);
    }
    else
    {
        check_near(output.x, expected.x);
        check_near(output.y, expected.y);
    }
}

template<class T>
void check_near_output(const T* d_output, const std::vector<T>& expected)
{
    const auto output = read_output(d_output, expected.size());
    if constexpr(std::is_integral_v<T>)
    {
        PRIMBENCH_ASSERT(output, expected);
    }
    else
    {
        for(size_t i = 0; i < expected.size(); ++i)
        {
            check_near(output[i], expected[i]);
        }
    }
}

template<class Iterator>
auto reference_sum(Iterator first, Iterator last)
{
    using T = typename std::iterator_traits<Iterator>::value_type;
    if constexpr(std::is_floating_point_v<T>)
    {
        return static_cast<T>(std::accumulate(first, last, 0.0));
    }
    else
    {
        return std::accumulate(first, last, T{}, add<T>);
    }
}

template<class Key, class Compare>
std::vector<Key>
    get_sorted_prefix(const std::vector<Key>& input, size_t begin, size_t end, Compare compare)
{
    std::vector<Key> expected(validation_size(end - begin));
    std::partial_sort_copy(input.begin() + begin,
                           input.begin() + end,
                           expected.begin(),
                           expected.end(),
                           compare);
    return expected;
}

template<class Key, class Compare>
void check_sorted_keys(
    const Key* d_output, const std::vector<Key>& input, size_t begin, size_t end, Compare compare)
{
    check_output(d_output + begin, get_sorted_prefix(input, begin, end, compare));
}

template<class Key, class Value, class Compare>
void check_sorted_pairs(const Key*                d_keys,
                        const Value*              d_values,
                        const std::vector<Key>&   keys,
                        const std::vector<Value>& values,
                        size_t                    begin,
                        size_t                    end,
                        Compare                   compare)
{
    const auto          count = validation_size(end - begin);
    std::vector<size_t> indices(count);
    std::iota(indices.begin(), indices.end(), begin);
    const auto index_compare = [&](size_t a, size_t b)
    {
        if(compare(keys[a], keys[b]))
            return true;
        if(compare(keys[b], keys[a]))
            return false;
        return a < b;
    };

    std::make_heap(indices.begin(), indices.end(), index_compare);
    for(size_t i = begin + count; i < end; ++i)
    {
        if(index_compare(i, indices.front()))
        {
            std::pop_heap(indices.begin(), indices.end(), index_compare);
            indices.back() = i;
            std::push_heap(indices.begin(), indices.end(), index_compare);
        }
    }
    std::sort_heap(indices.begin(), indices.end(), index_compare);

    std::vector<Key>   expected_keys;
    std::vector<Value> expected_values;
    expected_keys.reserve(count);
    expected_values.reserve(count);
    for(const auto i : indices)
    {
        expected_keys.push_back(keys[i]);
        expected_values.push_back(values[i]);
    }
    check_output(d_keys + begin, expected_keys);
    check_output(d_values + begin, expected_values);
}

template<class Key, class Value, class Compare>
void check_unstable_sorted_pairs(const Key*                d_keys,
                                 const Value*              d_values,
                                 const std::vector<Key>&   keys,
                                 const std::vector<Value>& values,
                                 size_t                    begin,
                                 size_t                    end,
                                 Compare                   compare)
{
    const auto expected_keys = get_sorted_prefix(keys, begin, end, compare);
    const auto count         = expected_keys.size();
    const auto output_keys   = read_output(d_keys + begin, count);
    const auto output_values = read_output(d_values + begin, count);
    check_output(output_keys, expected_keys);

    const auto valid_values
        = std::all_of(output_values.begin(),
                      output_values.end(),
                      [](const Value& value) { return std::equal_to<Value>{}(value, value); });
    PRIMBENCH_ASSERT(valid_values, true);

    std::multiset<std::pair<Key, Value>> remaining;
    for(size_t i = 0; i < count; ++i)
        remaining.emplace(output_keys[i], output_values[i]);

    for(size_t i = begin; i < end && !remaining.empty(); ++i)
    {
        if(compare(output_keys.back(), keys[i]))
            continue;
        const auto match = remaining.find({keys[i], values[i]});
        if(match != remaining.end())
            remaining.erase(match);
    }
    PRIMBENCH_ASSERT(remaining.empty(), true);
}

template<typename>
struct is_custom_type : std::false_type
{};

template<class T, class U>
struct is_custom_type<custom_type<T, U>> : std::true_type
{};

template<class CustomType>
struct custom_type_decomposer
{
    static_assert(is_custom_type<CustomType>::value,
                  "custom_type_decomposer can only be used with instantiations "
                  "of custom_type");

    using T = typename CustomType::first_type;
    using U = typename CustomType::second_type;

    HIPCUB_HOST_DEVICE
    ::hipcub::tuple<T&, U&> operator()(CustomType& key) const
    {
        return ::hipcub::tuple<T&, U&>{key.x, key.y};
    }
};

template<class T, class enable = void>
struct generate_limits;

template<class T>
struct generate_limits<T, std::enable_if_t<std::is_integral<T>::value>>
{
    static inline T min()
    {
        return std::numeric_limits<T>::min();
    }
    static inline T max()
    {
        return std::numeric_limits<T>::max();
    }
};

template<class T>
struct generate_limits<T, std::enable_if_t<is_custom_type<T>::value>>
{
    using F = typename T::first_type;
    using S = typename T::second_type;
    static inline T min()
    {
        return T(generate_limits<F>::min(), generate_limits<S>::min());
    }
    static inline T max()
    {
        return T(generate_limits<F>::max(), generate_limits<S>::max());
    }
};

template<class T>
struct generate_limits<T, std::enable_if_t<std::is_floating_point<T>::value>>
{
    static inline T min()
    {
        return T(-1000);
    }
    static inline T max()
    {
        return T(1000);
    }
};

template<class T>
inline auto get_random_data(size_t items, T min, T max, size_t max_random_size = 1024 * 1024) ->
    typename std::enable_if<is_custom_type<T>::value, std::vector<T>>::type
{
    using first_type  = typename T::first_type;
    using second_type = typename T::second_type;
    std::vector<T> data(items);
    auto           fdata = get_random_data<first_type>(items, min.x, max.x, max_random_size);
    auto           sdata = get_random_data<second_type>(items, min.y, max.y, max_random_size);
    for(size_t i = 0; i < items; i++)
    {
        data[i] = T(fdata[i], sdata[i]);
    }
    return data;
}

template<class T>
inline auto get_random_data(size_t items, T min, T max, size_t max_random_size = 1024 * 1024) ->
    typename std::enable_if<!is_custom_type<T>::value
                                && !std::is_same<decltype(max.x), void>::value,
                            std::vector<T>>::type
{

    using field_type = decltype(max.x);
    std::vector<T> data(items);
    auto           field_data = get_random_data<field_type>(items, min.x, max.x, max_random_size);
    for(size_t i = 0; i < items; i++)
    {
        data[i] = T(field_data[i]);
    }
    return data;
}

template<typename T>
std::vector<T>
    get_random_segments(const size_t items, const size_t max_segment_length, const int seed_value)
{
    static_assert(std::is_arithmetic<T>::value, "Key type must be arithmetic");

    std::default_random_engine            prng(seed_value);
    std::uniform_int_distribution<size_t> segment_length_distribution(max_segment_length);
    using key_distribution_type = std::conditional_t<std::is_integral<T>::value,
                                                     std::uniform_int_distribution<T>,
                                                     std::uniform_real_distribution<T>>;
    key_distribution_type key_distribution(std::numeric_limits<T>::max());
    std::vector<T>        keys(items);

    size_t keys_start_index = 0;
    while(keys_start_index < items)
    {
        const size_t new_segment_length = segment_length_distribution(prng);
        const size_t new_segment_end    = std::min(items, keys_start_index + new_segment_length);
        const T      key                = key_distribution(prng);
        std::fill(std::next(keys.begin(), keys_start_index),
                  std::next(keys.begin(), new_segment_end),
                  key);
        keys_start_index += new_segment_length;
    }
    return keys;
}

bool is_warp_size_supported(const unsigned required_warp_size)
{
    return HIPCUB_HOST_WARP_THREADS >= required_warp_size;
}

template<unsigned int LogicalWarpSize>
__device__ constexpr bool device_test_enabled_for_warp_size_v
    = HIPCUB_DEVICE_WARP_THREADS >= LogicalWarpSize;

template<class T>
__device__
inline constexpr bool is_power_of_two(const T x)
{
    static_assert(std::is_integral<T>::value, "T must be integer type");
    return (x > 0) && ((x & (x - 1)) == 0);
}

template<typename Iterator>
using it_value_t = typename std::iterator_traits<Iterator>::value_type;

using engine_type = std::default_random_engine;

// generate_random_data_n() generates only part of sequence and replicates it,
// because benchmarks usually do not need "true" random sequence.
template<class OutputIter, class U, class V, class Generator>
inline auto generate_random_data_n(
    OutputIter it, size_t items, U min, V max, Generator& gen, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if_t<std::is_integral<it_value_t<OutputIter>>::value, OutputIter>
{
    using T = it_value_t<OutputIter>;

    using dis_type = typename std::conditional<(sizeof(T) == 1), short, T>::type;
    std::uniform_int_distribution<dis_type> distribution((T)min, (T)max);
    std::generate_n(it, std::min(items, max_random_size), [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < items; i += max_random_size)
    {
        std::copy_n(it, std::min(items - i, max_random_size), it + i);
    }
    return it + items;
}

template<class OutputIterator, class U, class V, class Generator>
inline auto generate_random_data_n(OutputIterator it,
                                   size_t         items,
                                   U              min,
                                   V              max,
                                   Generator&     gen,
                                   size_t         max_random_size = 1024 * 1024)
    -> std::enable_if_t<std::is_floating_point<it_value_t<OutputIterator>>::value, OutputIterator>
{
    using T = typename std::iterator_traits<OutputIterator>::value_type;

    std::uniform_real_distribution<T> distribution((T)min, (T)max);
    std::generate_n(it, std::min(items, max_random_size), [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < items; i += max_random_size)
    {
        std::copy_n(it, std::min(items - i, max_random_size), it + i);
    }
    return it + items;
}

template<std::size_t Size, std::size_t Alignment>
struct alignas(Alignment) custom_aligned_type
{
    unsigned char data[Size];
};

template<typename T,
         typename U,
         std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
inline constexpr auto ceiling_div(const T a, const U b)
{
    return a / b + (a % b > 0 ? 1 : 0);
}

} // namespace benchmark_utils

// Need for hipcub::DeviceReduce::Min/Max etc.
namespace std
{
template<class T>
class numeric_limits<benchmark_utils::custom_type<T>>
{
    using value_type = benchmark_utils::custom_type<T>;

public:
    static constexpr value_type min()
    {
        return std::numeric_limits<T>::min();
    }

    static constexpr value_type max()
    {
        return std::numeric_limits<T>::max();
    }

    static constexpr value_type lowest()
    {
        return std::numeric_limits<T>::lowest();
    }
};
} // namespace std

PRIMBENCH_REGISTER_TYPE(int8_t, "i8")
PRIMBENCH_REGISTER_TYPE(int16_t, "i16")
PRIMBENCH_REGISTER_TYPE(int32_t, "i32")
PRIMBENCH_REGISTER_TYPE(int64_t, "i64")
PRIMBENCH_REGISTER_TYPE(uint8_t, "u8")
PRIMBENCH_REGISTER_TYPE(uint16_t, "u16")
PRIMBENCH_REGISTER_TYPE(uint32_t, "u32")
PRIMBENCH_REGISTER_TYPE(unsigned long long, "u64")
PRIMBENCH_REGISTER_TYPE(float, "f32")
PRIMBENCH_REGISTER_TYPE(double, "f64")
PRIMBENCH_REGISTER_TYPE(__half, "f16")

using custom_int_t       = benchmark_utils::custom_type<int>;
using custom_float2      = benchmark_utils::custom_type<float, float>;
using custom_double2     = benchmark_utils::custom_type<double, double>;
using custom_char_double = benchmark_utils::custom_type<char, double>;
using custom_double_char = benchmark_utils::custom_type<double, char>;
using custom_int_double  = benchmark_utils::custom_type<int, double>;

PRIMBENCH_REGISTER_TYPE(custom_int_t, "custom<i32>");
PRIMBENCH_REGISTER_TYPE(custom_float2, "custom<f32,f32>")
PRIMBENCH_REGISTER_TYPE(custom_double2, "custom<f64,f64>")
PRIMBENCH_REGISTER_TYPE(custom_char_double, "custom<i8,f64>")
PRIMBENCH_REGISTER_TYPE(custom_double_char, "custom<f64,i8>")
PRIMBENCH_REGISTER_TYPE(custom_int_double, "custom<i32,f64>")
