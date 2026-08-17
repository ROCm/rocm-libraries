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

#ifndef HIPCUB_BENCHMARK_UTILS_HPP_
#define HIPCUB_BENCHMARK_UTILS_HPP_

#ifndef BENCHMARK_UTILS_INCLUDE_GUARD
    #error benchmark_utils.hpp must ONLY be included by common_benchmark_header.hpp. Please include common_benchmark_header.hpp instead.
#endif

// hipCUB API
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

namespace benchmark_utils
{
const size_t default_max_random_size = 1024 * 1024;
// get_random_data() generates only part of sequence and replicates it,
// because benchmarks usually do not need "true" random sequence.
template<class T>
inline auto
    get_random_data(size_t size, T min, T max, size_t max_random_size = default_max_random_size) ->
    typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
{
    std::random_device         rd;
    std::default_random_engine gen(rd());
    using distribution_type = typename std::conditional<(sizeof(T) == 1), short, T>::type;
    std::uniform_int_distribution<distribution_type> distribution(min, max);
    std::vector<T>                                   data(size);
    std::generate(data.begin(),
                  data.begin() + std::min(size, max_random_size),
                  [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline auto
    get_random_data(size_t size, T min, T max, size_t max_random_size = default_max_random_size) ->
    typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
{
    std::random_device                rd;
    std::default_random_engine        gen(rd());
    std::uniform_real_distribution<T> distribution(min, max);
    std::vector<T>                    data(size);
    std::generate(data.begin(),
                  data.begin() + std::min(size, max_random_size),
                  [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline std::vector<T>
    get_random_data01(size_t size, float p, size_t max_random_size = default_max_random_size)
{
    std::random_device          rd;
    std::default_random_engine  gen(rd());
    std::bernoulli_distribution distribution(p);
    std::vector<T>              data(size);
    std::generate(data.begin(),
                  data.begin() + std::min(size, max_random_size),
                  [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
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
        } else
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

    HIPCUB_HOST_DEVICE inline custom_type& operator=(const custom_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    HIPCUB_HOST_DEVICE inline custom_type operator+(const custom_type& rhs) const
    {
        return custom_type(x + rhs.x, y + rhs.y);
    }

    HIPCUB_HOST_DEVICE inline custom_type operator-(const custom_type& other) const
    {
        return custom_type(x - other.x, y - other.y);
    }

    HIPCUB_HOST_DEVICE inline bool operator<(const custom_type& rhs) const
    {
        // intentionally suboptimal choice for short-circuting,
        // required to generate more performant device code
        return ((x == rhs.x && y < rhs.y) || x < rhs.x);
    }

    HIPCUB_HOST_DEVICE inline bool operator>(const custom_type& other) const
    {
        return (x > other.x || (x == other.x && y > other.y));
    }

    HIPCUB_HOST_DEVICE inline bool operator==(const custom_type& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }

    HIPCUB_HOST_DEVICE inline bool operator!=(const custom_type& other) const
    {
        return !(*this == other);
    }

    HIPCUB_HOST_DEVICE custom_type& operator+=(const custom_type& rhs)
    {
        this->x += rhs.x;
        this->y += rhs.y;
        return *this;
    }
};

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

    HIPCUB_HOST_DEVICE ::hipcub::tuple<T&, U&> operator()(CustomType& key) const
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
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024) ->
    typename std::enable_if<is_custom_type<T>::value, std::vector<T>>::type
{
    using first_type  = typename T::first_type;
    using second_type = typename T::second_type;
    std::vector<T> data(size);
    auto           fdata = get_random_data<first_type>(size, min.x, max.x, max_random_size);
    auto           sdata = get_random_data<second_type>(size, min.y, max.y, max_random_size);
    for(size_t i = 0; i < size; i++)
    {
        data[i] = T(fdata[i], sdata[i]);
    }
    return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024) ->
    typename std::enable_if<!is_custom_type<T>::value
                                && !std::is_same<decltype(max.x), void>::value,
                            std::vector<T>>::type
{

    using field_type = decltype(max.x);
    std::vector<T> data(size);
    auto           field_data = get_random_data<field_type>(size, min.x, max.x, max_random_size);
    for(size_t i = 0; i < size; i++)
    {
        data[i] = T(field_data[i]);
    }
    return data;
}

template<typename T>
std::vector<T>
    get_random_segments(const size_t size, const size_t max_segment_length, const int seed_value)
{
    static_assert(std::is_arithmetic<T>::value, "Key type must be arithmetic");

    std::default_random_engine            prng(seed_value);
    std::uniform_int_distribution<size_t> segment_length_distribution(max_segment_length);
    using key_distribution_type = std::conditional_t<std::is_integral<T>::value,
                                                     std::uniform_int_distribution<T>,
                                                     std::uniform_real_distribution<T>>;
    key_distribution_type key_distribution(std::numeric_limits<T>::max());
    std::vector<T>        keys(size);

    size_t keys_start_index = 0;
    while(keys_start_index < size)
    {
        const size_t new_segment_length = segment_length_distribution(prng);
        const size_t new_segment_end    = std::min(size, keys_start_index + new_segment_length);
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
    OutputIter it, size_t size, U min, V max, Generator& gen, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if_t<std::is_integral<it_value_t<OutputIter>>::value, OutputIter>
{
    using T = it_value_t<OutputIter>;

    using dis_type = typename std::conditional<(sizeof(T) == 1), short, T>::type;
    std::uniform_int_distribution<dis_type> distribution((T)min, (T)max);
    std::generate_n(it, std::min(size, max_random_size), [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(it, std::min(size - i, max_random_size), it + i);
    }
    return it + size;
}

template<class OutputIterator, class U, class V, class Generator>
inline auto generate_random_data_n(OutputIterator it,
                                   size_t         size,
                                   U              min,
                                   V              max,
                                   Generator&     gen,
                                   size_t         max_random_size = 1024 * 1024)
    -> std::enable_if_t<std::is_floating_point<it_value_t<OutputIterator>>::value, OutputIterator>
{
    using T = typename std::iterator_traits<OutputIterator>::value_type;

    std::uniform_real_distribution<T> distribution((T)min, (T)max);
    std::generate_n(it, std::min(size, max_random_size), [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(it, std::min(size - i, max_random_size), it + i);
    }
    return it + size;
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
template<>
class numeric_limits<benchmark_utils::custom_type<int>>
{
    using T = typename benchmark_utils::custom_type<int>;

public:
    static constexpr inline T min()
    {
        return std::numeric_limits<typename T::first_type>::min();
    }

    static constexpr inline T max()
    {
        return std::numeric_limits<typename T::first_type>::max();
    }

    static constexpr inline T lowest()
    {
        return std::numeric_limits<typename T::first_type>::lowest();
    }
};

template<>
class numeric_limits<benchmark_utils::custom_type<float>>
{
    using T = typename benchmark_utils::custom_type<float>;

public:
    static constexpr inline T min()
    {
        return std::numeric_limits<typename T::first_type>::min();
    }

    static constexpr inline T max()
    {
        return std::numeric_limits<typename T::first_type>::max();
    }

    static constexpr inline T lowest()
    {
        return std::numeric_limits<typename T::first_type>::lowest();
    }
};
} // namespace std

#endif // HIPCUB_BENCHMARK_UTILS_HPP_
