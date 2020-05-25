// MIT License
//
// Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef BENCHMARK_UTILS_INCLUDE_GAURD
    #error benchmark_utils.hpp must ONLY be included by common_benchmark_header.hpp. Please include common_benchmark_header.hpp instead.
#endif

// hipCUB API
#ifdef __HIP_PLATFORM_HCC__
    #include "hipcub/backend/rocprim/util_ptx.hpp"
#elif defined(__HIP_PLATFORM_NVCC__)
    #include "hipcub/config.hpp"
    #include <cub/util_ptx.cuh>
#endif

#ifndef HIPCUB_CUB_API
#define HIPCUB_WARP_THREADS_MACRO warpSize
#else
#define HIPCUB_WARP_THREADS_MACRO HIPCUB_WARP_THREADS
#endif

namespace benchmark_utils
{

// get_random_data() generates only part of sequence and replicates it,
// because benchmarks usually do not need "true" random sequence.
template<class T>
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<T> distribution(min, max);
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
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<T> distribution(min, max);
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
inline std::vector<T> get_random_data01(size_t size, float p, size_t max_random_size = 1024 * 1024)
{
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
inline T get_random_value(T min, T max)
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

template<class T, class U = T>
struct custom_type
{
    using first_type = T;
    using second_type = U;

    T x;
    U y;

    HIPCUB_HOST_DEVICE inline
    constexpr custom_type() {}

    HIPCUB_HOST_DEVICE inline
    constexpr custom_type(T xx, U yy) : x(xx), y(yy)
    {
    }

    HIPCUB_HOST_DEVICE inline
    constexpr custom_type(T xy) : x(xy), y(xy)
    {
    }

    template<class V, class W = V>
    HIPCUB_HOST_DEVICE inline
    custom_type(const custom_type<V,W>& other)
    {
        x = other.x;
        y = other.y;
    }

    #ifndef HIPCUB_CUB_API
        HIPCUB_HOST_DEVICE inline
        ~custom_type() = default;
    #endif

    HIPCUB_HOST_DEVICE inline
    custom_type& operator=(const custom_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    HIPCUB_HOST_DEVICE inline
    custom_type operator+(const custom_type& rhs) const
    {
        return custom_type(x + rhs.x, y + rhs.y);
    }

    HIPCUB_HOST_DEVICE inline
    custom_type operator-(const custom_type& other) const
    {
        return custom_type(x - other.x, y - other.y);
    }

    HIPCUB_HOST_DEVICE inline
    bool operator<(const custom_type& rhs) const
    {
        return (x < rhs.x || (x == rhs.x && y < rhs.y));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator>(const custom_type& other) const
    {
        return (x > other.x || (x == other.x && y > other.y));
    }

    HIPCUB_HOST_DEVICE inline
    bool operator==(const custom_type& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }

    HIPCUB_HOST_DEVICE inline
    bool operator!=(const custom_type& other) const
    {
       return !(*this == other);
    }


};

template<typename>
struct is_custom_type : std::false_type {};

template<class T, class U>
struct is_custom_type<custom_type<T,U>> : std::true_type {};

template<class T>
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<is_custom_type<T>::value, std::vector<T>>::type
{
    using first_type = typename T::first_type;
    using second_type = typename T::second_type;
    std::vector<T> data(size);
    auto fdata = get_random_data<first_type>(size, min.x, max.x, max_random_size);
    auto sdata = get_random_data<second_type>(size, min.y, max.y, max_random_size);
    for(size_t i = 0; i < size; i++)
    {
        data[i] = T(fdata[i], sdata[i]);
    }
    return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<!is_custom_type<T>::value && !std::is_same<decltype(max.x), void>::value, std::vector<T>>::type
{
    // NOTE 1: post-increment operator required, because HIP has different typedefs for vector field types
    //         when using HCC or HIP-Clang. Using HIP-Clang members are accessed as fields of a struct via
    //         a union, but in HCC mode they are proxy types (Scalar_accessor). STL algorithms don't
    //         always tolerate proxies. Unfortunately, Scalar_accessor doesn't have any member typedefs to
    //         conveniently obtain the inner stored type. All operations on it (operator+, operator+=,
    //         CTOR, etc.) return a reference to an accessor, it is only the post-increment operator that
    //         returns a copy of the stored type, hence we take the decltype of that.
    //
    // NOTE 2: decltype() is unevaluated context. We don't really modify max, just compute the type of the
    //         expression if we were to actually call it.
    using field_type = decltype(max.x++);
    std::vector<T> data(size);
    auto field_data = get_random_data<field_type>(size, min.x, max.x, max_random_size);
    for(size_t i = 0; i < size; i++)
    {
        data[i] = T(field_data[i]);
    }
    return data;
}

} // end benchmark_util namespace

// Need for hipcub::DeviceReduce::Min/Max etc.
namespace std
{
    template<>
    class numeric_limits<benchmark_utils::custom_type<int>>
    {
        using T = typename benchmark_utils::custom_type<int>;

        public:

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

        static constexpr inline T max()
        {
            return std::numeric_limits<typename T::first_type>::max();
        }

        static constexpr inline T lowest()
        {
            return std::numeric_limits<typename T::first_type>::lowest();
        }
    };
}

#endif // HIPCUB_BENCHMARK_UTILS_HPP_
