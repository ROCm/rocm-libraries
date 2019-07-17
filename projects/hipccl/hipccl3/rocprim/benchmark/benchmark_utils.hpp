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

#ifndef ROCPRIM_BENCHMARK_UTILS_HPP_
#define ROCPRIM_BENCHMARK_UTILS_HPP_

#include <algorithm>
#include <vector>
#include <random>
#include <type_traits>

#include <rocprim/rocprim.hpp>

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

// get_random_data() generates only part of sequence and replicates it,
// because benchmarks usually do not need "true" random sequence.
template<class T>
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<rocprim::is_integral<T>::value, std::vector<T>>::type
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
    -> typename std::enable_if<rocprim::is_floating_point<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    using dis_type = typename std::conditional<std::is_same<rocprim::half, T>::value, float, T>::type;
    std::uniform_real_distribution<dis_type> distribution(min, max);
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

template<class T, class U = T>
struct custom_type
{
    using first_type = T;
    using second_type = U;

    T x;
    U y;

    ROCPRIM_HOST_DEVICE inline
    custom_type(T xx = 0, U yy = 0) : x(xx), y(yy)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~custom_type() = default;

    ROCPRIM_HOST_DEVICE inline
    custom_type operator+(const custom_type& rhs) const
    {
        return custom_type(x + rhs.x, y + rhs.y);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(const custom_type& rhs) const
    {
        return (x < rhs.x || (x == rhs.x && y < rhs.y));
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(const custom_type& rhs) const
    {
        return x == rhs.x && y == rhs.y;
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

#endif // ROCPRIM_BENCHMARK_UTILS_HPP_
