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

#ifndef ROCPRIM_TEST_TEST_UTILS_HPP_
#define ROCPRIM_TEST_TEST_UTILS_HPP_

#include <algorithm>
#include <vector>
#include <random>
#include <type_traits>

#include <rocprim.hpp>

#ifdef ROCPRIM_HC_API
size_t get_max_tile_size(hc::accelerator acc = hc::accelerator())
{
    return acc.get_max_tile_static_size();
}
#endif

#if defined(ROCPRIM_HC_API) || defined(ROCPRIM_HIP_API)
// Custom type used in tests
template<class T>
struct custom_test_type
{
    T x;
    T y;

    ROCPRIM_HOST_DEVICE
    custom_test_type(T xx = 0, T yy = 0) : x(xx), y(yy) {}

    ROCPRIM_HOST_DEVICE
    ~custom_test_type() {}

    ROCPRIM_HOST_DEVICE
    custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(x + other.x, y + other.y);
    }

    ROCPRIM_HOST_DEVICE
    bool operator==(const custom_test_type& other) const
    {
        return (x == other.x && y == other.y);
    }
};
#endif

template<class T>
inline auto get_random_data(size_t size, T min, T max)
    -> typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
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
    -> typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<T> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
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
    if (first == last) return d_first;

    typename std::iterator_traits<OutputIt>::value_type sum = *first;
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
    if (first == last) return d_first;

    typename std::iterator_traits<OutputIt>::value_type sum = initial_value;
    *d_first = initial_value;

    while ((first+1) != last)
    {
       sum = op(sum, *first);
       *++d_first = sum;
       first++;
    }
    return ++d_first;
}

#endif // ROCPRIM_TEST_TEST_UTILS_HPP_
