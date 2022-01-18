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

#include "test_utils_half.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_sort_comparator.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_assertions.hpp"

// Seed values
#include "test_seed.hpp"

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

template<class InputIt, class KeyIt, class OutputIt, class BinaryOperation, class KeyCompare>
OutputIt host_inclusive_scan_by_key(InputIt first, InputIt last, KeyIt k_first,
                                    OutputIt d_first, BinaryOperation op, KeyCompare key_compare_op)
{
    using input_type = typename std::iterator_traits<InputIt>::value_type;
    using output_type = typename std::iterator_traits<OutputIt>::value_type;
    using result_type =
        typename std::conditional<
            std::is_void<output_type>::value, input_type, output_type
        >::type;

    if (first == last)
    {
        return d_first;
    }

    result_type sum = *first;
    *d_first = sum;

    while (++first != last)
    {
        if (key_compare_op(*k_first, *++k_first))
        {
            sum = op(sum, static_cast<result_type>(*first));
        }
        else
        {
            sum = *first;
        }
        *++d_first = sum;
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

#define SKIP_IF_UNSUPPORTED_WARP_SIZE(test_warp_size) { \
    const auto host_warp_size = HIPCUB_HOST_WARP_THREADS; \
    if (host_warp_size < (test_warp_size)) \
    { \
        GTEST_SKIP() << "Cannot run test of warp size " \
            << (test_warp_size) \
            << " on a device with warp size " \
            << host_warp_size; \
    } \
}

template<unsigned LogicalWarpSize>
struct DeviceSelectWarpSize
{
    static constexpr unsigned value = HIPCUB_DEVICE_WARP_THREADS >= LogicalWarpSize
        ? LogicalWarpSize
        : HIPCUB_DEVICE_WARP_THREADS;
};

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
}

#endif // HIPCUB_TEST_HIPCUB_TEST_UTILS_HPP_
