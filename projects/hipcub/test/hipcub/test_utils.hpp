// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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
    #include <hipcub/backend/rocprim/util_ptx.hpp>
#elif defined(__HIP_PLATFORM_NVIDIA__)
    #include <cub/util_ptx.cuh>
    #include <hipcub/config.hpp>
#endif

#include "test_utils_assertions.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_functional.hpp"
#include "test_utils_half.hpp"
#include "test_utils_hipgraphs.hpp"
#include "test_utils_sort_comparator.hpp"

// Seed values
#include "test_seed.hpp"

#include <type_traits>

namespace test_utils
{

// Values of relative error for non-assotiative operations
// (+, -, *) and type conversions for floats
// They are doubled from 1 / (1 << mantissa_bits) as we compare in tests
// the results of _two_ sequences of operations with different order
// For all other operations (i.e. integer arithmetics) default 0 is used
template<typename T>
struct precision
{
    static constexpr float value = 0;
};

template<>
struct precision<double>
{
    static constexpr float value = 2.0f / static_cast<float>(1ll << 52);
};

template<>
struct precision<float>
{
    static constexpr float value = 2.0f / static_cast<float>(1ll << 23);
};

template<>
struct precision<test_utils::half>
{
    static constexpr float value = 2.0f / static_cast<float>(1ll << 10);
};

template<>
struct precision<test_utils::bfloat16>
{
    static constexpr float value = 2.0f / static_cast<float>(1ll << 7);
};

template<typename T>
struct precision<const T>
{
    static constexpr float value = precision<T>::value;
};

template<typename T>
struct precision<custom_test_type<T>>
{
    static constexpr float value = precision<T>::value;
};

template<class T>
struct is_add_operator : std::false_type
{
    using value_type = uint8_t;
};

template<class T>
struct is_add_operator<test_utils::plus(T)> : std::true_type
{
    using value_type = T;
};

template<class T>
struct is_add_operator<test_utils::minus(T)> : std::true_type
{
    using value_type = T;
};

template<class T>
struct is_multiply_operator : std::false_type
{
    using value_type = uint8_t;
};

template<class T>
struct is_multiply_operator<test_utils::multiplies(T)> : std::true_type
{
    using value_type = T;
};

/* Plus to operator selector for host-side
 * On host-side we use `double` as accumulator and `test_utils::plus<double>` as operator
 * for bfloat16 and half types. This is because additions of floating-point types are not
 * associative. This would result in wrong output rather quickly for reductions and scan-algorithms
 * on host-side for bfloat16 and half because of their low-precision.
 */
template<typename T>
struct select_plus_operator_host
{
    using type     = test_utils::plus;
    using acc_type = T;
};

template<bool UseInitialValue = false,
         class InputIt,
         class OutputIt,
         class BinaryOperation,
         class acc_type>
OutputIt host_inclusive_scan_impl(
    InputIt first, InputIt last, OutputIt d_first, BinaryOperation op, acc_type init_value)
{
    acc_type sum = UseInitialValue ? op(init_value, *first) : *first;
    *d_first     = sum;

    if(first == last)
    {
        return d_first;
    }

    while(++first != last)
    {
        sum        = op(sum, *first);
        *++d_first = sum;
    }
    return ++d_first;
}

template<class InputIt, class OutputIt, class BinaryOperation>
OutputIt host_inclusive_scan(InputIt first, InputIt last, OutputIt d_first, BinaryOperation op)
{
    using acc_type = typename std::iterator_traits<InputIt>::value_type;
    return host_inclusive_scan_impl(first, last, d_first, op, acc_type{});
}

template<class InputIt, class OutputIt, class BinaryOperation, class InitType>
OutputIt host_inclusive_scan_init(
    InputIt first, InputIt last, OutputIt d_first, InitType init_value, BinaryOperation op)
{
    return host_inclusive_scan_impl<true>(first, last, d_first, op, init_value);
}

template<class InputIt,
         class OutputIt,
         class T,
         std::enable_if_t<
             std::is_same<typename std::iterator_traits<InputIt>::value_type,
                          test_utils::bfloat16>::value
                 || std::is_same<typename std::iterator_traits<InputIt>::value_type,
                                 test_utils::half>::value
                 || std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
             bool>
         = true>
OutputIt host_inclusive_scan(InputIt first, InputIt last, OutputIt d_first, test_utils::plus)
{
    using acc_type = double;
    return host_inclusive_scan_impl(first, last, d_first, test_utils::plus(), acc_type{});
}

template<class InputIt,
         class OutputIt,
         class InitType,
         class T,
         std::enable_if_t<
             std::is_same<typename std::iterator_traits<InputIt>::value_type,
                          test_utils::bfloat16>::value
                 || std::is_same<typename std::iterator_traits<InputIt>::value_type,
                                 test_utils::half>::value
                 || std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
             bool>
         = true>
OutputIt host_inclusive_scan_init(
    InputIt first, InputIt last, OutputIt d_first, InitType init_value, test_utils::plus)
{
    using acc_type = double;
    return host_inclusive_scan_impl<true>(first,
                                          last,
                                          d_first,
                                          init_value,
                                          test_utils::plus(),
                                          acc_type{});
}

template<class InputIt, class T, class OutputIt, class BinaryOperation, class acc_type>
OutputIt host_exclusive_scan_impl(
    InputIt first, InputIt last, T initial_value, OutputIt d_first, BinaryOperation op, acc_type)
{
    if(first == last)
        return d_first;

    acc_type sum = initial_value;
    *d_first     = initial_value;

    while((first + 1) != last)
    {
        sum        = op(sum, *first);
        *++d_first = sum;
        first++;
    }
    return ++d_first;
}

template<class InputIt, class T, class OutputIt, class BinaryOperation>
OutputIt host_exclusive_scan(
    InputIt first, InputIt last, T initial_value, OutputIt d_first, BinaryOperation op)
{
    using acc_type = typename std::iterator_traits<InputIt>::value_type;
    return host_exclusive_scan_impl(first, last, initial_value, d_first, op, acc_type{});
}

template<class InputIt,
         class T,
         class OutputIt,
         class U,
         std::enable_if_t<
             std::is_same<typename std::iterator_traits<InputIt>::value_type,
                          test_utils::bfloat16>::value
                 || std::is_same<typename std::iterator_traits<InputIt>::value_type,
                                 test_utils::half>::value
                 || std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
             bool>
         = true>
OutputIt host_exclusive_scan(
    InputIt first, InputIt last, T initial_value, OutputIt d_first, test_utils::plus)
{
    using acc_type = double;
    return host_exclusive_scan_impl(first,
                                    last,
                                    initial_value,
                                    d_first,
                                    test_utils::plus(),
                                    acc_type{});
}

template<class InputIt,
         class KeyIt,
         class T,
         class OutputIt,
         class BinaryOperation,
         class KeyCompare,
         class acc_type>
OutputIt host_exclusive_scan_by_key_impl(InputIt         first,
                                         InputIt         last,
                                         KeyIt           k_first,
                                         T               initial_value,
                                         OutputIt        d_first,
                                         BinaryOperation op,
                                         KeyCompare      key_compare_op,
                                         acc_type)
{
    if(first == last)
        return d_first;

    acc_type sum = initial_value;
    *d_first     = initial_value;

    while((first + 1) != last)
    {
        if(key_compare_op(*k_first, *(k_first + 1)))
        {
            sum = op(sum, *first);
        } else
        {
            sum = initial_value;
        }
        k_first++;
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
    using acc_type = typename std::iterator_traits<InputIt>::value_type;
    return host_exclusive_scan_by_key_impl(first,
                                           last,
                                           k_first,
                                           initial_value,
                                           d_first,
                                           op,
                                           key_compare_op,
                                           acc_type{});
}

template<class InputIt,
         class KeyIt,
         class T,
         class OutputIt,
         class U,
         class KeyCompare,
         std::enable_if_t<
             std::is_same<typename std::iterator_traits<InputIt>::value_type,
                          test_utils::bfloat16>::value
                 || std::is_same<typename std::iterator_traits<InputIt>::value_type,
                                 test_utils::half>::value
                 || std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
             bool>
         = true>
OutputIt host_exclusive_scan_by_key(InputIt  first,
                                    InputIt  last,
                                    KeyIt    k_first,
                                    T        initial_value,
                                    OutputIt d_first,
                                    test_utils::plus,
                                    KeyCompare key_compare_op)
{
    using acc_type = double;
    return host_exclusive_scan_by_key_impl(first,
                                           last,
                                           k_first,
                                           initial_value,
                                           d_first,
                                           test_utils::plus(),
                                           key_compare_op,
                                           acc_type{});
}

template<class InputIt,
         class KeyIt,
         class OutputIt,
         class BinaryOperation,
         class KeyCompare,
         class acc_type>
OutputIt host_inclusive_scan_by_key_impl(InputIt         first,
                                         InputIt         last,
                                         KeyIt           k_first,
                                         OutputIt        d_first,
                                         BinaryOperation op,
                                         KeyCompare      key_compare_op,
                                         acc_type)
{
    if(first == last)
        return d_first;

    acc_type sum = *first;
    *d_first     = sum;

    while(++first != last)
    {
        if(key_compare_op(*k_first, *(k_first + 1)))
        {
            sum = op(sum, *first);
        } else
        {
            sum = *first;
        }
        k_first++;
        *++d_first = sum;
    }
    return ++d_first;
}
template<class InputIt, class KeyIt, class OutputIt, class BinaryOperation, class KeyCompare>
OutputIt host_inclusive_scan_by_key(InputIt         first,
                                    InputIt         last,
                                    KeyIt           k_first,
                                    OutputIt        d_first,
                                    BinaryOperation op,
                                    KeyCompare      key_compare_op)
{
    using acc_type = typename std::iterator_traits<InputIt>::value_type;
    return host_inclusive_scan_by_key_impl(first,
                                           last,
                                           k_first,
                                           d_first,
                                           op,
                                           key_compare_op,
                                           acc_type{});
}

template<class InputIt,
         class KeyIt,
         class OutputIt,
         class U,
         class KeyCompare,
         std::enable_if_t<
             std::is_same<typename std::iterator_traits<InputIt>::value_type,
                          test_utils::bfloat16>::value
                 || std::is_same<typename std::iterator_traits<InputIt>::value_type,
                                 test_utils::half>::value
                 || std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
             bool>
         = true>
OutputIt host_inclusive_scan_by_key(InputIt  first,
                                    InputIt  last,
                                    KeyIt    k_first,
                                    OutputIt d_first,
                                    test_utils::plus,
                                    KeyCompare key_compare_op)
{
    using acc_type = double;
    return host_inclusive_scan_by_key_impl(first,
                                           last,
                                           k_first,
                                           d_first,
                                           test_utils::plus(),
                                           key_compare_op,
                                           acc_type{});
}

template<class T, class U = T>
HIPCUB_HOST_DEVICE inline constexpr std::common_type_t<T, U> max(const T& t, const U& u)
{
    using common_type = std::common_type_t<T, U>;
    return static_cast<common_type>(t) < static_cast<common_type>(u) ? u : t;
}

HIPCUB_HOST_DEVICE inline test_utils::half max(const test_utils::half& a, const test_utils::half& b)
{
    return test_utils::half_maximum{}(a, b);
}

template<class T>
HIPCUB_HOST_DEVICE inline constexpr T max(const T& t, const test_utils::half& u)
{
    return test_utils::max(t, static_cast<T>(u));
}

template<class T>
HIPCUB_HOST_DEVICE inline constexpr T max(const test_utils::half& t, const T& u)
{
    return test_utils::max(static_cast<T>(t), u);
}

HIPCUB_HOST_DEVICE inline test_utils::bfloat16 max(const test_utils::bfloat16& a,
                                                   const test_utils::bfloat16& b)
{
    return test_utils::bfloat16_maximum{}(a, b);
}

template<class T>
HIPCUB_HOST_DEVICE inline constexpr T max(const T& t, const test_utils::bfloat16& u)
{
    return test_utils::max(t, static_cast<T>(u));
}

template<class T>
HIPCUB_HOST_DEVICE inline constexpr T max(const test_utils::bfloat16& t, const T& u)
{
    return test_utils::max(static_cast<T>(t), u);
}

template<class T, class U>
HIPCUB_HOST_DEVICE inline constexpr std::common_type_t<test_utils::custom_test_type<T>,
                                                       test_utils::custom_test_type<U>>
    min(const test_utils::custom_test_type<T>& t, const test_utils::custom_test_type<U>& u)
{
    using common_type
        = std::common_type_t<test_utils::custom_test_type<T>, test_utils::custom_test_type<U>>;
    const common_type common_t(t);
    const common_type common_u(u);

    return common_t < common_u ? common_t : common_u;
}

template<class T, class U = T>
HIPCUB_HOST_DEVICE inline constexpr std::common_type_t<T, U> min(const T& t, const U& u)
{
    using common_type = std::common_type_t<T, U>;
    return static_cast<common_type>(t) < static_cast<common_type>(u) ? t : u;
}

template<class T>
HIPCUB_HOST_DEVICE inline constexpr T min(const T& t, const test_utils::half& u)
{
    return test_utils::min(t, static_cast<T>(u));
}

template<class T>
HIPCUB_HOST_DEVICE inline constexpr T min(const test_utils::half& t, const T& u)
{
    return test_utils::min(static_cast<T>(t), u);
}

HIPCUB_HOST_DEVICE inline test_utils::half min(const test_utils::half& a, const test_utils::half& b)
{
    return test_utils::half_minimum{}(a, b);
}

template<class T>
HIPCUB_HOST_DEVICE inline constexpr T min(const T& t, const test_utils::bfloat16& u)
{
    return test_utils::min(t, static_cast<T>(u));
}

template<class T>
HIPCUB_HOST_DEVICE inline constexpr T min(const test_utils::bfloat16& t, const T& u)
{
    return test_utils::min(static_cast<T>(t), u);
}

HIPCUB_HOST_DEVICE inline test_utils::bfloat16 min(const test_utils::bfloat16& a,
                                                   const test_utils::bfloat16& b)
{
    return test_utils::bfloat16_minimum{}(a, b);
}

template<class T, class U>
HIPCUB_HOST_DEVICE inline constexpr typename std::common_type<test_utils::custom_test_type<T>,
                                                              test_utils::custom_test_type<U>>::type
    max(const test_utils::custom_test_type<T>& t, const test_utils::custom_test_type<U>& u)
{
    using common_type = typename std::common_type<test_utils::custom_test_type<T>,
                                                  test_utils::custom_test_type<U>>::type;
    const common_type common_t(t);
    const common_type common_u(u);

    return common_t < common_u ? common_u : common_t;
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

template<unsigned int LogicalWarpSize>
__device__ constexpr bool device_test_enabled_for_warp_size_v
    = HIPCUB_DEVICE_WARP_THREADS >= LogicalWarpSize;

template<typename T,
         typename U,
         std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
inline constexpr auto ceiling_div(const T a, const U b)
{
    return a / b + (a % b > 0 ? 1 : 0);
}

} // namespace test_utils

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
