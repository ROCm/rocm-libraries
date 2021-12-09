// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprim/types.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/detail/match_result_type.hpp>

// Identity iterator
#include "identity_iterator.hpp"
// Bounds checking iterator
#include "bounds_checking_iterator.hpp"
// Seed values
#include "test_seed.hpp"

// For better Google Test reporting and debug output of half values
inline
std::ostream& operator<<(std::ostream& stream, const rocprim::half& value)
{
    stream << static_cast<float>(value);
    return stream;
}

namespace test_utils
{

static constexpr uint32_t random_data_generation_segments = 32;
static constexpr uint32_t random_data_generation_repeat_strides = 4;

template<class T>
struct precision_threshold
{
    static constexpr float percentage = 0.0002f;
};

template<>
struct precision_threshold<rocprim::half>
{
    static constexpr float percentage = 0.01f;
};

template<>
struct precision_threshold<rocprim::bfloat16>
{
    static constexpr float percentage = 0.02f;
};

// Support half operators on host side

ROCPRIM_HOST inline
rocprim::native_half half_to_native(const rocprim::half& x)
{
    return *reinterpret_cast<const rocprim::native_half *>(&x);
}

ROCPRIM_HOST inline
rocprim::half native_to_half(const rocprim::native_half& x)
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

// Support bfloat16 operators on host side

ROCPRIM_HOST inline
rocprim::native_bfloat16 bfloat16_to_native(const rocprim::bfloat16& x)
{
    return rocprim::native_bfloat16(x);
}

ROCPRIM_HOST inline
rocprim::bfloat16 native_to_bfloat16(const rocprim::native_bfloat16& x)
{
    return rocprim::bfloat16(x);
}

struct bfloat16_less
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b;
        #else
        return bfloat16_to_native(a) < bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_less_equal
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a <= b;
        #else
        return bfloat16_to_native(a) <= bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_greater
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a > b;
        #else
        return bfloat16_to_native(a) > bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_greater_equal
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a >= b;
        #else
        return bfloat16_to_native(a) >= bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a == b;
        #else
        return bfloat16_to_native(a) == bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_not_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a != b;
        #else
        return bfloat16_to_native(a) != bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_plus
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::bfloat16 operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a + b;
        #else
        return native_to_bfloat16(bfloat16_to_native(a) + bfloat16_to_native(b));
        #endif
    }
};

struct bfloat16_minus
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::bfloat16 operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a - b;
        #else
        return native_to_bfloat16(bfloat16_to_native(a) - bfloat16_to_native(b));
        #endif
    }
};

struct bfloat16_multiplies
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::bfloat16 operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a * b;
        #else
        return native_to_bfloat16(bfloat16_to_native(a) * bfloat16_to_native(b));
        #endif
    }
};

struct bfloat16_maximum
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::bfloat16 operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b ? b : a;
        #else
        return bfloat16_to_native(a) < bfloat16_to_native(b) ? b : a;
        #endif
    }
};

struct bfloat16_minimum
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::bfloat16 operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b ? a : b;
        #else
        return bfloat16_to_native(a) < bfloat16_to_native(b) ? a : b;
        #endif
    }
};

// Less operator selector
template<typename T>
struct select_less_operator
{
    typedef ::rocprim::less<T> type;
};

template<>
struct select_less_operator<::rocprim::half>
{
    typedef half_less type;
};

template<>
struct select_less_operator<::rocprim::bfloat16>
{
    typedef bfloat16_less type;
};

// Equal to operator selector
template<typename T>
struct select_equal_to_operator
{
    typedef ::rocprim::equal_to<T> type;
};

template<>
struct select_equal_to_operator<::rocprim::half>
{
    typedef half_equal_to type;
};

template<>
struct select_equal_to_operator<::rocprim::bfloat16>
{
    typedef bfloat16_equal_to type;
};

// Greator to operator selector
template<typename T>
struct select_greater_operator
{
    typedef ::rocprim::greater<T> type;
};

template<>
struct select_greater_operator<::rocprim::half>
{
    typedef half_greater type;
};

template<>
struct select_greater_operator<::rocprim::bfloat16>
{
    typedef bfloat16_greater type;
};

// Not equeal to operator selector
template<typename T>
struct select_not_equal_to_operator
{
    typedef ::rocprim::not_equal_to<T> type;
};

template<>
struct select_not_equal_to_operator<::rocprim::half>
{
    typedef half_not_equal_to type;
};

template<>
struct select_not_equal_to_operator<::rocprim::bfloat16>
{
    typedef bfloat16_not_equal_to type;
};

/* Plus to operator selector for host-side
 * On host-side we use `double` as accumulator and `rocprim::plus<double>` as operator
 * for bfloat16 and half types. This is because additions of floating-point types are not
 * associative. This would result in wrong output rather quickly for reductions and scan-algorithms
 * on host-side for bfloat16 and half because of their low-precision.
 */
template<typename T>
struct select_plus_operator_host
{
    typedef ::rocprim::plus<T> type;
    typedef T acc_type;
};

template<>
struct select_plus_operator_host<::rocprim::half>
{
    typedef ::rocprim::plus<double> type;
    typedef double acc_type;
};

template<>
struct select_plus_operator_host<::rocprim::bfloat16>
{
    typedef ::rocprim::plus<double> type;
    typedef double acc_type;
};

// Minimum to operator selector
template<typename T>
struct select_minimum_operator
{
    typedef ::rocprim::minimum<T> type;
};

template<>
struct select_minimum_operator<::rocprim::half>
{
    typedef half_minimum type;
};

template<>
struct select_minimum_operator<::rocprim::bfloat16>
{
    typedef bfloat16_minimum type;
};

// Maximum to operator selector
template<typename T>
struct select_maximum_operator
{
    typedef ::rocprim::maximum<T> type;
};

template<>
struct select_maximum_operator<::rocprim::half>
{
    typedef half_maximum type;
};

template<>
struct select_maximum_operator<::rocprim::bfloat16>
{
    typedef bfloat16_maximum type;
};

// std::uniform_int_distribution is undefined for anything other than
// short, int, long, long long, unsigned short, unsigned int, unsigned long, or unsigned long long.
template <typename T>
struct is_valid_for_int_distribution :
    std::integral_constant<bool,
        std::is_same<short, T>::value ||
        std::is_same<unsigned short, T>::value ||
        std::is_same<int, T>::value ||
        std::is_same<unsigned int, T>::value ||
        std::is_same<long, T>::value ||
        std::is_same<unsigned long, T>::value ||
        std::is_same<long long, T>::value ||
        std::is_same<unsigned long long, T>::value
    > {};

template<class T, class U, class V>
inline auto get_random_data(size_t size, U min, V max, seed_type seed_value)
    -> typename std::enable_if<rocprim::is_integral<T>::value, std::vector<T>>::type
{
    engine_type gen{seed_value};
    using dis_type = typename std::conditional<
        is_valid_for_int_distribution<T>::value,
        T,
        typename std::conditional<std::is_signed<T>::value,
            int,
            unsigned int>::type
        >::type;
    std::uniform_int_distribution<dis_type> distribution((dis_type)min, (dis_type)max);
    std::vector<T> data(size);
    size_t segment_size = size / random_data_generation_segments;
    if(segment_size != 0)
    {
        for(uint32_t segment_index = 0; segment_index < random_data_generation_segments; segment_index++)
        {
            if(segment_index % random_data_generation_repeat_strides == 0)
            {
                T repeated_value = static_cast<T>(distribution(gen));
                std::fill(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    repeated_value);

            }
            else
            {
                std::generate(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    [&]() { return static_cast<T>(distribution(gen)); });
            }
        }
    }
    else
    {
        std::generate(data.begin(), data.end(), [&]() { return static_cast<T>(distribution(gen)); });
    }
    return data;
}

template<class T, class U, class V>
inline auto get_random_data(size_t size, U min, V max, seed_type seed_value)
    -> typename std::enable_if<rocprim::is_floating_point<T>::value, std::vector<T>>::type
{
    engine_type gen{seed_value};
    // Generate floats when T is half or bfloat16
    using dis_type = typename std::conditional<std::is_same<rocprim::half, T>::value || std::is_same<rocprim::bfloat16, T>::value, float, T>::type;
    std::uniform_real_distribution<dis_type> distribution((dis_type)min, (dis_type)max);
    std::vector<T> data(size);
    size_t segment_size = size / random_data_generation_segments;
    if(segment_size != 0)
    {
        for(uint32_t segment_index = 0; segment_index < random_data_generation_segments; segment_index++)
        {
            if(segment_index % random_data_generation_repeat_strides == 0)
            {
                T repeated_value = static_cast<T>(distribution(gen));
                std::fill(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    repeated_value);

            }
            else
            {
                std::generate(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    [&]() { return static_cast<T>(distribution(gen)); });
            }
        }
    }
    else
    {
        std::generate(data.begin(), data.end(), [&]() { return static_cast<T>(distribution(gen)); });

    }
    return data;
}

template<class T>
inline std::vector<T> get_random_data01(size_t size, float p, seed_type seed_value)
{
    const size_t max_random_size = 1024 * 1024;
    engine_type gen{seed_value};
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

template<class T, class U, class V>
inline auto get_random_value(U min, V max, seed_type seed_value)
    -> typename std::enable_if<rocprim::is_arithmetic<T>::value, T>::type
{
    return get_random_data<T>(random_data_generation_segments, min, max, seed_value)[0];
}

template<class InputIt, class T,
    std::enable_if_t<std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value ||
                     std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::half>::value ||
                     std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value
                     , bool> = true>
constexpr T host_reduce(InputIt first, InputIt last, rocprim::plus<T>)
{
    using accumulator_type = double;
    // Calculate expected results on host
    accumulator_type expected = accumulator_type(0);
    rocprim::plus<accumulator_type> bin_op;
    for(InputIt it = first; it != last; it++)
    {
        expected = bin_op(expected, static_cast<accumulator_type>(*it));
    }
    return static_cast<T>(expected);
}

template<class InputIt, class T,
          std::enable_if_t<!std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value &&
                           !std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::half>::value &&
                           !std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value
                           , bool> = true>
constexpr T host_reduce(InputIt first, InputIt last, rocprim::plus<T> op)
{
    using acc_type = T;
    // Calculate expected results on host
    acc_type expected = acc_type(0);
    for(InputIt it = first; it != last; it++)
    {
        expected = op(expected, *it);
    }
    return expected;
}

template<class InputIt, class OutputIt, class BinaryOperation, class acc_type>
OutputIt host_inclusive_scan_impl(InputIt first, InputIt last,
                             OutputIt d_first, BinaryOperation op, acc_type)
{
    if (first == last) return d_first;

    acc_type sum = *first;
    *d_first = sum;

    while (++first != last) {
        sum = op(sum, *first);
        *++d_first = sum;
    }
    return ++d_first;
}

template<class InputIt, class OutputIt, class BinaryOperation>
OutputIt host_inclusive_scan(InputIt first, InputIt last,
                             OutputIt d_first, BinaryOperation op)
{
    using acc_type = typename std::iterator_traits<InputIt>::value_type;
    return host_inclusive_scan_impl(first, last, d_first, op, acc_type{});
}

template<class InputIt, class OutputIt, class T,
          std::enable_if_t<std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value ||
                           std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::half>::value ||
                           std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value
                           , bool> = true>
OutputIt host_inclusive_scan(InputIt first, InputIt last,
                             OutputIt d_first, rocprim::plus<T>)
{
    using acc_type = double;
    return host_inclusive_scan_impl(first, last, d_first, rocprim::plus<acc_type>(), acc_type{});
}

template<class InputIt, class T, class OutputIt, class BinaryOperation, class acc_type>
OutputIt host_exclusive_scan_impl(InputIt first, InputIt last,
                             T initial_value, OutputIt d_first,
                             BinaryOperation op, acc_type)
{
    if (first == last) return d_first;

    acc_type sum = initial_value;
    *d_first = initial_value;

    while ((first+1) != last)
    {
        sum = op(sum, *first);
        *++d_first = sum;
        first++;
    }
    return ++d_first;
}

template<class InputIt, class T, class OutputIt, class BinaryOperation>
OutputIt host_exclusive_scan(InputIt first, InputIt last,
                             T initial_value, OutputIt d_first,
                             BinaryOperation op)
{
    using acc_type = typename std::iterator_traits<InputIt>::value_type;
    return host_exclusive_scan_impl(first, last, initial_value, d_first, op, acc_type{});
}

template<class InputIt, class T, class OutputIt, class U,
          std::enable_if_t<std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value ||
                               std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::half>::value ||
                               std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value
                           , bool> = true>
OutputIt host_exclusive_scan(InputIt first, InputIt last,
                             T initial_value, OutputIt d_first,
                             rocprim::plus<U>)
{
    using acc_type = double;
    return host_exclusive_scan_impl(first, last, initial_value, d_first, rocprim::plus<acc_type>(), acc_type{});
}

template<class InputIt, class KeyIt, class T, class OutputIt, class BinaryOperation, class KeyCompare, class acc_type>
OutputIt host_exclusive_scan_by_key_impl(InputIt first, InputIt last, KeyIt k_first,
                                         T initial_value, OutputIt d_first,
                                         BinaryOperation op, KeyCompare key_compare_op, acc_type)
{
    if (first == last) return d_first;

    acc_type sum = initial_value;
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
template<class InputIt, class KeyIt, class T, class OutputIt, class BinaryOperation, class KeyCompare>
OutputIt host_exclusive_scan_by_key(InputIt first, InputIt last, KeyIt k_first,
                                    T initial_value, OutputIt d_first,
                                    BinaryOperation op, KeyCompare key_compare_op)
{
    using acc_type = typename std::iterator_traits<InputIt>::value_type;
    return host_exclusive_scan_by_key_impl(first, last, k_first, initial_value, d_first, op, key_compare_op, acc_type{});
}

template<class InputIt, class KeyIt, class T, class OutputIt, class U, class KeyCompare,
          std::enable_if_t<std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value ||
                               std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::half>::value ||
                               std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value
                           , bool> = true>
OutputIt host_exclusive_scan_by_key(InputIt first, InputIt last, KeyIt k_first,
                                    T initial_value, OutputIt d_first,
                                    rocprim::plus<U>, KeyCompare key_compare_op)
{
    using acc_type = double;
    return host_exclusive_scan_by_key_impl(first, last, k_first, initial_value, d_first, rocprim::plus<acc_type>(), key_compare_op, acc_type{});
}


template<class InputIt, class KeyIt, class OutputIt, class BinaryOperation, class KeyCompare, class acc_type>
OutputIt host_inclusive_scan_by_key_impl(InputIt first, InputIt last, KeyIt k_first,
                                         OutputIt d_first,
                                         BinaryOperation op, KeyCompare key_compare_op, acc_type)
{
    if (first == last) return d_first;

    acc_type sum = *first;
    *d_first = sum;

    while (++first != last)
    {
        if(key_compare_op(*k_first, *++k_first))
        {
            sum = op(sum, *first);
        }
        else
        {
            sum = *first;
        }
        *++d_first = sum;
    }
    return ++d_first;
}
template<class InputIt, class KeyIt, class OutputIt, class BinaryOperation, class KeyCompare>
OutputIt host_inclusive_scan_by_key(InputIt first, InputIt last, KeyIt k_first,
                                    OutputIt d_first,
                                    BinaryOperation op, KeyCompare key_compare_op)
{
    using acc_type = typename std::iterator_traits<InputIt>::value_type;
    return host_inclusive_scan_by_key_impl(first, last, k_first, d_first, op, key_compare_op, acc_type{});
}

template<class InputIt, class KeyIt, class OutputIt, class U, class KeyCompare,
          std::enable_if_t<std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value ||
                               std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::half>::value ||
                               std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value
                           , bool> = true>
OutputIt host_inclusive_scan_by_key(InputIt first, InputIt last, KeyIt k_first,
                                    OutputIt d_first,
                                    rocprim::plus<U>, KeyCompare key_compare_op)
{
    using acc_type = double;
    return host_inclusive_scan_by_key_impl(first, last, k_first, d_first, rocprim::plus<acc_type>(), key_compare_op, acc_type{});
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

template<class T>
struct precision_threshold<custom_test_type<T>>
{
    static constexpr float percentage = 0.01f;
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

//Overload for rocprim::bfloat16
template<>
struct custom_test_type<rocprim::bfloat16>
{
    using value_type = rocprim::bfloat16;

    rocprim::bfloat16 x;
    rocprim::bfloat16 y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_test_type(), value) != value
    ROCPRIM_HOST_DEVICE inline
    custom_test_type() : x(12), y(34) {}

    ROCPRIM_HOST_DEVICE inline
    custom_test_type(rocprim::bfloat16 x, rocprim::bfloat16 y) : x(x), y(y) {}

    ROCPRIM_HOST_DEVICE inline
    custom_test_type(rocprim::bfloat16 xy) : x(xy), y(xy) {}

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
        return custom_test_type(bfloat16_plus()(x, other.x), bfloat16_plus()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
    custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(bfloat16_minus()(x, other.x), bfloat16_minus()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(const custom_test_type& other) const
    {
        return (bfloat16_less()(x, other.x) || (bfloat16_equal_to()(x, other.x) && bfloat16_less()(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(const custom_test_type& other) const
    {
        return (bfloat16_greater()(x, other.x) || (bfloat16_equal_to()(x, other.x) && bfloat16_greater()(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(const custom_test_type& other) const
    {
        return (bfloat16_equal_to()(x, other.x) && bfloat16_equal_to()(y, other.y));
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

template<> struct numeric_limits<rocprim::half> : public std::numeric_limits<rocprim::half> {
public:
    static rocprim::half min() {
        return rocprim::half(0.00006104f);
    };
};

template<> class numeric_limits<rocprim::bfloat16> : public std::numeric_limits<rocprim::bfloat16> {
public:
    static rocprim::bfloat16 min() {
        rocprim::bfloat16 a;
        a.data = 0x0080;
        return a;
    };
};

template<class T>
inline auto get_random_data(size_t size, T min, T max, seed_type seed_value)
    -> typename std::enable_if<
           is_custom_test_type<T>::value && std::is_integral<typename T::value_type>::value,
           std::vector<T>
       >::type
{
    engine_type gen(seed_value);
    std::uniform_int_distribution<typename T::value_type> distribution(min.x, max.x);
    std::vector<T> data(size);
    size_t segment_size = size / random_data_generation_segments;
    if(segment_size != 0)
    {
        for(uint32_t segment_index = 0; segment_index < random_data_generation_segments; segment_index++)
        {
            if(segment_index % random_data_generation_repeat_strides == 0)
            {
                T repeated_value = T(distribution(gen), distribution(gen));
                std::fill(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    repeated_value);

            }
            else
            {
                std::generate(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    [&]() { return T(distribution(gen), distribution(gen)); });
            }
        }
    }
    else
    {
        std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen), distribution(gen)); });
    }
    return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max, seed_type seed_value)
    -> typename std::enable_if<
           is_custom_test_type<T>::value && std::is_floating_point<typename T::value_type>::value,
           std::vector<T>
       >::type
{
    engine_type gen(seed_value);
    std::uniform_real_distribution<typename T::value_type> distribution(min.x, max.x);
    std::vector<T> data(size);
    size_t segment_size = size / random_data_generation_segments;
    if(segment_size != 0)
    {
        for(uint32_t segment_index = 0; segment_index < random_data_generation_segments; segment_index++)
        {
            if(segment_index % random_data_generation_repeat_strides == 0)
            {
                T repeated_value = T(distribution(gen), distribution(gen));
                std::fill(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    repeated_value);

            }
            else
            {
                std::generate(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    [&]() { return T(distribution(gen), distribution(gen)); });
            }
        }
    }
    else
    {
        std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen), distribution(gen)); });
    }
    return data;
}

template<class T>
inline auto get_random_data(size_t size, typename T::value_type min, typename T::value_type max, seed_type seed_value)
    -> typename std::enable_if<
           is_custom_test_array_type<T>::value && std::is_integral<typename T::value_type>::value,
           std::vector<T>
       >::type
{
    engine_type gen(seed_value);
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
inline auto get_random_value(typename T::value_type min, typename T::value_type max, seed_type seed_value)
    -> typename std::enable_if<is_custom_test_type<T>::value || is_custom_test_array_type<T>::value, T>::type
{
    return get_random_data<typename T::value_type>(random_data_generation_segments, min, max, seed_value)[0];
}

template<class T>
auto assert_near(const std::vector<T>& result, const std::vector<T>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff = std::abs(percent * expected[i]);
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
        if(static_cast<float>(result[i])==static_cast<float>(expected[i])) continue;
        auto diff = std::abs(percent * static_cast<float>(expected[i]));
        ASSERT_NEAR(static_cast<float>(result[i]), static_cast<float>(expected[i]), diff) << "where index = " << i;
    }
}

void assert_near(const std::vector<rocprim::bfloat16>& result, const std::vector<rocprim::bfloat16>& expected, float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        if(result[i]==expected[i]) continue;
        auto diff = std::abs(percent * static_cast<float>(expected[i]));
        ASSERT_NEAR(static_cast<float>(result[i]), static_cast<float>(expected[i]), diff) << "where index = " << i;
    }
}

void assert_near(const std::vector<custom_test_type<rocprim::half>>& result, const std::vector<custom_test_type<rocprim::half>>& expected, const float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff1 = std::abs(percent * static_cast<float>(expected[i].x));
        auto diff2 = std::abs(percent * static_cast<float>(expected[i].y));
        if(static_cast<float>(result[i].x)!=static_cast<float>(expected[i].x)) ASSERT_NEAR(static_cast<float>(result[i].x), static_cast<float>(expected[i].x), diff1) << "where index = " << i;
        if(static_cast<float>(result[i].y)!=static_cast<float>(expected[i].y)) ASSERT_NEAR(static_cast<float>(result[i].y), static_cast<float>(expected[i].y), diff2) << "where index = " << i;
    }
}

void assert_near(const std::vector<custom_test_type<rocprim::bfloat16>>& result, const std::vector<custom_test_type<rocprim::bfloat16>>& expected, const float percent)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff1 = std::abs(percent * static_cast<float>(expected[i].x));
        auto diff2 = std::abs(percent * static_cast<float>(expected[i].y));
        if(result[i].x!=expected[i].x) ASSERT_NEAR(static_cast<float>(result[i].x), static_cast<float>(expected[i].x), diff1) << "where index = " << i;
        if(result[i].y!=expected[i].y) ASSERT_NEAR(static_cast<float>(result[i].y), static_cast<float>(expected[i].y), diff2) << "where index = " << i;
    }
}


template<class T>
auto assert_near(const std::vector<custom_test_type<T>>& result, const std::vector<custom_test_type<T>>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        auto diff1 = std::abs(percent * expected[i].x);
        auto diff2 = std::abs(percent * expected[i].y);
        ASSERT_NEAR(result[i].x, expected[i].x, diff1) << "where index = " << i;
        ASSERT_NEAR(result[i].y, expected[i].y, diff2) << "where index = " << i;
    }
}

template<class T>
auto assert_near(const T& result, const T& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    auto diff = std::abs(percent * expected);
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
    if(static_cast<float>(result)==static_cast<float>(expected)) return;
    auto diff = std::abs(percent * static_cast<float>(expected));
    ASSERT_NEAR(static_cast<float>(result), static_cast<float>(expected), diff);
}

void assert_near(const rocprim::bfloat16& result, const rocprim::bfloat16& expected, float percent)
{
    if(result==expected) return;
    auto diff = std::abs(percent * static_cast<float>(expected));
    ASSERT_NEAR(static_cast<float>(result), static_cast<float>(expected), diff);
}

template<class T>
auto assert_near(const custom_test_type<T>& result, const custom_test_type<T>& expected, const float percent)
    -> typename std::enable_if<std::is_floating_point<T>::value>::type
{
    auto diff1 = std::abs(percent * expected.x);
    auto diff2 = std::abs(percent * expected.y);
    ASSERT_NEAR(result.x, expected.x, diff1);
    ASSERT_NEAR(result.y, expected.y, diff2);
}

template<class T>
void assert_bit_eq(const std::vector<T>& result, const std::vector<T>& expected)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        bool the_same = true;
        for(size_t j = 0; j < sizeof(T); j++)
            the_same &= ((reinterpret_cast<const uint8_t*>(&result[i]))[j] == (reinterpret_cast<const uint8_t*>(&expected[i]))[j]);
        ASSERT_EQ(true, the_same) << "where index = " << i;
    }
}

template<class T>
auto assert_eq(const std::vector<T>& result, const std::vector<T>& expected)
    -> typename std::enable_if<!rocprim::is_floating_point<T>::value, void>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
    }
}

template<class T>
auto assert_eq(const std::vector<T>& result, const std::vector<T>& expected)
    -> typename std::enable_if<rocprim::is_floating_point<T>::value, void>::type
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        if( result[i] != result[i] )
            ASSERT_EQ(result[i] != result[i], expected[i] != expected[i]) << "NAN check failed where index = " << i;
        else
            ASSERT_EQ(result[i], expected[i]) << "where index = " << i;
    }
}

auto assert_eq(const std::vector<rocprim::half>& result, const std::vector<rocprim::half>& expected)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(half_to_native(result[i]), half_to_native(expected[i])) << "where index = " << i;
    }
}

auto assert_eq(const std::vector<rocprim::bfloat16>& result, const std::vector<rocprim::bfloat16>& expected)
{
    ASSERT_EQ(result.size(), expected.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        ASSERT_EQ(bfloat16_to_native(result[i]), bfloat16_to_native(expected[i])) << "where index = " << i;
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

void custom_assert_eq(const std::vector<rocprim::bfloat16>& result, const std::vector<rocprim::bfloat16>& expected, size_t size)
{
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(bfloat16_to_native(result[i]), bfloat16_to_native(expected[i])) << "where index = " << i;
    }
}

template<class T>
auto assert_eq(const T& result, const T& expected)
    -> typename std::enable_if<!rocprim::is_floating_point<T>::value, void>::type
{
    ASSERT_EQ(result, expected);
}

template<class T>
auto assert_eq(const T& result, const T& expected)
    -> typename std::enable_if<rocprim::is_floating_point<T>::value, void>::type
{
    if( result != result )
        ASSERT_EQ( result != result, expected != expected);
    else
        ASSERT_EQ(result, expected);
}

auto assert_eq(const rocprim::half& result, const rocprim::half& expected)
{
    ASSERT_EQ(half_to_native(result), half_to_native(expected));
}

auto assert_eq(const rocprim::bfloat16& result, const rocprim::bfloat16& expected)
{
    ASSERT_EQ(bfloat16_to_native(result), bfloat16_to_native(expected));
}

//TODO: Use custom iota until the follwing PR merge: https://github.com/ROCm-Developer-Tools/HIP/pull/2303
template<class ForwardIt, class T>
void iota(ForwardIt first, ForwardIt last, T value)
{
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;
    while(first != last) {
        *first++ = static_cast<value_type>(value);
        ++value;
    }
}

} // end test_utils namespace

#endif // TEST_TEST_UTILS_HPP_
