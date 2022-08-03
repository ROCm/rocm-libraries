// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprim/intrinsics.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/detail/match_result_type.hpp>

// Identity iterator
#include "identity_iterator.hpp"
// Bounds checking iterator
#include "bounds_checking_iterator.hpp"
// Seed values
#include "test_seed.hpp"

#include "test_utils_half.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_assertions.hpp"

// Helper macros to disable warnings in clang
#ifdef __clang__
#define ROCPRIM_PRAGMA_TO_STR(x) _Pragma(#x)
#define ROCPRIM_CLANG_SUPPRESS_WARNING_PUSH _Pragma("clang diagnostic push")
#define ROCPRIM_CLANG_SUPPRESS_WARNING(w) ROCPRIM_PRAGMA_TO_STR(clang diagnostic ignored w)
#define ROCPRIM_CLANG_SUPPRESS_WARNING_POP _Pragma("clang diagnostic pop")
#define ROCPRIM_CLANG_SUPPRESS_WARNING_WITH_PUSH(w) \
    ROCPRIM_CLANG_SUPPRESS_WARNING_PUSH ROCPRIM_CLANG_SUPPRESS_WARNING(w)
#else // __clang__
#define ROCPRIM_CLANG_SUPPRESS_WARNING_PUSH
#define ROCPRIM_CLANG_SUPPRESS_WARNING(w)
#define ROCPRIM_CLANG_SUPPRESS_WARNING_POP
#define ROCPRIM_CLANG_SUPPRESS_WARNING_WITH_PUSH(w)
#endif // __clang__

namespace test_utils
{

template<class T>
struct precision_threshold
{
    static constexpr float percentage = 0.0008f;
};

template<>
struct precision_threshold<rocprim::half>
{
    static constexpr float percentage = 0.02f;
};

template<>
struct precision_threshold<rocprim::bfloat16>
{
    static constexpr float percentage = 0.08f;
};

template<class T>
struct precision_threshold<custom_test_type<T>>
{
    static constexpr float percentage = 0.01f;
};

template<class T>
struct is_plus_operator : std::false_type {
    typedef uint8_t value_type;
};

template<class T>
struct is_plus_operator<rocprim::plus<T>> : std::true_type {
    typedef T value_type;
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

template<class acc_type, class InputIt, class OutputIt, class FlagsIt, class BinaryOperation>
OutputIt host_inclusive_segmented_scan_headflags(InputIt first, InputIt last, FlagsIt flags,
                                                 OutputIt d_first, BinaryOperation op)
{
    if (first == last) return d_first;

    acc_type sum = *first;
    *d_first = sum;

    while (++first != last) {
        ++flags;
        sum = *flags ? *first : op(sum, *first);
        *++d_first = sum;
    }
    return ++d_first;
}

template<class InputIt, class OutputIt, class FlagsIt, class BinaryOperation, class acc_type>
OutputIt host_exclusive_segmented_scan_headflags(InputIt first, InputIt last, FlagsIt flags,
                                                 OutputIt d_first, BinaryOperation op, acc_type init)
{
    if (first == last) return d_first;

    acc_type sum = init;
    *d_first = sum;

    while ((first+1) != last){
        ++flags;
        sum = *flags ? init : op(sum, *first);
        *++d_first = static_cast<acc_type>(sum);
        first++;
    }
    return ++d_first;
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
        if(key_compare_op(*k_first, *(k_first+1)))
        {
            sum = op(sum, *first);
        }
        else
        {
            sum = initial_value;
        }
        k_first++;
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
        if(key_compare_op(*k_first, *(k_first+1)))
        {
            sum = op(sum, *first);
        }
        else
        {
            sum = *first;
        }
        k_first++;
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

// std::iota causes problems with __half and bfloat16 and custom_test_type because of a missing ++increment operator
template<class ForwardIt, class T>
void iota(ForwardIt first, ForwardIt last, T value)
{
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;
    while(first != last) {
        *first++ = static_cast<value_type>(value);
        ++value;
    }
}

#define SKIP_IF_UNSUPPORTED_WARP_SIZE(test_warp_size) { \
    const auto host_warp_size = ::rocprim::host_warp_size(); \
    if (host_warp_size < (test_warp_size)) \
    { \
        GTEST_SKIP() << "Cannot run test of warp size " \
            << (test_warp_size) \
            << " on a device with warp size " \
            << host_warp_size; \
    } \
}

template<unsigned int LogicalWarpSize>
struct DeviceSelectWarpSize
{
    static constexpr unsigned value = ::rocprim::device_warp_size() >= LogicalWarpSize
        ? LogicalWarpSize
        : ::rocprim::device_warp_size();
};

} // end test_utils namespace

#endif // TEST_TEST_UTILS_HPP_
