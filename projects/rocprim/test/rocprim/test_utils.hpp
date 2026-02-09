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

#ifndef TEST_TEST_UTILS_HPP_
#define TEST_TEST_UTILS_HPP_

#include "../common_test_header.hpp"

#include "../../common/utils.hpp"
#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_data_generation.hpp"
#include "../../common/utils_device_ptr.hpp"
#include "../../common/utils_half.hpp"

// Identity iterator
#include "identity_iterator.hpp"
// Bounds checking iterator
#include "bounds_checking_iterator.hpp"
// Seed values
#include "test_seed.hpp"

#include "test_utils_assertions.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_get_random_data.hpp"
#include "test_utils_hipgraphs.hpp"

#include <rocprim/device/config_types.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdint.h>
#include <type_traits>

namespace test_utils
{

// Values of relative error for non-assotiative operations
// (+, -, *) and type conversions for floats
// They are doubled from 1 / (1 << mantissa_bits) as we compare in tests
// the results of _two_ sequences of operations with different order
// For all other operations (i.e. integer arithmetics) default 0 is used
template<class T>
inline constexpr float precision = 0;

template<>
inline constexpr float precision<double> = 2.0f / (1ll << 52);

template<>
inline constexpr float precision<float> = 2.0f / (1ll << 23);

template<>
inline constexpr float precision<rocprim::half> = 2.0f / (1ll << 10);

template<>
inline constexpr float precision<rocprim::bfloat16> = 2.0f / (1ll << 7);

template<class T>
inline constexpr float precision<const T> = precision<T>;

template<class T>
inline constexpr float precision<common::custom_type<T, T, true>> = precision<T>;

template<class T, int N>
inline constexpr float precision<custom_test_array_type<T, N>> = precision<T>;

template<class T>
struct is_plus_operator : std::false_type
{
    using value_type = uint8_t;
};

template<class T>
struct is_plus_operator<rocprim::plus<T>> : std::true_type
{
    using value_type = T;
};

template<class T>
struct is_add_operator : std::false_type
{
    using value_type = uint8_t;
};

template<class T>
struct is_add_operator<rocprim::plus<T>> : std::true_type
{
    using value_type = T;
};

template<class T>
struct is_add_operator<rocprim::minus<T>> : std::true_type
{
    using value_type = T;
};

template<class T>
struct is_multiply_operator : std::false_type
{
    using value_type = uint8_t;
};

template<class T>
struct is_multiply_operator<rocprim::multiplies<T>> : std::true_type
{
    using value_type = T;
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
    using type     = ::rocprim::plus<T>;
    using acc_type = T;
};

template<>
struct select_plus_operator_host<::rocprim::half>
{
    using type     = ::rocprim::plus<double>;
    using acc_type = double;
};

template<>
struct select_plus_operator_host<::rocprim::bfloat16>
{
    using type     = ::rocprim::plus<double>;
    using acc_type = double;
};

template<
    class InputIt,
    class T,
    std::enable_if_t<
        std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value
            || std::is_same<typename std::iterator_traits<InputIt>::value_type,
                            rocprim::half>::value
            || std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
        bool>
    = true>
constexpr std::vector<T> host_reduce(InputIt first, InputIt last, rocprim::plus<T>)
{
    using accumulator_type = double;
    size_t         size    = std::distance(first, last);
    std::vector<T> result(size);
    if(size == 0)
    {
        return result;
    }
    // Calculate expected results on host
    accumulator_type                expected = accumulator_type(0);
    rocprim::plus<accumulator_type> bin_op;
    for(int i = size - 1; i >= 0; --i)
    {
        expected  = bin_op(expected, static_cast<accumulator_type>(*(first + i)));
        result[i] = static_cast<T>(expected);
    }
    return result;
}

template<
    class InputIt,
    class T,
    std::enable_if_t<
        !std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value
            && !std::is_same<typename std::iterator_traits<InputIt>::value_type,
                             rocprim::half>::value
            && !std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
        bool>
    = true>
constexpr std::vector<T> host_reduce(InputIt first, InputIt last, rocprim::plus<T> op)
{
    using acc_type      = T;
    size_t         size = std::distance(first, last);
    std::vector<T> result(size);
    if(size == 0)
    {
        return result;
    }
    // Calculate expected results on host
    acc_type expected = acc_type(0);
    for(int i = size - 1; i >= 0; --i)
    {
        expected  = op(expected, *(first + i));
        result[i] = expected;
    }
    return result;
}

template<class acc_type, class InputIt, class OutputIt, class FlagsIt, class BinaryOperation>
OutputIt host_inclusive_segmented_scan_headflags(
    InputIt first, InputIt last, FlagsIt flags, OutputIt d_first, BinaryOperation op)
{
    if(first == last)
        return d_first;

    acc_type sum = *first;
    *d_first     = sum;

    while(++first != last)
    {
        ++flags;
        sum        = *flags ? acc_type(*first) : acc_type(op(sum, *first));
        *++d_first = sum;
    }
    return ++d_first;
}

template<class InputIt, class OutputIt, class FlagsIt, class BinaryOperation, class acc_type>
OutputIt host_exclusive_segmented_scan_headflags(
    InputIt first, InputIt last, FlagsIt flags, OutputIt d_first, BinaryOperation op, acc_type init)
{
    if(first == last)
        return d_first;

    acc_type sum = init;
    *d_first     = sum;

    while((first + 1) != last)
    {
        ++flags;
        sum        = *flags ? acc_type(init) : acc_type(op(sum, *first));
        *++d_first = sum;
        first++;
    }
    return ++d_first;
}

template<bool UseInitialValue = false,
         class InputIt,
         class OutputIt,
         class BinaryOperation,
         class acc_type>
OutputIt host_inclusive_scan_impl(
    InputIt first, InputIt last, OutputIt d_first, BinaryOperation op, acc_type initial_value)
{
    if(first == last)
        return d_first;

    acc_type sum = UseInitialValue ? op(initial_value, *first) : static_cast<acc_type>(*first);
    *d_first     = sum;

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
    using acc_type = ::rocprim::accumulator_t<BinaryOperation,
                                              typename std::iterator_traits<InputIt>::value_type>;
    return host_inclusive_scan_impl(first, last, d_first, op, acc_type{});
}

template<class InputIt, class OutputIt, class BinaryOperation, class InitValueType>
OutputIt host_inclusive_scan(
    InputIt first, InputIt last, OutputIt d_first, BinaryOperation op, InitValueType initial_value)
{
    return host_inclusive_scan_impl<true>(first, last, d_first, op, initial_value);
}

template<
    class InputIt,
    class OutputIt,
    class T,
    std::enable_if_t<
        std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value
            || std::is_same<typename std::iterator_traits<InputIt>::value_type,
                            rocprim::half>::value
            || std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
        bool>
    = true>
OutputIt host_inclusive_scan(InputIt first, InputIt last, OutputIt d_first, rocprim::plus<T>)
{
    using acc_type = double;
    return host_inclusive_scan_impl(first, last, d_first, rocprim::plus<acc_type>(), acc_type{});
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
    using acc_type = ::rocprim::accumulator_t<BinaryOperation, rocprim::detail::input_type_t<T>>;
    return host_exclusive_scan_impl(first, last, initial_value, d_first, op, acc_type{});
}

template<
    class InputIt,
    class T,
    class OutputIt,
    class U,
    std::enable_if_t<
        std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value
            || std::is_same<typename std::iterator_traits<InputIt>::value_type,
                            rocprim::half>::value
            || std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
        bool>
    = true>
OutputIt host_exclusive_scan(
    InputIt first, InputIt last, T initial_value, OutputIt d_first, rocprim::plus<U>)
{
    using acc_type = double;
    return host_exclusive_scan_impl(first,
                                    last,
                                    initial_value,
                                    d_first,
                                    rocprim::plus<acc_type>(),
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

template<
    class InputIt,
    class KeyIt,
    class T,
    class OutputIt,
    class U,
    class KeyCompare,
    std::enable_if_t<
        std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value
            || std::is_same<typename std::iterator_traits<InputIt>::value_type,
                            rocprim::half>::value
            || std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
        bool>
    = true>
OutputIt host_exclusive_scan_by_key(InputIt  first,
                                    InputIt  last,
                                    KeyIt    k_first,
                                    T        initial_value,
                                    OutputIt d_first,
                                    rocprim::plus<U>,
                                    KeyCompare key_compare_op)
{
    using acc_type = double;
    return host_exclusive_scan_by_key_impl(first,
                                           last,
                                           k_first,
                                           initial_value,
                                           d_first,
                                           rocprim::plus<acc_type>(),
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

template<
    class InputIt,
    class KeyIt,
    class OutputIt,
    class U,
    class KeyCompare,
    std::enable_if_t<
        std::is_same<typename std::iterator_traits<InputIt>::value_type, rocprim::bfloat16>::value
            || std::is_same<typename std::iterator_traits<InputIt>::value_type,
                            rocprim::half>::value
            || std::is_same<typename std::iterator_traits<InputIt>::value_type, float>::value,
        bool>
    = true>
OutputIt host_inclusive_scan_by_key(InputIt  first,
                                    InputIt  last,
                                    KeyIt    k_first,
                                    OutputIt d_first,
                                    rocprim::plus<U>,
                                    KeyCompare key_compare_op)
{
    using acc_type = double;
    return host_inclusive_scan_by_key_impl(first,
                                           last,
                                           k_first,
                                           d_first,
                                           rocprim::plus<acc_type>(),
                                           key_compare_op,
                                           acc_type{});
}

inline size_t get_max_block_size()
{

    int max_threads_blocks{};
    HIP_CHECK(hipDeviceGetAttribute(&max_threads_blocks, hipDeviceAttributeMaxThreadsPerBlock, 0));
    return static_cast<size_t>(max_threads_blocks);
}

// std::iota causes problems with __half and bfloat16 and common::custom_type because of a missing ++increment operator
template<class ForwardIt, class T>
void iota(ForwardIt first, ForwardIt last, T value)
{
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;
    while(first != last)
    {
        *first++ = static_cast<value_type>(value);
        ++value;
    }
}

// Like test_utils::iota but applies module 'ubound' to the values generated.
template<class ForwardIt,
         class T,
         typename std::enable_if<!std::is_same<typename std::iterator_traits<ForwardIt>::value_type,
                                               rocprim::half>::value,
                                 bool>::type
         = false>
void iota_modulo(ForwardIt first, ForwardIt last, T lbound, const size_t ubound)
{
    const T value_mod = static_cast<size_t>(lbound) < ubound ? lbound : 0;
    using value_type  = typename std::iterator_traits<ForwardIt>::value_type;

    for(T value = value_mod; first != last; value++, *first++)
    {
        if(static_cast<size_t>(value) >= ubound)
        {
            value = value_mod;
        }
        *first = static_cast<value_type>(value);
    }
}

// Necessary because for rocprim::half even though lbound < ubound it gets cast as a greater
// value, as precision is bigger for values closer to the maximum.
template<class ForwardIt,
         class T,
         typename std::enable_if<std::is_same<typename std::iterator_traits<ForwardIt>::value_type,
                                              rocprim::half>::value,
                                 bool>::type
         = true>
void iota_modulo(ForwardIt first, ForwardIt last, T lbound, const size_t ubound)
{
    const T value_mod = static_cast<size_t>(lbound) < ubound ? lbound : 0;
    using value_type  = rocprim::half;

    for(T value = value_mod; first != last; value++, *first++)
    {
        if(static_cast<float>(static_cast<value_type>(value)) >= ubound)
        {
            value = value_mod;
        }
        *first = static_cast<value_type>(value);
    }
}

#define SKIP_IF_UNSUPPORTED_WARP_SIZE(test_warp_size, device_id)                \
    {                                                                           \
        unsigned int host_warp_size;                                            \
        HIP_CHECK(::rocprim::host_warp_size(device_id, host_warp_size));        \
        if(host_warp_size < (test_warp_size))                                   \
        {                                                                       \
            GTEST_SKIP() << "Cannot run test of warp size " << (test_warp_size) \
                         << " on a device with warp size " << host_warp_size;   \
        }                                                                       \
    }

template<bool MakeConst, typename T>
inline auto wrap_in_const(T* ptr) -> typename std::enable_if_t<MakeConst, const T*>
{
    return ptr;
}

template<bool MakeConst, typename T>
inline auto wrap_in_const(T* ptr) -> typename std::enable_if_t<!MakeConst, T*>
{
    return ptr;
}

template<typename F>
inline auto test_kernel_wrapper(F func, hipStream_t stream, const bool use_graphs = false)
{
    // temp storage
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    HIP_CHECK(func(nullptr, temp_storage_size_bytes));

    // temp_storage_size_bytes must be >0
    ASSERT_GT(temp_storage_size_bytes, 0);

    // allocate temporary storage
    common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

    test_utils::GraphHelper gHelper;
    if(use_graphs)
    {
        gHelper.startStreamCapture(stream);
    }

    // Run
    HIP_CHECK(func(d_temp_storage.get(), temp_storage_size_bytes));

    if(use_graphs)
    {
        gHelper.createAndLaunchGraph(stream);
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if(use_graphs)
    {
        gHelper.cleanupGraphHelper();
    }
}

} // namespace test_utils

#endif // TEST_TEST_UTILS_HPP_
