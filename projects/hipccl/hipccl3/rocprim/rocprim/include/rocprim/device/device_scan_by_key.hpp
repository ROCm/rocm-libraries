// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SCAN_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DEVICE_SCAN_BY_KEY_HPP_

#include "../config.hpp"
#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../types/future_value.hpp"
#include "../types/tuple.hpp"
#include "config_types.hpp"
#include "detail/config/device_scan_by_key.hpp"
#include "detail/device_config_helper.hpp"
#include "detail/device_scan_by_key.hpp"
#include "detail/lookback_scan_state.hpp"
#include "device_scan_by_key_config.hpp"

#include <hip/hip_runtime.h>

#include <iostream>
#include <iterator>
#include <type_traits>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<bool Exclusive,
         typename Config,
         typename KeyInputIterator,
         typename InputIterator,
         typename OutputIterator,
         typename InitialValueType,
         typename CompareFunction,
         typename BinaryFunction,
         typename LookbackScanState,
         typename ResultType>
void __global__ __launch_bounds__(device_params<Config>().kernel_config.block_size)
    device_scan_by_key_kernel(const KeyInputIterator                          keys,
                              const InputIterator                             values,
                              const OutputIterator                            output,
                              const InitialValueType                          initial_value,
                              const CompareFunction                           compare,
                              const BinaryFunction                            scan_op,
                              const LookbackScanState                         scan_state,
                              const size_t                                    size,
                              const size_t                                    starting_block,
                              const size_t                                    number_of_blocks,
                              const ::rocprim::tuple<ResultType, bool>* const previous_last_value)
{
    device_scan_by_key_kernel_impl<Exclusive, Config>(keys,
                                                      values,
                                                      output,
                                                      get_input_value(initial_value),
                                                      compare,
                                                      scan_op,
                                                      scan_state,
                                                      size,
                                                      starting_block,
                                                      number_of_blocks,
                                                      previous_last_value);
}

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    do                                                                                           \
    {                                                                                            \
        auto _error = hipGetLastError();                                                         \
        if(_error != hipSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            auto __error = hipStreamSynchronize(stream);                                         \
            if(__error != hipSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::high_resolution_clock::now();                               \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    } while(false)

template<bool Exclusive,
         typename Config,
         typename KeysInputIterator,
         typename InputIterator,
         typename OutputIterator,
         typename InitValueType,
         typename BinaryFunction,
         typename CompareFunction>
inline hipError_t scan_by_key_impl(void* const           temporary_storage,
                                   size_t&               storage_size,
                                   KeysInputIterator     keys,
                                   InputIterator         input,
                                   OutputIterator        output,
                                   const InitValueType   initial_value,
                                   const size_t          size,
                                   const BinaryFunction  scan_op,
                                   const CompareFunction compare,
                                   const hipStream_t     stream,
                                   const bool            debug_synchronous)
{
    using key_type             = typename std::iterator_traits<KeysInputIterator>::value_type;
    using real_init_value_type = input_type_t<InitValueType>;

    using config = wrapped_scan_by_key_config<Config, key_type, real_init_value_type>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const scan_by_key_config_params params = dispatch_target_arch<config>(target_arch);

    using wrapped_type = ::rocprim::tuple<real_init_value_type, bool>;

    using scan_state_type            = detail::lookback_scan_state<wrapped_type>;
    using scan_state_with_sleep_type = detail::lookback_scan_state<wrapped_type, true>;

    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const unsigned int items_per_block  = block_size * items_per_thread;

    const unsigned int size_limit = params.kernel_config.size_limit;
    const unsigned int aligned_size_limit
        = std::max(size_limit - size_limit % items_per_block, items_per_block);

    const unsigned int limited_size
        = static_cast<unsigned int>(std::min<size_t>(size, aligned_size_limit));
    const bool use_limited_size = limited_size == aligned_size_limit;

    // Number of blocks in a single launch (or the only launch if it fits)
    const unsigned int number_of_blocks = ceiling_div(limited_size, items_per_block);

    void*         scan_state_storage;
    wrapped_type* previous_last_value;

    const hipError_t partition_result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            // This is valid even with offset_scan_state_with_sleep_type
            detail::temp_storage::make_partition(
                &scan_state_storage,
                scan_state_type::get_temp_storage_layout(number_of_blocks)),
            detail::temp_storage::ptr_aligned_array(&previous_last_value,
                                                    use_limited_size ? 1 : 0)));
    if(partition_result != hipSuccess || temporary_storage == nullptr)
    {
        return partition_result;
    }

    if(number_of_blocks == 0u)
    {
        return hipSuccess;
    }

    bool use_sleep;
    if(const hipError_t error = is_sleep_scan_state_used(use_sleep))
    {
        return error;
    }

    // Call the provided function with either scan_state or scan_state_with_sleep based on
    // the value of use_sleep_scan_state
    auto with_scan_state
        = [use_sleep,
           scan_state = scan_state_type::create(scan_state_storage, number_of_blocks),
           scan_state_with_sleep
           = scan_state_with_sleep_type::create(scan_state_storage, number_of_blocks)](
              auto&& func) mutable -> decltype(auto)
    {
        if(use_sleep)
        {
            return func(scan_state_with_sleep);
        }
        else
        {
            return func(scan_state);
        }
    };

    // Total number of blocks in all launches
    const auto   total_number_of_blocks = ceiling_div(size, items_per_block);
    const size_t number_of_launch       = ceiling_div(size, limited_size);

    if(debug_synchronous)
    {
        std::cout << "----------------------------------\n";
        std::cout << "size:               " << size << '\n';
        std::cout << "aligned_size_limit: " << aligned_size_limit << '\n';
        std::cout << "use_limited_size:   " << std::boolalpha << use_limited_size << '\n';
        std::cout << "number_of_launch:   " << number_of_launch << '\n';
        std::cout << "block_size:         " << block_size << '\n';
        std::cout << "items_per_block:    " << items_per_block << '\n';
        std::cout << "----------------------------------\n";
    }

    for(size_t i = 0, offset = 0; i < number_of_launch; i++, offset += limited_size)
    {
        // limited_size is of type unsigned int, so current_size also fits in an unsigned int
        // size_t is necessary as type of std::min because 'size - offset' can exceed the
        // upper limit of unsigned int and converting it can lead to wrong results
        const unsigned int current_size
            = static_cast<unsigned int>(std::min<size_t>(size - offset, limited_size));
        const unsigned int scan_blocks    = ceiling_div(current_size, items_per_block);
        const unsigned int init_grid_size = ceiling_div(scan_blocks, block_size);

        // Start point for time measurements
        std::chrono::high_resolution_clock::time_point start;
        if(debug_synchronous)
        {
            std::cout << "index:            " << i << '\n';
            std::cout << "current_size:     " << current_size << '\n';
            std::cout << "number of blocks: " << scan_blocks << '\n';

            start = std::chrono::high_resolution_clock::now();
        }

        with_scan_state(
            [&](const auto scan_state)
            {
                hipLaunchKernelGGL(init_lookback_scan_state_kernel,
                                   dim3(init_grid_size),
                                   dim3(block_size),
                                   0,
                                   stream,
                                   scan_state,
                                   scan_blocks,
                                   number_of_blocks - 1,
                                   i > 0 ? previous_last_value : nullptr);
            });
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_lookback_scan_state_kernel",
                                                    scan_blocks,
                                                    start);

        if(debug_synchronous)
        {
            start = std::chrono::high_resolution_clock::now();
        }
        with_scan_state(
            [&](auto& scan_state)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(device_scan_by_key_kernel<Exclusive, config>),
                                   dim3(scan_blocks),
                                   dim3(block_size),
                                   0,
                                   stream,
                                   keys + offset,
                                   input + offset,
                                   output + offset,
                                   static_cast<real_init_value_type>(initial_value),
                                   compare,
                                   scan_op,
                                   scan_state,
                                   size,
                                   i * number_of_blocks,
                                   total_number_of_blocks,
                                   i > 0 ? as_const_ptr(previous_last_value) : nullptr);
            });
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("device_scan_by_key_kernel",
                                                    current_size,
                                                    start);
    }
    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
}

/// \addtogroup devicemodule
/// @{

/// \brief Parallel inclusive scan-by-key primitive for device level.
///
/// inclusive_scan_by_key function performs a device-wide inclusive prefix scan-by-key
/// operation using binary \p scan_op operator.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p keys_input, \p values_input, and \p values_output must have
/// at least \p size elements.
///
/// \tparam Config - [optional] configuration of the primitive, should be \p scan_by_key_config_v2.
/// \tparam KeysInputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for scan. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
/// \tparam KeyCompareFunction - type of binary function used to determine keys equality. Default type
/// is \p rocprim::equal_to<T>, where \p T is a \p value_type of \p KeysInputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - iterator to the first element in the range of keys.
/// \param [in] values_input - iterator to the first element in the range of values to scan.
/// \param [out] values_output - iterator to the first element in the output value range.
/// \param [in] size - number of element in the input range.
/// \param [in] scan_op - binary operation function object that will be used for scanning
/// input values.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] key_compare_op - binary operation function object that will be used to determine keys equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is KeyCompareFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level inclusive sum-by-key operation is performed on an array of
/// integer values (<tt>short</tt>s are scanned into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;           // e.g., 8
/// int *   keys_input;    // e.g., [1, 1, 2, 2, 3, 3, 3, 5]
/// short * values_input;  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int *   values_output; // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::inclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, size,
///     rocprim::plus<int>()
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan-by-key
/// rocprim::inclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, size,
///     rocprim::plus<int>()
/// );
/// // values_output: [1, 3, 3, 7, 5, 11, 18, 8]
/// \endcode
/// \endparblock
template<typename Config = default_config,
         typename KeysInputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
         typename KeyCompareFunction
         = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>>
inline hipError_t inclusive_scan_by_key(void* const                temporary_storage,
                                        size_t&                    storage_size,
                                        const KeysInputIterator    keys_input,
                                        const ValuesInputIterator  values_input,
                                        const ValuesOutputIterator values_output,
                                        const size_t               size,
                                        const BinaryFunction       scan_op = BinaryFunction(),
                                        const KeyCompareFunction   key_compare_op
                                        = KeyCompareFunction(),
                                        const hipStream_t stream            = 0,
                                        const bool        debug_synchronous = false)
{
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    return detail::scan_by_key_impl<false, Config>(temporary_storage,
                                                   storage_size,
                                                   keys_input,
                                                   values_input,
                                                   values_output,
                                                   value_type(),
                                                   size,
                                                   scan_op,
                                                   key_compare_op,
                                                   stream,
                                                   debug_synchronous);
}

/// \brief Parallel exclusive scan-by-key primitive for device level.
///
/// inclusive_scan_by_key function performs a device-wide exclusive prefix scan-by-key
/// operation using binary \p scan_op operator.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p keys_input, \p values_input, and \p values_output must have
/// at least \p size elements.
///
/// \tparam Config - [optional] configuration of the primitive, should be \p scan_by_key_config_v2.
/// \tparam KeysInputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam InitValueType - type of the initial value.
/// \tparam BinaryFunction - type of binary function used for scan. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
/// \tparam KeyCompareFunction - type of binary function used to determine keys equality. Default type
/// is \p rocprim::equal_to<T>, where \p T is a \p value_type of \p KeysInputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - iterator to the first element in the range of keys.
/// \param [in] values_input - iterator to the first element in the range of values to scan.
/// \param [out] values_output - iterator to the first element in the output value range.
/// \param [in] initial_value - initial value to start the scan.
/// A rocpim::future_value may be passed to use a value that will be later computed.
/// \param [in] size - number of element in the input range.
/// \param [in] scan_op - binary operation function object that will be used for scanning
/// input values.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] key_compare_op - binary operation function object that will be used to determine keys equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is KeyCompareFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level inclusive sum-by-key operation is performed on an array of
/// integer values (<tt>short</tt>s are scanned into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;           // e.g., 8
/// int *   keys_input;    // e.g., [1, 1, 1, 2, 2, 3, 3, 4]
/// short * values_input;  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int start_value;       // e.g., 9
/// int *   values_output; // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::exclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, start_value,
///     size,rocprim::plus<int>()
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan-by-key
/// rocprim::exclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, start_value,
///     size,rocprim::plus<int>()
/// );
/// // values_output: [9, 10, 12, 9, 13, 9, 15, 9]
/// \endcode
/// \endparblock
template<typename Config = default_config,
         typename KeysInputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename InitialValueType,
         typename BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
         typename KeyCompareFunction
         = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>>
inline hipError_t exclusive_scan_by_key(void* const                temporary_storage,
                                        size_t&                    storage_size,
                                        const KeysInputIterator    keys_input,
                                        const ValuesInputIterator  values_input,
                                        const ValuesOutputIterator values_output,
                                        const InitialValueType     initial_value,
                                        const size_t               size,
                                        const BinaryFunction       scan_op = BinaryFunction(),
                                        const KeyCompareFunction   key_compare_op
                                        = KeyCompareFunction(),
                                        const hipStream_t stream            = 0,
                                        const bool        debug_synchronous = false)
{
    return detail::scan_by_key_impl<true, Config>(temporary_storage,
                                                  storage_size,
                                                  keys_input,
                                                  values_input,
                                                  values_output,
                                                  initial_value,
                                                  size,
                                                  scan_op,
                                                  key_compare_op,
                                                  stream,
                                                  debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SCAN_BY_KEY_HPP_
