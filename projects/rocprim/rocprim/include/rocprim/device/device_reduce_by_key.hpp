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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_

#include "config_types.hpp"
#include "device_reduce_by_key_config.hpp"
#include "device_transform.hpp"

#include "detail/device_config_helper.hpp"
#include "detail/device_reduce_by_key.hpp"
#include "detail/device_scan_common.hpp"
#include "detail/lookback_scan_state.hpp"

#include "../common.hpp"
#include "../config.hpp"
#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../intrinsics/thread.hpp"
#include "../iterator/constant_iterator.hpp"
#include "../type_traits.hpp"

#include <chrono>
#include <iostream>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<typename LookBackScanState, typename AccumulatorType>
ROCPRIM_KERNEL ROCPRIM_LAUNCH_BOUNDS(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE) void
    reduce_by_key_init_kernel(LookBackScanState      lookback_scan_state,
                              const unsigned int     number_of_blocks,
                              const bool             is_first_launch,
                              const unsigned int     block_save_idx,
                              std::size_t* const     global_head_count,
                              AccumulatorType* const previous_accumulated)
{
    const unsigned int block_id        = ::rocprim::detail::block_id<0>();
    const unsigned int block_size      = ::rocprim::detail::block_size<0>();
    const unsigned int block_thread_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_thread_id  = (block_id * block_size) + block_thread_id;

    if(is_first_launch)
    {
        if(global_head_count != nullptr && flat_thread_id == 0)
        {
            // If there are subsequent launches, initialize the accumulated head flags
            // over previous launches to zero.
            *global_head_count = 0;
        }
    }
    else
    {
        // Use the reduction of the last launch to update the across-launch variables.
        const auto update_func = [&](typename LookBackScanState::value_type value)
        {
            *global_head_count += ::rocprim::get<0>(value);
            *previous_accumulated = ::rocprim::get<1>(value);
        };
        access_indexed_lookback_value(lookback_scan_state,
                                      number_of_blocks,
                                      block_save_idx,
                                      flat_thread_id,
                                      update_func);
    }

    init_lookback_scan_state(lookback_scan_state, number_of_blocks, flat_thread_id);
}

template<lookback_scan_determinism Determinism,
         typename Config,
         typename AccumulatorType,
         typename KeyIterator,
         typename ValueIterator,
         typename UniqueIterator,
         typename ReductionIterator,
         typename UniqueCountIterator,
         typename CompareFunction,
         typename BinaryOp,
         typename LookbackScanState>
ROCPRIM_KERNEL ROCPRIM_LAUNCH_BOUNDS(device_params<Config>().kernel_config.block_size) void
    reduce_by_key_kernel(const KeyIterator            keys_input,
                         const ValueIterator          values_input,
                         const UniqueIterator         unique_keys,
                         const ReductionIterator      reductions,
                         const UniqueCountIterator    unique_count,
                         const BinaryOp               reduce_op,
                         const CompareFunction        compare,
                         const LookbackScanState      scan_state,
                         const std::size_t            starting_block,
                         const std::size_t            total_number_of_blocks,
                         const std::size_t            size,
                         const std::size_t* const     global_head_count,
                         const AccumulatorType* const previous_accumulated)
{
    reduce_by_key::kernel_impl<Determinism, Config>(keys_input,
                                                    values_input,
                                                    unique_keys,
                                                    reductions,
                                                    unique_count,
                                                    reduce_op,
                                                    compare,
                                                    scan_state,
                                                    starting_block,
                                                    total_number_of_blocks,
                                                    size,
                                                    global_head_count,
                                                    previous_accumulated);
}

template<lookback_scan_determinism Determinism,
         typename config,
         typename KeysInputIterator,
         typename ValuesInputIterator,
         typename UniqueOutputIterator,
         typename AggregatesOutputIterator,
         typename UniqueCountOutputIterator,
         typename BinaryFunction,
         typename KeyCompareFunction>
hipError_t reduce_by_key_impl_wrapped_config(void*                     temporary_storage,
                                             size_t&                   storage_size,
                                             KeysInputIterator         keys_input,
                                             ValuesInputIterator       values_input,
                                             const size_t              size,
                                             UniqueOutputIterator      unique_output,
                                             AggregatesOutputIterator  aggregates_output,
                                             UniqueCountOutputIterator unique_count_output,
                                             BinaryFunction            reduce_op,
                                             KeyCompareFunction        key_compare_op,
                                             const hipStream_t         stream,
                                             const bool                debug_synchronous)
{
    using accumulator_type = reduce_by_key::accumulator_type_t<ValuesInputIterator, BinaryFunction>;
    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const reduce_by_key_config_params params = dispatch_target_arch<config>(target_arch);

    using scan_state_type
        = reduce_by_key::lookback_scan_state_t<accumulator_type, /*UseSleep=*/false>;
    using scan_state_with_sleep_type
        = reduce_by_key::lookback_scan_state_t<accumulator_type, /*UseSleep=*/true>;

    const unsigned int block_size      = params.kernel_config.block_size;
    const unsigned int items_per_block = block_size * params.kernel_config.items_per_thread;

    const size_t size_limit = params.kernel_config.size_limit;
    const size_t aligned_size_limit
        = ::rocprim::max<size_t>(size_limit - size_limit % items_per_block, items_per_block);

    const size_t limited_size     = std::min<size_t>(size, aligned_size_limit);
    const bool   use_limited_size = limited_size == aligned_size_limit;

    // Number of blocks in a single launch (or the only launch if it fits)
    const std::size_t number_of_blocks = detail::ceiling_div(limited_size, items_per_block);

    // Calculate required temporary storage
    void* scan_state_storage;
    // The number of segment heads in previous launches.
    std::size_t* d_global_head_count = nullptr;
    // The running accumulation across the launch boundary.
    accumulator_type* d_previous_accumulated = nullptr;

    detail::temp_storage::layout layout{};
    result = scan_state_type::get_temp_storage_layout(number_of_blocks, stream, layout);
    if(result != hipSuccess)
    {
        return result;
    }

    result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            // This is valid even with scan_state_with_sleep_type
            detail::temp_storage::make_partition(&scan_state_storage, layout),
            detail::temp_storage::ptr_aligned_array(&d_global_head_count, use_limited_size ? 1 : 0),
            detail::temp_storage::ptr_aligned_array(&d_previous_accumulated,
                                                    use_limited_size ? 1 : 0)));
    if(result != hipSuccess || temporary_storage == nullptr)
    {
        return result;
    }

    bool use_sleep;
    result = detail::is_sleep_scan_state_used(stream, use_sleep);
    if(result != hipSuccess)
    {
        return result;
    }
    scan_state_type scan_state{};
    result = scan_state_type::create(scan_state, scan_state_storage, number_of_blocks, stream);
    if(result != hipSuccess)
    {
        return result;
    }
    scan_state_with_sleep_type scan_state_with_sleep{};
    result = scan_state_with_sleep_type::create(scan_state_with_sleep,
                                                scan_state_storage,
                                                number_of_blocks,
                                                stream);
    if(result != hipSuccess)
    {
        return result;
    }

    auto with_scan_state
        = [use_sleep, scan_state, scan_state_with_sleep](auto&& func) mutable -> decltype(auto)
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

    if(size == 0)
    {
        // Fill out unique_count_output with zero
        return rocprim::transform(rocprim::constant_iterator<std::size_t>(0),
                                  unique_count_output,
                                  1,
                                  rocprim::identity<std::size_t>{},
                                  stream,
                                  debug_synchronous);
    }

    // Total number of blocks in all launches
    const std::size_t total_number_of_blocks = ceiling_div(size, items_per_block);
    const std::size_t number_of_launch       = ceiling_div(size, limited_size);

    if(debug_synchronous)
    {
        std::cout << "size:               " << size << '\n';
        std::cout << "aligned_size_limit: " << aligned_size_limit << '\n';
        std::cout << "use_limited_size:   " << std::boolalpha << use_limited_size << '\n';
        std::cout << "number_of_launch:   " << number_of_launch << '\n';
        std::cout << "block_size:         " << block_size << '\n';
        std::cout << "number_of_blocks:   " << number_of_blocks << '\n';
        std::cout << "items_per_block:    " << items_per_block << '\n';
    }

    for(size_t i = 0, offset = 0; i < number_of_launch; ++i, offset += limited_size)
    {
        const std::size_t current_size = std::min<std::size_t>(size - offset, limited_size);
        const std::size_t number_of_blocks_launch = ceiling_div(current_size, items_per_block);

        // Start point for time measurements
        std::chrono::steady_clock::time_point start;
        if(debug_synchronous)
        {
            std::cout << "index:            " << i << '\n';
            std::cout << "current_size:     " << current_size << '\n';
            std::cout << "number of blocks: " << number_of_blocks_launch << '\n';

            start = std::chrono::steady_clock::now();
        }

        with_scan_state(
            [&](const auto scan_state)
            {
                const unsigned int block_size = ROCPRIM_DEFAULT_MAX_BLOCK_SIZE;
                const std::size_t  grid_size
                    = detail::ceiling_div(number_of_blocks_launch, block_size);

                reduce_by_key_init_kernel<<<dim3(grid_size), dim3(block_size), 0, stream>>>(
                    scan_state,
                    number_of_blocks_launch,
                    i == 0,
                    number_of_blocks - 1,
                    d_global_head_count,
                    d_previous_accumulated);
            });
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("reduce_by_key_init_kernel",
                                                    number_of_blocks_launch,
                                                    start);

        with_scan_state(
            [&](const auto scan_state)
            {
                reduce_by_key_kernel<Determinism, config>
                    <<<dim3(number_of_blocks_launch), dim3(block_size), 0, stream>>>(
                        keys_input + offset,
                        values_input + offset,
                        unique_output,
                        aggregates_output,
                        unique_count_output,
                        reduce_op,
                        key_compare_op,
                        scan_state,
                        i * number_of_blocks,
                        total_number_of_blocks,
                        size,
                        i > 0 ? d_global_head_count : nullptr,
                        i > 0 ? d_previous_accumulated : nullptr);
            });
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("reduce_by_key_kernel", current_size, start);
    }

    return hipSuccess;
}

template<lookback_scan_determinism Determinism,
         typename Config,
         typename KeysInputIterator,
         typename ValuesInputIterator,
         typename UniqueOutputIterator,
         typename AggregatesOutputIterator,
         typename UniqueCountOutputIterator,
         typename BinaryFunction,
         typename KeyCompareFunction>
hipError_t reduce_by_key_impl(void*                     temporary_storage,
                              size_t&                   storage_size,
                              KeysInputIterator         keys_input,
                              ValuesInputIterator       values_input,
                              const size_t              size,
                              UniqueOutputIterator      unique_output,
                              AggregatesOutputIterator  aggregates_output,
                              UniqueCountOutputIterator unique_count_output,
                              BinaryFunction            reduce_op,
                              KeyCompareFunction        key_compare_op,
                              const hipStream_t         stream,
                              const bool                debug_synchronous)
{
    using key_type         = ::rocprim::detail::value_type_t<KeysInputIterator>;
    using accumulator_type = reduce_by_key::accumulator_type_t<ValuesInputIterator, BinaryFunction>;

    using config = wrapped_reduce_by_key_config<Config, key_type, accumulator_type, BinaryFunction>;

    return detail::reduce_by_key_impl_wrapped_config<Determinism, config>(temporary_storage,
                                                                          storage_size,
                                                                          keys_input,
                                                                          values_input,
                                                                          size,
                                                                          unique_output,
                                                                          aggregates_output,
                                                                          unique_count_output,
                                                                          reduce_op,
                                                                          key_compare_op,
                                                                          stream,
                                                                          debug_synchronous);
}

} // namespace detail

/// \brief Parallel reduce-by-key primitive for device level.
///
/// reduce_by_key function performs a device-wide reduction operation on groups
/// of consecutive values having the same key using binary \p reduce_op operator. The first key of each group
/// is copied to \p unique_output and the reduction of the group is written to \p aggregates_output.
/// The total number of groups is written to \p unique_count_output.
///
/// \par Overview
/// * Supports non-commutative reduction operators. However, a reduction operator should be
/// associative.
/// * When used with non-associative functions (e.g. floating point arithmetic operations):
///    - the results may be non-deterministic and/or vary in precision,
///    - and bit-wise reproducibility is not guaranteed, that is, results from multiple runs
///      using the same input values on the same device may not be bit-wise identical.
///   If deterministic behavior is required, Use \link deterministic_reduce_by_key()
///   rocprim::deterministic_reduce_by_key \endlink instead.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p keys_input and \p values_input must have at least \p size elements.
/// * Range specified by \p unique_count_output must have at least 1 element.
/// * Ranges specified by \p unique_output and \p aggregates_output must have at least
/// <tt>*unique_count_output</tt> (i.e. the number of unique keys) elements.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `reduce_by_key_config`.
/// \tparam KeysInputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam UniqueOutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam AggregatesOutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam UniqueCountOutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction type of binary function used for reduction. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p ValuesInputIterator.
/// \tparam KeyCompareFunction type of binary function used to determine keys equality. Default type
/// is \p rocprim::equal_to<T>, where \p T is a \p value_type of \p KeysInputIterator.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input iterator to the first element in the range of keys.
/// \param [in] values_input iterator to the first element in the range of values to reduce.
/// \param [in] size number of element in the input range.
/// \param [out] unique_output iterator to the first element in the output range of unique keys.
/// \param [out] aggregates_output iterator to the first element in the output range of reductions.
/// \param [out] unique_count_output iterator to total number of groups.
/// \param [in] reduce_op binary operation function object that will be used for reduction.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it and must not have
/// any side effects since the function may be called on uninitalized data.
/// Default is BinaryFunction().
/// \param [in] key_compare_op binary operation function object that will be used to determine key equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it and must not have
/// any side effects since the function may be called on uninitalized data.
/// Default is KeyCompareFunction().
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful reduction; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level sum operation is performed on an array of
/// integer values and integer keys.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// int * keys_input;           // e.g., [1, 1, 1, 2, 10, 10, 10, 88]
/// int * values_input;         // e.g., [1, 2, 3, 4,  5,  6,  7,  8]
/// int * unique_output;        // empty array of at least 4 elements
/// int * aggregates_output;    // empty array of at least 4 elements
/// int * unique_count_output;  // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::reduce_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input, input_size,
///     unique_output, aggregates_output, unique_count_output
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform reduction
/// rocprim::reduce_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input, input_size,
///     unique_output, aggregates_output, unique_count_output
/// );
/// // unique_output:       [1, 2, 10, 88]
/// // aggregates_output:   [6, 4, 18,  8]
/// // unique_count_output: [4]
/// \endcode
/// \endparblock
template<typename Config = default_config,
         typename KeysInputIterator,
         typename ValuesInputIterator,
         typename UniqueOutputIterator,
         typename AggregatesOutputIterator,
         typename UniqueCountOutputIterator,
         typename BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
         typename KeyCompareFunction
         = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>>
inline hipError_t reduce_by_key(void*                     temporary_storage,
                                size_t&                   storage_size,
                                KeysInputIterator         keys_input,
                                ValuesInputIterator       values_input,
                                const size_t              size,
                                UniqueOutputIterator      unique_output,
                                AggregatesOutputIterator  aggregates_output,
                                UniqueCountOutputIterator unique_count_output,
                                BinaryFunction            reduce_op         = BinaryFunction(),
                                KeyCompareFunction        key_compare_op    = KeyCompareFunction(),
                                hipStream_t               stream            = 0,
                                bool                      debug_synchronous = false)
{
    return detail::reduce_by_key_impl<detail::lookback_scan_determinism::default_determinism,
                                      Config>(temporary_storage,
                                              storage_size,
                                              keys_input,
                                              values_input,
                                              size,
                                              unique_output,
                                              aggregates_output,
                                              unique_count_output,
                                              reduce_op,
                                              key_compare_op,
                                              stream,
                                              debug_synchronous);
}

/// \brief Bitwise-reproducible parallel reduce-by-key primitive for device level.
///
/// This function behaves the same as <tt>reduce_by_key()</tt>, except that unlike
/// <tt>reduce_by_key()</tt>, it provides run-to-run deterministic behavior for
/// non-associative scan operators like floating point arithmetic operations.
/// Refer to the documentation for \link reduce_by_key() rocprim::reduce_by_key \endlink
/// for a detailed description of this function.
template<typename Config = default_config,
         typename KeysInputIterator,
         typename ValuesInputIterator,
         typename UniqueOutputIterator,
         typename AggregatesOutputIterator,
         typename UniqueCountOutputIterator,
         typename BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
         typename KeyCompareFunction
         = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>>
inline hipError_t deterministic_reduce_by_key(void*                     temporary_storage,
                                              size_t&                   storage_size,
                                              KeysInputIterator         keys_input,
                                              ValuesInputIterator       values_input,
                                              const size_t              size,
                                              UniqueOutputIterator      unique_output,
                                              AggregatesOutputIterator  aggregates_output,
                                              UniqueCountOutputIterator unique_count_output,
                                              BinaryFunction     reduce_op = BinaryFunction(),
                                              KeyCompareFunction key_compare_op
                                              = KeyCompareFunction(),
                                              hipStream_t stream            = 0,
                                              bool        debug_synchronous = false)
{
    return detail::reduce_by_key_impl<detail::lookback_scan_determinism::deterministic, Config>(
        temporary_storage,
        storage_size,
        keys_input,
        values_input,
        size,
        unique_output,
        aggregates_output,
        unique_count_output,
        reduce_op,
        key_compare_op,
        stream,
        debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_
