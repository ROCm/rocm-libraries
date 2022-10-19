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

#ifndef ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_
#define ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_

#include <algorithm>
#include <iostream>
#include <iterator>
#include <type_traits>

#include "../config.hpp"
#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../type_traits.hpp"
#include "../types.hpp"

#include "device_select_config.hpp"
#include "detail/device_scan_common.hpp"
#include "detail/device_partition.hpp"
#include "device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<select_method SelectMethod,
         bool          OnlySelected,
         class Config,
         class KeyIterator,
         class ValueIterator,
         class FlagIterator,
         class OutputKeyIterator,
         class OutputValueIterator,
         class InequalityOp,
         class OffsetLookbackScanState,
         class... UnaryPredicates>
ROCPRIM_KERNEL __launch_bounds__(Config::block_size) void partition_kernel(
    KeyIterator                    keys_input,
    ValueIterator                  values_input,
    FlagIterator                   flags,
    OutputKeyIterator              keys_output,
    OutputValueIterator            values_output,
    size_t*                        selected_count,
    size_t*                        prev_selected_count,
    size_t                         prev_processed,
    const size_t                   total_size,
    InequalityOp                   inequality_op,
    OffsetLookbackScanState        offset_scan_state,
    const unsigned int             number_of_blocks,
    ordered_block_id<unsigned int> ordered_bid,
    UnaryPredicates... predicates)
{
    partition_kernel_impl<SelectMethod, OnlySelected, Config>(keys_input,
                                                              values_input,
                                                              flags,
                                                              keys_output,
                                                              values_output,
                                                              selected_count,
                                                              prev_selected_count,
                                                              prev_processed,
                                                              total_size,
                                                              inequality_op,
                                                              offset_scan_state,
                                                              number_of_blocks,
                                                              ordered_bid,
                                                              predicates...);
}

#define ROCPRIM_DETAIL_HIP_SYNC(name, size, start) \
    if(debug_synchronous) \
    { \
        std::cout << name << "(" << size << ")"; \
        auto error = hipStreamSynchronize(stream); \
        if(error != hipSuccess) return error; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
        std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
    }

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
        auto _error = hipGetLastError(); \
        if(_error != hipSuccess) return _error; \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            auto __error = hipStreamSynchronize(stream); \
            if(__error != hipSuccess) return __error; \
            auto _end = std::chrono::high_resolution_clock::now(); \
            auto _d = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n'; \
        } \
    }

template<
    // Method of selection: flag, predicate, unique
    select_method SelectMethod,
     // if true, it doesn't copy rejected values to output
    bool OnlySelected,
    class Config,
    class OffsetT,
    class KeyIterator,
    class ValueIterator, // can be rocprim::empty_type* for key only
    class FlagIterator,
    class OutputKeyIterator,
    class OutputValueIterator, // can be rocprim::empty_type* for key only
    class InequalityOp,
    class SelectedCountOutputIterator,
    class... UnaryPredicates
>
inline
hipError_t partition_impl(void * temporary_storage,
                          size_t& storage_size,
                          KeyIterator keys_input,
                          ValueIterator values_input,
                          FlagIterator flags,
                          OutputKeyIterator keys_output,
                          OutputValueIterator values_output,
                          SelectedCountOutputIterator selected_count_output,
                          const size_t size,
                          InequalityOp inequality_op,
                          const hipStream_t stream,
                          bool debug_synchronous,
                          UnaryPredicates... predicates)
{
    using offset_type = OffsetT;
    using key_type = typename std::iterator_traits<KeyIterator>::value_type;
    using value_type = typename std::iterator_traits<ValueIterator>::value_type;

    // Get default config if Config is default_config
    using config = default_or_custom_config<
        Config,
        default_select_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    using offset_scan_state_type = detail::lookback_scan_state<offset_type>;
    using offset_scan_state_with_sleep_type = detail::lookback_scan_state<offset_type, true>;
    using ordered_block_id_type = detail::ordered_block_id<unsigned int>;


    static constexpr unsigned int block_size = config::block_size;
    static constexpr unsigned int items_per_thread = config::items_per_thread;
    static constexpr auto items_per_block = block_size * items_per_thread;

    static constexpr bool is_three_way = sizeof...(UnaryPredicates) == 2;
    static constexpr const size_t selected_count_size = is_three_way ? 2 : 1;

    static constexpr size_t size_limit = config::size_limit;
    static constexpr size_t aligned_size_limit = ::rocprim::max<size_t>(size_limit - (size_limit % items_per_block), items_per_block);
    const size_t limited_size = std::min<size_t>(size, aligned_size_limit);
    const bool use_limited_size = limited_size == aligned_size_limit;

    const unsigned int number_of_blocks
        = static_cast<unsigned int>(::rocprim::detail::ceiling_div(limited_size, items_per_block));

    // Calculate required temporary storage
    void*                           offset_scan_state_storage;
    ordered_block_id_type::id_type* ordered_bid_storage;
    size_t*                         selected_count;
    size_t*                         prev_selected_count;

    const hipError_t partition_result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            // This is valid even with offset_scan_state_with_sleep_type
            detail::temp_storage::make_partition(
                &offset_scan_state_storage,
                offset_scan_state_type::get_temp_storage_layout(number_of_blocks)),
            detail::temp_storage::make_partition(&ordered_bid_storage,
                                                 ordered_block_id_type::get_temp_storage_layout()),
            // Note: the following two are to be allocated continuously, so that they can be initialized
            // simultaneously.
            // They have the same base type, so there is no padding between the types.
            detail::temp_storage::ptr_aligned_array(&selected_count, selected_count_size),
            detail::temp_storage::ptr_aligned_array(&prev_selected_count, selected_count_size)));
    if(partition_result != hipSuccess || temporary_storage == nullptr)
    {
        return partition_result;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    // Create and initialize lookback_scan_state obj
    auto offset_scan_state
        = offset_scan_state_type::create(offset_scan_state_storage, number_of_blocks);
    auto offset_scan_state_with_sleep
        = offset_scan_state_with_sleep_type::create(offset_scan_state_storage, number_of_blocks);

    // Create and initialize ordered_block_id obj
    auto ordered_bid = ordered_block_id_type::create(ordered_bid_storage);

    hipError_t error;

    // Memset selected_count and prev_selected_count at once
    error = hipMemsetAsync(selected_count,
                           0,
                           sizeof(*selected_count) * 2 * selected_count_size,
                           stream);
    if (error != hipSuccess) return error;

    hipDeviceProp_t prop;
    int deviceId;
    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&prop, deviceId));

#if HIP_VERSION >= 307
    int asicRevision = prop.asicRevision;
#else
    int asicRevision = 0;
#endif

    const size_t number_of_launches = ::rocprim::detail::ceiling_div(size, aligned_size_limit);

    if(debug_synchronous)
    {
        std::cout << "use_limited_size " << use_limited_size << '\n';
        std::cout << "aligned_size_limit " << aligned_size_limit << '\n';
        std::cout << "number_of_launches " << number_of_launches << '\n';
        std::cout << "size " << size << '\n';
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    for(size_t i = 0, prev_processed = 0; i < number_of_launches;
        i++, prev_processed += limited_size)
    {
        const unsigned int current_size
            = static_cast<unsigned int>(std::min<size_t>(size - prev_processed, limited_size));

        const unsigned int current_number_of_blocks = ::rocprim::detail::ceiling_div(current_size, items_per_block);

        auto grid_size = ::rocprim::detail::ceiling_div(number_of_blocks, block_size);

        if(debug_synchronous)
        {
            std::cout << "current size " << current_size << '\n';
            std::cout << "current number of blocks " << current_number_of_blocks << '\n';

            start = std::chrono::high_resolution_clock::now();
        }

        if (prop.gcnArch == 908 && asicRevision < 2)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(init_lookback_scan_state_kernel<offset_scan_state_with_sleep_type>),
                dim3(grid_size), dim3(block_size), 0, stream,
                offset_scan_state_with_sleep, current_number_of_blocks, ordered_bid
            );
        } else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(init_lookback_scan_state_kernel<offset_scan_state_type>),
                dim3(grid_size), dim3(block_size), 0, stream,
                offset_scan_state, current_number_of_blocks, ordered_bid
            );
        }

        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_offset_scan_state_kernel", current_number_of_blocks, start)

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();

        grid_size = current_number_of_blocks;

        if (prop.gcnArch == 908 && asicRevision < 2)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(partition_kernel<SelectMethod, OnlySelected, config>),
                dim3(grid_size),
                dim3(block_size),
                0,
                stream,
                keys_input + prev_processed,
                values_input + prev_processed,
                flags + prev_processed,
                keys_output,
                values_output,
                selected_count,
                prev_selected_count,
                prev_processed,
                size,
                inequality_op,
                offset_scan_state_with_sleep,
                current_number_of_blocks,
                ordered_bid,
                predicates...);
        } else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(partition_kernel<SelectMethod, OnlySelected, config>),
                dim3(grid_size),
                dim3(block_size),
                0,
                stream,
                keys_input + prev_processed,
                values_input + prev_processed,
                flags + prev_processed,
                keys_output,
                values_output,
                selected_count,
                prev_selected_count,
                prev_processed,
                size,
                inequality_op,
                offset_scan_state,
                current_number_of_blocks,
                ordered_bid,
                predicates...);
        }

        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("partition_kernel", size, start)

        std::swap(selected_count, prev_selected_count);
    }

    error = ::rocprim::transform(prev_selected_count,
                                 selected_count_output,
                                 (is_three_way ? 2 : 1),
                                 ::rocprim::identity<>{},
                                 stream,
                                 debug_synchronous);
    if (error != hipSuccess) return error;

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

} // end of detail namespace

/// \brief Parallel select primitive for device level using range of flags.
///
/// Performs a device-wide partition based on input \p flags. Partition copies
/// the values from \p input to \p output in such a way that all values for which the corresponding
/// items from /p flags are \p true (or can be implicitly converted to \p true) precede
/// the elements for which the corresponding items from /p flags are \p false.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input, \p flags and \p output must have at least \p size elements.
/// * Range specified by \p selected_count_output must have at least 1 element.
/// * Values of \p flag range should be implicitly convertible to `bool` type.
/// * Relative order is preserved for the elements for which the corresponding values from \p flags
/// are \p true. Other elements are copied in reverse order.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p select_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam FlagIterator - random-access iterator type of the flag range. It can be
/// a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator - random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to select values from.
/// \param [in] flags - iterator to the selection flag corresponding to the first element from \p input range.
/// \param [out] output - iterator to the first element in the output range.
/// \param [out] selected_count_output - iterator to the total number of selected values (length of \p output).
/// \param [in] size - number of element in the input range.
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level partition operation is performed on an array of
/// integer values with array of <tt>char</tt>s used as flags.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;     // e.g., 8
/// int * input;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// char * flags;          // e.g., [0, 1, 1, 0, 0, 1, 0, 1]
/// int * output;          // empty array of 8 elements
/// size_t * output_count; // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::partition(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, flags,
///     output, output_count,
///     input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform partition
/// rocprim::partition(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, flags,
///     output, output_count,
///     input_size
/// );
/// // output: [2, 3, 6, 8, 7, 5, 4, 1]
/// // output_count: 4
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
inline
hipError_t partition(void * temporary_storage,
                     size_t& storage_size,
                     InputIterator input,
                     FlagIterator flags,
                     OutputIterator output,
                     SelectedCountOutputIterator selected_count_output,
                     const size_t size,
                     const hipStream_t stream = 0,
                     const bool debug_synchronous = false)
{
    // Dummy unary predicate
    using unary_predicate_type = ::rocprim::empty_type;
    // Dummy inequality operation
    using inequality_op_type = ::rocprim::empty_type;
    using offset_type = unsigned int;
    rocprim::empty_type* const no_values = nullptr; // key only

    return detail::partition_impl<detail::select_method::flag, false, Config, offset_type>(
        temporary_storage, storage_size, input, no_values, flags, output, no_values, selected_count_output,
        size, inequality_op_type(), stream, debug_synchronous, unary_predicate_type()
    );
}

/// \brief Parallel select primitive for device level using selection predicate.
///
/// Performs a device-wide partition using selection predicate. Partition copies
/// the values from \p input to \p output  in such a way that all values for which
/// the \p predicate returns \p true precede the elements for which it returns \p false.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input, \p flags and \p output must have at least \p size elements.
/// * Range specified by \p selected_count_output must have at least 1 element.
/// * Relative order is preserved for the elements for which the \p predicate returns \p true. Other
/// elements are copied in reverse order.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p select_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator - random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam UnaryPredicate - type of a unary selection predicate.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to select values from.
/// \param [out] output - iterator to the first element in the output range.
/// \param [out] selected_count_output - iterator to the total number of selected values (length of \p output).
/// \param [in] size - number of element in the input range.
/// \param [in] predicate - unary function object which returns /p true if the element should be
/// ordered before other elements.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level partition operation is performed on an array of
/// integer values, even values are copied before odd values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>///
///
/// auto predicate =
///     [] __device__ (int a) -> bool
///     {
///         return (a%2) == 0;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;     // e.g., 8
/// int * input;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * output;          // empty array of 8 elements
/// size_t * output_count; // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::partition(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input,
///     output, output_count,
///     input_size,
///     predicate
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform partition
/// rocprim::partition(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input,
///     output, output_count,
///     input_size,
///     predicate
/// );
/// // output: [2, 4, 6, 8, 7, 5, 3, 1]
/// // output_count: 4
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class UnaryPredicate
>
inline
hipError_t partition(void * temporary_storage,
                     size_t& storage_size,
                     InputIterator input,
                     OutputIterator output,
                     SelectedCountOutputIterator selected_count_output,
                     const size_t size,
                     UnaryPredicate predicate,
                     const hipStream_t stream = 0,
                     const bool debug_synchronous = false)
{
    // Dummy flag type
    using flag_type = ::rocprim::empty_type;
    flag_type * flags = nullptr;
    // Dummy inequality operation
    using inequality_op_type = ::rocprim::empty_type;
    using offset_type = unsigned int;
    rocprim::empty_type* const no_values = nullptr; // key only

    return detail::partition_impl<detail::select_method::predicate, false, Config, offset_type>(
        temporary_storage, storage_size, input, no_values, flags, output, no_values, selected_count_output,
        size, inequality_op_type(), stream, debug_synchronous, predicate
    );
}

/// \brief Parallel select primitive for device level using two selection predicates.
///
/// Performs a device-wide three-way partition using two selection predicates. Partition copies
/// the values from \p input to either \p output_first_part or \p output_second_part or
/// \p output_unselected according to the following criteria:
/// The value is copied to \p output_first_part if the predicate \p select_first_part_op invoked
/// with the value returns \p true. It is copied to \p output_second_part if \p select_first_part_op
/// returns \p false and \p select_second_part_op returns \p true, and it is copied to
/// \p output_unselected otherwise.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage is a null pointer.
/// * Range specified by \p selected_count_output must have at least 2 elements.
/// * Relative order is preserved for the elements.
/// * The number of elements written to \p output_first_part is equal to the number of elements
/// in the input for which \p select_first_part_op returned \p true.
/// * The number of elements written to \p output_second_part is equal to the number of elements
/// in the input for which \p select_first_part_op returned \p false and \p select_second_part_op
/// returned \p true.
/// * The number of elements written to \p output_unselected is equal to the number of input elements
/// minus the number of elements written to \p output_first_part minus the number of elements written
/// to \p output_second_part.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p select_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam FirstOutputIterator - random-access iterator type of the first output range. It can be
/// a simple pointer type.
/// \tparam SecondOutputIterator - random-access iterator type of the second output range. It can be
/// a simple pointer type.
/// \tparam UnselectedOutputIterator - random-access iterator type of the unselected output range.
/// It can be a simple pointer type.
/// \tparam SelectedCountOutputIterator - random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam FirstUnaryPredicate - type of the first unary selection predicate.
/// \tparam SecondUnaryPredicate - type of the second unary selection predicate.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to select values from.
/// \param [out] output_first_part - iterator to the first element in the first output range.
/// \param [out] output_second_part - iterator to the first element in the second output range.
/// \param [out] output_unselected - iterator to the first element in the unselected output range.
/// \param [out] selected_count_output - iterator to the total number of selected values in
/// \p output_first_part and \p output_second_part respectively.
/// \param [in] size - number of element in the input range.
/// \param [in] select_first_part_op - unary function object which returns \p true if the element
/// should be in \p output_first_part range
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] select_second_part_op - unary function object which returns \p true if the element
/// should be in \p output_second_part range (given that \p select_first_part_op returned \p false)
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level three-way partition operation is performed on an array of
/// integer values, even values are copied to the first partition, odd and 3-divisible values
/// are copied to the second partition, and the rest of the values are copied to the
/// unselected partition
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// auto first_predicate =
///     [] __device__ (int a) -> bool
///     {
///         return (a%2) == 0;
///     };
/// auto second_predicate =
///     [] __device__ (int a) -> bool
///     {
///         return (a%3) == 0;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// int * input;                // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * output_first_part;    // array of 8 elements
/// int * output_second_part;   // array of 8 elements
/// int * output_unselected;    // array of 8 elements
/// size_t * output_count;      // array of 2 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::partition_three_way(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input,
///     output_first_part, output_second_part, output_unselected,
///     output_count,
///     input_size,
///     first_predicate,
///     second_predicate
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform partition
/// rocprim::partition_three_way(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input,
///     output_first_part, output_second_part, output_unselected,
///     output_count,
///     input_size,
///     first_predicate,
///     second_predicate
/// );
/// // elements denoted by '*' were not modified
/// // output_first_part:  [2, 4, 6, 8, *, *, *, *]
/// // output_second_part: [3, *, *, *, *, *, *, *]
/// // output_unselected:  [1, 5, 7, *, *, *, *, *]
/// // output_count:       [4, 1]
/// \endcode
/// \endparblock
template <
    class Config = default_config,
    typename InputIterator,
    typename FirstOutputIterator,
    typename SecondOutputIterator,
    typename UnselectedOutputIterator,
    typename SelectedCountOutputIterator,
    typename FirstUnaryPredicate,
    typename SecondUnaryPredicate>
inline
hipError_t partition_three_way(void * temporary_storage,
                               size_t& storage_size,
                               InputIterator input,
                               FirstOutputIterator output_first_part,
                               SecondOutputIterator output_second_part,
                               UnselectedOutputIterator output_unselected,
                               SelectedCountOutputIterator selected_count_output,
                               const size_t size,
                               FirstUnaryPredicate select_first_part_op,
                               SecondUnaryPredicate select_second_part_op,
                               const hipStream_t stream = 0,
                               const bool debug_synchronous = false)
{
    // Dummy flag type
    using flag_type = ::rocprim::empty_type;
    flag_type * flags = nullptr;
    // Dummy inequality operation
    using inequality_op_type = ::rocprim::empty_type;
    using offset_type = uint2;
    using output_key_iterator_tuple = tuple<
        FirstOutputIterator,
        SecondOutputIterator,
        UnselectedOutputIterator>;
    using output_value_iterator_tuple
        = tuple<::rocprim::empty_type*, ::rocprim::empty_type*, ::rocprim::empty_type*>;
    rocprim::empty_type* const no_input_values = nullptr; // key only
    const output_value_iterator_tuple no_output_values {nullptr, nullptr, nullptr}; // key only

    output_key_iterator_tuple output{ output_first_part, output_second_part, output_unselected };

    return detail::partition_impl<detail::select_method::predicate, false, Config, offset_type>(
        temporary_storage, storage_size, input, no_input_values, flags, output, no_output_values, selected_count_output,
        size, inequality_op_type(), stream, debug_synchronous,
        select_first_part_op, select_second_part_op
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_
