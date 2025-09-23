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

#ifndef ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_
#define ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_

#include <algorithm>
#include <iostream>
#include <iterator>
#include <optional>
#include <type_traits>

#include "../common.hpp"
#include "../config.hpp"
#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../iterator/reverse_iterator.hpp"
#include "../type_traits.hpp"
#include "../types.hpp"

#include "detail/device_partition.hpp"
#include "detail/device_scan_common.hpp"
#include "device_partition_config.hpp"
#include "device_transform.hpp"
#include "rocprim/detail/virtual_shared_memory.hpp"
#include "rocprim/device/config_types.hpp"
#include "rocprim/device/detail/ordered_block_id.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<class Config,
         select_method SelectMethod,
         bool          OnlySelected,
         class KeyIterator,
         class ValueIterator,
         class FlagIterator,
         class OutputKeyIterator,
         class OutputValueIterator,
         class InequalityOp,
         class OffsetLookbackScanState,
         class BlockIdWrapper,
         class... UnaryPredicates>
inline hipError_t launch_partition(detail::target_arch     arch,
                                   KeyIterator             keys_input,
                                   ValueIterator           values_input,
                                   FlagIterator            flags,
                                   OutputKeyIterator       keys_output,
                                   OutputValueIterator     values_output,
                                   size_t*                 selected_count,
                                   size_t*                 prev_selected_count,
                                   size_t                  prev_processed,
                                   const size_t            total_size,
                                   InequalityOp            inequality_op,
                                   OffsetLookbackScanState offset_scan_state,
                                   const unsigned int      number_of_blocks,
                                   detail::vsmem_t         vsmem,
                                   BlockIdWrapper          block_id,
                                   dim3                    grid,
                                   dim3                    block,
                                   size_t                  shmem,
                                   hipStream_t             stream,
                                   UnaryPredicates... predicates)
{
    auto kernel = [=](auto arch_config) mutable
    {
        using offset_type = typename OffsetLookbackScanState::value_type;
        using key_type    = typename std::iterator_traits<KeyIterator>::value_type;
        using value_type  = typename std::iterator_traits<ValueIterator>::value_type;
        using flag_type =
            typename std::conditional<SelectMethod == select_method::predicated_flag,
                                      typename std::iterator_traits<FlagIterator>::value_type,
                                      bool>::type;

        using partition_kernel_impl_t = partition_kernel_impl_<decltype(arch_config),
                                                               SelectMethod,
                                                               OnlySelected,
                                                               key_type,
                                                               value_type,
                                                               flag_type,
                                                               offset_type,
                                                               BlockIdWrapper>;

        using VSmemHelperT = detail::vsmem_helper_impl<partition_kernel_impl_t>;
        ROCPRIM_SHARED_MEMORY typename VSmemHelperT::static_temp_storage_t static_temp_storage;
        // Get temporary storage
        typename partition_kernel_impl_t::storage_type& storage
            = VSmemHelperT::get_temp_storage(static_temp_storage, vsmem);

        partition_kernel_impl_t().partition(keys_input,
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
                                            block_id,
                                            storage,
                                            predicates...);
    };

    return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
}

template<class Config,
         select_method SelectMethod,
         bool          OnlySelected,
         class Key,
         class Value,
         class FlagType,
         class OffsetLookbackScanState,
         class BlockIdWrapper>
inline size_t get_partition_vsmem_size_per_block(detail::target_arch arch)
{
    using offset_type = typename OffsetLookbackScanState::value_type;
    std::optional<size_t> vsmem_per_block;
    for_each_arch(
        [&](auto arch_tag)
        {
            constexpr target_arch Arch = decltype(arch_tag)::value;
            if(Arch != arch || vsmem_per_block)
                return;

            using ArchConfig               = typename Config::template architecture_config<Arch>;
            using partition_kernel_impl_t  = partition_kernel_impl_<ArchConfig,
                                                                   SelectMethod,
                                                                   OnlySelected,
                                                                   Key,
                                                                   Value,
                                                                   FlagType,
                                                                   offset_type,
                                                                   BlockIdWrapper>;
            using partition_vsmem_helper_t = detail::vsmem_helper_impl<partition_kernel_impl_t>;

            vsmem_per_block = partition_vsmem_helper_t::vsmem_per_block;
        });
    if(!vsmem_per_block)
    {
        using ArchConfig = typename Config::template architecture_config<target_arch::unknown>;
        using partition_kernel_impl_t  = partition_kernel_impl_<ArchConfig,
                                                               SelectMethod,
                                                               OnlySelected,
                                                               Key,
                                                               Value,
                                                               FlagType,
                                                               offset_type,
                                                               BlockIdWrapper>;
        using partition_vsmem_helper_t = detail::vsmem_helper_impl<partition_kernel_impl_t>;

        vsmem_per_block = partition_vsmem_helper_t::vsmem_per_block;
    }
    return vsmem_per_block.value();
}

template<partition_subalgo SubAlgo,
         bool              UsingOrderedBlockId,
         class Config,
         class OffsetT,
         class KeyIterator,
         class ValueIterator, // can be rocprim::empty_type* for key only
         class FlagIterator,
         class OutputKeyIterator,
         class OutputValueIterator, // can be rocprim::empty_type* for key only
         class InequalityOp,
         class SelectedCountOutputIterator,
         class... UnaryPredicates>
inline hipError_t partition_impl(void*                       temporary_storage,
                                 size_t&                     storage_size,
                                 KeyIterator                 keys_input,
                                 ValueIterator               values_input,
                                 FlagIterator                flags,
                                 OutputKeyIterator           keys_output,
                                 OutputValueIterator         values_output,
                                 SelectedCountOutputIterator selected_count_output,
                                 const size_t                size,
                                 InequalityOp                inequality_op,
                                 const hipStream_t           stream,
                                 bool                        debug_synchronous,
                                 UnaryPredicates... predicates)
{
    using offset_type = OffsetT;
    using key_type    = typename std::iterator_traits<KeyIterator>::value_type;
    using value_type  = typename std::iterator_traits<ValueIterator>::value_type;

    using config = wrapped_partition_config<Config, SubAlgo, key_type, value_type>;

    constexpr bool write_only_selected = SubAlgo == partition_subalgo::select_flag
                                         || SubAlgo == partition_subalgo::select_predicate
                                         || SubAlgo == partition_subalgo::select_predicated_flag
                                         || SubAlgo == partition_subalgo::select_unique
                                         || SubAlgo == partition_subalgo::select_unique_by_key;

    constexpr bool is_unique = SubAlgo == partition_subalgo::select_unique
                               || SubAlgo == partition_subalgo::select_unique_by_key;
    constexpr bool is_flag = SubAlgo == partition_subalgo::partition_two_way_flag
                             || SubAlgo == partition_subalgo::partition_flag
                             || SubAlgo == partition_subalgo::select_flag;
    constexpr bool is_predicated_flag = SubAlgo == partition_subalgo::select_predicated_flag;
    constexpr select_method method
        = is_unique
              ? select_method::unique
              : (is_predicated_flag ? select_method::predicated_flag
                                    : (is_flag ? select_method::flag : select_method::predicate));

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const partition_config_params params = dispatch_target_arch<config, false>(target_arch);

    using offset_scan_state_type            = detail::lookback_scan_state<offset_type>;
    using offset_scan_state_with_sleep_type = detail::lookback_scan_state<offset_type, true>;

    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const auto         items_per_block  = block_size * items_per_thread;

    static constexpr bool         is_three_way        = sizeof...(UnaryPredicates) == 2;
    static constexpr const size_t selected_count_size = is_three_way ? 2 : 1;

    const size_t size_limit = params.kernel_config.size_limit;
    const size_t aligned_size_limit
        = ::rocprim::max<size_t>(size_limit - (size_limit % items_per_block), items_per_block);
    const size_t limited_size     = std::min<size_t>(size, aligned_size_limit);
    const bool   use_limited_size = limited_size == aligned_size_limit;

    const unsigned int number_of_blocks
        = static_cast<unsigned int>(::rocprim::detail::ceiling_div(limited_size, items_per_block));

    // Calculate required temporary storage
    void*   offset_scan_state_storage;
    size_t* selected_count;
    size_t* prev_selected_count;

    detail::temp_storage::layout layout{};
    result = offset_scan_state_type::get_temp_storage_layout(number_of_blocks, stream, layout);
    if(result != hipSuccess)
    {
        return result;
    }

    using block_id_wrapper_type = block_id_wrapper<uint32_t, UsingOrderedBlockId>;

    typename block_id_wrapper_type::id_type* block_id_pool = nullptr;

    bool use_sleep;
    ROCPRIM_RETURN_ON_ERROR(is_sleep_scan_state_used(stream, use_sleep));

    // vsmem size
    void*  vsmem                      = nullptr;
    size_t virtual_shared_memory_size = 0;
    using flag_type =
        typename std::conditional<method == select_method::predicated_flag,
                                  typename std::iterator_traits<FlagIterator>::value_type,
                                  bool>::type;
    if(use_sleep)
    {
        virtual_shared_memory_size
            = get_partition_vsmem_size_per_block<config,
                                                 method,
                                                 write_only_selected,
                                                 key_type,
                                                 value_type,
                                                 flag_type,
                                                 offset_scan_state_with_sleep_type,
                                                 block_id_wrapper_type>(target_arch);
    }
    else
    {
        virtual_shared_memory_size
            = get_partition_vsmem_size_per_block<config,
                                                 method,
                                                 write_only_selected,
                                                 key_type,
                                                 value_type,
                                                 flag_type,
                                                 offset_scan_state_type,
                                                 block_id_wrapper_type>(target_arch);
    }
    virtual_shared_memory_size *= number_of_blocks;

    // temporary storage partition

    result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            // This is valid even with offset_scan_state_with_sleep_type
            detail::temp_storage::make_partition(&offset_scan_state_storage, layout),
            // Note: the following two are to be allocated continuously, so that they can be initialized
            // simultaneously.
            // They have the same base type, so there is no padding between the types.
            detail::temp_storage::ptr_aligned_array(&selected_count, selected_count_size),
            detail::temp_storage::ptr_aligned_array(&prev_selected_count, selected_count_size),
            detail::temp_storage::ptr_aligned_array(&block_id_pool,
                                                    block_id_wrapper_type::get_storage_size()),
            // vsmem
            detail::temp_storage::make_partition(&vsmem,
                                                 virtual_shared_memory_size,
                                                 cache_line_size)));

    if(result != hipSuccess || temporary_storage == nullptr)
    {
        return result;
    }

    auto block_id = block_id_wrapper_type::create(block_id_pool);

    // Start point for time measurements
    std::chrono::steady_clock::time_point start;

    // Create and initialize lookback_scan_state obj
    offset_scan_state_type offset_scan_state{};
    result = offset_scan_state_type::create(offset_scan_state,
                                            offset_scan_state_storage,
                                            number_of_blocks,
                                            stream);
    if(result != hipSuccess)
    {
        return result;
    }
    offset_scan_state_with_sleep_type offset_scan_state_with_sleep{};
    result = offset_scan_state_with_sleep_type::create(offset_scan_state_with_sleep,
                                                       offset_scan_state_storage,
                                                       number_of_blocks,
                                                       stream);
    if(result != hipSuccess)
    {
        return result;
    }

    // Call the provided function with either offset_scan_state or offset_scan_state_with_sleep based on
    // the value of use_sleep
    auto with_scan_state = [use_sleep, offset_scan_state, offset_scan_state_with_sleep](
                               auto&& func) mutable -> decltype(auto)
    {
        if(use_sleep)
        {
            return func(offset_scan_state_with_sleep);
        }
        else
        {
            return func(offset_scan_state);
        }
    };

    // Memset selected_count and prev_selected_count at once
    result = hipMemsetAsync(selected_count,
                            0,
                            sizeof(*selected_count) * 2 * selected_count_size,
                            stream);
    if(result != hipSuccess)
    {
        return result;
    }

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

        const unsigned int current_number_of_blocks
            = ::rocprim::detail::ceiling_div(current_size, items_per_block);

        if(debug_synchronous)
        {
            std::cout << "current size " << current_size << '\n';
            std::cout << "current number of blocks " << current_number_of_blocks << '\n';

            start = std::chrono::steady_clock::now();
        }

        with_scan_state(
            [&](const auto scan_state)
            {
                const unsigned int block_size = ROCPRIM_DEFAULT_MAX_BLOCK_SIZE;
                const unsigned int grid_size
                    = ::rocprim::detail::ceiling_div(current_number_of_blocks, block_size);
                init_lookback_scan_state_kernel<decltype(scan_state)>
                    <<<dim3(grid_size), dim3(block_size), 0, stream>>>(scan_state,
                                                                       current_number_of_blocks);
            });
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_offset_scan_state_kernel",
                                                    current_number_of_blocks,
                                                    start);

        if(UsingOrderedBlockId)
        {
            result = hipMemsetAsync(block_id_pool, 0, sizeof(unsigned int), stream);
            if(result != hipSuccess)
            {
                return result;
            }
        }

        if(debug_synchronous)
            start = std::chrono::steady_clock::now();

        ROCPRIM_RETURN_ON_ERROR(with_scan_state(
            [&](const auto scan_state)
            {
                return launch_partition<config, method, write_only_selected>(
                    target_arch,
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
                    scan_state,
                    current_number_of_blocks,
                    detail::vsmem_t{vsmem},
                    block_id,
                    dim3(current_number_of_blocks),
                    dim3(block_size),
                    0,
                    stream,
                    predicates...);
            }));
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("partition_kernel", size, start);

        std::swap(selected_count, prev_selected_count);
    }

    result = ::rocprim::transform(prev_selected_count,
                                  selected_count_output,
                                  (is_three_way ? 2 : 1),
                                  ::rocprim::identity<>{},
                                  stream,
                                  debug_synchronous);
    if(result != hipSuccess)
    {
        return result;
    }

    return hipSuccess;
}

} // namespace detail

/// \brief Two-way parallel select primitive for device level using selection predicate.
///
/// Performs a device-wide partition using selection predicate. Partition copies the values from
/// \p input to \p output_selected and \p output_rejected for all values for which the \p predicate
/// returns \p true and \p false respectively.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The number of elements written to \p output_selected is equal to the number elements
/// in the input for which \p predicate returns \p true.
/// * The number of elements written to \p output_rejected is equal to the number elements
/// in the input for which \p predicate returns \p false.
/// * Range specified by \p selected_count_output must have at least 1 element.
/// * Relative order is preserved.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `select_config`.
/// \tparam InputIterator random-access iterator type of the input range. It can be a simple
/// pointer type.
/// \tparam SelectedOutputIterator random-access iterator type of the selected output range. It
/// can be a simple pointer type.
/// \tparam RejectedOutputIterator random-access iterator type of the rejected output range. It
/// can be a simple pointer type.
/// \tparam SelectedCountOutputIterator random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam Predicate type of the selection predicate.
/// \tparam UsingOrderedBlockId If true, uses an atomic counter to assign block id instead of natural
/// blockIdx-based ordering.  Can increase performance on MI3xx architectures when using streams. The
/// default is false.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When a null
/// pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to select values from.
/// \param [out] output_selected iterator to the first element in the selected output range.
/// \param [out] output_rejected iterator to the first element in the rejected output range.
/// \param [out] selected_count_output iterator to the total number of selected values (length of
/// \p output_selected ).
/// \param [in] size number of elements in the input range.
/// \param [in] predicate the unary selection predicate to select values into the select and reject
/// outputs.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel launch is
/// forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level two-way partition operation is performed on an array of integer
/// values, even values are copied into the selected output and odd values are copied into rejected
/// output.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>///
///
/// auto predicate =
///     [] (int a) -> bool
///     {
///         return (a%2) == 0;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;     // e.g., 8
/// int * input;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * selected_output; // empty array of at least 4 elements
/// int * rejected_output; // empty array of at least 4 elements
/// size_t * output_count; // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::partition_two_way(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input,
///     selected_output,
///     rejected_output,
///     selected_output_count,
///     input_size,
///     predicate
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform partition
/// rocprim::partition_two_way(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input,
///     selected_output,
///     rejected_output,
///     selected_output_count,
///     input_size,
///     predicate
/// );
/// // output_selected: [2, 4, 6, 8]
/// // output_rejected: [1, 3, 5, 7]
/// // selected_output_count: 4
/// \endcode
/// \endparblock
template<class Config = default_config,
         class InputIterator,
         class SelectedOutputIterator,
         class RejectedOutputIterator,
         class SelectedCountOutputIterator,
         class Predicate,
         bool UsingOrderedBlockId = false>
inline hipError_t partition_two_way(void*                       temporary_storage,
                                    size_t&                     storage_size,
                                    InputIterator               input,
                                    SelectedOutputIterator      output_selected,
                                    RejectedOutputIterator      output_rejected,
                                    SelectedCountOutputIterator selected_count_output,
                                    const size_t                size,
                                    Predicate                   predicate,
                                    const hipStream_t           stream            = 0,
                                    const bool                  debug_synchronous = false)
{
    using flag_type          = ::rocprim::empty_type; //dummy
    using inequality_op_type = ::rocprim::empty_type; //dummy
    using offset_type        = unsigned int;

    flag_type* flags = nullptr;

    rocprim::empty_type* const no_input_values = nullptr; // key only

    using output_key_iterator_tuple = tuple<SelectedOutputIterator, RejectedOutputIterator>;
    output_key_iterator_tuple output{output_selected, output_rejected};

    using output_value_iterator_tuple = tuple<::rocprim::empty_type*, ::rocprim::empty_type*>;
    const output_value_iterator_tuple no_output_values{nullptr, nullptr}; // key only

    return detail::partition_impl<detail::partition_subalgo::partition_two_way_predicate,
                                  UsingOrderedBlockId,
                                  Config,
                                  offset_type>(temporary_storage,
                                               storage_size,
                                               input,
                                               no_input_values,
                                               flags,
                                               output,
                                               no_output_values,
                                               selected_count_output,
                                               size,
                                               inequality_op_type(),
                                               stream,
                                               debug_synchronous,
                                               predicate);
}

/// \brief Two-way parallel select primitive for device level using range of flags.
///
/// Performs a device-wide partition based on input \p flags. Partition copies the values from \p
/// input to \p output_selected and \p output_rejected in such a way that all values for which the
/// corresponding items from \p flags are \p true (or can be implicitly converted to \p true )
/// are copied to \p output_selected, and to \p output rejected if the flag is \p false .
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size if \p temporary_storage
/// in a null pointer.
/// * Ranges specified by \p input and \p flags must have at least \p size elements.
/// * The number of elements written to \p output_selected is equal to the number of \p true
/// elements in \p flags .
/// * The number of elements written to \p output_rejected is equal to the number of \p false
/// elements in \p flags .
/// * Range specified by \p selected_count_output must have at least 1 element.
/// * Values of \p flag range should be implicitly convertible to `bool` type.
/// * The relative order of elements in both output ranges matches the input range.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `select_config`.
/// \tparam InputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam FlagIterator random-access iterator type of the flag range. It can be
/// a simple pointer type.
/// \tparam SelectedOutputIterator random-access iterator type of the selected output range. It
/// can be a simple pointer type.
/// \tparam RejectedOutputIterator random-access iterator type of the rejected output range. It
/// can be a simple pointer type
/// \tparam SelectedCountOutputIterator random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam UsingOrderedBlockId If true, uses an atomic counter to assign block id instead of natural
/// blockIdx-based ordering.  Can increase performance on MI3xx architectures when using streams. The
/// default is false.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to select values from.
/// \param [in] flags iterator to the selection flag corresponding to the first element from \p
/// input range.
/// \param [out] output_selected iterator to the first element in the selected output range.
/// \param [out] output_rejected iterator to the first element in the rejected output range.
/// \param [out] selected_count_output iterator to the total number of selected values (length of
/// \p output_selected).
/// \param [in] size number of elements in the input range.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level two-way partition operation is performed on an array of
/// integer values with array of <tt>char</tt>s used as flags.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;     // e.g., 8
/// int * input;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// char * flags;          // e.g., [0, 1, 1, 0, 0, 1, 0, 1]
/// int * output_selected; // empty array of at least 4 elements
/// int * output_rejected; // empty array of at least 4 elements
/// size_t * output_count; // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::partition(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, flags,
///     output_selected,
///     output_rejected,
///     output_count,
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
///     output_selected,
///     output_rejected,
///     output_count,
///     input_size
/// );
/// // output_selected: [2, 3, 6, 8]
/// // output_rejected: [1, 4, 5, 7]
/// // output_count: 4
/// \endcode
/// \endparblock
template<class Config = default_config,
         typename InputIterator,
         typename FlagIterator,
         typename SelectedOutputIterator,
         typename RejectedOutputIterator,
         typename SelectedCountOutputIterator,
         bool UsingOrderedBlockId = false>
inline hipError_t partition_two_way(void*                       temporary_storage,
                                    size_t&                     storage_size,
                                    InputIterator               input,
                                    FlagIterator                flags,
                                    SelectedOutputIterator      output_selected,
                                    RejectedOutputIterator      output_rejected,
                                    SelectedCountOutputIterator selected_count_output,
                                    const size_t                size,
                                    const hipStream_t           stream            = 0,
                                    const bool                  debug_synchronous = false)
{
    using unary_predicate_type = ::rocprim::empty_type; // dummy
    using inequality_op_type   = ::rocprim::empty_type; // dummy
    using offset_type          = unsigned int;

    rocprim::empty_type* const no_input_values = nullptr; // key only

    using output_key_iterator_tuple = tuple<SelectedOutputIterator, RejectedOutputIterator>;
    output_key_iterator_tuple keys_output{output_selected, output_rejected};

    using output_value_iterator_tuple = tuple<::rocprim::empty_type*, ::rocprim::empty_type*>;
    const output_value_iterator_tuple no_output_values{nullptr, nullptr}; // key only

    return detail::partition_impl<detail::partition_subalgo::partition_two_way_flag,
                                  UsingOrderedBlockId,
                                  Config,
                                  offset_type>(temporary_storage,
                                               storage_size,
                                               input,
                                               no_input_values,
                                               flags,
                                               keys_output,
                                               no_output_values,
                                               selected_count_output,
                                               size,
                                               inequality_op_type(),
                                               stream,
                                               debug_synchronous,
                                               unary_predicate_type());
}

/// \brief Parallel select primitive for device level using range of flags.
///
/// Performs a device-wide partition based on input \p flags. Partition copies
/// the values from \p input to \p output in such a way that all values for which the corresponding
/// items from \p flags are \p true (or can be implicitly converted to \p true) precede
/// the elements for which the corresponding items from \p flags are \p false.
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
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `select_config`.
/// \tparam InputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam FlagIterator random-access iterator type of the flag range. It can be
/// a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam UsingOrderedBlockId If true, uses an atomic counter to assign block id instead of natural
/// blockIdx-based ordering.  Can increase performance on MI3xx architectures when using streams. The
/// default is false.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to select values from.
/// \param [in] flags iterator to the selection flag corresponding to the first element from \p input range.
/// \param [out] output iterator to the first element in the output range.
/// \param [out] selected_count_output iterator to the total number of selected values (length of \p output).
/// \param [in] size number of elements in the input range.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
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
template<class Config = default_config,
         class InputIterator,
         class FlagIterator,
         class OutputIterator,
         class SelectedCountOutputIterator,
         bool UsingOrderedBlockId = false>
inline hipError_t partition(void*                       temporary_storage,
                            size_t&                     storage_size,
                            InputIterator               input,
                            FlagIterator                flags,
                            OutputIterator              output,
                            SelectedCountOutputIterator selected_count_output,
                            const size_t                size,
                            const hipStream_t           stream            = 0,
                            const bool                  debug_synchronous = false)
{
    using unary_predicate_type = ::rocprim::empty_type; // dummy
    using inequality_op_type   = ::rocprim::empty_type; // dummy
    using offset_type          = unsigned int;

    rocprim::empty_type* const no_input_values = nullptr; // key only

    using output_key_iterator_tuple = tuple<OutputIterator, ::rocprim::empty_type*>;
    output_key_iterator_tuple keys_output{output, nullptr};

    using output_value_iterator_tuple = tuple<::rocprim::empty_type*, ::rocprim::empty_type*>;
    const output_value_iterator_tuple no_output_values{nullptr, nullptr}; // key only

    return detail::partition_impl<detail::partition_subalgo::partition_flag,
                                  UsingOrderedBlockId,
                                  Config,
                                  offset_type>(temporary_storage,
                                               storage_size,
                                               input,
                                               no_input_values,
                                               flags,
                                               keys_output,
                                               no_output_values,
                                               selected_count_output,
                                               size,
                                               inequality_op_type(),
                                               stream,
                                               debug_synchronous,
                                               unary_predicate_type());
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
/// * Range specified by \p selected_count_output must have at least 1 element.
/// * Relative order is preserved for the elements for which the \p predicate returns \p true. Other
/// elements are copied in reverse order.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `select_config`.
/// \tparam InputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam UnaryPredicate type of a unary selection predicate.
/// \tparam UsingOrderedBlockId If true, uses an atomic counter to assign block id instead of natural
/// blockIdx-based ordering.  Can increase performance on MI3xx architectures when using streams. The
/// default is false.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to select values from.
/// \param [out] output iterator to the first element in the output range.
/// \param [out] selected_count_output iterator to the total number of selected values (length of \p output).
/// \param [in] size number of elements in the input range.
/// \param [in] predicate unary function object which returns \p true if the element should be
/// ordered before other elements.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
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
///     [] (int a) -> bool
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
template<class Config = default_config,
         class InputIterator,
         class OutputIterator,
         class SelectedCountOutputIterator,
         class UnaryPredicate,
         bool UsingOrderedBlockId = false>
inline hipError_t partition(void*                       temporary_storage,
                            size_t&                     storage_size,
                            InputIterator               input,
                            OutputIterator              output,
                            SelectedCountOutputIterator selected_count_output,
                            const size_t                size,
                            UnaryPredicate              predicate,
                            const hipStream_t           stream            = 0,
                            const bool                  debug_synchronous = false)
{
    using flag_type          = ::rocprim::empty_type; //dummy
    using inequality_op_type = ::rocprim::empty_type; //dummy
    using offset_type        = unsigned int;

    flag_type* flags = nullptr;

    rocprim::empty_type* const no_input_values = nullptr; // key only

    using output_key_iterator_tuple = tuple<OutputIterator, ::rocprim::empty_type*>;
    output_key_iterator_tuple keys_output{output, nullptr};

    using output_value_iterator_tuple = tuple<::rocprim::empty_type*, ::rocprim::empty_type*>;
    const output_value_iterator_tuple no_output_values{nullptr, nullptr}; // key only

    return detail::partition_impl<detail::partition_subalgo::partition_predicate,
                                  UsingOrderedBlockId,
                                  Config,
                                  offset_type>(temporary_storage,
                                               storage_size,
                                               input,
                                               no_input_values,
                                               flags,
                                               keys_output,
                                               no_output_values,
                                               selected_count_output,
                                               size,
                                               inequality_op_type(),
                                               stream,
                                               debug_synchronous,
                                               predicate);
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
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `select_config`.
/// \tparam InputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam FirstOutputIterator random-access iterator type of the first output range. It can be
/// a simple pointer type.
/// \tparam SecondOutputIterator random-access iterator type of the second output range. It can be
/// a simple pointer type.
/// \tparam UnselectedOutputIterator random-access iterator type of the unselected output range.
/// It can be a simple pointer type.
/// \tparam SelectedCountOutputIterator random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam FirstUnaryPredicate type of the first unary selection predicate.
/// \tparam SecondUnaryPredicate type of the second unary selection predicate.
/// \tparam UsingOrderedBlockId If true, uses an atomic counter to assign block id instead of natural
/// blockIdx-based ordering.  Can increase performance on MI3xx architectures when using streams. The
/// default is false.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to select values from.
/// \param [out] output_first_part iterator to the first element in the first output range.
/// \param [out] output_second_part iterator to the first element in the second output range.
/// \param [out] output_unselected iterator to the first element in the unselected output range.
/// \param [out] selected_count_output iterator to the total number of selected values in
/// \p output_first_part and \p output_second_part respectively.
/// \param [in] size number of elements in the input range.
/// \param [in] select_first_part_op unary function object which returns \p true if the element
/// should be in \p output_first_part range
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] select_second_part_op unary function object which returns \p true if the element
/// should be in \p output_second_part range (given that \p select_first_part_op returned \p false)
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
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
///     [] (int a) -> bool
///     {
///         return (a%2) == 0;
///     };
/// auto second_predicate =
///     [] (int a) -> bool
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
template<class Config = default_config,
         typename InputIterator,
         typename FirstOutputIterator,
         typename SecondOutputIterator,
         typename UnselectedOutputIterator,
         typename SelectedCountOutputIterator,
         typename FirstUnaryPredicate,
         typename SecondUnaryPredicate,
         bool UsingOrderedBlockId = false>
inline hipError_t partition_three_way(void*                       temporary_storage,
                                      size_t&                     storage_size,
                                      InputIterator               input,
                                      FirstOutputIterator         output_first_part,
                                      SecondOutputIterator        output_second_part,
                                      UnselectedOutputIterator    output_unselected,
                                      SelectedCountOutputIterator selected_count_output,
                                      const size_t                size,
                                      FirstUnaryPredicate         select_first_part_op,
                                      SecondUnaryPredicate        select_second_part_op,
                                      const hipStream_t           stream            = 0,
                                      const bool                  debug_synchronous = false)
{
    // Dummy flag type
    using flag_type  = ::rocprim::empty_type;
    flag_type* flags = nullptr;
    // Dummy inequality operation
    using inequality_op_type = ::rocprim::empty_type;
    using offset_type        = uint2;
    using output_key_iterator_tuple
        = tuple<FirstOutputIterator, SecondOutputIterator, UnselectedOutputIterator>;
    using output_value_iterator_tuple
        = tuple<::rocprim::empty_type*, ::rocprim::empty_type*, ::rocprim::empty_type*>;
    rocprim::empty_type* const        no_input_values = nullptr; // key only
    const output_value_iterator_tuple no_output_values{nullptr, nullptr, nullptr}; // key only

    output_key_iterator_tuple output{output_first_part, output_second_part, output_unselected};

    return detail::partition_impl<detail::partition_subalgo::partition_three_way,
                                  UsingOrderedBlockId,
                                  Config,
                                  offset_type>(temporary_storage,
                                               storage_size,
                                               input,
                                               no_input_values,
                                               flags,
                                               output,
                                               no_output_values,
                                               selected_count_output,
                                               size,
                                               inequality_op_type(),
                                               stream,
                                               debug_synchronous,
                                               select_first_part_op,
                                               select_second_part_op);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_
