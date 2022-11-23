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

#ifndef ROCPRIM_DEVICE_DEVICE_SORT_HPP_
#define ROCPRIM_DEVICE_DEVICE_SORT_HPP_

#include <iostream>
#include <iterator>
#include <type_traits>

#include "../config.hpp"
#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"

#include "detail/config/device_merge_sort.hpp"
#include "detail/device_merge.hpp"
#include "detail/device_merge_sort.hpp"
#include "detail/device_merge_sort_mergepath.hpp"
#include "device_merge_sort_config.hpp"
#include "device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<class Config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetT,
         class BinaryFunction>
ROCPRIM_KERNEL
    __launch_bounds__(device_params<Config>().block_sort_config.block_size) void block_sort_kernel(
        KeysInputIterator    keys_input,
        KeysOutputIterator   keys_output,
        ValuesInputIterator  values_input,
        ValuesOutputIterator values_output,
        const OffsetT        sorted_block_size,
        BinaryFunction       compare_function)
{
    static constexpr merge_sort_block_sort_config_params params = device_params<Config>();
    block_sort_kernel_impl<params.block_sort_config.block_size,
                           params.block_sort_config.items_per_thread,
                           params.block_sort_method>(keys_input,
                                                     keys_output,
                                                     values_input,
                                                     values_output,
                                                     sorted_block_size,
                                                     compare_function);
}

template<class Config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetT,
         class BinaryFunction>
ROCPRIM_KERNEL __launch_bounds__(
    device_params<Config>()
        .merge_oddeven_config
        .block_size) void device_block_merge_oddeven_kernel(KeysInputIterator    keys_input,
                                                            KeysOutputIterator   keys_output,
                                                            ValuesInputIterator  values_input,
                                                            ValuesOutputIterator values_output,
                                                            const OffsetT        input_size,
                                                            const OffsetT        sorted_block_size,
                                                            BinaryFunction       compare_function)
{
    static constexpr merge_sort_block_merge_config_params params = device_params<Config>();
    block_merge_oddeven_kernel<params.merge_oddeven_config.block_size,
                               params.merge_oddeven_config.items_per_thread>(keys_input,
                                                                             keys_output,
                                                                             values_input,
                                                                             values_output,
                                                                             input_size,
                                                                             sorted_block_size,
                                                                             compare_function);
}

template<class Config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetT,
         class BinaryFunction>
ROCPRIM_KERNEL __launch_bounds__(
    device_params<Config>()
        .merge_mergepath_config
        .block_size) void device_block_merge_mergepath_kernel(KeysInputIterator    keys_input,
                                                              KeysOutputIterator   keys_output,
                                                              ValuesInputIterator  values_input,
                                                              ValuesOutputIterator values_output,
                                                              const OffsetT        input_size,
                                                              const OffsetT  sorted_block_size,
                                                              BinaryFunction compare_function,
                                                              const OffsetT* merge_partitions)
{
    static constexpr merge_sort_block_merge_config_params params = device_params<Config>();
    block_merge_mergepath_kernel<params.merge_mergepath_config.block_size,
                                 params.merge_mergepath_config.items_per_thread>(keys_input,
                                                                                 keys_output,
                                                                                 values_input,
                                                                                 values_output,
                                                                                 input_size,
                                                                                 sorted_block_size,
                                                                                 compare_function,
                                                                                 merge_partitions);
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

template<typename Config, typename KeysInputIterator, typename OffsetT, typename CompareOpT>
ROCPRIM_KERNEL __launch_bounds__(
    device_params<Config>()
        .merge_mergepath_partition_config
        .block_size) void device_block_merge_mergepath_partition_kernel(KeysInputIterator keys,
                                                                        const OffsetT input_size,
                                                                        const unsigned int
                                                                                 num_partitions,
                                                                        OffsetT* merge_partitions,
                                                                        const CompareOpT compare_op,
                                                                        const OffsetT
                                                                            sorted_block_size)
{
    static constexpr merge_sort_block_merge_config_params params = device_params<Config>();
    static constexpr unsigned int                         ItemsPerTile
        = params.merge_mergepath_config.block_size * params.merge_mergepath_config.items_per_thread;
    const OffsetT partition_id
        = blockIdx.x * params.merge_mergepath_partition_config.block_size + threadIdx.x;

    if (partition_id >= num_partitions)
    {
        return;
    }

    const unsigned int merged_tiles = sorted_block_size / ItemsPerTile;
    const unsigned int target_merged_tiles = merged_tiles * 2;
    const unsigned int mask = target_merged_tiles - 1;
    const unsigned int tilegroup_start_id = ~mask & partition_id; // id of the first tile in the current tile-group
    const OffsetT tilegroup_start = ItemsPerTile * tilegroup_start_id; // index of the first item in the current tile-group

    const unsigned int local_tile_id = mask & partition_id; // id of the current tile in the current tile-group

    const OffsetT keys1_beg = rocprim::min(input_size, tilegroup_start);
    const OffsetT keys1_end = rocprim::min(input_size, tilegroup_start + sorted_block_size);
    const OffsetT keys2_beg = keys1_end;
    const OffsetT keys2_end = rocprim::min(input_size, keys2_beg + sorted_block_size);

    const OffsetT partition_at = rocprim::min<OffsetT>(keys2_end - keys1_beg, ItemsPerTile * local_tile_id);

    const OffsetT partition_diag = ::rocprim::detail::merge_path(keys + keys1_beg,
                                                                 keys + keys2_beg,
                                                                 keys1_end - keys1_beg,
                                                                 keys2_end - keys2_beg,
                                                                 partition_at,
                                                                 compare_op);

    merge_partitions[partition_id] = keys1_beg + partition_diag;
}

template<class Config,
         class KeysIterator,
         class ValuesIterator,
         class OffsetT,
         class BinaryFunction>
inline hipError_t merge_sort_block_merge(
    void*                                                      temporary_storage,
    size_t&                                                    storage_size,
    KeysIterator                                               keys,
    ValuesIterator                                             values,
    const OffsetT                                              size,
    unsigned int                                               sorted_block_size,
    BinaryFunction                                             compare_function,
    const hipStream_t                                          stream,
    bool                                                       debug_synchronous,
    typename std::iterator_traits<KeysIterator>::value_type*   keys_double_buffer   = nullptr,
    typename std::iterator_traits<ValuesIterator>::value_type* values_double_buffer = nullptr)
{
    using key_type             = typename std::iterator_traits<KeysIterator>::value_type;
    using value_type           = typename std::iterator_traits<ValuesIterator>::value_type;
    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    using config = wrapped_merge_sort_block_merge_config<Config, key_type, value_type>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const merge_sort_block_merge_config_params params = dispatch_target_arch<config>(target_arch);

    const unsigned int merge_oddeven_block_size = params.merge_oddeven_config.block_size;
    const unsigned int merge_oddeven_items_per_thread
        = params.merge_oddeven_config.items_per_thread;
    const unsigned int merge_oddeven_items_per_block
        = merge_oddeven_block_size * merge_oddeven_items_per_thread;

    const unsigned int merge_partition_block_size
        = params.merge_mergepath_partition_config.block_size;
    const unsigned int merge_mergepath_block_size = params.merge_mergepath_config.block_size;
    const unsigned int merge_mergepath_items_per_thread
        = params.merge_mergepath_config.items_per_thread;
    const unsigned int merge_mergepath_items_per_block
        = merge_mergepath_block_size * merge_mergepath_items_per_thread;

    const unsigned int sort_number_of_blocks = ceiling_div(size, sorted_block_size);
    const unsigned int merge_oddeven_number_of_blocks
        = ceiling_div(size, merge_oddeven_items_per_block);
    const unsigned int merge_mergepath_number_of_blocks = ceiling_div(size, merge_mergepath_items_per_block);

    const bool use_mergepath = size > params.merge_oddeven_config.size_limit;
    // variables below used for mergepath
    const unsigned int merge_num_partitions = merge_mergepath_number_of_blocks + 1;
    const unsigned int merge_partition_number_of_blocks
        = ceiling_div(merge_num_partitions, merge_partition_block_size);

    OffsetT*    d_merge_partitions = nullptr;
    key_type*   keys_buffer        = nullptr;
    value_type* values_buffer      = nullptr;

    hipError_t partition_result;
    if(keys_double_buffer == nullptr)
    {
        partition_result = detail::temp_storage::partition(
            temporary_storage,
            storage_size,
            detail::temp_storage::make_linear_partition(
                detail::temp_storage::ptr_aligned_array(&keys_buffer, size),
                detail::temp_storage::ptr_aligned_array(&values_buffer, with_values ? size : 0),
                detail::temp_storage::ptr_aligned_array(&d_merge_partitions,
                                                        use_mergepath ? merge_num_partitions : 0)));
    }
    else
    {
        partition_result = detail::temp_storage::partition(
            temporary_storage,
            storage_size,
            detail::temp_storage::make_linear_partition(
                detail::temp_storage::ptr_aligned_array(&d_merge_partitions,
                                                        use_mergepath ? merge_num_partitions : 0)));
        keys_buffer   = keys_double_buffer;
        values_buffer = values_double_buffer;
    }

    if(partition_result != hipSuccess || temporary_storage == nullptr)
    {
        return partition_result;
    }

    if( size == size_t(0) )
        return hipSuccess;

    if(sorted_block_size < std::max(merge_mergepath_items_per_block, merge_oddeven_block_size))
    {
        return hipError_t::hipErrorAssert;
    }

    if(debug_synchronous)
    {
        std::cout << "-----" << '\n';
        std::cout << "size: " << size << '\n';
        std::cout << "sorted_block_size: " << sorted_block_size << '\n';
        std::cout << "sort_number_of_blocks: " << sort_number_of_blocks << '\n';
        std::cout << "merge_oddeven_block_size: " << merge_oddeven_block_size << '\n';
        std::cout << "merge_oddeven_number_of_blocks: " << merge_oddeven_number_of_blocks << '\n';
        std::cout << "merge_oddeven_items_per_thread: " << merge_oddeven_items_per_thread << '\n';
        std::cout << "merge_oddeven_items_per_block: " << merge_oddeven_items_per_block << '\n';
        std::cout << "merge_mergepath_block_size: " << merge_mergepath_block_size << '\n';
        std::cout << "merge_mergepath_number_of_blocks: " << merge_mergepath_number_of_blocks << '\n';
        std::cout << "merge_mergepath_items_per_thread: " << merge_mergepath_items_per_thread << '\n';
        std::cout << "merge_mergepath_items_per_block: " << merge_mergepath_items_per_block << '\n';
        std::cout << "num_partitions: " << merge_num_partitions << '\n';
        std::cout << "merge_mergepath_partition_block_size: " << merge_partition_block_size << '\n';
        std::cout << "merge_mergepath_partition_number_of_blocks: " << merge_partition_number_of_blocks << '\n';
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    bool temporary_store = true;
    for(OffsetT block = sorted_block_size; block < size; block *= 2)
    {
        temporary_store = !temporary_store;

        const auto merge_step = [&](auto keys_input_,
                                    auto keys_output_,
                                    auto values_input_,
                                    auto values_output_) -> hipError_t
        {
            if(use_mergepath && block >= merge_mergepath_items_per_block)
            {
                if(debug_synchronous)
                    start = std::chrono::high_resolution_clock::now();
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(device_block_merge_mergepath_partition_kernel<config>),
                    dim3(merge_partition_number_of_blocks),
                    dim3(merge_partition_block_size),
                    0,
                    stream,
                    keys_input_,
                    size,
                    merge_num_partitions,
                    d_merge_partitions,
                    compare_function,
                    block);
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(
                    "device_block_merge_mergepath_partition_kernel",
                    size,
                    start);

                if(debug_synchronous)
                    start = std::chrono::high_resolution_clock::now();
                hipLaunchKernelGGL(HIP_KERNEL_NAME(device_block_merge_mergepath_kernel<config>),
                                   dim3(merge_mergepath_number_of_blocks),
                                   dim3(merge_mergepath_block_size),
                                   0,
                                   stream,
                                   keys_input_,
                                   keys_output_,
                                   values_input_,
                                   values_output_,
                                   size,
                                   block,
                                   compare_function,
                                   d_merge_partitions);
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("device_block_merge_mergepath_kernel",
                                                            size,
                                                            start);
            }
            else
            {
                if(debug_synchronous)
                    start = std::chrono::high_resolution_clock::now();
                hipLaunchKernelGGL(HIP_KERNEL_NAME(device_block_merge_oddeven_kernel<config>),
                                   dim3(merge_oddeven_number_of_blocks),
                                   dim3(merge_oddeven_block_size),
                                   0,
                                   stream,
                                   keys_input_,
                                   keys_output_,
                                   values_input_,
                                   values_output_,
                                   size,
                                   block,
                                   compare_function);
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("device_block_merge_oddeven_kernel",
                                                            size,
                                                            start)
            }
            return hipSuccess;
        };

        hipError_t error;
        if(temporary_store)
        {
            error = merge_step(keys_buffer, keys, values_buffer, values);
        }
        else
        {
            error = merge_step(keys, keys_buffer, values, values_buffer);
        }
        if(error != hipSuccess) return error;
    }

    if(!temporary_store)
    {
        hipError_t error = ::rocprim::transform(keys_buffer,
                                                keys,
                                                size,
                                                ::rocprim::identity<key_type>(),
                                                stream,
                                                debug_synchronous);
        if(error != hipSuccess) return error;

        if(with_values)
        {
            hipError_t error = ::rocprim::transform(values_buffer,
                                                    values,
                                                    size,
                                                    ::rocprim::identity<value_type>(),
                                                    stream,
                                                    debug_synchronous);
            if(error != hipSuccess) return error;
        }
    }

    return hipSuccess;
}

template<class Config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class BinaryFunction>
inline hipError_t merge_sort_block_sort(KeysInputIterator    keys_input,
                                        KeysOutputIterator   keys_output,
                                        ValuesInputIterator  values_input,
                                        ValuesOutputIterator values_output,
                                        const unsigned int   size,
                                        unsigned int&        sort_items_per_block,
                                        BinaryFunction       compare_function,
                                        const hipStream_t    stream,
                                        bool                 debug_synchronous)
{
    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using config = wrapped_merge_sort_block_sort_config<Config, key_type, value_type>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const merge_sort_block_sort_config_params params = dispatch_target_arch<config>(target_arch);

    sort_items_per_block
        = params.block_sort_config.block_size * params.block_sort_config.items_per_thread;
    const unsigned int sort_number_of_blocks = ceiling_div(size, sort_items_per_block);

    if(debug_synchronous)
    {
        std::cout << "-----" << '\n';
        std::cout << "size: " << size << '\n';
        std::cout << "sort_block_size: " << params.block_sort_config.block_size << '\n';
        std::cout << "sort_items_per_thread: " << params.block_sort_config.items_per_thread << '\n';
        std::cout << "sort_items_per_block: " << sort_items_per_block << '\n';
        std::cout << "sort_number_of_blocks: " << sort_number_of_blocks << '\n';
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;
    if(debug_synchronous)
        start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(HIP_KERNEL_NAME(block_sort_kernel<config>),
                       dim3(sort_number_of_blocks),
                       dim3(params.block_sort_config.block_size),
                       0,
                       stream,
                       keys_input,
                       keys_output,
                       values_input,
                       values_output,
                       size,
                       compare_function);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_sort_kernel", size, start);

    return hipSuccess;
}

// Helpful function that actually prints the values when static_assert fails
template<unsigned int A, unsigned int B>
ROCPRIM_DEVICE void TAssertEqualGreater()
{
    static_assert(A >= B, "A not greater or equal to B");
};

template<class BlockSortConfig, class BlockMergeConfig>
ROCPRIM_KERNEL void device_merge_sort_compile_time_verifier()
{
    static constexpr merge_sort_block_sort_config_params bs_params
        = device_params<BlockSortConfig>();
    static constexpr merge_sort_block_merge_config_params bm_params
        = device_params<BlockMergeConfig>();
    static constexpr unsigned int sort_items_per_block
        = bs_params.block_sort_config.block_size * bs_params.block_sort_config.items_per_thread;
    static constexpr unsigned int merge_oddeven_items_per_block
        = bm_params.merge_oddeven_config.block_size
          * bm_params.merge_oddeven_config.items_per_thread;
    static constexpr unsigned int merge_mergepath_items_per_block
        = bm_params.merge_mergepath_config.block_size
          * bm_params.merge_mergepath_config.items_per_thread;
    static_assert(is_power_of_two(sort_items_per_block),
                  "device_merge_sort block_sort items_per_block must be power of two");
    static_assert(is_power_of_two(merge_oddeven_items_per_block),
                  "device_merge_sort merge_oddeven items_per_block must be power of two");
    static_assert(is_power_of_two(merge_mergepath_items_per_block),
                  "device_merge_sort merge_mergepath items_per_block must be power of two");
    (void)TAssertEqualGreater<sort_items_per_block, merge_oddeven_items_per_block>();
    (void)TAssertEqualGreater<sort_items_per_block, merge_mergepath_items_per_block>();
    static_assert(sort_items_per_block >= merge_oddeven_items_per_block,
                  "device_merge_sort sort_items_per_block must be larger or equal to "
                  "merge_oddeven_items_per_block");
    static_assert(sort_items_per_block >= merge_mergepath_items_per_block,
                  "device_merge_sort sort_items_per_block must be larger or equal to "
                  "merge_mergepath_items_per_block");
}

template<class Config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class BinaryFunction>
inline hipError_t merge_sort_impl(
    void*                                                           temporary_storage,
    size_t&                                                         storage_size,
    KeysInputIterator                                               keys_input,
    KeysOutputIterator                                              keys_output,
    ValuesInputIterator                                             values_input,
    ValuesOutputIterator                                            values_output,
    const unsigned int                                              size,
    BinaryFunction                                                  compare_function,
    const hipStream_t                                               stream,
    bool                                                            debug_synchronous,
    typename std::iterator_traits<KeysInputIterator>::value_type*   keys_buffer   = nullptr,
    typename std::iterator_traits<ValuesInputIterator>::value_type* values_buffer = nullptr)
{
    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    static constexpr bool with_custom_config = !std::is_same<Config, default_config>::value;

    using block_sort_config = typename std::
        conditional<with_custom_config, typename Config::block_sort_config, default_config>::type;
    using block_merge_config = typename std::
        conditional<with_custom_config, typename Config::block_merge_config, default_config>::type;
    using wrapped_bs_config
        = wrapped_merge_sort_block_sort_config<block_sort_config, key_type, value_type>;
    using wrapped_bm_config
        = wrapped_merge_sort_block_merge_config<block_merge_config, key_type, value_type>;

    (void)device_merge_sort_compile_time_verifier<
        wrapped_bs_config,
        wrapped_bm_config>; // Some helpful checks during compile-time

    unsigned int sort_items_per_block = 1; // We will get this later from the block_sort algorithm

    if(temporary_storage == nullptr)
    {
        return merge_sort_block_merge<block_merge_config>(temporary_storage,
                                                          storage_size,
                                                          keys_output,
                                                          values_output,
                                                          size,
                                                          sort_items_per_block,
                                                          compare_function,
                                                          stream,
                                                          debug_synchronous,
                                                          keys_buffer,
                                                          values_buffer);
    }

    if(size == size_t(0))
    {
        return hipSuccess;
    }

    merge_sort_block_sort<block_sort_config>(keys_input,
                                             keys_output,
                                             values_input,
                                             values_output,
                                             size,
                                             sort_items_per_block,
                                             compare_function,
                                             stream,
                                             debug_synchronous);
    // ^ sort_items_per_block is now updated
    if(size > sort_items_per_block)
    {
        return merge_sort_block_merge<block_merge_config>(temporary_storage,
                                                          storage_size,
                                                          keys_output,
                                                          values_output,
                                                          size,
                                                          sort_items_per_block,
                                                          compare_function,
                                                          stream,
                                                          debug_synchronous,
                                                          keys_buffer,
                                                          values_buffer);
    }
    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

} // end of detail namespace

/// \brief Parallel merge sort primitive for device level.
///
/// \p merge_sort function performs a device-wide merge sort
/// of keys. Function sorts input keys based on comparison function.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Accepts custom compare_functions for sorting across the device.
///
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] compare_function - binary operation function object that will be used for comparison.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending merge sort is performed on an array of
/// \p float values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;      // e.g., 8
/// float * input;          // e.g., [0.6, 0.3, 0.65, 0.4, 0.2, 0.08, 1, 0.7]
/// float * output;         // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::merge_sort(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::merge_sort(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
/// // keys_output: [0.08, 0.2, 0.3, 0.4, 0.6, 0.65, 0.7, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
hipError_t merge_sort(void * temporary_storage,
                      size_t& storage_size,
                      KeysInputIterator keys_input,
                      KeysOutputIterator keys_output,
                      const size_t size,
                      BinaryFunction compare_function = BinaryFunction(),
                      const hipStream_t stream = 0,
                      bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    return detail::merge_sort_impl<Config>(
        temporary_storage, storage_size,
        keys_input, keys_output, values, values, size,
        compare_function, stream, debug_synchronous
    );
}

/// \brief Parallel ascending merge sort-by-key primitive for device level.
///
/// \p merge_sort function performs a device-wide merge sort
/// of (key, value) pairs. Function sorts input pairs based on comparison function.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Accepts custom compare_functions for sorting across the device.
///
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] values_input - pointer to the first element in the range to sort.
/// \param [out] values_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] compare_function - binary operation function object that will be used for comparison.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending merge sort is performed where input keys are
/// represented by an array of unsigned integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// unsigned int * keys_input;  // e.g., [ 6, 3,  5, 4,  1,  8,  2, 7]
/// double * values_input;      // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// unsigned int * keys_output; // empty array of 8 elements
/// double * values_output;     // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::merge_sort(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::merge_sort(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size
/// );
/// // keys_output:   [ 1,  2, 3, 4,  5,  6, 7,  8]
/// // values_output: [-1, -2, 2, 3, -4, -5, 7, -8]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
hipError_t merge_sort(void * temporary_storage,
                      size_t& storage_size,
                      KeysInputIterator keys_input,
                      KeysOutputIterator keys_output,
                      ValuesInputIterator values_input,
                      ValuesOutputIterator values_output,
                      const size_t size,
                      BinaryFunction compare_function = BinaryFunction(),
                      const hipStream_t stream = 0,
                      bool debug_synchronous = false)
{
    return detail::merge_sort_impl<Config>(
        temporary_storage, storage_size,
        keys_input, keys_output, values_input, values_output, size,
        compare_function, stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SORT_HPP_
