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

#ifndef ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_
#define ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_

#include <iostream>
#include <iterator>
#include <type_traits>
#include <utility>

#include "../config.hpp"
#include "../detail/radix_sort.hpp"
#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "detail/config/device_radix_sort_onesweep.hpp"
#include "detail/device_radix_sort.hpp"
#include "device_transform.hpp"
#include "specialization/device_radix_block_sort.hpp"
#include "specialization/device_radix_merge_sort.hpp"

/// \addtogroup devicemodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#ifndef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

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

#endif

template<class Size>
using offset_type_t = std::conditional_t<
    sizeof(Size) <= 4,
    unsigned int,
    size_t
>;

template<class Config, bool Descending, class KeysInputIterator, class Offset>
ROCPRIM_KERNEL
    __launch_bounds__(device_params<Config>().histogram.block_size) void onesweep_histograms_kernel(
        KeysInputIterator  keys_input,
        Offset*            global_digit_counts,
        const Offset       size,
        const Offset       full_blocks,
        const unsigned int begin_bit,
        const unsigned int end_bit)
{
    static constexpr radix_sort_onesweep_config_params params = device_params<Config>();
    onesweep_histograms<params.histogram.block_size,
                        params.histogram.items_per_thread,
                        params.radix_bits_per_place,
                        Descending>(keys_input,
                                    global_digit_counts,
                                    size,
                                    full_blocks,
                                    begin_bit,
                                    end_bit);
}

template<class Config, class Offset>
ROCPRIM_KERNEL __launch_bounds__(
    device_params<Config>()
        .histogram.block_size) void onesweep_scan_histograms_kernel(Offset* global_digit_offsets)
{
    static constexpr radix_sort_onesweep_config_params params = device_params<Config>();
    onesweep_scan_histograms<params.histogram.block_size, params.radix_bits_per_place>(
        global_digit_offsets);
}

template<class Config,
         bool Descending,
         class KeysInputIterator,
         class ValuesInputIterator,
         class Offset>
inline hipError_t radix_sort_onesweep_global_offsets(KeysInputIterator keys_input,
                                                     ValuesInputIterator,
                                                     Offset*            global_digit_offsets,
                                                     const Offset       size,
                                                     const unsigned int digit_places,
                                                     const unsigned     begin_bit,
                                                     const unsigned     end_bit,
                                                     const hipStream_t  stream,
                                                     const bool         debug_synchronous)
{
    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using config     = wrapped_radix_sort_onesweep_config<Config, key_type, value_type>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const radix_sort_onesweep_config_params params = dispatch_target_arch<config>(target_arch);

    const unsigned int items_per_block
        = params.histogram.block_size * params.histogram.items_per_thread;

    const Offset blocks      = ::rocprim::detail::ceiling_div<Offset>(size, items_per_block);
    const Offset full_blocks = size % items_per_block == 0 ? blocks : blocks - 1;

    const unsigned int radix_size_per_place = 1u << params.radix_bits_per_place;
    const unsigned int places = ceiling_div(end_bit - begin_bit, params.radix_bits_per_place);
    const unsigned int bins   = radix_size_per_place * places;

    // Reset the histogram
    hipError_t error = hipMemsetAsync(global_digit_offsets, 0, sizeof(Offset) * bins, stream);
    if(error != hipSuccess)
        return error;

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous)
    {
        std::cout << "blocks " << blocks << '\n';
        std::cout << "full_blocks " << full_blocks << '\n';
        start = std::chrono::high_resolution_clock::now();
    }

    // Compute a histogram for each digit.
    hipLaunchKernelGGL(HIP_KERNEL_NAME(onesweep_histograms_kernel<config, Descending>),
                       dim3(blocks),
                       dim3(params.histogram.block_size),
                       0,
                       stream,
                       keys_input,
                       global_digit_offsets,
                       size,
                       full_blocks,
                       begin_bit,
                       end_bit);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("compute_global_digit_histograms", size, start);

    // Scan each histogram separately to get the final offsets.
    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }

    hipLaunchKernelGGL(HIP_KERNEL_NAME(onesweep_scan_histograms_kernel<config>),
                       dim3(digit_places), // One block for every digit place.
                       dim3(params.histogram.block_size),
                       0,
                       stream,
                       global_digit_offsets);

    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_global_digit_histograms", bins, start);
    return hipSuccess;
}

template<class Config,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class Offset>
ROCPRIM_KERNEL
    __launch_bounds__(device_params<Config>().sort.block_size) void onesweep_iteration_kernel(
        KeysInputIterator        keys_input,
        KeysOutputIterator       keys_output,
        ValuesInputIterator      values_input,
        ValuesOutputIterator     values_output,
        const unsigned int       size,
        Offset*                  global_digit_offsets_in,
        Offset*                  global_digit_offsets_out,
        onesweep_lookback_state* lookback_states,
        const unsigned int       bit,
        const unsigned int       current_radix_bits,
        const unsigned int       full_blocks)
{
    static constexpr radix_sort_onesweep_config_params params = device_params<Config>();
    onesweep_iteration<params.sort.block_size,
                       params.sort.items_per_thread,
                       params.radix_bits_per_place,
                       Descending,
                       params.radix_rank_algorithm>(keys_input,
                                                    keys_output,
                                                    values_input,
                                                    values_output,
                                                    size,
                                                    global_digit_offsets_in,
                                                    global_digit_offsets_out,
                                                    lookback_states,
                                                    bit,
                                                    current_radix_bits,
                                                    full_blocks);
}

template<class Config,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class Offset>
inline hipError_t radix_sort_onesweep_iteration(
    KeysInputIterator                                               keys_input,
    typename std::iterator_traits<KeysInputIterator>::value_type*   keys_tmp,
    KeysOutputIterator                                              keys_output,
    ValuesInputIterator                                             values_input,
    typename std::iterator_traits<ValuesInputIterator>::value_type* values_tmp,
    ValuesOutputIterator                                            values_output,
    const Offset                                                    size,
    Offset*                                                         global_digit_offsets_in,
    Offset*                                                         global_digit_offsets_out,
    onesweep_lookback_state*                                        lookback_states,
    const bool                                                      from_input,
    const bool                                                      to_output,
    const unsigned int                                              bit,
    const unsigned int                                              end_bit,
    const hipStream_t                                               stream,
    const bool                                                      debug_synchronous)
{
    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using config     = wrapped_radix_sort_onesweep_config<Config, key_type, value_type>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const radix_sort_onesweep_config_params params = dispatch_target_arch<config>(target_arch);

    const unsigned int items_per_block = params.sort.block_size * params.sort.items_per_thread;
    const unsigned int current_radix_bits
        = ::rocprim::min(params.radix_bits_per_place, end_bit - bit);

    const unsigned int radix_size_per_place     = 1u << params.radix_bits_per_place;
    const unsigned int max_items_per_full_batch = 1u << 30;
    const unsigned int items_per_full_batch
        = max_items_per_full_batch - max_items_per_full_batch % items_per_block;

    const unsigned int batches = ceiling_div(size, items_per_full_batch);
    const unsigned int items_per_batch
        = static_cast<unsigned int>(::rocprim::min<size_t>(size, items_per_full_batch));

    for(Offset batch = 0; batch < batches; ++batch)
    {
        const Offset       offset             = batch * items_per_batch;
        const Offset       items_left         = size - offset;
        const unsigned int current_batch_size
            = static_cast<unsigned int>(::rocprim::min<Offset>(items_left, items_per_batch));
        const unsigned int blocks
            = ::rocprim::detail::ceiling_div<Offset>(current_batch_size, items_per_block);
        const unsigned int full_blocks
            = current_batch_size % items_per_block == 0 ? blocks : blocks - 1;
        const unsigned int num_lookback_states = radix_size_per_place * blocks;

        // Reset lookback scan states to zero, indicating empty prefix.
        hipError_t error = hipMemsetAsync(lookback_states,
                                          0,
                                          sizeof(onesweep_lookback_state) * num_lookback_states,
                                          stream);
        if(error != hipSuccess)
            return error;

        std::chrono::high_resolution_clock::time_point start;
        if(debug_synchronous)
        {
            std::cout << "radix_bits " << params.radix_bits_per_place << '\n';
            std::cout << "items_per_block " << items_per_block << '\n';
            std::cout << "items_per_full_batch " << items_per_full_batch << '\n';
            std::cout << "bit " << bit << '\n';
            std::cout << "current_radix_bits " << current_radix_bits << '\n';
            std::cout << "batches " << batches << '\n';
            std::cout << "batch " << batch << '\n';
            std::cout << "items_left " << items_left << '\n';
            std::cout << "current_batch_size " << current_batch_size << '\n';
            std::cout << "offset " << offset << '\n';
            std::cout << "blocks " << blocks << '\n';
            std::cout << "full_blocks " << full_blocks << '\n';
            start = std::chrono::high_resolution_clock::now();
        }

        if(from_input && to_output)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(onesweep_iteration_kernel<config, Descending>),
                               dim3(blocks),
                               dim3(params.sort.block_size),
                               0,
                               stream,
                               keys_input + offset,
                               keys_output,
                               values_input + offset,
                               values_output,
                               current_batch_size,
                               global_digit_offsets_in,
                               global_digit_offsets_out,
                               lookback_states,
                               bit,
                               current_radix_bits,
                               full_blocks);
        }
        else if(from_input)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(onesweep_iteration_kernel<config, Descending>),
                               dim3(blocks),
                               dim3(params.sort.block_size),
                               0,
                               stream,
                               keys_input + offset,
                               keys_tmp,
                               values_input + offset,
                               values_tmp,
                               current_batch_size,
                               global_digit_offsets_in,
                               global_digit_offsets_out,
                               lookback_states,
                               bit,
                               current_radix_bits,
                               full_blocks);
        }
        else if(to_output)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(onesweep_iteration_kernel<config, Descending>),
                               dim3(blocks),
                               dim3(params.sort.block_size),
                               0,
                               stream,
                               keys_tmp + offset,
                               keys_output,
                               values_tmp + offset,
                               values_output,
                               current_batch_size,
                               global_digit_offsets_in,
                               global_digit_offsets_out,
                               lookback_states,
                               bit,
                               current_radix_bits,
                               full_blocks);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(onesweep_iteration_kernel<config, Descending>),
                               dim3(blocks),
                               dim3(params.sort.block_size),
                               0,
                               stream,
                               keys_output + offset,
                               keys_tmp,
                               values_output + offset,
                               values_tmp,
                               current_batch_size,
                               global_digit_offsets_in,
                               global_digit_offsets_out,
                               lookback_states,
                               bit,
                               current_radix_bits,
                               full_blocks);
        }

        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("onesweep_iteration", size, start);

        std::swap(global_digit_offsets_in, global_digit_offsets_out);
    }
    return hipSuccess;
}

template<class Config,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class Size>
inline hipError_t radix_sort_onesweep_impl(
    void*                                                           temporary_storage,
    size_t&                                                         storage_size,
    KeysInputIterator                                               keys_input,
    typename std::iterator_traits<KeysInputIterator>::value_type*   keys_tmp,
    KeysOutputIterator                                              keys_output,
    ValuesInputIterator                                             values_input,
    typename std::iterator_traits<ValuesInputIterator>::value_type* values_tmp,
    ValuesOutputIterator                                            values_output,
    const Size                                                      size,
    bool&                                                           is_result_in_output,
    const unsigned int                                              begin_bit,
    const unsigned int                                              end_bit,
    const hipStream_t                                               stream,
    const bool                                                      debug_synchronous)
{
    using key_type    = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type  = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using offset_type = offset_type_t<Size>;

    using config = wrapped_radix_sort_onesweep_config<Config, key_type, value_type>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const radix_sort_onesweep_config_params params = dispatch_target_arch<config>(target_arch);

    const unsigned int sort_items_per_block = params.sort.block_size * params.sort.items_per_thread;
    const unsigned int radix_size_per_place = 1u << params.radix_bits_per_place;
    const unsigned int max_items_per_full_batch = 1u << 30;
    const unsigned int items_per_full_batch
        = max_items_per_full_batch - max_items_per_full_batch % sort_items_per_block;

    const unsigned int places = ceiling_div(end_bit - begin_bit, params.radix_bits_per_place);
    const unsigned int bins   = radix_size_per_place * places;
    const unsigned int items_per_batch
        = static_cast<unsigned int>(::rocprim::min<size_t>(size, items_per_full_batch));
    const unsigned int num_lookback_states
        = radix_size_per_place * ceiling_div(items_per_batch, sort_items_per_block);

    constexpr bool with_values        = !std::is_same<value_type, ::rocprim::empty_type>::value;
    const bool     with_double_buffer = keys_tmp != nullptr;

    offset_type*             global_digit_offsets;
    offset_type*             global_digit_offsets_tmp;
    onesweep_lookback_state* lookback_states;
    key_type*                keys_tmp_storage;
    value_type*              values_tmp_storage;

    const hipError_t partition_result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::ptr_aligned_array(&global_digit_offsets, bins),
            detail::temp_storage::ptr_aligned_array(&global_digit_offsets_tmp,
                                                    radix_size_per_place),
            detail::temp_storage::ptr_aligned_array(&lookback_states, num_lookback_states),
            detail::temp_storage::ptr_aligned_array(&keys_tmp_storage,
                                                    !with_double_buffer ? size : 0),
            detail::temp_storage::ptr_aligned_array(&values_tmp_storage,
                                                    !with_double_buffer && with_values ? size
                                                                                       : 0)));

    if(partition_result != hipSuccess || temporary_storage == nullptr)
    {
        return partition_result;
    }

    if(size == 0)
        return hipSuccess;

    if(debug_synchronous)
    {
        std::cout << "radix_size " << radix_size_per_place << '\n';
        std::cout << "digit_places " << places << '\n';
        std::cout << "histograms_size " << bins << '\n';
        std::cout << "num_lookback_states " << num_lookback_states << '\n';
        hipError_t error = hipStreamSynchronize(stream);
        if(error != hipSuccess)
            return error;
    }

    // Compute the global digit offset, for each digit and for each digit place.
    {
        hipError_t error
            = radix_sort_onesweep_global_offsets<Config, Descending>(keys_input,
                                                                     values_input,
                                                                     global_digit_offsets,
                                                                     static_cast<offset_type>(size),
                                                                     places,
                                                                     begin_bit,
                                                                     end_bit,
                                                                     stream,
                                                                     debug_synchronous);
        if(error != hipSuccess)
            return error;
    }

    if(!with_double_buffer)
    {
        keys_tmp   = keys_tmp_storage;
        values_tmp = values_tmp_storage;
    }

    // Copy input keys and values if necessary (in-place sorting: input and output iterators are equal).
    bool to_output  = with_double_buffer || (places - 1) % 2 == 0;
    bool from_input = true;
    if(!with_double_buffer && to_output)
    {
        const bool keys_alias
            = ::rocprim::detail::can_iterators_alias(keys_input, keys_output, size);
        const bool values_alias
            = with_values
              && ::rocprim::detail::can_iterators_alias(values_input, values_output, size);
        if(keys_alias || values_alias)
        {
            hipError_t error = ::rocprim::transform(keys_input,
                                                    keys_tmp,
                                                    size,
                                                    ::rocprim::identity<key_type>(),
                                                    stream,
                                                    debug_synchronous);
            if(error != hipSuccess)
                return error;

            if(with_values)
            {
                hipError_t error = ::rocprim::transform(values_input,
                                                        values_tmp,
                                                        size,
                                                        ::rocprim::identity<value_type>(),
                                                        stream,
                                                        debug_synchronous);
                if(error != hipSuccess)
                    return error;
            }

            from_input = false;
        }
    }

    // Sort each digit place iteratively.
    for(unsigned bit = begin_bit, place = 0; bit < end_bit;
        bit += params.radix_bits_per_place, ++place)
    {
        hipError_t error = radix_sort_onesweep_iteration<Config, Descending>(
            keys_input,
            keys_tmp,
            keys_output,
            values_input,
            values_tmp,
            values_output,
            static_cast<offset_type>(size),
            global_digit_offsets + place * radix_size_per_place,
            global_digit_offsets_tmp,
            lookback_states,
            from_input,
            to_output,
            bit,
            end_bit,
            stream,
            debug_synchronous);
        if(error != hipSuccess)
            return error;

        is_result_in_output = to_output;
        from_input          = false;
        to_output           = !to_output;
    }

    return hipSuccess;
}

template<class Config,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class Size>
inline hipError_t
    radix_sort_impl(void*                                                         temporary_storage,
                    size_t&                                                       storage_size,
                    KeysInputIterator                                             keys_input,
                    typename std::iterator_traits<KeysInputIterator>::value_type* keys_tmp,
                    KeysOutputIterator                                            keys_output,
                    ValuesInputIterator                                           values_input,
                    typename std::iterator_traits<ValuesInputIterator>::value_type* values_tmp,
                    ValuesOutputIterator                                            values_output,
                    Size                                                            size,
                    bool&        is_result_in_output,
                    unsigned int begin_bit,
                    unsigned int end_bit,
                    hipStream_t  stream,
                    bool         debug_synchronous)
{
    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    static_assert(
        std::is_same<key_type,
                     typename std::iterator_traits<KeysOutputIterator>::value_type>::value,
        "KeysInputIterator and KeysOutputIterator must have the same value_type");
    static_assert(
        std::is_same<value_type,
                     typename std::iterator_traits<ValuesOutputIterator>::value_type>::value,
        "ValuesInputIterator and ValuesOutputIterator must have the same value_type");

    constexpr bool is_default_config = std::is_same<Config, default_config>::value;
    // if config is not custom, provide default value for merge sort limit
    constexpr size_t merge_sort_limit = std::
        conditional<is_default_config, radix_sort_config_v2<>, Config>::type::merge_sort_limit;

    // Instantiate single sort config to find the threshold that determines which algorithm is used.

    // In the case that the user provides no custom config for the single sort,
    // instead of using the autotuned merge_sort_block_sort_config, use a hard-coded config that
    // significantly improves performance in the case that only a single block is launched.
    // Higher performance is achieved by increasing compute unit utilization.
    // Use <256u, 4u>, unless smaller is needed to not exceed shared memory maximum.
    constexpr bool use_default_small_block_sort
        = is_default_config
          || std::is_same<typename Config::single_sort_config, default_config>::value;
    using default_radix_sort_block_sort_config =
        typename radix_sort_block_sort_config_base<key_type, value_type>::type;
    using default_block_sort_config
        = kernel_config<rocprim::min(256u, default_radix_sort_block_sort_config::block_size),
                        rocprim::min(4u, default_radix_sort_block_sort_config::items_per_thread)>;
    using block_sort_config = typename std::conditional<use_default_small_block_sort,
                                                        default_block_sort_config,
                                                        typename Config::single_sort_config>::type;

    unsigned int single_sort_items_per_block
        = block_sort_config::block_size * block_sort_config::items_per_thread;
    if(size <= single_sort_items_per_block)
    {
        if(temporary_storage == nullptr)
        {
            storage_size = ::rocprim::detail::align_size(1);
            return hipSuccess;
        }

        if(size == 0u)
        {
            is_result_in_output = true;
            return hipSuccess;
        }
        is_result_in_output = true;
        // block_sort_config is never default_config
        return radix_sort_block_sort<block_sort_config, Descending>(keys_input,
                                                                    keys_output,
                                                                    values_input,
                                                                    values_output,
                                                                    static_cast<unsigned int>(size),
                                                                    single_sort_items_per_block,
                                                                    begin_bit,
                                                                    end_bit,
                                                                    stream,
                                                                    debug_synchronous);
    }
    // For sizeof(key_type) <= 2, onesweep is 2x/3x faster (also with values) when
    // input_size > 100K, so don't use radix_sort_merge_sort then.
    else if(size <= merge_sort_limit && (sizeof(key_type) > 2 || size < 100000))
    {
        is_result_in_output = true;
        // note: Config::merge_sort_config may be default_config
        using merge_sort_config = typename Config::merge_sort_config;
        return radix_sort_merge_impl<merge_sort_config, Descending>(temporary_storage,
                                                                    storage_size,
                                                                    keys_input,
                                                                    keys_tmp,
                                                                    keys_output,
                                                                    values_input,
                                                                    values_tmp,
                                                                    values_output,
                                                                    static_cast<unsigned int>(size),
                                                                    begin_bit,
                                                                    end_bit,
                                                                    stream,
                                                                    debug_synchronous);
    }
    else
    {
        // note: Config::onesweep_config may be default_config
        using onesweep_config = typename Config::onesweep_config;
        return radix_sort_onesweep_impl<onesweep_config, Descending>(temporary_storage,
                                                                     storage_size,
                                                                     keys_input,
                                                                     keys_tmp,
                                                                     keys_output,
                                                                     values_input,
                                                                     values_tmp,
                                                                     values_output,
                                                                     size,
                                                                     is_result_in_output,
                                                                     begin_bit,
                                                                     end_bit,
                                                                     stream,
                                                                     debug_synchronous);
    }
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // end namespace detail

/// \brief Parallel ascending radix sort primitive for device level.
///
/// \p radix_sort_keys function performs a device-wide radix sort
/// of keys. Function sorts input keys in ascending order.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input and \p keys_output must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam Size - integral type that represents the problem size.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed on an array of
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
/// rocprim::radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_keys(
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
    class Size,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t radix_sort_keys(void * temporary_storage,
                           size_t& storage_size,
                           KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           Size size,
                           unsigned int begin_bit = 0,
                           unsigned int end_bit = 8 * sizeof(Key),
                           hipStream_t stream = 0,
                           bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    empty_type * values = nullptr;
    bool ignored;
    return detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values, nullptr, values,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

/// \brief Parallel descending radix sort primitive for device level.
///
/// \p radix_sort_keys_desc function performs a device-wide radix sort
/// of keys. Function sorts input keys in descending order.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input and \p keys_output must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam Size - integral type that represents the problem size.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;    // e.g., 8
/// int * input;          // e.g., [6, 3, 5, 4, 2, 8, 1, 7]
/// int * output;         // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
/// // keys_output: [8, 7, 6, 5, 4, 3, 2, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class Size,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t radix_sort_keys_desc(void * temporary_storage,
                                size_t& storage_size,
                                KeysInputIterator keys_input,
                                KeysOutputIterator keys_output,
                                Size size,
                                unsigned int begin_bit = 0,
                                unsigned int end_bit = 8 * sizeof(Key),
                                hipStream_t stream = 0,
                                bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    empty_type * values = nullptr;
    bool ignored;
    return detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values, nullptr, values,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

/// \brief Parallel ascending radix sort-by-key primitive for device level.
///
/// \p radix_sort_pairs_desc function performs a device-wide radix sort
/// of (key, value) pairs. Function sorts input pairs in ascending order of keys.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input, \p keys_output, \p values_input and \p values_output must
/// have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam Size - integral type that represents the problem size.
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
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed where input keys are
/// represented by an array of unsigned integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// unsigned int * keys_input;  // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;      // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// unsigned int * keys_output; // empty array of 8 elements
/// double * values_output;     // empty array of 8 elements
///
/// // Keys are in range [0; 8], so we can limit compared bit to bits on indexes
/// // 0, 1, 2, 3, and 4. In order to do this begin_bit is set to 0 and end_bit
/// // is set to 5.
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size, 0, 5
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size, 0, 5
/// );
/// // keys_output:   [ 1,  1, 3, 4,  5,  6, 7,  8]
/// // values_output: [-1, -2, 2, 3, -4, -5, 7, -8]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class Size,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t radix_sort_pairs(void * temporary_storage,
                            size_t& storage_size,
                            KeysInputIterator keys_input,
                            KeysOutputIterator keys_output,
                            ValuesInputIterator values_input,
                            ValuesOutputIterator values_output,
                            Size size,
                            unsigned int begin_bit = 0,
                            unsigned int end_bit = 8 * sizeof(Key),
                            hipStream_t stream = 0,
                            bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    bool ignored;
    return detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values_input, nullptr, values_output,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

/// \brief Parallel descending radix sort-by-key primitive for device level.
///
/// \p radix_sort_pairs_desc function performs a device-wide radix sort
/// of (key, value) pairs. Function sorts input pairs in descending order of keys.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input, \p keys_output, \p values_input and \p values_output must
/// have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam Size - integral type that represents the problem size.
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
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed where input keys are
/// represented by an array of integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;       // e.g., 8
/// int * keys_input;        // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;   // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// int * keys_output;       // empty array of 8 elements
/// double * values_output;  // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size
/// );
/// // keys_output:   [ 8, 7,  6,  5, 4, 3,  1,  1]
/// // values_output: [-8, 7, -5, -4, 3, 2, -1, -2]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class Size,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t radix_sort_pairs_desc(void * temporary_storage,
                                 size_t& storage_size,
                                 KeysInputIterator keys_input,
                                 KeysOutputIterator keys_output,
                                 ValuesInputIterator values_input,
                                 ValuesOutputIterator values_output,
                                 Size size,
                                 unsigned int begin_bit = 0,
                                 unsigned int end_bit = 8 * sizeof(Key),
                                 hipStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    bool ignored;
    return detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values_input, nullptr, values_output,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

/// \brief Parallel ascending radix sort primitive for device level.
///
/// \p radix_sort_keys function performs a device-wide radix sort
/// of keys. Function sorts input keys in ascending order.
///
/// \par Overview
/// * The contents of both buffers of \p keys may be altered by the sorting function.
/// * \p current() of \p keys is used as the input.
/// * The function will update \p current() of \p keys to point to the buffer
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam Key - key type. Must be an integral type or a floating-point type.
/// \tparam Size - integral type that represents the problem size.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys - reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed on an array of
/// \p float values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;  // e.g., 8
/// float * input;      // e.g., [0.6, 0.3, 0.65, 0.4, 0.2, 0.08, 1, 0.7]
/// float * tmp;        // empty array of 8 elements
/// // Create double-buffer
/// rocprim::double_buffer<float> keys(input, tmp);
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size
/// );
/// // keys.current(): [0.08, 0.2, 0.3, 0.4, 0.6, 0.65, 0.7, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class Key,
    class Size
>
inline
hipError_t radix_sort_keys(void * temporary_storage,
                           size_t& storage_size,
                           double_buffer<Key>& keys,
                           Size size,
                           unsigned int begin_bit = 0,
                           unsigned int end_bit = 8 * sizeof(Key),
                           hipStream_t stream = 0,
                           bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    empty_type * values = nullptr;
    bool         is_result_in_output;
    hipError_t error = detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values, values, values,
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && error == hipSuccess && is_result_in_output)
    {
        keys.swap();
    }
    return error;
}

/// \brief Parallel descending radix sort primitive for device level.
///
/// \p radix_sort_keys_desc function performs a device-wide radix sort
/// of keys. Function sorts input keys in descending order.
///
/// \par Overview
/// * The contents of both buffers of \p keys may be altered by the sorting function.
/// * \p current() of \p keys is used as the input.
/// * The function will update \p current() of \p keys to point to the buffer
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam Key - key type. Must be an integral type or a floating-point type.
/// \tparam Size - integral type that represents the problem size.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys - reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;  // e.g., 8
/// int * input;        // e.g., [6, 3, 5, 4, 2, 8, 1, 7]
/// int * tmp;          // empty array of 8 elements
/// // Create double-buffer
/// rocprim::double_buffer<int> keys(input, tmp);
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size
/// );
/// // keys.current(): [8, 7, 6, 5, 4, 3, 2, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class Key,
    class Size
>
inline
hipError_t radix_sort_keys_desc(void * temporary_storage,
                                size_t& storage_size,
                                double_buffer<Key>& keys,
                                Size size,
                                unsigned int begin_bit = 0,
                                unsigned int end_bit = 8 * sizeof(Key),
                                hipStream_t stream = 0,
                                bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    empty_type * values = nullptr;
    bool         is_result_in_output;
    hipError_t error = detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values, values, values,
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && error == hipSuccess && is_result_in_output)
    {
        keys.swap();
    }
    return error;
}

/// \brief Parallel ascending radix sort-by-key primitive for device level.
///
/// \p radix_sort_pairs_desc function performs a device-wide radix sort
/// of (key, value) pairs. Function sorts input pairs in ascending order of keys.
///
/// \par Overview
/// * The contents of both buffers of \p keys and \p values may be altered by the sorting function.
/// * \p current() of \p keys and \p values are used as the input.
/// * The function will update \p current() of \p keys and \p values to point to buffers
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam Key - key type. Must be an integral type or a floating-point type.
/// \tparam Value - value type.
/// \tparam Size - integral type that represents the problem size.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys - reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in,out] values - reference to the double-buffer of values, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed where input keys are
/// represented by an array of unsigned integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// unsigned int * keys_input;  // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;      // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// unsigned int * keys_tmp;    // empty array of 8 elements
/// double*  values_tmp;        // empty array of 8 elements
/// // Create double-buffers
/// rocprim::double_buffer<unsigned int> keys(keys_input, keys_tmp);
/// rocprim::double_buffer<double> values(values_input, values_tmp);
///
/// // Keys are in range [0; 8], so we can limit compared bit to bits on indexes
/// // 0, 1, 2, 3, and 4. In order to do this begin_bit is set to 0 and end_bit
/// // is set to 5.
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size,
///     0, 5
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size,
///     0, 5
/// );
/// // keys.current():   [ 1,  1, 3, 4,  5,  6, 7,  8]
/// // values.current(): [-1, -2, 2, 3, -4, -5, 7, -8]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class Key,
    class Value,
    class Size
>
inline
hipError_t radix_sort_pairs(void * temporary_storage,
                            size_t& storage_size,
                            double_buffer<Key>& keys,
                            double_buffer<Value>& values,
                            Size size,
                            unsigned int begin_bit = 0,
                            unsigned int end_bit = 8 * sizeof(Key),
                            hipStream_t stream = 0,
                            bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    bool       is_result_in_output;
    hipError_t error = detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values.current(), values.current(), values.alternate(),
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && error == hipSuccess && is_result_in_output)
    {
        keys.swap();
        values.swap();
    }
    return error;
}

/// \brief Parallel descending radix sort-by-key primitive for device level.
///
/// \p radix_sort_pairs_desc function performs a device-wide radix sort
/// of (key, value) pairs. Function sorts input pairs in descending order of keys.
///
/// \par Overview
/// * The contents of both buffers of \p keys and \p values may be altered by the sorting function.
/// * \p current() of \p keys and \p values are used as the input.
/// * The function will update \p current() of \p keys and \p values to point to buffers
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam Key - key type. Must be an integral type or a floating-point type.
/// \tparam Value - value type.
/// \tparam Size - integral type that represents the problem size.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys - reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in,out] values - reference to the double-buffer of values, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed where input keys are
/// represented by an array of integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;       // e.g., 8
/// int * keys_input;        // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;   // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// int * keys_tmp;          // empty array of 8 elements
/// double * values_tmp;     // empty array of 8 elements
/// // Create double-buffers
/// rocprim::double_buffer<int> keys(keys_input, keys_tmp);
/// rocprim::double_buffer<double> values(values_input, values_tmp);
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size
/// );
/// // keys.current():   [ 8, 7,  6,  5, 4, 3,  1,  1]
/// // values.current(): [-8, 7, -5, -4, 3, 2, -1, -2]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class Key,
    class Value,
    class Size
>
inline
hipError_t radix_sort_pairs_desc(void * temporary_storage,
                                 size_t& storage_size,
                                 double_buffer<Key>& keys,
                                 double_buffer<Value>& values,
                                 Size size,
                                 unsigned int begin_bit = 0,
                                 unsigned int end_bit = 8 * sizeof(Key),
                                 hipStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    bool       is_result_in_output;
    hipError_t error = detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values.current(), values.current(), values.alternate(),
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && error == hipSuccess && is_result_in_output)
    {
        keys.swap();
        values.swap();
    }
    return error;
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group devicemodule

#endif // ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_
