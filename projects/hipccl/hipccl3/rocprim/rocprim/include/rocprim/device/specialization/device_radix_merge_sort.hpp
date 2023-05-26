// Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_MERGE_SORT_HPP_
#define ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_MERGE_SORT_HPP_

#include "../detail/device_radix_sort.hpp"
#include "../device_merge_sort.hpp"
#include "device_radix_block_sort.hpp"

#include <type_traits>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

/// In device_radix_sort, we use this device_radix_sort_merge_sort specialization only
/// for low input sizes (< 1M elements).
template<class Config,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator>
inline hipError_t radix_sort_merge_impl(
    void*                                                           temporary_storage,
    size_t&                                                         storage_size,
    KeysInputIterator                                               keys_input,
    typename std::iterator_traits<KeysInputIterator>::value_type*   keys_buffer,
    KeysOutputIterator                                              keys_output,
    ValuesInputIterator                                             values_input,
    typename std::iterator_traits<ValuesInputIterator>::value_type* values_buffer,
    ValuesOutputIterator                                            values_output,
    unsigned int                                                    size,
    unsigned int                                                    bit,
    unsigned int                                                    end_bit,
    hipStream_t                                                     stream,
    bool                                                            debug_synchronous)
{
    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    const unsigned int current_radix_bits = end_bit - bit;

    constexpr bool is_default_config = std::is_same<Config, default_config>::value;
    // no need to provide default values for merge_sort_config as it only
    // contains two autotuned subalgorithms

    // In the case that the user provides no custom config for merge sort block sort,
    // instead of using the autotuned merge_sort_block_sort_config, use a hard-coded config that
    // a power-of-two items sorted per block.
    using default_block_sort_config =
        typename radix_sort_block_sort_config_base<key_type, value_type>::type;
    using radix_sort_block_sort_config =
        typename std::conditional<is_default_config,
                                  default_block_sort_config,
                                  // extract the relevant config from merge_sort_block_sort_config
                                  typename Config::block_sort_config::sort_config>::type;
    static_assert(
        is_power_of_two(radix_sort_block_sort_config::block_size
                        * radix_sort_block_sort_config::items_per_thread),
        "The sorted items per block of the radix sort merge sort must be a power of two.");

    using merge_sort_block_merge_config = typename std::
        conditional<is_default_config, default_config, typename Config::block_merge_config>::type;

    // Wrap our radix_sort_block_sort kernel config in a merge_sort_block_sort_config
    // just so device_merge_sort_compile_time_verifier can check.
    using merge_sort_block_sort_config
        = merge_sort_block_sort_config<radix_sort_block_sort_config::block_size,
                                       radix_sort_block_sort_config::items_per_thread,
                                       block_sort_algorithm::default_algorithm>;
    using wrapped_bs_config
        = wrapped_merge_sort_block_sort_config<merge_sort_block_sort_config, key_type, value_type>;
    using wrapped_bm_config = wrapped_merge_sort_block_merge_config<merge_sort_block_merge_config,
                                                                    key_type,
                                                                    value_type>;

    // Some helpful checks during compile-time
    (void)device_merge_sort_compile_time_verifier<wrapped_bs_config, wrapped_bm_config>;

    // We will get this later from the block_sort algorithm
    unsigned int sort_items_per_block
        = radix_sort_block_sort_config::block_size * radix_sort_block_sort_config::items_per_thread;

    if(temporary_storage == nullptr)
    {
        return merge_sort_block_merge<merge_sort_block_merge_config>(
            temporary_storage,
            storage_size,
            keys_output,
            values_output,
            size,
            sort_items_per_block,
            radix_merge_compare<Descending, false, key_type>(),
            stream,
            debug_synchronous,
            keys_buffer,
            values_buffer);
    }

    if(size == size_t(0))
    {
        return hipSuccess;
    }

    radix_sort_block_sort<radix_sort_block_sort_config, Descending>(keys_input,
                                                                    keys_output,
                                                                    values_input,
                                                                    values_output,
                                                                    size,
                                                                    sort_items_per_block,
                                                                    bit,
                                                                    end_bit,
                                                                    stream,
                                                                    debug_synchronous);
    // ^ sort_items_per_block is now updated
    if(size > sort_items_per_block)
    {
        if(current_radix_bits == sizeof(key_type) * 8)
        {
            return merge_sort_block_merge<merge_sort_block_merge_config>(
                temporary_storage,
                storage_size,
                keys_output,
                values_output,
                size,
                sort_items_per_block,
                radix_merge_compare<Descending, false, key_type>(),
                stream,
                debug_synchronous,
                keys_buffer,
                values_buffer);
        }
        else
        {
            return merge_sort_block_merge<merge_sort_block_merge_config>(
                temporary_storage,
                storage_size,
                keys_output,
                values_output,
                size,
                sort_items_per_block,
                radix_merge_compare<Descending, true, key_type>(bit, current_radix_bits),
                stream,
                debug_synchronous,
                keys_buffer,
                values_buffer);
        }
    }
    return hipSuccess;
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_MERGE_SORT_HPP_
