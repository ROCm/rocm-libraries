// Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

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

    static constexpr bool with_custom_config = !std::is_same<Config, default_config>::value;

    // In device_radix_sort, we use this device_radix_sort_merge_sort specialization only
    // for low input sizes (< 1M elements), so we hardcode a kernel configuration most
    // suitable for this (maximum: <256u, 4u>).
    // Use <256u, 4u>, unless smaller is needed to not exceed shared memory maximum.
    using default_radix_sort_block_sort_config =
        typename rocprim::detail::radix_sort_block_sort_config_base<key_type, value_type>::type;
    using block_sort_config
        = kernel_config<rocprim::min(256u, default_radix_sort_block_sort_config::block_size),
                        rocprim::min(4u, default_radix_sort_block_sort_config::items_per_thread)>;

    using block_merge_config = typename std::
        conditional<with_custom_config, typename Config::block_merge_config, default_config>::type;

    // Wrap our radix_sort_block_sort kernel config in a merge_sort_block_sort_config
    // just so device_merge_sort_compile_time_verifier can check.
    using wrapped_bs_config = wrapped_merge_sort_block_sort_config<
        merge_sort_block_sort_config<block_sort_config::block_size,
                                     block_sort_config::items_per_thread,
                                     block_sort_algorithm::default_algorithm>,
        key_type,
        value_type>;
    using wrapped_bm_config
        = wrapped_merge_sort_block_merge_config<block_merge_config, key_type, value_type>;

    (void)device_merge_sort_compile_time_verifier<
        wrapped_bs_config,
        wrapped_bm_config>; // Some helpful checks during compile-time

    unsigned int sort_items_per_block
        = block_sort_config::block_size
          * block_sort_config::
              items_per_thread; // We will get this later from the block_sort algorithm

    if(temporary_storage == nullptr)
    {
        return merge_sort_block_merge<block_merge_config>(
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

    radix_sort_block_sort<block_sort_config, Descending>(keys_input,
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
            return merge_sort_block_merge<block_merge_config>(
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
            return merge_sort_block_merge<block_merge_config>(
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
