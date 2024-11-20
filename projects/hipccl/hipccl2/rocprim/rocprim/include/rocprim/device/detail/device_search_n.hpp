// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEARCH_N_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEARCH_N_HPP_

#include "../../common.hpp"
#include "../../config.hpp"
#include "../config_types.hpp"
#include "../device_reduce.hpp"
#include "../device_search_n_config.hpp"
#include "../device_transform.hpp"

#include <iterator>

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

inline void search_n_start_timer(std::chrono::steady_clock::time_point& start,
                                 const bool                             debug_synchronous)
{
    if(debug_synchronous)
    {
        start = std::chrono::steady_clock::now();
    }
}

template<class SizeType>
ROCPRIM_KERNEL __launch_bounds__(1)
void search_n_init_kernel(SizeType* __restrict__ output, const SizeType target)
{
    *output = target;
}

/// \brief Supports all forms of search_n operations,
/// but the efficiency is insufficient when `items_per_block` is too large.
template<class Config, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL
#ifndef DOXYGEN_DOCUMENTATION_BUILD
__launch_bounds__(device_params<Config>().kernel_config.block_size)
#endif
void search_n_normal_kernel(InputIterator input,
                            size_t* __restrict__ output,
                            const size_t                                                    size,
                            const size_t                                                    count,
                            const typename std::iterator_traits<InputIterator>::value_type* value,
                            const BinaryPredicate binary_predicate)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const size_t this_thread_start_idx
        = (block_id<0>() * items_per_block) + (items_per_thread * block_thread_id<0>());

    // TODO: This could cause load imbalance among threads
    // So maybe there is a better way to do this
    if(size < count + this_thread_start_idx)
    { // not able to find a sequence equal to or longer than count
        return;
    }

    size_t remaining_count    = count;
    size_t sequence_start_idx = this_thread_start_idx;

    const size_t items_this_thread
        = std::min<size_t>(size - this_thread_start_idx, items_per_thread);
    for(size_t i = this_thread_start_idx;
        sequence_start_idx - this_thread_start_idx < items_this_thread
        && i + remaining_count <= size;
        ++i)
    {
        if(binary_predicate(input[i], *value))
        {
            if(--remaining_count == 0)
            {
                atomic_min(output, sequence_start_idx);
                return;
            }
        }
        else
        {
            remaining_count    = count;
            sequence_start_idx = i + 1;
        }
    }
}

template<class Config, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL
#ifndef DOXYGEN_DOCUMENTATION_BUILD
__launch_bounds__(device_params<Config>().kernel_config.block_size)
#endif
void search_n_find_heads_kernel(
    InputIterator                                                   input,
    const size_t                                                    size,
    const typename std::iterator_traits<InputIterator>::value_type* value,
    const BinaryPredicate                                           binary_predicate,
    size_t* __restrict__ unfiltered_heads,
    const size_t group_size)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const size_t this_thread_start_idx
        = (block_id<0>() * items_per_block) + (items_per_thread * block_thread_id<0>());
    const size_t items_this_thread
        = std::min<size_t>(this_thread_start_idx < size ? size - this_thread_start_idx : 0,
                           items_per_thread);

    for(size_t i = this_thread_start_idx; i < this_thread_start_idx + items_this_thread; i++)
    {
        if(binary_predicate(input[i], *value))
        {
            if(i == 0)
            { // is head // `size - i - 1` is the distance to the end
                atomic_min(&(unfiltered_heads[i / group_size]), size - i - 1);
            }
            else if(!binary_predicate(input[i - 1], *value))
            { // is head
                atomic_min(&(unfiltered_heads[i / group_size]), size - i - 1);
            }
        }
    }
}

template<class Config>
ROCPRIM_KERNEL
#ifndef DOXYGEN_DOCUMENTATION_BUILD
__launch_bounds__(device_params<Config>().kernel_config.block_size)
#endif
void search_n_heads_filter_kernel(const size_t size,
                                  const size_t count,
                                  const size_t* __restrict__ heads,
                                  const size_t heads_size,
                                  size_t* __restrict__ filtered_heads,
                                  size_t* __restrict__ filtered_heads_size)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const size_t this_thread_start_idx
        = (block_id<0>() * items_per_block) + (block_thread_id<0>() * items_per_thread);
    const size_t this_thread_end_idx
        = std::min(items_per_thread + this_thread_start_idx, heads_size);
    for(size_t i = this_thread_start_idx; i < this_thread_end_idx; ++i)
    {
        const auto cur_val = heads[i];
        if(cur_val == (size_t)-1)
        { // this is not a valid head
            continue;
        }
        const size_t this_head = size - cur_val - 1;
        if(i + 1 == heads_size)
        { // head of last group
            if(size - this_head < count)
            { // cannot make it to count
                continue;
            }
        }
        else if(i + 2 == heads_size)
        { // the head before last head (last group might be incomplete so, the head before last head can be invalid)
            const auto next_val = heads[i + 1];
            if(((next_val != (size_t)-1) ? ((size - next_val - 1) - this_head - 1)
                                         : (size - this_head))
               < count)
            { // cannot make it to count
                continue;
            }
        }
        else
        { // other heads
            const auto next_val = heads[i + 1];
            if((next_val != (size_t)-1) && (((size - next_val - 1) - this_head - 1) < count))
            { // if next head is invalid, the limit of this head should the next head, else it is possible to make the sequence to count
                continue;
            }
        }
        filtered_heads[atomic_add(filtered_heads_size, 1)] = this_head;
    }
}

template<class Config, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL
#ifndef DOXYGEN_DOCUMENTATION_BUILD
__launch_bounds__(device_params<Config>().kernel_config.block_size)
#endif
void search_n_discard_heads_kernel(
    InputIterator                                                   input,
    const size_t                                                    size,
    const size_t                                                    count,
    const typename std::iterator_traits<InputIterator>::value_type* value,
    const BinaryPredicate                                           binary_predicate,
    size_t* __restrict__ heads,
    size_t* num_heads)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const size_t heads_size = *num_heads;
    if(heads_size == 0)
    {
        return; // should return
    }

    size_t num_blocks_needed = ceiling_div(heads_size * count /*group_size*/, items_per_block);
    if(block_id<0>() >= num_blocks_needed)
    {
        return;
    }

    const size_t this_thread_start_idx
        = (block_id<0>() * items_per_block) + (block_thread_id<0>() * items_per_thread);

    for(size_t global_idx = this_thread_start_idx;
        global_idx < this_thread_start_idx + items_per_block;
        global_idx++)
    {
        const size_t g_id /*group id*/ = global_idx / count /*group_size*/;
        if(g_id >= heads_size)
        {
            return;
        }
        const size_t check_head
            = heads[g_id] + 1; // the `head` is already checked, so we check the next value here
        const size_t check_count = count - 1;
        const size_t idx         = check_head + (global_idx % count);

        if((idx >= size) || (idx >= (check_head + check_count)))
        {
            return;
        }
        if(!binary_predicate(input[idx], *value))
        {
            heads[g_id] = size;
            return;
        }
    }
}

template<class Config, class InputIterator, class OutputIterator, class BinaryPredicate>
ROCPRIM_INLINE
hipError_t search_n_impl(void*          temporary_storage,
                         size_t&        storage_size,
                         InputIterator  input,
                         OutputIterator output,
                         const size_t   size,
                         const size_t   count,
                         const typename std::iterator_traits<InputIterator>::value_type* value,
                         const BinaryPredicate binary_predicate,
                         const hipStream_t     stream,
                         const bool            debug_synchronous)
{
    using input_type  = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using config      = wrapped_search_n_config<Config, input_type>;

    if(count > size)
    { // size must greater than or equal to count
        return hipErrorInvalidValue;
    }

    target_arch target_arch;
    ROCPRIM_RETURN_ON_ERROR(host_target_arch(stream, target_arch));

    const auto         params           = dispatch_target_arch<config>(target_arch);
    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const unsigned int items_per_block  = block_size * items_per_thread;
    const unsigned int num_blocks       = ceiling_div(size, items_per_block);

    std::chrono::steady_clock::time_point start;

    size_t* tmp_output = reinterpret_cast<size_t*>(temporary_storage);

    if(size == 0 || count <= 0)
    { // to be consist to the std::search_n
        // calculate size
        if(tmp_output == nullptr)
        {
            storage_size = sizeof(size_t);
            return hipSuccess;
        }

        // return end or begin
        search_n_start_timer(start, debug_synchronous);
        search_n_init_kernel<<<1, 1, 0, stream>>>(tmp_output, count <= 0 ? 0 : size);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_init_kernel", 1, start);
        ROCPRIM_RETURN_ON_ERROR(
            transform(tmp_output, output, 1, identity<output_type>(), stream, debug_synchronous));
        return hipSuccess;
    }
    else if(count <= params.threshold)
    { // reduce search_n will have a maximum access time of params.threshold
        // So if the count is equals to or smaller than params.threshold, `normal_search_n` should be faster
        // calculate size
        if(tmp_output == nullptr)
        {
            storage_size = sizeof(size_t);
            return hipSuccess;
        }

        // do `normal_search_n`
        search_n_start_timer(start, debug_synchronous);
        search_n_init_kernel<<<1, 1, 0, stream>>>(tmp_output, size);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_init_kernel", 1, start);
        // TODO: There can be overlapping between threads, this probably can be optimized
        search_n_normal_kernel<config><<<num_blocks, block_size, 0, stream>>>(input,
                                                                              tmp_output,
                                                                              size,
                                                                              count,
                                                                              value,
                                                                              binary_predicate);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_normal_kernel", size, start);
        ROCPRIM_RETURN_ON_ERROR(
            transform(tmp_output, output, 1, identity<output_type>(), stream, debug_synchronous));
        return hipSuccess;
    }
    else
    { // the count is greater than params.threshold
        // group_size is equal to `count`
        const size_t num_groups = ceiling_div(size, count /*group_size*/);
        size_t       reduce_storage_size;
        ROCPRIM_RETURN_ON_ERROR(reduce(nullptr,
                                       reduce_storage_size,
                                       reinterpret_cast<size_t*>(0),
                                       output,
                                       size,
                                       num_groups,
                                       minimum<size_t>{},
                                       stream,
                                       debug_synchronous));
        size_t front_size
            = std::max<size_t>(sizeof(size_t) + (sizeof(size_t) * num_groups), reduce_storage_size);
        if(tmp_output == nullptr)
        {
            storage_size = front_size + (sizeof(size_t) * num_groups);
            return hipSuccess;
        }

        const size_t num_blocks_for_heads_filter = ceiling_div(num_groups, items_per_block);
        const size_t num_blocks_for_discard_heads
            = ceiling_div(num_groups * count, items_per_block);

        auto unfiltered_heads = reinterpret_cast<size_t*>(reinterpret_cast<char*>(temporary_storage)
                                                          + sizeof(size_t));
        auto filtered_heads
            = reinterpret_cast<size_t*>(reinterpret_cast<char*>(temporary_storage) + front_size);

        search_n_start_timer(start, debug_synchronous);
        // initialization
        ROCPRIM_RETURN_ON_ERROR(hipMemsetAsync(tmp_output, 0, sizeof(size_t), stream));
        ROCPRIM_RETURN_ON_ERROR(
            hipMemsetAsync(unfiltered_heads, -1, sizeof(size_t) * num_groups * 2, stream));

        // find the thread heads of each group
        search_n_find_heads_kernel<config>
            <<<num_blocks, block_size, 0, stream>>>(input,
                                                    size,
                                                    value,
                                                    binary_predicate,
                                                    unfiltered_heads,
                                                    count /*group_size*/);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_find_heads_kernel", size, start);

        // filter heads
        // move valid heads into filtered_heads, and set the size of filtered_heads to tmp_output
        search_n_heads_filter_kernel<config>
            <<<num_blocks_for_heads_filter, block_size, 0, stream>>>(size,
                                                                     count,
                                                                     unfiltered_heads,
                                                                     num_groups,
                                                                     filtered_heads,
                                                                     tmp_output);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_heads_filter_kernel",
                                                    num_groups,
                                                    start);

        // check if any valid heads make a valid sequence
        // max access time for each item is 1
        // TODO: num_blocks_for_discard_heads is actually graeter than the actural valid filtered_heads_size
        // so the actural num_blocks_for_discard_heads needed is smaller than the current value
        search_n_discard_heads_kernel<config>
            <<<num_blocks_for_discard_heads, block_size, 0, stream>>>(
                input,
                size,
                count,
                value,
                binary_predicate,
                filtered_heads,
                tmp_output); // currently the tmp_output contains the actual size of filtered_heads
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_discard_heads_kernel ",
                                                    num_groups,
                                                    start);

        // calculate the minimum valid head
        ROCPRIM_RETURN_ON_ERROR(reduce(temporary_storage,
                                       reduce_storage_size,
                                       filtered_heads,
                                       output,
                                       size, // original value
                                       num_groups,
                                       minimum<size_t>{},
                                       stream,
                                       debug_synchronous));
        return hipSuccess; // no needs to call transform, return directly
    }
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SEARCH_N_HPP_
