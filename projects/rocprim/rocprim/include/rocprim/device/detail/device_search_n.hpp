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

template<class Config, class InputIterator, class BinaryPredicate>
struct search_n_impl_kernels
{
    inline hipError_t launch_search_n_normal(
        detail::target_arch arch,
        InputIterator       input,
        size_t* __restrict__ output,
        const size_t                                                    size,
        const size_t                                                    possible_head_exist_size,
        const size_t                                                    count,
        const typename std::iterator_traits<InputIterator>::value_type* value,
        const BinaryPredicate                                           binary_predicate,
        dim3                                                            grid,
        dim3                                                            block,
        size_t                                                          shmem,
        hipStream_t                                                     stream)
    {
        auto kernel = [=](auto arch_config)
        {
            static constexpr auto params           = decltype(arch_config)::params;
            static constexpr auto block_size       = params.kernel_config.block_size;
            static constexpr auto items_per_thread = params.kernel_config.items_per_thread;
            static constexpr auto items_per_block  = block_size * items_per_thread;

            const size_t this_thread_start_idx
                = (block_id<0>() * items_per_block) + (items_per_thread * block_thread_id<0>());

            // Not able to find a sequence equal to or longer than count
            if(this_thread_start_idx >= possible_head_exist_size)
            {
                return;
            }

            size_t       remaining_count = count;
            size_t       head            = this_thread_start_idx;
            const size_t this_thread_end_idx
                = std::min<size_t>(this_thread_start_idx + items_per_thread, size);

            for(size_t i = this_thread_start_idx;
                head < this_thread_end_idx && i + remaining_count <= size;
                ++i)
            {
                if(binary_predicate(input[i], *value))
                {
                    if(--remaining_count == 0)
                    {
                        atomic_min(output, head);
                        return;
                    }
                }
                else
                {
                    remaining_count = count;
                    head            = i + 1;
                }
            }
        };
        return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
    }

    template<class SizeType>
    static ROCPRIM_KERNEL ROCPRIM_LAUNCH_BOUNDS(1) void search_n_init_kernel(SizeType* __restrict__ output,
                                                       const SizeType target)
    {
        *output = target;
    }

    inline hipError_t launch_search_n_find_heads(
        detail::target_arch                                             arch,
        InputIterator                                                   input,
        const size_t                                                    input_size,
        const size_t                                                    possible_head_exist_size,
        const typename std::iterator_traits<InputIterator>::value_type* value,
        const BinaryPredicate                                           binary_predicate,
        size_t* __restrict__ unfiltered_heads,
        const size_t group_size,
        dim3         grid,
        dim3         block,
        size_t       shmem,
        hipStream_t  stream)
    {
        auto kernel = [=](auto arch_config)
        {
            static constexpr auto params           = decltype(arch_config)::params;
            static constexpr auto block_size       = params.kernel_config.block_size;
            static constexpr auto items_per_thread = params.kernel_config.items_per_thread;
            static constexpr auto items_per_block  = block_size * items_per_thread;

            const size_t this_thread_start_idx
                = (block_id<0>() * items_per_block) + (items_per_thread * block_thread_id<0>());
            const size_t this_thread_end_idx
                = std::min<size_t>(this_thread_start_idx + items_per_thread,
                                   possible_head_exist_size);
            for(size_t i = this_thread_start_idx; i < this_thread_end_idx; i++)
            {
                if(binary_predicate(input[i], *value))
                {
                    if(i == 0)
                    {
                        // This item is the first head
                        // `input_size - i - 1` is the distance to the end
                        atomic_min(&(unfiltered_heads[i / group_size]), input_size - i - 1);
                    }
                    else if(!binary_predicate(input[i - 1], *value))
                    {
                        // This item is head
                        atomic_min(&(unfiltered_heads[i / group_size]), input_size - i - 1);
                    }
                }
            }
        };
        return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
    }

    inline hipError_t launch_search_n_heads_filter(detail::target_arch arch,
                                                   const size_t        input_size,
                                                   const size_t        count,
                                                   const size_t* __restrict__ unfiltered_heads,
                                                   const size_t unfiltered_heads_size,
                                                   size_t* __restrict__ filtered_heads,
                                                   size_t* __restrict__ filtered_heads_size,
                                                   dim3        grid,
                                                   dim3        block,
                                                   size_t      shmem,
                                                   hipStream_t stream)
    {
        auto kernel = [=](auto arch_config)
        {
            static constexpr auto params           = decltype(arch_config)::params;
            static constexpr auto block_size       = params.kernel_config.block_size;
            static constexpr auto items_per_thread = params.kernel_config.items_per_thread;
            static constexpr auto items_per_block  = block_size * items_per_thread;

            const size_t this_thread_start_idx
                = (block_id<0>() * items_per_block) + (block_thread_id<0>() * items_per_thread);
            const size_t this_thread_end_idx
                = std::min<size_t>(items_per_thread + this_thread_start_idx, unfiltered_heads_size);
            for(size_t i = this_thread_start_idx; i < this_thread_end_idx; ++i)
            {
                const auto cur_val = unfiltered_heads[i];
                // This is not a valid head
                if(cur_val == (size_t)-1)
                {
                    continue;
                }
                const size_t this_head = input_size - cur_val - 1;
                // Other heads
                if(i + 1 < unfiltered_heads_size)
                {
                    const auto next_val = unfiltered_heads[i + 1];
                    // Check if this head is valid by calculating the distance between this head and the next head.
                    if((next_val != (size_t)-1)
                       && (((input_size - next_val - 1) - this_head - 1) < count))
                    {
                        continue;
                    }
                }
                filtered_heads[atomic_add(filtered_heads_size, 1)] = this_head;
            }
        };
        return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
    }

    inline hipError_t launch_search_n_discard_heads(
        detail::target_arch                                             arch,
        InputIterator                                                   input,
        const size_t                                                    input_size,
        const size_t                                                    count,
        const typename std::iterator_traits<InputIterator>::value_type* value,
        const BinaryPredicate                                           binary_predicate,
        size_t* __restrict__ filtered_heads,
        size_t*     filtered_heads_size,
        dim3        grid,
        dim3        block,
        size_t      shmem,
        hipStream_t stream)
    {
        auto kernel = [=](auto arch_config)
        {
            static constexpr auto params           = decltype(arch_config)::params;
            static constexpr auto block_size       = params.kernel_config.block_size;
            static constexpr auto items_per_thread = params.kernel_config.items_per_thread;
            static constexpr auto items_per_block  = block_size * items_per_thread;

            const size_t heads_size = *filtered_heads_size;
            const auto   block_idx  = block_id<0>();
            if(heads_size == 0)
            {
                return;
            }
            const size_t total_check_size  = heads_size * count /*group_size*/;
            const size_t num_blocks_needed = ceiling_div(total_check_size, items_per_block);
            if(block_idx >= num_blocks_needed)
            {
                return;
            }

            const size_t this_thread_start_idx
                = (block_idx * items_per_block) + (block_thread_id<0>() * items_per_thread);
            const size_t this_thread_end_idx
                = std::min<size_t>(items_per_thread + this_thread_start_idx, total_check_size);
            // The `global_idx` is the index of item in the whole `input`
            for(size_t global_idx = this_thread_start_idx; global_idx < this_thread_end_idx;
                global_idx++)
            {
                // The id of the group who contains the item on global_idx
                const size_t group_id = global_idx / count /*group_size*/;
                if(group_id >= heads_size)
                {
                    return;
                }
                const size_t check_head
                    = filtered_heads[group_id]
                      + 1; // The `head` is already checked, so we check the next value here
                const size_t check_count = count - 1;
                const size_t idx         = check_head + (global_idx % count);

                if((idx >= input_size) || (idx >= (check_head + check_count)))
                {
                    return;
                }
                if(!binary_predicate(input[idx], *value))
                {
                    filtered_heads[group_id] = input_size;
                    return;
                }
            }
        };
        return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
    }
};

inline void search_n_start_timer(std::chrono::steady_clock::time_point& start,
                                 const bool                             debug_synchronous)
{
    if(debug_synchronous)
    {
        start = std::chrono::steady_clock::now();
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
    using input_type       = typename std::iterator_traits<InputIterator>::value_type;
    using output_type      = typename std::iterator_traits<OutputIterator>::value_type;
    using config           = wrapped_search_n_config<Config, input_type>;
    using search_n_kernels = search_n_impl_kernels<config, InputIterator, BinaryPredicate>;

    // The `size` must greater than or equal to `count`
    if(count > size)
    {
        return hipErrorInvalidValue;
    }

    target_arch target_arch;
    ROCPRIM_RETURN_ON_ERROR(host_target_arch(stream, target_arch));

    const auto         params           = dispatch_target_arch<config, false>(target_arch);
    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const unsigned int items_per_block  = block_size * items_per_thread;

    std::chrono::steady_clock::time_point start;

    size_t* tmp_output = reinterpret_cast<size_t*>(temporary_storage);

    // Only the items on index that smaller than possible_head_exist_size are possible heads
    const size_t possible_head_exist_size = size - count + 1;

    // To be consistent with std::search_n
    if(size == 0 || count <= 0)
    {
        // Calculate size
        if(tmp_output == nullptr)
        {
            storage_size = sizeof(size_t);
            return hipSuccess;
        }

        // Return end or begin
        search_n_start_timer(start, debug_synchronous);

        search_n_kernels::search_n_init_kernel<<<1, 1, 0, stream>>>(tmp_output,
                                                                    count <= 0 ? 0 : size);
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

        const unsigned int num_blocks = ceiling_div(possible_head_exist_size, items_per_block);

        // do `normal_search_n`
        search_n_start_timer(start, debug_synchronous);
        search_n_kernels::search_n_init_kernel<<<1, 1, 0, stream>>>(tmp_output, size);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_init_kernel", 1, start);

        ROCPRIM_RETURN_ON_ERROR(search_n_kernels{}.launch_search_n_normal(target_arch,
                                                                          input,
                                                                          tmp_output,
                                                                          size,
                                                                          possible_head_exist_size,
                                                                          count,
                                                                          value,
                                                                          binary_predicate,
                                                                          num_blocks,
                                                                          block_size,
                                                                          0,
                                                                          stream));
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_normal_kernel", size, start);
        ROCPRIM_RETURN_ON_ERROR(
            transform(tmp_output, output, 1, identity<output_type>(), stream, debug_synchronous));
        return hipSuccess;
    }

    const size_t num_groups = ceiling_div(possible_head_exist_size, count /*group_size*/);

    size_t reduce_storage_size{};
    ROCPRIM_RETURN_ON_ERROR(reduce(nullptr,
                                   reduce_storage_size,
                                   reinterpret_cast<size_t*>(0),
                                   output,
                                   size, // Original value
                                   num_groups,
                                   minimum<size_t>{},
                                   stream,
                                   debug_synchronous));
    const size_t front_size
        = std::max<size_t>(sizeof(size_t) + (sizeof(size_t) * num_groups), reduce_storage_size);
    if(tmp_output == nullptr)
    {
        storage_size = front_size + (sizeof(size_t) * num_groups);
        return hipSuccess;
    }

    // Prepare device variables
    auto unfiltered_heads
        = reinterpret_cast<size_t*>(reinterpret_cast<char*>(temporary_storage) + sizeof(size_t));
    auto filtered_heads
        = reinterpret_cast<size_t*>(reinterpret_cast<char*>(temporary_storage) + front_size);

    // check if is capturing
    hipStreamCaptureStatus is_capturing = hipStreamCaptureStatusNone;
    ROCPRIM_RETURN_ON_ERROR(hipStreamIsCapturing(stream, &is_capturing));

    search_n_start_timer(start, debug_synchronous);
    // Initialization
    ROCPRIM_RETURN_ON_ERROR(hipMemsetAsync(tmp_output, 0, sizeof(size_t), stream));
    ROCPRIM_RETURN_ON_ERROR(
        hipMemsetAsync(unfiltered_heads, -1, sizeof(size_t) * num_groups * 2, stream));

    // This function processes `possible_head_exist_size` items
    const size_t num_blocks_for_find_heads = ceiling_div(possible_head_exist_size, items_per_block);

    ROCPRIM_RETURN_ON_ERROR(search_n_kernels{}.launch_search_n_find_heads(target_arch,
                                                                          input,
                                                                          size,
                                                                          possible_head_exist_size,
                                                                          value,
                                                                          binary_predicate,
                                                                          unfiltered_heads,
                                                                          count /*group_size*/,
                                                                          num_blocks_for_find_heads,
                                                                          block_size,
                                                                          0,
                                                                          stream));
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_find_heads_kernel",
                                                possible_head_exist_size,
                                                start);

    // This function processes `num_groups` items
    const size_t num_blocks_for_heads_filter = ceiling_div(num_groups, items_per_block);

    ROCPRIM_RETURN_ON_ERROR(search_n_kernels{}.launch_search_n_heads_filter(
        target_arch,
        size, // It is just a value to decode the heads' index value
        count, // It is just a value to determine whether a certain head is invalid
        unfiltered_heads, // Input heads
        num_groups, // Size of unfiltered_heads
        filtered_heads, // Output
        tmp_output, // Used to store the number of items in `filtered_heads`
        num_blocks_for_heads_filter,
        block_size,
        0,
        stream));
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_heads_filter_kernel", num_groups, start);

    size_t h_filtered_heads_size = 0;
    if(is_capturing != hipStreamCaptureStatusActive)
    {
        ROCPRIM_RETURN_ON_ERROR(hipMemcpyAsync(&h_filtered_heads_size,
                                               tmp_output,
                                               sizeof(size_t),
                                               hipMemcpyDeviceToHost,
                                               stream));
        if(h_filtered_heads_size == 0)
        {
            // Return end
            search_n_start_timer(start, debug_synchronous);
            search_n_kernels::search_n_init_kernel<<<1, 1, 0, stream>>>(tmp_output, size);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_init_kernel", 1, start);
            ROCPRIM_RETURN_ON_ERROR(transform(tmp_output,
                                              output,
                                              1,
                                              identity<output_type>(),
                                              stream,
                                              debug_synchronous));
            return hipSuccess;
        }
    }
    else
    {
        h_filtered_heads_size = num_groups;
    }

    // Max access time for each item is 1
    // So the actual num_blocks_for_discard_heads needed is smaller than the current value
    const size_t num_blocks_for_discard_heads
        = ceiling_div(h_filtered_heads_size * count, items_per_block);

    ROCPRIM_RETURN_ON_ERROR(search_n_kernels{}.launch_search_n_discard_heads(
        target_arch,
        input,
        size,
        count,
        value,
        binary_predicate,
        filtered_heads,
        tmp_output, // Currently, `tmp_output` contains the actual size of `filtered_heads`
        num_blocks_for_discard_heads,
        block_size,
        0,
        stream));
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_discard_heads_kernel ",
                                                h_filtered_heads_size,
                                                start);

    // Calculate the minimum valid head
    ROCPRIM_RETURN_ON_ERROR(reduce(temporary_storage,
                                   reduce_storage_size,
                                   filtered_heads,
                                   output,
                                   size, // Original value
                                   h_filtered_heads_size,
                                   minimum<size_t>{},
                                   stream,
                                   debug_synchronous));
    return hipSuccess; // No needs to call transform, return directly
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SEARCH_N_HPP_
