// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEARCH_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEARCH_HPP_

#include "../../detail/temp_storage.hpp"

#include "../../common.hpp"
#include "../../config.hpp"

#include "../../intrinsics.hpp"
#include "../../iterator/reverse_iterator.hpp"
#include "../config_types.hpp"
#include "../device_search_config.hpp"
#include "../device_transform.hpp"

#include <iostream>
#include <iterator>

#include <cstddef>
#include <cstdio>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class ArchConfig, class InputIterator1, class InputIterator2, class BinaryFunction>
ROCPRIM_DEVICE
void search_kernel_impl(InputIterator1 input,
                        InputIterator2 keys,
                        size_t*        output,
                        size_t         size,
                        size_t         keys_size,
                        BinaryFunction compare_function)
{
    constexpr search_config_params params = ArchConfig::params;

    constexpr unsigned int block_size       = params.kernel_config.block_size;
    constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;
    constexpr unsigned int items_per_block  = block_size * items_per_thread;

    const unsigned int flat_id       = rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = rocprim::detail::block_id<0>();

    const size_t offset       = flat_id * items_per_thread + flat_block_id * items_per_block;
    bool         find_pattern = false;

    // Check if it can have fit a key and a key has not yet be found with a lower index.
    if(offset + keys_size > size || offset > atomic_load(output))
    {
        return;
    }

    size_t index = 0;
    for(size_t id = offset; id < offset + items_per_thread; id++)
    {
        size_t i          = 0;
        size_t current_id = id;
        for(; i < keys_size - 1 && current_id < size; i++, current_id++)
        {
            if(!compare_function(input[current_id], keys[i]))
            {
                break;
            }
        }

        // If the i is the last value for the key and the compare is also true,
        // the pattern is found.
        if(current_id < size && i == (keys_size - 1)
           && compare_function(input[current_id], keys[i]))
        {
            index        = id;
            find_pattern = true;
            break;
        }
    }

    // Construct a mask of threads in this wave which have the same digit.
    lane_mask_type peer_mask = ballot(find_pattern);

    wave_barrier();

    // The number of threads in the warp that have the same digit AND whose lane id is lower
    // than the current thread's.
    const unsigned int peer_digit_prefix = masked_bit_count(peer_mask);

    if(find_pattern && (peer_digit_prefix == 0))
    {
        atomic_min(output, index);
    }
}

template<class ArchConfig, class InputIterator1, class InputIterator2, class BinaryFunction>
ROCPRIM_DEVICE
void search_kernel_shared_impl(InputIterator1 input,
                               InputIterator2 keys,
                               size_t*        output,
                               size_t         size,
                               size_t         keys_size,
                               BinaryFunction compare_function)
{
    using value_type = typename std::iterator_traits<InputIterator1>::value_type;
    using key_type   = typename std::iterator_traits<InputIterator2>::value_type;

    constexpr search_config_params params = ArchConfig::params;

    constexpr unsigned int block_size       = params.kernel_config.block_size;
    constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;
    constexpr unsigned int items_per_block  = block_size * items_per_thread;
    constexpr unsigned int max_shared_key   = params.max_shared_key_bytes / sizeof(key_type);

    const unsigned int flat_id       = rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = rocprim::detail::block_id<0>();

    const size_t block_offset = flat_block_id * items_per_block;
    const size_t offset       = flat_id * items_per_thread;
    bool find_pattern = false;

    ROCPRIM_SHARED_MEMORY uninitialized_array<key_type, max_shared_key> local_keys_;
    ROCPRIM_SHARED_MEMORY uninitialized_array<value_type, items_per_block> local_input_;

    // Check if a key was already found in a place before this block
    if(block_offset > atomic_load(output))
    {
        return;
    }

    // Load in key in shared memory
    const size_t batch_size = ceiling_div(keys_size, block_size);
    for(size_t i = 0; i < batch_size; i++)
    {
        const size_t index = flat_id * batch_size + i;
        if(index < keys_size)
        {
            local_keys_.emplace(index, keys[index]);
        }
    }

    using block_load_input = block_load<value_type, items_per_thread, items_per_thread>;

    value_type elements[items_per_thread];

    const bool is_complete_block = block_offset + items_per_block <= size;

    // Load in all the input values that are guaranteed to be loaded.
    if(is_complete_block)
    {
        block_load_input().load(input + block_offset, elements);
        for(size_t i = 0; i < items_per_thread; i++)
        {
            const size_t index = flat_id * items_per_thread + i;
            local_input_.emplace(index, elements[i]);
        }
    }
    else
    {
        block_load_input().load(input + block_offset, elements, size - block_offset);
        for(size_t i = 0; i < items_per_thread; i++)
        {
            const size_t index       = flat_id * items_per_thread + i;
            const size_t index_value = block_offset + index;
            if(index_value < size)
            {
                local_input_.emplace(index, elements[i]);
            }
        }
    }

    const key_type*   local_keys  = local_keys_.get_unsafe_array();
    const value_type* local_input = local_input_.get_unsafe_array();

    syncthreads();

    // Check if it can have fit a key and a key has not yet be found with a lower index.
    if(offset + block_offset + keys_size > size || offset > atomic_load(output))
    {
        return;
    }

    size_t       index      = 0;
    const size_t check      = size - block_offset;
    const size_t check_both = rocprim::min(check, size_t(items_per_block));
    for(size_t id = offset; id < offset + items_per_thread; id++)
    {
        size_t i          = 0;
        size_t current_id = id;
        // Values till the items_per_block are in shared_memory
        for(; i < keys_size - 1 && current_id < check_both; i++, current_id++)
        {
            if(!compare_function(local_input[current_id], local_keys[i]))
            {
                break;
            }
        }
        // Compare values that are not in the shared memory
        for(; current_id >= items_per_block && i < keys_size - 1 && current_id < check;
            i++, current_id++)
        {
            if(!compare_function(input[current_id + block_offset], local_keys[i]))
            {
                break;
            }
        }

        // If the i is the last value for the key and the compare is also true,
        // the pattern is found.
        if(current_id + block_offset < size && i == (keys_size - 1)
           && compare_function(current_id < items_per_block ? local_input[current_id]
                                                            : input[current_id + block_offset],
                               local_keys[i]))
        {
            index        = id + block_offset;
            find_pattern = true;
            // Want to find the first occurance, do not need to search further.
            break;
        }
    }

    // Construct a mask of threads in this wave which have the same digit.
    lane_mask_type peer_mask = ballot(find_pattern);

    wave_barrier();

    // The number of threads in the warp that have the same digit AND whose lane id is lower
    // than the current thread's.
    const unsigned int peer_digit_prefix = masked_bit_count(peer_mask);

    if(find_pattern && (peer_digit_prefix == 0))
    {
        atomic_min(output, index);
    }
}

template<class Config, class InputIterator1, class InputIterator2, class BinaryFunction>
struct search_impl_kernels
{
    inline static hipError_t launch_search_shared(detail::target_arch arch,
                                                  InputIterator1      input,
                                                  InputIterator2      keys,
                                                  size_t*             output,
                                                  size_t              size,
                                                  size_t              keys_size,
                                                  BinaryFunction      compare_function,
                                                  dim3                grid,
                                                  dim3                block,
                                                  size_t              shmem,
                                                  hipStream_t         stream)
    {
        auto kernel = [=](auto arch_config)
        {
            search_kernel_shared_impl<decltype(arch_config)>(input,
                                                             keys,
                                                             output,
                                                             size,
                                                             keys_size,
                                                             compare_function);
        };

        return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
    }

    inline static hipError_t launch_search(detail::target_arch arch,
                                           InputIterator1      input,
                                           InputIterator2      keys,
                                           size_t*             output,
                                           size_t              size,
                                           size_t              keys_size,
                                           BinaryFunction      compare_function,
                                           dim3                grid,
                                           dim3                block,
                                           size_t              shmem,
                                           hipStream_t         stream)
    {
        auto kernel = [=](auto arch_config)
        {
            search_kernel_impl<decltype(arch_config)>(input,
                                                      keys,
                                                      output,
                                                      size,
                                                      keys_size,
                                                      compare_function);
        };

        return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
    }

    inline static hipError_t launch_search_shared(detail::target_arch              arch,
                                                  reverse_iterator<InputIterator1> input,
                                                  reverse_iterator<InputIterator2> keys,
                                                  size_t*                          output,
                                                  size_t                           size,
                                                  size_t                           keys_size,
                                                  BinaryFunction                   compare_function,
                                                  dim3                             grid,
                                                  dim3                             block,
                                                  size_t                           shmem,
                                                  hipStream_t                      stream)
    {
        auto kernel = [=](auto arch_config)
        {
            search_kernel_shared_impl<decltype(arch_config)>(input,
                                                             keys,
                                                             output,
                                                             size,
                                                             keys_size,
                                                             compare_function);
        };

        return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
    }

    inline static hipError_t launch_search(detail::target_arch              arch,
                                           reverse_iterator<InputIterator1> input,
                                           reverse_iterator<InputIterator2> keys,
                                           size_t*                          output,
                                           size_t                           size,
                                           size_t                           keys_size,
                                           BinaryFunction                   compare_function,
                                           dim3                             grid,
                                           dim3                             block,
                                           size_t                           shmem,
                                           hipStream_t                      stream)
    {
        auto kernel = [=](auto arch_config)
        {
            search_kernel_impl<decltype(arch_config)>(input,
                                                      keys,
                                                      output,
                                                      size,
                                                      keys_size,
                                                      compare_function);
        };

        return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
    }

    template<class T>
    static ROCPRIM_KERNEL
    void set_output_kernel(T* output, T value)
    {
        *output = value;
    }

    template<class T>
    static ROCPRIM_KERNEL
    void reverse_index_kernel(T* output, T size, T keys_size)
    {
        // Return the reverse index as long as the index is lower than the size.
        if(*output < size)
        {
            *output = size - keys_size - *output;
        }
    }
};

template<class Config,
         bool find_first,
         class InputIterator1,
         class InputIterator2,
         class OutputIterator,
         class BinaryFunction>
ROCPRIM_INLINE
hipError_t search_impl(void*          temporary_storage,
                       size_t&        storage_size,
                       InputIterator1 input,
                       InputIterator2 keys,
                       OutputIterator output,
                       size_t         size,
                       size_t         keys_size,
                       BinaryFunction compare_function,
                       hipStream_t    stream,
                       bool           debug_synchronous)
{
    using input_type  = typename std::iterator_traits<InputIterator1>::value_type;
    using key_type    = typename std::iterator_traits<InputIterator2>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;

    using config = wrapped_search_config<Config, input_type>;
    using search_kernels
        = search_impl_kernels<config, InputIterator1, InputIterator2, BinaryFunction>;

    target_arch target_arch;
    ROCPRIM_RETURN_ON_ERROR(host_target_arch(stream, target_arch));

    const search_config_params params = dispatch_target_arch<config, false>(target_arch);

    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const unsigned int items_per_block  = block_size * items_per_thread;

    const unsigned int shared_key_mem_size_bytes = params.max_shared_key_bytes;
    const unsigned int key_size_bytes            = keys_size * sizeof(key_type);

    // Start point for time measurements
    std::chrono::steady_clock::time_point start;

    const auto start_timer = [&start, debug_synchronous]()
    {
        if(debug_synchronous)
        {
            start = std::chrono::steady_clock::now();
        }
    };

    if(temporary_storage == nullptr)
    {
        storage_size = sizeof(size_t);
        return hipSuccess;
    }

    if(keys_size > size)
    {
        return hipErrorInvalidValue;
    }

    size_t* tmp_output = reinterpret_cast<size_t*>(temporary_storage);

    start_timer();
    search_kernels::set_output_kernel<<<1, 1, 0, stream>>>(tmp_output,
                                                           find_first && keys_size <= 0 ? 0 : size);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("set_output_kernel", 1, start);

    if(size > 0 && keys_size > 0)
    {
        const unsigned int num_blocks = ceiling_div(size, items_per_block);
        if(key_size_bytes < shared_key_mem_size_bytes)
        {
            if constexpr(find_first)
            {
                start_timer();
                ROCPRIM_RETURN_ON_ERROR(search_kernels::launch_search_shared(target_arch,
                                                                             input,
                                                                             keys,
                                                                             tmp_output,
                                                                             size,
                                                                             keys_size,
                                                                             compare_function,
                                                                             num_blocks,
                                                                             block_size,
                                                                             0,
                                                                             stream));
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_kernel_shared", size, start);
            }
            else
            {
                start_timer();
                ROCPRIM_RETURN_ON_ERROR(search_kernels::launch_search_shared(
                    target_arch,
                    rocprim::make_reverse_iterator(input + size),
                    rocprim::make_reverse_iterator(keys + keys_size),
                    tmp_output,
                    size,
                    keys_size,
                    compare_function,
                    num_blocks,
                    block_size,
                    0,
                    stream));
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_kernel_shared", size, start);
            }
        }
        else
        {
            if constexpr(find_first)
            {
                start_timer();
                ROCPRIM_RETURN_ON_ERROR(search_kernels::launch_search(target_arch,
                                                                      input,
                                                                      keys,
                                                                      tmp_output,
                                                                      size,
                                                                      keys_size,
                                                                      compare_function,
                                                                      num_blocks,
                                                                      block_size,
                                                                      0,
                                                                      stream));
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_kernel", size, start);
            }
            else
            {
                start_timer();
                ROCPRIM_RETURN_ON_ERROR(
                    search_kernels::launch_search(target_arch,
                                                  rocprim::make_reverse_iterator(input + size),
                                                  rocprim::make_reverse_iterator(keys + keys_size),
                                                  tmp_output,
                                                  size,
                                                  keys_size,
                                                  compare_function,
                                                  num_blocks,
                                                  block_size,
                                                  0,
                                                  stream));
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_kernel", size, start);
            }
        }

        if constexpr(!find_first)
        {
            start_timer();
            search_kernels::reverse_index_kernel<<<1, 1, 0, stream>>>(tmp_output, size, keys_size);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("reverse_index_kernel", 1, start);
        }
    }

    ROCPRIM_RETURN_ON_ERROR(transform(tmp_output,
                                      output,
                                      1,
                                      rocprim::identity<output_type>(),
                                      stream,
                                      debug_synchronous));

    return hipSuccess;
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SEARCH_HPP_
