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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_NTH_ELEMENT_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_NTH_ELEMENT_HPP_

#include "../../block/block_load.hpp"
#include "../../block/block_radix_rank.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_sort.hpp"
#include "../../block/block_store.hpp"

#include "../../common.hpp"
#include "../../config.hpp"
#include "../../intrinsics.hpp"
#include "../../type_traits.hpp"

#include "../device_transform.hpp"

#include "device_config_helper.hpp"

#include <cstdint>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

#include <iostream>

#include <cstddef>
#include <cstdio>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
struct nth_element_onesweep_lookback_state
{
    // The two most significant bits are used to indicate the status of the prefix - leaving the other 30 bits for the
    // counter value.
    using underlying_type = uint32_t;

    static constexpr unsigned int state_bits = 8u * sizeof(underlying_type);

    enum prefix_flag : underlying_type
    {
        EMPTY    = underlying_type(0),
        PARTIAL  = underlying_type(1) << (state_bits - 2),
        COMPLETE = underlying_type(2) << (state_bits - 2)
    };

    static constexpr underlying_type status_mask = underlying_type(3) << (state_bits - 2);
    static constexpr underlying_type value_mask  = ~status_mask;

    underlying_type state;

    ROCPRIM_DEVICE ROCPRIM_INLINE explicit nth_element_onesweep_lookback_state(underlying_type state)
        : state(state)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE nth_element_onesweep_lookback_state(prefix_flag     status,
                                                                     underlying_type value)
        : state(static_cast<underlying_type>(status) | value)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE
    underlying_type value() const
    {
        return this->state & value_mask;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    prefix_flag status() const
    {
        return static_cast<prefix_flag>(this->state & status_mask);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static nth_element_onesweep_lookback_state load(nth_element_onesweep_lookback_state* ptr)
    {
        underlying_type state = atomic_load(&ptr->state);
        return nth_element_onesweep_lookback_state(state);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(nth_element_onesweep_lookback_state* ptr) const
    {
        atomic_store(&ptr->state, this->state);
    }

    static hipError_t reset(nth_element_onesweep_lookback_state* states,
                            unsigned int                         num_total_states,
                            hipStream_t                          stream)
    {
        // All zeroes is equivalent to the empty prefix
        return hipMemsetAsync(states, 0, num_total_states * sizeof(*states), stream);
    }
};

struct n_th_element_iteration_data
{
    unsigned int bucket_idx;
    unsigned int offset;
    unsigned int size;
    bool         equality_bucket;
};

template<class ArchConfig, class KeysIterator, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    block_sort_kernel_impl(KeysIterator keys, const unsigned int size, BinaryFunction compare_function)
{
    constexpr nth_element_config_params params = ArchConfig::params;

    constexpr unsigned int stop_recursion_size = params.stop_recursion_size;

    using key_type = typename std::iterator_traits<KeysIterator>::value_type;

    using block_load_key  = block_load<key_type, stop_recursion_size, 1>;
    using block_sort_key  = block_sort<key_type, stop_recursion_size>;
    using block_store_key = block_store<key_type, stop_recursion_size, 1>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_key::storage_type  load;
        typename block_sort_key::storage_type  sort;
        typename block_store_key::storage_type store;
    } storage;

    key_type sample_buffer[1];

    block_load_key().load(keys, sample_buffer, size, storage.load);

    syncthreads();

    block_sort_key().sort(sample_buffer, storage.sort, size, compare_function);

    syncthreads();

    block_store_key().store(keys, sample_buffer, size, storage.store);
}

template<class Config, class KeysIterator, class BinaryFunction>
inline hipError_t launch_block_sort(detail::target_arch arch,
                                    KeysIterator        keys,
                                    const unsigned int  size,
                                    BinaryFunction      compare_function,
                                    dim3                grid,
                                    dim3                block,
                                    size_t              shmem,
                                    hipStream_t         stream)
{
    auto kernel = [=](auto arch_config)
    { block_sort_kernel_impl<decltype(arch_config)>(keys, size, compare_function); };

    return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
}

template<class ArchConfig, class KeysIterator, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void find_splitters_kernel_impl(
                               KeysIterator                                             keys,
                               typename std::iterator_traits<KeysIterator>::value_type* tree,
                               bool*          equality_buckets,
                               const unsigned int   size,
                               BinaryFunction compare_function)
{
    constexpr nth_element_config_params params        = ArchConfig::params;
    constexpr unsigned int              num_splitters = params.number_of_buckets - 1;

    using key_type = typename std::iterator_traits<KeysIterator>::value_type;

    using block_sort_key = block_sort<key_type, num_splitters>;

    ROCPRIM_SHARED_MEMORY typename block_sort_key::storage_type storage;

    const unsigned int stride = size / num_splitters;
    const unsigned int idx    = threadIdx.x;

    // Find values to split data in buckets
    key_type sample_buffer = keys[stride * idx];
    // Sort the splitters
    block_sort_key().sort(sample_buffer, storage, compare_function);

    tree[idx] = sample_buffer;

    syncthreads();

    bool equality_bucket = false;
    if(idx > 0)
    {
        // Check if the splitter value before has the same value and the value after is different
        // If so use the bucket for items equal to the splitter value.
        equality_bucket
            = tree[idx - 1] == sample_buffer
              && (idx == num_splitters - 1 || compare_function(sample_buffer, tree[idx + 1]));
    }

    equality_buckets[idx] = equality_bucket;
}

template<class Config, class KeysIterator, class BinaryFunction>
inline hipError_t
    launch_find_splitters(detail::target_arch                                      arch,
                          KeysIterator                                             keys,
                          typename std::iterator_traits<KeysIterator>::value_type* tree,
                          bool*                                                    equality_buckets,
                          const unsigned int                                       size,
                          BinaryFunction                                           compare_function,
                          dim3                                                     grid,
                          dim3                                                     block,
                          size_t                                                   shmem,
                          hipStream_t                                              stream)
{
    auto kernel = [=](auto arch_config)
    {
        find_splitters_kernel_impl<decltype(arch_config)>(keys,
                                                          tree,
                                                          equality_buckets,
                                                          size,
                                                          compare_function);
    };

    return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
}

template<class ArchConfig, class KeysIterator, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    count_bucket_sizes_kernel_impl(KeysIterator                                             keys,
                                   typename std::iterator_traits<KeysIterator>::value_type* tree,
                                   const unsigned int                                             size,
                                   unsigned int*                                                  buckets,
                                   bool*          equality_buckets,
                                   BinaryFunction compare_function)
{
    constexpr nth_element_config_params params = ArchConfig::params;

    constexpr unsigned int num_buckets           = params.number_of_buckets;
    constexpr unsigned int num_threads_per_block = params.kernel_config.block_size;
    constexpr unsigned int num_items_per_thread  = params.kernel_config.items_per_thread;
    constexpr unsigned int num_splitters         = num_buckets - 1;
    constexpr unsigned int num_items_per_block   = num_threads_per_block * num_items_per_thread;

    // It needs enough splitters to choose from the input
    static_assert(params.stop_recursion_size >= num_buckets,
                  "stop_recursion_size should be larger or equal than the number_of_buckets");
    // It loads in shared memory for the number_of_buckets for every thread
    static_assert(num_threads_per_block >= num_buckets,
                  "num_threads_per_block should be larger or equal than the number_of_buckets");
    // It assumes the number_of_buckets of buckets is a power of two in the traversal
    static_assert(detail::is_power_of_two(num_buckets),
                  "number_of_buckets should be a power of two");
    static_assert(num_buckets >= 2, "number_of_buckets should be larger than 2");

    using key_type = typename std::iterator_traits<KeysIterator>::value_type;

    using block_load_key = block_load<key_type, num_threads_per_block, num_items_per_thread>;

    ROCPRIM_SHARED_MEMORY struct
    {
        uninitialized_array<key_type, num_splitters> buffer;
        unsigned int                                 shared_buckets[num_buckets];
        bool                                         shared_equality_buckets[num_buckets];
    } storage;

    if(threadIdx.x < num_buckets)
    {
        storage.shared_buckets[threadIdx.x]          = 0;
        storage.shared_equality_buckets[threadIdx.x] = equality_buckets[threadIdx.x];
    }

    if(threadIdx.x < num_splitters)
    {
        storage.buffer.emplace(threadIdx.x, tree[threadIdx.x]);
    }
    const key_type* search_tree = storage.buffer.get_unsafe_array();

    key_type           elements[num_items_per_thread];
    const unsigned int offset            = blockIdx.x * num_items_per_block;
    const bool         is_complete_block = offset + num_items_per_block <= size;

    if(is_complete_block)
    {
        block_load_key().load(keys + offset, elements);
    }
    else
    {
        block_load_key().load(keys + offset, elements, size - offset);
    }

    syncthreads();

    for(unsigned int item = 0; item < num_items_per_thread; item++)
    {
        unsigned int idx = offset + threadIdx.x * num_items_per_thread + item;
        if(idx < size)
        {
            const key_type element = elements[item];
            unsigned int   left    = 0;
            unsigned int   right   = num_splitters;
            unsigned int   bucket;
            // Binary search through splitters to put in bucket
            for(unsigned int i = 0; i < Log2<static_cast<int>(num_buckets)>::VALUE; i++)
            {
                const unsigned int mid  = (left + right) >> 1;
                const bool         comp = compare_function(element, search_tree[mid]);
                right                   = comp ? mid : right;
                left                    = comp ? left : mid;
                bucket                  = right;
            }

            // Checks if the bucket before is an equality bucket for the current value
            if(bucket > 0 && storage.shared_equality_buckets[bucket - 1]
               && element == search_tree[bucket - 1])
            {
                bucket = bucket - 1;
            }

            atomic_add(&storage.shared_buckets[bucket], 1);
        }
    }

    syncthreads();

    if(threadIdx.x < num_buckets)
    {
        atomic_add(&buckets[threadIdx.x], storage.shared_buckets[threadIdx.x]);
    }
}

template<class Config, class KeysIterator, class BinaryFunction>
inline hipError_t
    launch_count_bucket_sizes(detail::target_arch                                      arch,
                              KeysIterator                                             keys,
                              typename std::iterator_traits<KeysIterator>::value_type* tree,
                              const unsigned int                                       size,
                              unsigned int*                                            buckets,
                              bool*          equality_buckets,
                              BinaryFunction compare_function,
                              dim3           grid,
                              dim3           block,
                              size_t         shmem,
                              hipStream_t    stream)
{
    auto kernel = [=](auto arch_config)
    {
        count_bucket_sizes_kernel_impl<decltype(arch_config)>(keys,
                                                              tree,
                                                              size,
                                                              buckets,
                                                              equality_buckets,
                                                              compare_function);
    };

    return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
}

template<class ArchConfig>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    find_nth_element_bucket_kernel_impl(unsigned int*                      buckets,
                                        n_th_element_iteration_data* nth_element_data,
                                        bool*                        equality_buckets,
                                        const unsigned int           rank)

{
    constexpr nth_element_config_params params = ArchConfig::params;

    constexpr unsigned int num_buckets = params.number_of_buckets;

    using block_scan_buckets = block_scan<unsigned int, num_buckets>;

    ROCPRIM_SHARED_MEMORY struct
    {
        typename block_scan_buckets::storage_type scan;
        unsigned int                              bucket_offsets[num_buckets];
    } storage;

    unsigned int bucket_size = buckets[threadIdx.x];
    unsigned int bucket_offset;
    // Calculate the global offset of the buckets based on bucket sizes
    block_scan_buckets().exclusive_scan(bucket_size, bucket_offset, 0, storage.scan);

    storage.bucket_offsets[threadIdx.x] = bucket_offset;

    syncthreads();

    unsigned int num_buckets_before;

    // Find in which bucket the nth element sits
    bool in_nth = storage.bucket_offsets[threadIdx.x] <= rank;
    block_scan_buckets().inclusive_scan(in_nth, num_buckets_before, storage.scan);

    if(threadIdx.x == (num_buckets - 1))
    {
        // Store nth_element data
        unsigned int nth_element          = num_buckets_before - 1;
        nth_element_data->offset          = storage.bucket_offsets[nth_element];
        nth_element_data->size            = buckets[nth_element];
        nth_element_data->equality_bucket = equality_buckets[nth_element];
        nth_element_data->bucket_idx      = nth_element;
    }
}

template<class Config>
inline hipError_t launch_find_nth_element_bucket(detail::target_arch          arch,
                                                 unsigned int*                buckets,
                                                 n_th_element_iteration_data* nth_element_data,
                                                 bool*                        equality_buckets,
                                                 const unsigned int           rank,
                                                 dim3                         grid,
                                                 dim3                         block,
                                                 size_t                       shmem,
                                                 hipStream_t                  stream)

{
    auto kernel = [=](auto arch_config)
    {
        find_nth_element_bucket_kernel_impl<decltype(arch_config)>(buckets,
                                                                   nth_element_data,
                                                                   equality_buckets,
                                                                   rank);
    };

    return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
}

template<class ArchConfig, unsigned int NumPartitions, class KeysIterator, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    copy_buckets_kernel_impl(KeysIterator                                             keys,
                             typename std::iterator_traits<KeysIterator>::value_type* tree,
                             const unsigned int                                       size,
                             nth_element_onesweep_lookback_state* lookback_states,
                             n_th_element_iteration_data*         nth_element_data,
                             typename std::iterator_traits<KeysIterator>::value_type* keys_buffer,
                             bool*          equality_buckets,
                             BinaryFunction compare_function)
{
    using key_type = typename std::iterator_traits<KeysIterator>::value_type;
    using state    = nth_element_onesweep_lookback_state;

    constexpr nth_element_config_params params = ArchConfig::params;

    constexpr unsigned int num_buckets           = params.number_of_buckets;
    constexpr unsigned int num_splitters         = num_buckets - 1;
    constexpr unsigned int num_threads_per_block = params.kernel_config.block_size;
    constexpr unsigned int num_items_per_thread  = params.kernel_config.items_per_thread;
    constexpr unsigned int num_items_per_block   = num_threads_per_block * num_items_per_thread;
    constexpr unsigned int num_partitions        = NumPartitions;

    using block_rank = rocprim::block_radix_rank<num_threads_per_block,
                                                 Log2<static_cast<int>(num_partitions + 1)>::VALUE,
                                                 params.radix_rank_algorithm>;

    using block_load_element = block_load<key_type, num_threads_per_block, num_items_per_thread>;

    static_assert(block_rank::digits_per_thread == 1,
                  "The digits_per_thread is assumed to be one.");

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_rank::storage_type         rank;
        typename block_load_element::storage_type load_element;
        size_t                                    buckets_block_offsets_shared[num_partitions];
    } storage;

    uint8_t buckets[num_items_per_thread];

    const unsigned int nth_element = nth_element_data->bucket_idx;
    const key_type     lower_bound = nth_element > 0 ? tree[nth_element - 1] : key_type(0);
    const key_type     upper_bound = nth_element < num_splitters ? tree[nth_element] : key_type(0);
    const bool         equality_bucket = nth_element_data->equality_bucket;
    const bool equality_bucket_before  = nth_element > 0 && equality_buckets[nth_element - 1];

    const unsigned int offset            = blockIdx.x * num_items_per_block;
    const bool         is_complete_block = offset + num_items_per_block <= size;

    key_type elements[num_items_per_thread];
    if(is_complete_block)
    {
        block_load_element().load(keys + offset, elements, storage.load_element);
    }
    else
    {
        block_load_element().load(keys + offset, elements, size - offset, 0, storage.load_element);
    }

    const unsigned int thread_id = (threadIdx.x * num_items_per_thread) + offset;

    if(is_complete_block)
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < num_items_per_thread; item++)
        {
            const key_type element = elements[item];
            // Assign items to the offset of buckets:
            // * 0 if the item is < lower bound and thus a bucket before nth-element
            // * 1 if the item is in bounds and thus in the same bucket at nth-element
            // * 2 if the item is > upper bound and thus in a bucket after nth-element
            // If there is an equality bucket then the comparison needs to be inclusive.
            buckets[item] = ((nth_element > 0)
                             && (equality_bucket_before ? compare_function(lower_bound, element)
                                                        : !compare_function(element, lower_bound)))
                            + ((nth_element < num_splitters)
                               && (equality_bucket ? compare_function(upper_bound, element)
                                                   : !compare_function(element, upper_bound)));
        }
    }
    else
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < num_items_per_thread; item++)
        {
            const key_type     element = elements[item];
            const unsigned int idx     = item + thread_id;
            if(idx < size)
            {
                buckets[item]
                    = ((nth_element > 0)
                       && (equality_bucket_before ? compare_function(lower_bound, element)
                                                  : !compare_function(element, lower_bound)))
                      + ((nth_element < num_splitters)
                         && (equality_bucket ? compare_function(upper_bound, element)
                                             : !compare_function(element, upper_bound)));
            }
            else
            {
                buckets[item] = num_partitions;
            }
        }
    }

    syncthreads();

    unsigned int ranks[num_items_per_thread];
    unsigned int partition_prefix[block_rank::digits_per_thread];
    unsigned int partition_counts[block_rank::digits_per_thread];
    block_rank().rank_keys(
        buckets,
        ranks,
        storage.rank,
        [](const uint8_t& key) { return key; },
        partition_prefix,
        partition_counts);

    syncthreads();

    const unsigned int partition = threadIdx.x;
    if(partition < num_partitions)
    {
        state* block_state = &lookback_states[blockIdx.x * num_partitions + partition];
        state(state::PARTIAL, partition_counts[0]).store(block_state);

        unsigned int exclusive_prefix  = 0;
        unsigned int lookback_block_id = blockIdx.x;
        // The main back tracking loop.
        while(lookback_block_id > 0)
        {
            --lookback_block_id;
            state* lookback_state_ptr
                = &lookback_states[lookback_block_id * num_partitions + partition];
            state lookback_state = state::load(lookback_state_ptr);
            while(lookback_state.status() == state::EMPTY)
            {
                lookback_state = state::load(lookback_state_ptr);
            }

            exclusive_prefix += lookback_state.value();
            if(lookback_state.status() == state::COMPLETE)
            {
                break;
            }
        }

        // Update the state for the current block.
        const unsigned int inclusive_digit_prefix = exclusive_prefix + partition_counts[0];
        // Note that this should not deadlock, as HSA guarantees that block with a lowest block ID
        // of all not finished yet is progressing.
        state(state::COMPLETE, inclusive_digit_prefix).store(block_state);

        unsigned int global_offset = 0;

        // The global offsets based on the nth element bucket.
        if(partition > 0)
        {
            const unsigned int nth_bucket_offset = nth_element_data->offset;
            global_offset += nth_bucket_offset;
        }

        if((partition > 0 && nth_element == 0) || partition > 1)
        {
            const unsigned int nth_bucket_size = nth_element_data->size;
            global_offset += nth_bucket_size;
        }

        // Subtract the exclusive digit prefix from the global offset here, since we already ordered the keys in shared
        // memory.
        storage.buckets_block_offsets_shared[partition]
            = global_offset - partition_prefix[0] + exclusive_prefix;
    }

    syncthreads();

    // Scatter the keys based on their placement in the buckets
    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < num_items_per_thread; item++)
    {
        const unsigned int idx = item + thread_id;
        if(idx < size)
        {
            const uint8_t      bucket = buckets[item];
            const unsigned int index  = storage.buckets_block_offsets_shared[bucket] + ranks[item];
            keys_buffer[index]        = elements[item];
        }
    }
}

template<class Config, unsigned int NumPartitions, class KeysIterator, class BinaryFunction>
inline hipError_t
    launch_copy_buckets(detail::target_arch                                      arch,
                        KeysIterator                                             keys,
                        typename std::iterator_traits<KeysIterator>::value_type* tree,
                        const unsigned int                                       size,
                        nth_element_onesweep_lookback_state*                     lookback_states,
                        n_th_element_iteration_data*                             nth_element_data,
                        typename std::iterator_traits<KeysIterator>::value_type* keys_buffer,
                        bool*                                                    equality_buckets,
                        BinaryFunction                                           compare_function,
                        dim3                                                     grid,
                        dim3                                                     block,
                        size_t                                                   shmem,
                        hipStream_t                                              stream)
{
    auto kernel = [=](auto arch_config)
    {
        copy_buckets_kernel_impl<decltype(arch_config), NumPartitions>(keys,
                                                                       tree,
                                                                       size,
                                                                       lookback_states,
                                                                       nth_element_data,
                                                                       keys_buffer,
                                                                       equality_buckets,
                                                                       compare_function);
    };

    return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
}

template<class Config, unsigned int NumPartitions, class KeysIterator, class BinaryFunction>
ROCPRIM_INLINE
hipError_t
    nth_element_keys_impl(detail::target_arch                                      arch,
                          KeysIterator                                             keys,
                          typename std::iterator_traits<KeysIterator>::value_type* keys_buffer,
                          typename std::iterator_traits<KeysIterator>::value_type* tree,
                          unsigned int                                             rank,
                          unsigned int                                             size,
                          unsigned int*                                            buckets,
                          bool*                                                    equality_buckets,
                          nth_element_onesweep_lookback_state*                     lookback_states,
                          const unsigned int                                       num_buckets,
                          const unsigned int           stop_recursion_size,
                          const unsigned int           num_threads_per_block,
                          const unsigned int           num_items_per_thread,
                          n_th_element_iteration_data* nth_element_data,
                          BinaryFunction               compare_function,
                          hipStream_t                  stream,
                          bool                         debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysIterator>::value_type;

    constexpr unsigned int num_partitions      = NumPartitions;
    const unsigned int     num_splitters       = num_buckets - 1;
    const unsigned int     num_items_per_block = num_threads_per_block * num_items_per_thread;
    unsigned int           iteration           = 0;

    // Start point for time measurements
    std::chrono::steady_clock::time_point start;

    const auto start_timer = [&start, debug_synchronous]()
    {
        if(debug_synchronous)
        {
            start = std::chrono::steady_clock::now();
        }
    };

    while(size >= stop_recursion_size)
    {
        const unsigned int num_blocks = ceiling_div(size, num_items_per_block);

        if(debug_synchronous)
        {
            std::cout << "-----" << '\n';
            std::cout << "size: " << size << '\n';
            std::cout << "rank: " << rank << '\n';
            std::cout << "iteration: " << iteration++ << '\n';
        }

        ROCPRIM_RETURN_ON_ERROR(hipMemsetAsync(buckets, 0, sizeof(*buckets) * num_buckets, stream));

        ROCPRIM_RETURN_ON_ERROR(
            hipMemsetAsync(equality_buckets, 0, sizeof(*equality_buckets) * num_buckets, stream));

        // Reset lookback scan states to zero, indicating empty prefix.
        ROCPRIM_RETURN_ON_ERROR(
            nth_element_onesweep_lookback_state::reset(lookback_states,
                                                       num_partitions * num_blocks,
                                                       stream));

        start_timer();
        ROCPRIM_RETURN_ON_ERROR(launch_find_splitters<Config>(arch,
                                                              keys,
                                                              tree,
                                                              equality_buckets,
                                                              size,
                                                              compare_function,
                                                              1,
                                                              num_splitters,
                                                              0,
                                                              stream));
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("find_splitters_kernel", size, start);

        start_timer();
        ROCPRIM_RETURN_ON_ERROR(launch_count_bucket_sizes<Config>(arch,
                                                                  keys,
                                                                  tree,
                                                                  size,
                                                                  buckets,
                                                                  equality_buckets,
                                                                  compare_function,
                                                                  num_blocks,
                                                                  num_threads_per_block,
                                                                  0,
                                                                  stream));
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("count_bucket_sizes_kernel", size, start);

        start_timer();
        ROCPRIM_RETURN_ON_ERROR(launch_find_nth_element_bucket<Config>(arch,
                                                                       buckets,
                                                                       nth_element_data,
                                                                       equality_buckets,
                                                                       rank,
                                                                       1,
                                                                       num_buckets,
                                                                       0,
                                                                       stream));
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("find_nth_element_bucket_kernel", size, start);

        start_timer();
        ROCPRIM_RETURN_ON_ERROR(launch_copy_buckets<Config, num_partitions>(arch,
                                                                            keys,
                                                                            tree,
                                                                            size,
                                                                            lookback_states,
                                                                            nth_element_data,
                                                                            keys_buffer,
                                                                            equality_buckets,
                                                                            compare_function,
                                                                            num_blocks,
                                                                            num_threads_per_block,
                                                                            0,
                                                                            stream));
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("copy_buckets_kernel", size, start);

        // Copy the results in keys_buffer back to the keys
        ROCPRIM_RETURN_ON_ERROR(transform(keys_buffer,
                                          keys,
                                          size,
                                          ::rocprim::identity<key_type>(),
                                          stream,
                                          debug_synchronous));

        n_th_element_iteration_data h_nth_element_data;
        ROCPRIM_RETURN_ON_ERROR(hipMemcpyAsync(&h_nth_element_data,
                                               nth_element_data,
                                               sizeof(h_nth_element_data),
                                               hipMemcpyDeviceToHost,
                                               stream));

        ROCPRIM_RETURN_ON_ERROR(hipStreamSynchronize(stream));

        unsigned int offset          = h_nth_element_data.offset;
        unsigned int bucket_size     = h_nth_element_data.size;
        bool         equality_bucket = h_nth_element_data.equality_bucket;

        // If all values are the same it is already sorted
        if(equality_bucket)
        {
            return hipSuccess;
        }

        size = bucket_size;
        // rank is the n from the nth-element, but it reduces based on the previous iteration
        rank = rank - offset;
        keys = keys + offset;
    }

    start_timer();
    ROCPRIM_RETURN_ON_ERROR(launch_block_sort<Config>(arch,
                                                      keys,
                                                      size,
                                                      compare_function,
                                                      1,
                                                      stop_recursion_size,
                                                      0,
                                                      stream));
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_block_sort", size, start);
    return hipSuccess;
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_NTH_ELEMENT_HPP_
