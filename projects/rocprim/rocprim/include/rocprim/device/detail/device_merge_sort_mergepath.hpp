/******************************************************************************
* Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
* Modifications Copyright (c) 2022-2026, Advanced Micro Devices, Inc.  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the NVIDIA CORPORATION nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_SORT_MERGEPATH_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_SORT_MERGEPATH_HPP_

#include <iterator>

#include "../../detail/various.hpp"

#include "device_merge.hpp"
#include "device_merge_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
// Load items from input1 and input2 from global memory
template<unsigned int ItemsPerThread, class KeyT, class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
void gmem_to_reg(KeyT (&output)[ItemsPerThread],
                 InputIterator input1,
                 InputIterator input2,
                 unsigned int  count1,
                 unsigned int  count2,
                 bool          IsLastTile)
{
    if(IsLastTile)
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            unsigned int idx = rocprim::flat_block_size() * item + threadIdx.x;
            if(idx < count1 + count2)
            {
                output[item] = (idx < count1) ? input1[idx] : input2[idx - count1];
            }
        }
    }
    else
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            unsigned int idx = rocprim::flat_block_size() * item + threadIdx.x;
            output[item]     = (idx < count1) ? input1[idx] : input2[idx - count1];
        }
    }
}

template<unsigned int BlockSize, unsigned int ItemsPerThread, class KeyT, class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
void reg_to_shared(OutputIterator output, KeyT (&input)[ItemsPerThread])
{
    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item)
    {
        unsigned int idx = BlockSize * item + threadIdx.x;
        output[idx]      = input[item];
    }
}

template<class Key,
         class Value,
         unsigned int            BlockSize,
         unsigned int            ItemsPerThread,
         arch::wavefront::target TargetWaveSize,
         typename Enable = void>
struct block_merge_impl;

template<class Key,
         class Value,
         unsigned int            BlockSize,
         unsigned int            ItemsPerThread,
         arch::wavefront::target TargetWaveSize>
struct block_merge_impl<
    Key,
    Value,
    BlockSize,
    ItemsPerThread,
    TargetWaveSize,
    std::enable_if_t<
        !std::is_trivially_copyable<Value>::value || rocprim::is_floating_point<Value>::value
        || rocprim::is_integral<Value>::value || std::is_same<Value, ::rocprim::empty_type>::value>>
{

    static constexpr bool         with_values = !std::is_same<Value, ::rocprim::empty_type>::value;
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using block_store
        = block_store_impl<with_values, BlockSize, ItemsPerThread, Key, Value, TargetWaveSize>;

    using keys_storage_   = Key[items_per_block + 1];
    using values_storage_ = Value[items_per_block + 1];

    union storage_type
    {
        typename block_store::storage_type store;
        ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
        detail::raw_storage<keys_storage_>   keys;
        detail::raw_storage<values_storage_> values;
        ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP
    };

    template<class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator,
             class OffsetT,
             class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void process_tile(KeysInputIterator    keys_input,
                                                          KeysOutputIterator   keys_output,
                                                          ValuesInputIterator  values_input,
                                                          ValuesOutputIterator values_output,
                                                          const OffsetT        input_size,
                                                          const OffsetT        current_run_length,
                                                          const unsigned int   num_blocks,
                                                          BinaryFunction       compare_function,
                                                          const OffsetT*       merge_partitions,
                                                          storage_type&        storage)
    {

        auto& keys_shared   = storage.keys.get();
        auto& values_shared = storage.values.get();

        const unsigned short flat_id       = block_thread_id<0>();
        const unsigned int   flat_block_id = ::rocprim::flat_block_id();
        if(flat_block_id >= num_blocks)
        {
            return;
        }

        const bool is_incomplete_tile = flat_block_id == (input_size / items_per_block);

        // Read global partition indices for current block.
        // The items in merge_partitions point are indices to items in keys_input, these indices are
        // partition point in Left Run, the indices of those in Right Run will be computed later.
        const OffsetT partition_beg = merge_partitions[flat_block_id];
        const OffsetT partition_end = merge_partitions[flat_block_id + 1];

        // The number of items in a single run is current_run_length, so the number of items in the
        // two merged runs will be 2 * current_run_length.
        const OffsetT merged_run_length = 2 * current_run_length;

        // The begin index of Left Run in keys_input.
        const OffsetT global_offset = static_cast<OffsetT>(flat_block_id) * items_per_block;

        // The pair of runs to be merged starts here.
        const OffsetT merge_run_base = (global_offset / merged_run_length) * merged_run_length;
        // diag is the output index relative to the current Merge Group. It represents the number of
        // items already consumed by previous blocks within this specific pair of runs.
        const OffsetT diag
            = static_cast<OffsetT>(flat_block_id) * items_per_block - merge_run_base;

        // For each pair of runs to be merged, the input keys for the Left the Right Runs are stored 
        // adjacent to eachother
        const OffsetT run_beg_L = partition_beg;
        OffsetT       run_end_L = partition_end;

        // The Left Run starts at merge_run_base.
        // The Right Run starts immediately after the Left Run.
        const OffsetT run_base_R = merge_run_base + current_run_length;

        // diag represents the number of items already consumed by previous blocks, and partition_beg
        // represents partition point in Left Run, which is
        // Principle: Consumed_Right = Total_Consumed - Consumed_Left
        const OffsetT consumed_beg_L = partition_beg - merge_run_base;
        const OffsetT consumed_beg_R = diag - consumed_beg_L;

        const OffsetT run_beg_R = rocprim::min(input_size, run_base_R + consumed_beg_R);

        // The total items consumed at the end of current block is (diag + items_per_block).
        const OffsetT consumed_total_end = diag + items_per_block;
        const OffsetT consumed_end_L     = partition_end - merge_run_base;
        const OffsetT consumed_end_R     = consumed_total_end - consumed_end_L;

        OffsetT run_end_R = rocprim::min(input_size, run_base_R + consumed_end_R);

        // Handle the boundary case where this block is the last one in the Merge Group
        // or covers the end of the input.
        if(global_offset + items_per_block >= merge_run_base + merged_run_length)
        {
            run_end_L = rocprim::min(input_size, merge_run_base + current_run_length);
            run_end_R = rocprim::min(input_size, merge_run_base + merged_run_length);
        }

        // Number of keys per block
        const unsigned int num_keys_L = static_cast<unsigned int>(run_end_L - run_beg_L);
        const unsigned int num_keys_R = static_cast<unsigned int>(run_end_R - run_beg_R);
        // Load keys_left & keys_right
        Key keys[ItemsPerThread];
        gmem_to_reg<ItemsPerThread>(keys,
                                    keys_input + run_beg_L,
                                    keys_input + run_beg_R,
                                    num_keys_L,
                                    num_keys_R,
                                    is_incomplete_tile);
        // Load keys into shared memory
        reg_to_shared<BlockSize, ItemsPerThread>(keys_shared, keys);

        Value values[ItemsPerThread];
        if constexpr(with_values)
        {
            gmem_to_reg<ItemsPerThread>(values,
                                        values_input + run_beg_L,
                                        values_input + run_beg_R,
                                        num_keys_L,
                                        num_keys_R,
                                        is_incomplete_tile);
        }
        rocprim::syncthreads();

        // diag_local is the number of items this thread needs to skip (output offset).
        const unsigned int diag_local
            = rocprim::min(num_keys_L + num_keys_R, ItemsPerThread * flat_id);

        // Search for the pivot in the Left Run.
        // Note: In shared memory, keys are stored as [Left Run ... | Right Run ...].
        // So, the first key for the right run is stored at &keys_shared[num_keys_L].
        const unsigned int consumed_L = merge_path(keys_shared,
                                                    &keys_shared[num_keys_L],
                                                    num_keys_L,
                                                    num_keys_R,
                                                    diag_local,
                                                    compare_function);

        const unsigned int consumed_R = diag_local - consumed_L;

        // Translate relative indices (0..len) into absolute shared memory indices.

        // Left Run is at the beginning of shared memory: [0, num_keys_L)
        const unsigned int idx_beg_L = consumed_L;
        const unsigned int idx_end_L = num_keys_L;

        // Right Run starts immediately after Left Run: [num_keys_L, num_keys_L + num_keys_R)
        const unsigned int shmem_base_R = num_keys_L;
        const unsigned int idx_beg_R    = shmem_base_R + consumed_R;
        const unsigned int idx_end_R    = shmem_base_R + num_keys_R;

        // range_t defines the available windows for this thread to merge from.
        range_t<> range_local{idx_beg_L, idx_end_L, idx_beg_R, idx_end_R};

        unsigned int indices[ItemsPerThread];

        serial_merge<false>(keys_shared, keys, indices, range_local, compare_function);
        rocprim::syncthreads();

        if constexpr(with_values)
        {
            reg_to_shared<BlockSize, ItemsPerThread>(values_shared, values);

            rocprim::syncthreads();

            ROCPRIM_UNROLL
            for(unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                values[item] = values_shared[indices[item]];
            }

            rocprim::syncthreads();
        }

        const OffsetT offset = static_cast<OffsetT>(flat_block_id) * items_per_block;
        block_store().store(offset,
                            input_size - offset,
                            is_incomplete_tile,
                            keys_output,
                            values_output,
                            keys,
                            values,
                            storage.store);
    }
};

// The specialization below exists because the compiler creates slow code for
// ValueTypes with misaligned datastructures in them (e.g. custom_char_double)
// when storing/loading those ValueTypes to/from registers.
// Thus this is a temporary workaround.
template<class Key,
         class Value,
         unsigned int            BlockSize,
         unsigned int            ItemsPerThread,
         arch::wavefront::target TargetWaveSize>
struct block_merge_impl<Key,
                        Value,
                        BlockSize,
                        ItemsPerThread,
                        TargetWaveSize,
                        std::enable_if_t<std::is_trivially_copyable<Value>::value
                                         && !rocprim::is_floating_point<Value>::value
                                         && !rocprim::is_integral<Value>::value
                                         && !std::is_same<Value, ::rocprim::empty_type>::value>>
{
    static constexpr bool         with_values = !std::is_same<Value, ::rocprim::empty_type>::value;
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using block_store = block_store_impl<false, BlockSize, ItemsPerThread, Key, Value>;

    using keys_storage_   = Key[items_per_block + 1];
    using values_storage_ = Value[items_per_block + 1];

    union storage_type
    {
        typename block_store::storage_type store;
        ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
        detail::raw_storage<keys_storage_>   keys;
        detail::raw_storage<values_storage_> values;
        ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP
    };

    template<class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator,
             class OffsetT,
             class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void process_tile(KeysInputIterator    keys_input,
                                                          KeysOutputIterator   keys_output,
                                                          ValuesInputIterator  values_input,
                                                          ValuesOutputIterator values_output,
                                                          const OffsetT        input_size,
                                                          const OffsetT        current_run_length,
                                                          const unsigned int   num_blocks,
                                                          BinaryFunction       compare_function,
                                                          const OffsetT*       merge_partitions,
                                                          storage_type&        storage)
    {
        auto& keys_shared   = storage.keys.get();
        auto& values_shared = storage.values.get();

        const unsigned short flat_id       = block_thread_id<0>();
        const unsigned int   flat_block_id = ::rocprim::flat_block_id();
        if(flat_block_id >= num_blocks)
        {
            return;
        }

        const bool is_incomplete_tile = flat_block_id == (input_size / items_per_block);

        const OffsetT partition_beg = merge_partitions[flat_block_id];
        const OffsetT partition_end = merge_partitions[flat_block_id + 1];

        const OffsetT merged_run_length = 2 * current_run_length;

        const OffsetT global_offset = static_cast<OffsetT>(flat_block_id) * items_per_block;

        const OffsetT merge_run_base = (global_offset / merged_run_length) * merged_run_length;
        const OffsetT diag
            = static_cast<OffsetT>(flat_block_id) * items_per_block - merge_run_base;

        const OffsetT run_beg_L = partition_beg;
        OffsetT       run_end_L = partition_end;

        const OffsetT run_base_R = merge_run_base + current_run_length;

        const OffsetT consumed_beg_L = partition_beg - merge_run_base;
        const OffsetT consumed_beg_R = diag - consumed_beg_L;

        const OffsetT run_beg_R = rocprim::min(input_size, run_base_R + consumed_beg_R);

        const OffsetT consumed_total_end = diag + items_per_block;
        const OffsetT consumed_end_L     = partition_end - merge_run_base;
        const OffsetT consumed_end_R     = consumed_total_end - consumed_end_L;

        OffsetT run_end_R = rocprim::min(input_size, run_base_R + consumed_end_R);

        if(global_offset + items_per_block >= merge_run_base + merged_run_length)
        {
            run_end_L = rocprim::min(input_size, merge_run_base + current_run_length);
            run_end_R = rocprim::min(input_size, merge_run_base + merged_run_length);
        }

        const unsigned int num_keys_L = static_cast<unsigned int>(run_end_L - run_beg_L);
        const unsigned int num_keys_R = static_cast<unsigned int>(run_end_R - run_beg_R);

        Key keys[ItemsPerThread];
        gmem_to_reg<ItemsPerThread>(keys,
                                    keys_input + run_beg_L,
                                    keys_input + run_beg_R,
                                    num_keys_L,
                                    num_keys_R,
                                    is_incomplete_tile);

        reg_to_shared<BlockSize, ItemsPerThread>(keys_shared, keys);

        rocprim::syncthreads();

        const unsigned int diag_local
            = rocprim::min(num_keys_L + num_keys_R, ItemsPerThread * flat_id);

        const unsigned int consumed_L = merge_path(keys_shared,
                                                    &keys_shared[num_keys_L],
                                                    num_keys_L,
                                                    num_keys_R,
                                                    diag_local,
                                                    compare_function);

        const unsigned int consumed_R = diag_local - consumed_L;

        const unsigned int idx_beg_L = consumed_L;
        const unsigned int idx_end_L = num_keys_L;

        const unsigned int shmem_base_R = num_keys_L;
        const unsigned int idx_beg_R    = shmem_base_R + consumed_R;
        const unsigned int idx_end_R    = shmem_base_R + num_keys_R;

        range_t<> range_local{idx_beg_L, idx_end_L, idx_beg_R, idx_end_R};

        unsigned int indices[ItemsPerThread];

        serial_merge<false>(keys_shared, keys, indices, range_local, compare_function);
        rocprim::syncthreads();

        if constexpr(with_values)
        {
            const ValuesInputIterator input_L = values_input + run_beg_L;
            const ValuesInputIterator input_R = values_input + run_beg_R;
            if(is_incomplete_tile)
            {
                ROCPRIM_UNROLL
                for(unsigned int item = 0; item < ItemsPerThread; ++item)
                {
                    const unsigned int idx = BlockSize * item + threadIdx.x;
                    if(idx < num_keys_L)
                    {
                        values_shared[idx] = input_L[idx];
                    }
                    else if(idx - num_keys_L < num_keys_R)
                    {
                        values_shared[idx] = input_R[idx - num_keys_L];
                    }
                }
            }
            else
            {
                ROCPRIM_UNROLL
                for(unsigned int item = 0; item < ItemsPerThread; ++item)
                {
                    const unsigned int idx = BlockSize * item + threadIdx.x;
                    if(idx < num_keys_L)
                    {
                        values_shared[idx] = input_L[idx];
                    }
                    else
                    {
                        values_shared[idx] = input_R[idx - num_keys_L];
                    }
                }
            }

            rocprim::syncthreads();
            const OffsetT thread_offset = items_per_block * static_cast<OffsetT>(flat_block_id)
                                            + ItemsPerThread * flat_id;
            if(is_incomplete_tile)
            {
                ROCPRIM_UNROLL
                for(unsigned int item = 0; item < ItemsPerThread; ++item)
                {
                    if(flat_id * ItemsPerThread + item < num_keys_L + num_keys_R)
                    {
                        values_output[thread_offset + item] = values_shared[indices[item]];
                    }
                }
            }
            else
            {
                ROCPRIM_UNROLL
                for(unsigned int item = 0; item < ItemsPerThread; ++item)
                {
                    values_output[thread_offset + item] = values_shared[indices[item]];
                }
            }

            rocprim::syncthreads();
        }

        const OffsetT offset = static_cast<OffsetT>(flat_block_id) * items_per_block;
        Value         values[ItemsPerThread];
        block_store().store(offset,
                            input_size - offset,
                            is_incomplete_tile,
                            keys_output,
                            values_output,
                            keys,
                            values,
                            storage.store);
    }
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_SORT_MERGEPATH_HPP_