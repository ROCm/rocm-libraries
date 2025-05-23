/******************************************************************************
* Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
* Modifications Copyright (c) 2022-2025, Advanced Micro Devices, Inc.  All rights reserved.
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
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         typename Enable = void>
struct block_merge_impl;

template<class Key, class Value, unsigned int BlockSize, unsigned int ItemsPerThread>
struct block_merge_impl<
    Key,
    Value,
    BlockSize,
    ItemsPerThread,
    std::enable_if_t<!std::is_trivially_copyable<Value>::value
                     || rocprim::is_floating_point<Value>::value || std::is_integral<Value>::value>>
{

    static constexpr bool         with_values = !std::is_same<Value, ::rocprim::empty_type>::value;
    static constexpr unsigned int items_per_tile = BlockSize * ItemsPerThread;

    using block_store = block_store_impl<with_values, BlockSize, ItemsPerThread, Key, Value>;

    using keys_storage_   = Key[items_per_tile + 1];
    using values_storage_ = Value[items_per_tile + 1];

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
                                                              const OffsetT        sorted_block_size,
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

            const bool is_incomplete_tile = flat_block_id == (input_size / items_per_tile);

            const OffsetT partition_beg = merge_partitions[flat_block_id];
            const OffsetT partition_end = merge_partitions[flat_block_id + 1];

            const unsigned int merged_tiles_number        = sorted_block_size / items_per_tile;
            const unsigned int target_merged_tiles_number = merged_tiles_number * 2;
            const unsigned int mask                       = target_merged_tiles_number - 1;
            const unsigned int tilegroup_start_id         = ~mask & flat_block_id;

            const OffsetT tilegroup_start = static_cast<OffsetT>(tilegroup_start_id)
                                            * items_per_tile; // Tile-group starts here
            const OffsetT diag
                = static_cast<OffsetT>(flat_block_id) * items_per_tile - tilegroup_start;

            const OffsetT keys1_beg = partition_beg;
            OffsetT       keys1_end = partition_end;
            const OffsetT keys2_beg
                = rocprim::min(input_size,
                               2 * tilegroup_start + sorted_block_size + diag - partition_beg);
            OffsetT keys2_end = rocprim::min(input_size,
                                             2 * tilegroup_start + sorted_block_size + diag
                                                 + items_per_tile - partition_end);

            if(mask == (mask & flat_block_id)) // If last tile in the tile-group
            {
                keys1_end = rocprim::min(input_size, tilegroup_start + sorted_block_size);
                keys2_end = rocprim::min(input_size, tilegroup_start + sorted_block_size * 2);
            }

            // Number of keys per tile
            const unsigned int num_keys1 = static_cast<unsigned int>(keys1_end - keys1_beg);
            const unsigned int num_keys2 = static_cast<unsigned int>(keys2_end - keys2_beg);
            // Load keys1 & keys2
            Key keys[ItemsPerThread];
            gmem_to_reg<ItemsPerThread>(keys,
                                        keys_input + keys1_beg,
                                        keys_input + keys2_beg,
                                        num_keys1,
                                        num_keys2,
                                        is_incomplete_tile);
            // Load keys into shared memory
            reg_to_shared<BlockSize, ItemsPerThread>(keys_shared, keys);

            Value values[ItemsPerThread];
            if constexpr(with_values)
            {
                gmem_to_reg<ItemsPerThread>(values,
                                            values_input + keys1_beg,
                                            values_input + keys2_beg,
                                            num_keys1,
                                            num_keys2,
                                            is_incomplete_tile);
            }
            rocprim::syncthreads();

            const unsigned int diag0_local
                = rocprim::min(num_keys1 + num_keys2, ItemsPerThread * flat_id);

            const unsigned int keys1_beg_local = merge_path(keys_shared,
                                                            &keys_shared[num_keys1],
                                                            num_keys1,
                                                            num_keys2,
                                                            diag0_local,
                                                            compare_function);
            const unsigned int keys1_end_local = num_keys1;
            const unsigned int keys2_beg_local = diag0_local - keys1_beg_local;
            const unsigned int keys2_end_local = num_keys2;

            range_t<> range_local{keys1_beg_local,
                                  keys1_end_local,
                                  keys2_beg_local + keys1_end_local,
                                  keys2_end_local + keys1_end_local};

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

            const OffsetT offset = static_cast<OffsetT>(flat_block_id) * items_per_tile;
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
template<class Key, class Value, unsigned int BlockSize, unsigned int ItemsPerThread>
struct block_merge_impl<Key,
                        Value,
                        BlockSize,
                        ItemsPerThread,
                        std::enable_if_t<std::is_trivially_copyable<Value>::value
                                         && !rocprim::is_floating_point<Value>::value
                                         && !std::is_integral<Value>::value>>
{

    static constexpr bool         with_values = !std::is_same<Value, ::rocprim::empty_type>::value;
    static constexpr unsigned int items_per_tile = BlockSize * ItemsPerThread;

    using block_store = block_store_impl<false, BlockSize, ItemsPerThread, Key, Value>;

    using keys_storage_   = Key[items_per_tile + 1];
    using values_storage_ = Value[items_per_tile + 1];

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
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void process_tile(KeysInputIterator  keys_input,
                                                              KeysOutputIterator   keys_output,
                                                              ValuesInputIterator  values_input,
                                                              ValuesOutputIterator values_output,
                                                              const OffsetT        input_size,
                                                              const OffsetT        sorted_block_size,
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

            const bool is_incomplete_tile = flat_block_id == (input_size / items_per_tile);

            const OffsetT partition_beg = merge_partitions[flat_block_id];
            const OffsetT partition_end = merge_partitions[flat_block_id + 1];

            const unsigned int merged_tiles_number        = sorted_block_size / items_per_tile;
            const unsigned int target_merged_tiles_number = merged_tiles_number * 2;
            const unsigned int mask                       = target_merged_tiles_number - 1;
            const unsigned int tilegroup_start_id         = ~mask & flat_block_id;
            const OffsetT      tilegroup_start            = static_cast<OffsetT>(tilegroup_start_id)
                                            * items_per_tile; // Tile-group starts here
            const OffsetT diag
                = static_cast<OffsetT>(flat_block_id) * items_per_tile - tilegroup_start;

            const OffsetT keys1_beg = partition_beg;
            OffsetT       keys1_end = partition_end;
            const OffsetT keys2_beg
                = rocprim::min(input_size,
                               2 * tilegroup_start + sorted_block_size + diag - partition_beg);
            OffsetT keys2_end = rocprim::min(input_size,
                                             2 * tilegroup_start + sorted_block_size + diag
                                                 + items_per_tile - partition_end);

            if(mask == (mask & flat_block_id)) // If last tile in the tile-group
            {
                keys1_end = rocprim::min(input_size, tilegroup_start + sorted_block_size);
                keys2_end = rocprim::min(input_size, tilegroup_start + sorted_block_size * 2);
            }

            // Number of keys per tile
            const unsigned int num_keys1 = static_cast<unsigned int>(keys1_end - keys1_beg);
            const unsigned int num_keys2 = static_cast<unsigned int>(keys2_end - keys2_beg);
            // Load keys1 & keys2
            Key keys[ItemsPerThread];
            gmem_to_reg<ItemsPerThread>(keys,
                                        keys_input + keys1_beg,
                                        keys_input + keys2_beg,
                                        num_keys1,
                                        num_keys2,
                                        is_incomplete_tile);
            // Load keys into shared memory
            reg_to_shared<BlockSize, ItemsPerThread>(keys_shared, keys);

            rocprim::syncthreads();

            const unsigned int diag0_local
                = rocprim::min(num_keys1 + num_keys2, ItemsPerThread * flat_id);

            const unsigned int keys1_beg_local = merge_path(keys_shared,
                                                            &keys_shared[num_keys1],
                                                            num_keys1,
                                                            num_keys2,
                                                            diag0_local,
                                                            compare_function);
            const unsigned int keys1_end_local = num_keys1;
            const unsigned int keys2_beg_local = diag0_local - keys1_beg_local;
            const unsigned int keys2_end_local = num_keys2;

            range_t<> range_local{keys1_beg_local,
                                  keys1_end_local,
                                  keys2_beg_local + keys1_end_local,
                                  keys2_end_local + keys1_end_local};

            unsigned int indices[ItemsPerThread];

            serial_merge<false>(keys_shared, keys, indices, range_local, compare_function);
            rocprim::syncthreads();

            if constexpr(with_values)
            {
                const ValuesInputIterator input1 = values_input + keys1_beg;
                const ValuesInputIterator input2 = values_input + keys2_beg;
                if(is_incomplete_tile)
                {
                    ROCPRIM_UNROLL
                    for(unsigned int item = 0; item < ItemsPerThread; ++item)
                    {
                        const unsigned int idx = BlockSize * item + threadIdx.x;
                        if(idx < num_keys1)
                        {
                            values_shared[idx] = input1[idx];
                        }
                        else if(idx - num_keys1 < num_keys2)
                        {
                            values_shared[idx] = input2[idx - num_keys1];
                        }
                    }
                }
                else
                {
                    ROCPRIM_UNROLL
                    for(unsigned int item = 0; item < ItemsPerThread; ++item)
                    {
                        const unsigned int idx = BlockSize * item + threadIdx.x;
                        if(idx < num_keys1)
                        {
                            values_shared[idx] = input1[idx];
                        }
                        else
                        {
                            values_shared[idx] = input2[idx - num_keys1];
                        }
                    }
                }

                rocprim::syncthreads();
                const OffsetT thread_offset = items_per_tile * static_cast<OffsetT>(flat_block_id)
                                              + ItemsPerThread * flat_id;
                if(is_incomplete_tile)
                {
                    ROCPRIM_UNROLL
                    for(unsigned int item = 0; item < ItemsPerThread; ++item)
                    {
                        if(flat_id * ItemsPerThread + item < num_keys1 + num_keys2)
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

            const OffsetT offset = static_cast<OffsetT>(flat_block_id) * items_per_tile;
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