// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_HPP_

#include <iterator>
#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"
#include "../config_types.hpp"
#include "../device_merge_config.hpp"
#include "device_config_helper.hpp"

#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "../../block/block_store.hpp"
#include "../../detail/merge_path.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

ROCPRIM_DEVICE ROCPRIM_INLINE
range_t<> compute_range(const unsigned int id,
                        const unsigned int size1,
                        const unsigned int size2,
                        const unsigned int spacing,
                        const unsigned int p1,
                        const unsigned int p2)
{
    unsigned int diag1 = id * spacing;
    unsigned int diag2 = min(size1 + size2, diag1 + spacing);

    return range_t<>{p1, p2, diag1 - p1, diag2 - p2};
}

template<class IndexIterator,
         class KeysInputIterator1,
         class KeysInputIterator2,
         class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
void partition_kernel_impl(IndexIterator      indices,
                           KeysInputIterator1 keys_input1,
                           KeysInputIterator2 keys_input2,
                           const size_t       input1_size,
                           const size_t       input2_size,
                           const unsigned int spacing,
                           BinaryFunction     compare_function)
{
    const unsigned int flat_id         = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id   = ::rocprim::detail::block_id<0>();
    const unsigned int flat_block_size = ::rocprim::detail::block_size<0>();
    const unsigned int input_size      = input1_size + input2_size;
    const unsigned int id              = flat_block_id * flat_block_size + flat_id;
    const unsigned int partition_id    = id * spacing;
    const unsigned int partitions      = (input_size + spacing - 1) / spacing;

    if(id > partitions)
    {
        return;
    }

    size_t diag = min(static_cast<size_t>(partition_id), input1_size + input2_size);

    unsigned int begin
        = merge_path(keys_input1, keys_input2, input1_size, input2_size, diag, compare_function);

    indices[id] = begin;
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class KeysInputIterator1,
         class KeysInputIterator2,
         class KeyType>
ROCPRIM_DEVICE ROCPRIM_INLINE
void load(unsigned int       flat_id,
          KeysInputIterator1 keys_input1,
          KeysInputIterator2 keys_input2,
          KeyType*           keys_shared,
          const size_t       input1_size,
          const size_t       input2_size)
{
    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        unsigned int index = BlockSize * i + flat_id;
        if(index < input1_size)
        {
            keys_shared[index] = keys_input1[index];
        }
        else if(index < input1_size + input2_size)
        {
            keys_shared[index] = static_cast<KeyType>(keys_input2[index - input1_size]);
        }
    }

    ::rocprim::syncthreads();
}

template<unsigned int BlockSize,
         class KeysInputIterator1,
         class KeysInputIterator2,
         class KeyType,
         unsigned int ItemsPerThread,
         class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
void merge_keys(unsigned int       flat_id,
                KeysInputIterator1 keys_input1,
                KeysInputIterator2 keys_input2,
                KeyType (&key_inputs)[ItemsPerThread],
                unsigned int (&index)[ItemsPerThread],
                KeyType*       keys_shared,
                range_t<>      range,
                BinaryFunction compare_function)
{
    load<BlockSize, ItemsPerThread>(flat_id,
                                    keys_input1 + range.begin1,
                                    keys_input2 + range.begin2,
                                    keys_shared,
                                    range.count1(),
                                    range.count2());

    range_t<> range_local{0, range.count1(), range.count1(), (range.count1() + range.count2())};

    unsigned int diag      = ItemsPerThread * flat_id;
    unsigned int partition = merge_path(keys_shared + range_local.begin1,
                                        keys_shared + range_local.begin2,
                                        range_local.count1(),
                                        range_local.count2(),
                                        diag,
                                        compare_function);

    range_t<> range_partition{range_local.begin1 + partition,
                              range_local.end1,
                              range_local.begin2 + diag - partition,
                              range_local.end2};

    serial_merge<false>(keys_shared, key_inputs, index, range_partition, compare_function);
}

template<bool         WithValues,
         unsigned int BlockSize,
         class ValuesInputIterator1,
         class ValuesInputIterator2,
         class ValuesOutputIterator,
         unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::enable_if<WithValues>::type merge_values(unsigned int         flat_id,
                                                       ValuesInputIterator1 values_input1,
                                                       ValuesInputIterator2 values_input2,
                                                       ValuesOutputIterator values_output,
                                                       unsigned int (&index)[ItemsPerThread],
                                                       const size_t input1_size,
                                                       const size_t input2_size)
{
    using value_type = typename std::iterator_traits<ValuesInputIterator1>::value_type;

    unsigned int count = input1_size + input2_size;

    value_type values[ItemsPerThread];

    if(count >= ItemsPerThread * BlockSize)
    {
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            values[i] = (index[i] < input1_size)
                            ? values_input1[index[i]]
                            : static_cast<value_type>(values_input2[index[i] - input1_size]);
        }
    }
    else
    {
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            if(flat_id * ItemsPerThread + i < count)
            {
                values[i] = (index[i] < input1_size)
                                ? values_input1[index[i]]
                                : static_cast<value_type>(values_input2[index[i] - input1_size]);
            }
        }
    }

    ::rocprim::syncthreads();

    block_store_direct_blocked(flat_id, values_output, values, count);
}

template<bool         WithValues,
         unsigned int BlockSize,
         class ValuesInputIterator1,
         class ValuesInputIterator2,
         class ValuesOutputIterator,
         unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::enable_if<!WithValues>::type merge_values(unsigned int         flat_id,
                                                        ValuesInputIterator1 values_input1,
                                                        ValuesInputIterator2 values_input2,
                                                        ValuesOutputIterator values_output,
                                                        unsigned int (&index)[ItemsPerThread],
                                                        const size_t input1_size,
                                                        const size_t input2_size)
{
    (void)flat_id;
    (void)values_input1;
    (void)values_input2;
    (void)values_output;
    (void)index;
    (void)input1_size;
    (void)input2_size;
}

template<class Config, class Key, class Value>
struct merge_kernel_impl_
{
    static constexpr merge_config_params params = device_params<Config>();

    static constexpr unsigned int block_size       = params.kernel_config.block_size;
    static constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;
    static constexpr unsigned int items_per_block  = block_size * items_per_thread;
    static constexpr unsigned int input_block_size = block_size * items_per_thread + 1;
    static constexpr bool         with_values = !std::is_same<Value, ::rocprim::empty_type>::value;

    // Block primitives
    using keys_store_type
        = ::rocprim::block_store<Key,
                                 block_size,
                                 items_per_thread,
                                 ::rocprim::block_store_method::block_store_transpose>;

    union storage_type
    {
        ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
        typename detail::raw_storage<Key[input_block_size]> keys_shared;
        ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP
        typename keys_store_type::storage_type keys_store;
    };

    template<
    class IndexIterator,
    class KeysInputIterator1,
    class KeysInputIterator2,
    class KeysOutputIterator,
    class ValuesInputIterator1,
    class ValuesInputIterator2,
    class ValuesOutputIterator,
    class BinaryFunction
>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void merge(IndexIterator indices,
                       KeysInputIterator1 keys_input1,
                       KeysInputIterator2 keys_input2,
                       KeysOutputIterator keys_output,
                       ValuesInputIterator1 values_input1,
                       ValuesInputIterator2 values_input2,
                       ValuesOutputIterator values_output,
                       const size_t input1_size,
                       const size_t input2_size,
                       BinaryFunction compare_function, 
                       storage_type&           storage)
    {
        using key_type1   = typename std::iterator_traits<KeysInputIterator1>::value_type;
        using key_type2   = typename std::iterator_traits<KeysInputIterator2>::value_type;
        using value_type1 = typename std::iterator_traits<ValuesInputIterator1>::value_type;
        using value_type2 = typename std::iterator_traits<ValuesInputIterator2>::value_type;

        if constexpr(with_values)
        {
            static_assert(std::is_convertible_v<value_type2, value_type1>,
                          "values_input2 must be convertible to values_input1");
        }

        static_assert(std::is_convertible_v<key_type2, key_type1>,
                      "Keys_input2 must be convertible to keys_input1");

        Key          input[items_per_thread];
        unsigned int index[items_per_thread];

        const unsigned int flat_id             = ::rocprim::detail::block_thread_id<0>();
        const unsigned int flat_block_id       = ::rocprim::detail::block_id<0>();
        const unsigned int block_offset        = flat_block_id * items_per_block;
        const unsigned int count               = input1_size + input2_size;
        const unsigned int valid_in_last_block = count - block_offset;
        const bool         is_incomplete_block = valid_in_last_block < items_per_block;

        const unsigned int partitions = (count + items_per_block - 1) / items_per_block;

        const unsigned int p1 = indices[rocprim::min(flat_block_id, partitions)];
        const unsigned int p2 = indices[rocprim::min(flat_block_id + 1, partitions)];

        range_t<> range
            = compute_range(flat_block_id, input1_size, input2_size, items_per_block, p1, p2);

        merge_keys<block_size>(flat_id,
                               keys_input1,
                               keys_input2,
                               input,
                               index,
                               storage.keys_shared.get(),
                               range,
                               compare_function);

        ::rocprim::syncthreads();

        if(is_incomplete_block) // # elements in last block may not equal items_per_block for the last block
        {
            keys_store_type().store(keys_output + block_offset,
                                    input,
                                    valid_in_last_block,
                                    storage.keys_store);
        }
        else
        {
            keys_store_type().store(keys_output + block_offset, input, storage.keys_store);
        }

        merge_values<with_values, block_size>(flat_id,
                                              values_input1 + range.begin1,
                                              values_input2 + range.begin2,
                                              values_output + block_offset,
                                              index,
                                              range.count1(),
                                              range.count2());
    }
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_HPP_
