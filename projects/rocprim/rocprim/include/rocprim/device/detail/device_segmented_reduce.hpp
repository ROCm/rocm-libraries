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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_REDUCE_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_REDUCE_HPP_

#include <iterator>
#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "../../block/block_load_func.hpp"
#include "../../block/block_reduce.hpp"
#include "../config_types.hpp"
#include "../device_segmented_reduce_config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class ArchConfig,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class ResultType,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void segmented_reduce(InputIterator input,
                      OutputIterator output,
                      OffsetIterator begin_offsets,
                      OffsetIterator end_offsets,
                      BinaryFunction reduce_op,
                      ResultType initial_value)
{
    using offset_type = typename std::iterator_traits<OffsetIterator>::value_type;

    static constexpr reduce_config_params params = ArchConfig::params;

    constexpr unsigned int block_size       = params.kernel_config.block_size;
    constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;
    constexpr unsigned int items_per_block  = block_size * items_per_thread;

    using reduce_type = ::rocprim::block_reduce<ResultType, block_size, params.block_reduce_method>;

    ROCPRIM_SHARED_MEMORY typename reduce_type::storage_type reduce_storage;

    const unsigned int flat_id    = ::rocprim::detail::block_thread_id<0>();
    const unsigned int segment_id = ::rocprim::detail::block_id<0>();

    const offset_type begin_offset = begin_offsets[segment_id];
    const offset_type end_offset   = end_offsets[segment_id];

    // Empty segment
    if(end_offset <= begin_offset)
    {
        if(flat_id == 0)
        {
            output[segment_id] = initial_value;
        }
        return;
    }

    ResultType  result;
    offset_type block_offset = begin_offset;
    if(block_offset + static_cast<offset_type>(items_per_block) > end_offset)
    {
        // Segment is shorter than items_per_block

        // Load the partial block and reduce the current thread's values.
        // valid_count is shorter than items_per_block, so it doesn't need to be
        // of type offset_type.
        const unsigned int valid_count = end_offset - block_offset;
        if(flat_id < valid_count)
        {
            offset_type offset = block_offset + flat_id;
            result             = input[offset];
            offset += block_size;
            while(offset < end_offset)
            {
                result = reduce_op(result, static_cast<ResultType>(input[offset]));
                offset += block_size;
            }
        }

        // Reduce threads' reductions to compute the final result
        if(valid_count >= block_size)
        {
            // All threads have at least one value, i.e. result has valid value
            reduce_type().reduce(result, result, reduce_storage, reduce_op);
        }
        else
        {
            reduce_type().reduce(result, result, valid_count, reduce_storage, reduce_op);
        }
    }
    else
    {
        // Long segments

        ResultType values[items_per_thread];

        // Load the first block and reduce the current thread's values
        block_load_direct_striped<block_size>(flat_id, input + block_offset, values);
        result = values[0];
        for(unsigned int i = 1; i < items_per_thread; i++)
        {
            result = reduce_op(result, values[i]);
        }
        block_offset += items_per_block;

        // Load next full blocks and continue reduction
        while(block_offset + static_cast<offset_type>(items_per_block) < end_offset)
        {
            block_load_direct_striped<block_size>(flat_id, input + block_offset, values);
            for(unsigned int i = 0; i < items_per_thread; i++)
            {
                result = reduce_op(result, values[i]);
            }
            block_offset += items_per_block;
        }

        // Load the last (probably partial) block and continue reduction
        // For this last block the valid_count is less or equal than items_per_block,
        // so it doesn't need to be of type offset_type.
        const unsigned int valid_count = end_offset - block_offset;
        block_load_direct_striped<block_size>(flat_id, input + block_offset, values, valid_count);
        for(unsigned int i = 0; i < items_per_thread; i++)
        {
            if(i * block_size + flat_id < valid_count)
            {
                result = reduce_op(result, values[i]);
            }
        }

        // Reduce threads' reductions to compute the final result
        reduce_type().reduce(result, result, reduce_storage, reduce_op);
    }

    if(flat_id == 0)
    {
        output[segment_id] = reduce_op(initial_value, result);
    }
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_REDUCE_HPP_
