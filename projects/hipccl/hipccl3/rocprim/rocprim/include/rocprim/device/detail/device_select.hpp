// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SELECT_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SELECT_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"
#include "../../block/block_discontinuity.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
ROCPRIM_DEVICE inline
void scatter_kernel_impl(InputIterator input,
                         size_t input_size,
                         FlagIterator flags,
                         unsigned int * output_indices,
                         OutputIterator output,
                         SelectedCountOutputIterator selected_count_output)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    // Fix for cases when output_type is void (there's no sizeof(void))
    using value_type = typename std::conditional<
        std::is_same<void, output_type>::value, input_type, output_type
    >::type;
    // Use smaller type for private storage
    using result_type = typename std::conditional<
        (sizeof(value_type) > sizeof(input_type)), input_type, value_type
    >::type;
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * items_per_block;
    const unsigned int number_of_blocks = (input_size + items_per_block - 1)/items_per_block;
    auto valid_in_last_block = input_size - items_per_block * (number_of_blocks - 1);

    result_type values[ItemsPerThread];
    unsigned int out_indices[ItemsPerThread];
    bool is_selected[ItemsPerThread];

    // load input values into values
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            values,
            valid_in_last_block
        );
        block_load_direct_striped<BlockSize>(
            flat_id,
            output_indices + block_offset,
            out_indices,
            valid_in_last_block
        );
        block_load_direct_striped<BlockSize>(
            flat_id,
            flags + block_offset,
            is_selected,
            valid_in_last_block
        );
    }
    else
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            values
        );
        block_load_direct_striped<BlockSize>(
            flat_id,
            output_indices + block_offset,
            out_indices
        );
        block_load_direct_striped<BlockSize>(
            flat_id,
            flags + block_offset,
            is_selected
        );
    }

    // Scatter
    unsigned int index = block_offset + flat_id;
    #pragma unroll
    for(auto i = 0U; i < ItemsPerThread; i++)
    {
        if(index < input_size)
        {
            if(is_selected[i])
            {
                output[out_indices[i]] = values[i];
            }
        }
        index += BlockSize;
    }

    // First thread in first block updates total number of selected values
    if((flat_id + block_offset) == 0)
    {
        *selected_count_output = output_indices[input_size - 1] + flags[input_size - 1];
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class SelectOp
>
ROCPRIM_DEVICE inline
void scatter_if_kernel_impl(InputIterator input,
                            size_t input_size,
                            unsigned int * output_indices,
                            OutputIterator output,
                            SelectedCountOutputIterator selected_count_output,
                            SelectOp select_op)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * items_per_block;
    const unsigned int number_of_blocks = (input_size + items_per_block - 1)/items_per_block;
    auto valid_in_last_block = input_size - items_per_block * (number_of_blocks - 1);

    input_type values[ItemsPerThread];
    unsigned int out_indices[ItemsPerThread];

    // load input values into values
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            values,
            valid_in_last_block
        );
        block_load_direct_striped<BlockSize>(
            flat_id,
            output_indices + block_offset,
            out_indices,
            valid_in_last_block
        );
    }
    else
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            values
        );
        block_load_direct_striped<BlockSize>(
            flat_id,
            output_indices + block_offset,
            out_indices
        );
    }

    // Scatter
    unsigned int index = block_offset + flat_id;
    #pragma unroll
    for(auto i = 0U; i < ItemsPerThread; i++)
    {
        if(index < input_size)
        {
            if(select_op(values[i]))
            {
                output[out_indices[i]] = values[i];
            }
        }
        index += BlockSize;
    }

    // First thread in first block updates total number of selected values
    if((flat_id + block_offset) == 0)
    {
        *selected_count_output =
            output_indices[input_size - 1] + select_op(input[input_size - 1]);
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OutputFlagIterator,
    class InequalityOp
>
ROCPRIM_DEVICE inline
void flag_unique_kernel_impl(InputIterator input,
                             size_t input_size,
                             OutputFlagIterator output_flags,
                             InequalityOp inequality_op)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using block_load_type = ::rocprim::block_load<
        input_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose
    >;
    using block_store_type = ::rocprim::block_store<
        bool, BlockSize, ItemsPerThread,
        ::rocprim::block_store_method::block_store_transpose
    >;
    using block_discontinuity_type = ::rocprim::block_discontinuity<
        input_type, BlockSize
    >;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
        typename block_discontinuity_type::storage_type discontinuity;
    } storage;

    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * items_per_block;
    const unsigned int number_of_blocks = (input_size + items_per_block - 1)/items_per_block;
    auto valid_in_last_block = input_size - items_per_block * (number_of_blocks - 1);

    input_type values[ItemsPerThread];
    input_type predecessor;

    // load input values into values
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_type()
            .load(
                input + block_offset,
                values,
                valid_in_last_block,
                storage.load
            );
    }
    else
    {
        block_load_type()
            .load(
                input + block_offset,
                values,
                storage.load
            );
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    bool flags[ItemsPerThread];
    if(flat_block_id > 0)
    {
        predecessor = input[block_offset - 1];
        block_discontinuity_type()
            .flag_heads(
                flags,
                predecessor,
                values,
                inequality_op
            );
    }
    else
    {
        block_discontinuity_type()
            .flag_heads(
                flags,
                values,
                inequality_op
            );
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    // Save values into output array
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_store_type()
            .store(
                output_flags + block_offset,
                flags,
                valid_in_last_block,
                storage.store
            );
    }
    else
    {
        block_store_type()
            .store(
                output_flags + block_offset,
                flags,
                storage.store
            );
    }
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SELECT_HPP_
