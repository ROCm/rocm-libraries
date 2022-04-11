// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_ADJACENT_DIFFERENCE_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_ADJACENT_DIFFERENCE_HPP_

#include "../../block/block_adjacent_difference.hpp"
#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"

#include "../../detail/various.hpp"

#include "../../config.hpp"

#include <hip/hip_runtime.h>

#include <type_traits>

#include <cstdint>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template <typename T, unsigned int BlockSize>
struct adjacent_diff_helper
{
    using adjacent_diff_type = ::rocprim::block_adjacent_difference<T, BlockSize>;
    using storage_type       = typename adjacent_diff_type::storage_type;

    template <unsigned int ItemsPerThread,
              typename Output,
              typename BinaryFunction,
              typename InputIt,
              bool InPlace>
    ROCPRIM_DEVICE void dispatch(const T (&input)[ItemsPerThread],
                                 Output (&output)[ItemsPerThread],
                                 const BinaryFunction op,
                                 const InputIt        previous_values,
                                 const unsigned int   block_id,
                                 const std::size_t    starting_block,
                                 const std::size_t    num_blocks,
                                 const std::size_t    size,
                                 storage_type&        storage,
                                 bool_constant<InPlace> /*in_place*/,
                                 std::false_type /*right*/)
    {
        static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

        // Not the first block, i.e. has a predecessor
        if(starting_block + block_id != 0)
        {
            // `previous_values` needs to be accessed with a stride of `items_per_block` if the
            // operation is out-of-place
            const unsigned int block_offset = InPlace ? block_id : block_id * items_per_block;
            const InputIt      block_previous_values = previous_values + block_offset;

            const T tile_predecessor = block_previous_values[-1];
            // Not the last (i.e. full block)
            if(starting_block + block_id != num_blocks - 1)
            {
                adjacent_diff_type {}.subtract_left(input, output, op, tile_predecessor, storage);
            }
            else
            {
                const unsigned int valid_items
                    = static_cast<unsigned int>(size - (num_blocks - 1) * items_per_block);
                adjacent_diff_type {}.subtract_left_partial(
                    input, output, op, tile_predecessor, valid_items, storage);
            }
        }
        else
        {
            // Not the last (i.e. full block)
            if(starting_block + block_id != num_blocks - 1)
            {
                adjacent_diff_type {}.subtract_left(input, output, op, storage);
            }
            else
            {
                const unsigned int valid_items
                    = static_cast<unsigned int>(size - (num_blocks - 1) * items_per_block);
                adjacent_diff_type {}.subtract_left_partial(
                    input, output, op, valid_items, storage);
            }
        }
    }

    template <unsigned int ItemsPerThread,
              typename Output,
              typename BinaryFunction,
              typename InputIt,
              bool InPlace>
    ROCPRIM_DEVICE void dispatch(const T (&input)[ItemsPerThread],
                                 Output (&output)[ItemsPerThread],
                                 const BinaryFunction op,
                                 const InputIt        previous_values,
                                 const unsigned int   block_id,
                                 const std::size_t    starting_block,
                                 const std::size_t    num_blocks,
                                 const std::size_t    size,
                                 storage_type&        storage,
                                 bool_constant<InPlace> /*in_place*/,
                                 std::true_type /*right*/)
    {
        static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

        // Not the last (i.e. full) block and has a successor
        if(starting_block + block_id != num_blocks - 1)
        {
            // `previous_values` needs to be accessed with a stride of `items_per_block` if the
            // operation is out-of-place
            // When in-place, the first block does not save its value (since it won't be used)
            // so the block values are shifted right one. This means that next block's first value
            // is in the position `block_id`
            const unsigned int block_offset = InPlace ? block_id : (block_id + 1) * items_per_block;

            const InputIt next_block_values = previous_values + block_offset;
            const T       tile_successor    = *next_block_values;

            adjacent_diff_type {}.subtract_right(input, output, op, tile_successor, storage);
        }
        else
        {
            const unsigned int valid_items
                = static_cast<unsigned int>(size - (num_blocks - 1) * items_per_block);
            adjacent_diff_type {}.subtract_right_partial(input, output, op, valid_items, storage);
        }
    }
};

template <typename T, typename InputIterator>
ROCPRIM_DEVICE ROCPRIM_INLINE auto select_previous_values_iterator(T* previous_values,
                                                                   InputIterator /*input*/,
                                                                   std::true_type /*in_place*/)
{
    return previous_values;
}

template <typename T, typename InputIterator>
ROCPRIM_DEVICE ROCPRIM_INLINE auto select_previous_values_iterator(T* /*previous_values*/,
                                                                   InputIterator input,
                                                                   std::false_type /*in_place*/)
{
    return input;
}

template <typename Config,
          bool InPlace,
          bool Right,
          typename InputIt,
          typename OutputIt,
          typename BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void adjacent_difference_kernel_impl(
    const InputIt                                             input,
    const OutputIt                                            output,
    const std::size_t                                         size,
    const BinaryFunction                                      op,
    const typename std::iterator_traits<InputIt>::value_type* previous_values,
    const std::size_t                                         starting_block)
{
    using input_type  = typename std::iterator_traits<InputIt>::value_type;
    using output_type = typename std::iterator_traits<OutputIt>::value_type;

    static constexpr unsigned int block_size       = Config::block_size;
    static constexpr unsigned int items_per_thread = Config::items_per_thread;
    static constexpr unsigned int items_per_block  = block_size * items_per_thread;

    using block_load_type
        = ::rocprim::block_load<input_type, block_size, items_per_thread, Config::load_method>;
    using block_store_type
        = ::rocprim::block_store<output_type, block_size, items_per_thread, Config::store_method>;

    using adjacent_helper = adjacent_diff_helper<input_type, block_size>;

    ROCPRIM_SHARED_MEMORY struct
    {
        typename block_load_type::storage_type  load;
        typename adjacent_helper::storage_type  adjacent_diff;
        typename block_store_type::storage_type store;
    } storage;

    const unsigned int block_id     = blockIdx.x;
    const unsigned int block_offset = block_id * items_per_block;

    const std::size_t num_blocks = ceiling_div(size, items_per_block);

    input_type thread_input[items_per_thread];
    if(starting_block + block_id < num_blocks - 1)
    {
        block_load_type {}.load(input + block_offset, thread_input, storage.load);
    }
    else
    {
        const unsigned int valid_items
            = static_cast<unsigned int>(size - (num_blocks - 1) * items_per_block);
        block_load_type {}.load(input + block_offset, thread_input, valid_items, storage.load);
    }
    ::rocprim::syncthreads();

    // Type tags for tag dispatch.
    static constexpr auto in_place = bool_constant<InPlace> {};
    static constexpr auto right    = bool_constant<Right> {};

    // When doing the operation in-place the last/first items of each block have been copied out
    // in advance and written to the contiguos locations, since accessing them would be a data race
    // with the writing of their new values. In this case `select_previous_values_iterator` returns
    // a pointer to the copied values, and it should be addressed by block_id.
    // Otherwise (when the transform is out-of-place) it just returns the input iterator, and the
    // first/last values of the blocks can be accessed with a stride of `items_per_block`
    const auto previous_values_it
        = select_previous_values_iterator(previous_values, input, in_place);

    output_type thread_output[items_per_thread];
    // Do tag dispatch on `right` to select either `subtract_right` or `subtract_left`.
    // Note that the function is overloaded on its last parameter.
    adjacent_helper {}.dispatch(thread_input,
                                thread_output,
                                op,
                                previous_values_it,
                                block_id,
                                starting_block,
                                num_blocks,
                                size,
                                storage.adjacent_diff,
                                in_place,
                                right);
    ::rocprim::syncthreads();

    if(starting_block + block_id < num_blocks - 1)
    {
        block_store_type {}.store(output + block_offset, thread_output, storage.store);
    }
    else
    {
        const unsigned int valid_items
            = static_cast<unsigned int>(size - (num_blocks - 1) * items_per_block);
        block_store_type {}.store(output + block_offset, thread_output, valid_items, storage.store);
    }
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_ADJACENT_DIFFERENCE_HPP_