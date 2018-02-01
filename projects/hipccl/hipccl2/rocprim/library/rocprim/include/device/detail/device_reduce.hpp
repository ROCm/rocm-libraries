// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_HPP_

#include <type_traits>
#include <iterator>

// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>

#include "../../detail/config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_reduce.hpp"


BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
    
template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class T
>
void block_reduce_kernel_impl(InputIterator input,
                              const size_t input_size,
                              OutputIterator output,
                              BinaryFunction reduce_op,
                              T init_value) [[hc]]
{
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    const unsigned int flat_id = ::rocprim::block_thread_id(0);
    const unsigned int flat_block_id = ::rocprim::block_id(0);
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = flat_block_id * BlockSize * ItemsPerThread;
    const unsigned int number_of_blocks = (input_size + items_per_block - 1)/items_per_block;
    auto valid_in_last_block = input_size - items_per_block * (number_of_blocks - 1);
    
    using block_reduce_type = ::rocprim::block_reduce<
        output_type, BlockSize,
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;

    output_type values[ItemsPerThread];
    output_type output_value;
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_direct_striped<BlockSize>(flat_id,
                                             input + block_offset,
                                             values,
                                             valid_in_last_block,
                                             init_value);
    }
    else
    {
        block_load_direct_striped<BlockSize>(flat_id,
                                             input + block_offset,
                                             values);
    }

    // load input values into values
    block_reduce_type()
        . reduce(
            values, // input
            output_value, // output
            reduce_op
        );

    // Save value into output
    if(flat_id == 0)
    {
        output[flat_block_id] = output_value;
    }
}

template<
    unsigned int BlockSize,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction
>
void final_reduce_kernel_impl(InputIterator input,
                              const size_t input_size,
                              OutputIterator output,
                              BinaryFunction reduce_op) [[hc]]
{
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using block_reduce_type = ::rocprim::block_reduce<
        output_type, BlockSize,
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
    
    const unsigned int flat_id = ::rocprim::block_thread_id(0);
    const unsigned int flat_block_id = ::rocprim::block_id(0);
    const unsigned int global_id = flat_block_id * ::rocprim::flat_tile_size() + flat_id;
    const unsigned int step = ::rocprim::flat_tile_size() * ::rocprim::grid_id(0);
    unsigned int valid = input_size;

    output_type output_value = input[global_id];
    
    for(size_t i = global_id + step; i < input_size; i += step)
    {
        output_value = reduce_op(output_value, input[i]);
    }
    
    block_reduce_type()
        .reduce(
            output_value, // input
            output_value, // output
            valid,
            reduce_op
        );
    
    if(flat_id == 0)
    {
        output[flat_block_id] = output_value;
    }
}

// Returns size of temporary storage in bytes.
template<class T>
size_t get_temporary_storage_bytes(size_t input_size,
                                   size_t items_per_block)
{
    if(input_size <= items_per_block)
    {
        return 0;
    }
    auto size = (input_size + items_per_block - 1)/(items_per_block);
    return size * sizeof(T) + get_temporary_storage_bytes<T>(size, items_per_block);
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_HPP_
