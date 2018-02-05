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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_TRANSFORM_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_TRANSFORM_HPP_

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
#include "../../block/block_store.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction
>
void transform_kernel_impl(InputIterator input,
                           const size_t input_size,
                           OutputIterator output,
                           BinaryFunction transform_op) [[hc]]
{
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using block_load_type = ::rocprim::block_load<
        output_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose
    >;
    using block_store_type = ::rocprim::block_store<
        output_type, BlockSize, ItemsPerThread,
        ::rocprim::block_store_method::block_store_transpose
    >;
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    
    __shared__ union
    {
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
    } storage;
    
    const unsigned int flat_block_id = ::rocprim::block_id(0);
    const unsigned int block_offset = flat_block_id * BlockSize * ItemsPerThread;
    const unsigned int number_of_blocks = (input_size + items_per_block - 1)/items_per_block;
    auto valid_in_last_block = input_size - items_per_block * (number_of_blocks - 1);

    output_type values[ItemsPerThread];
    
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
    
    #pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        values[i] = transform_op(values[i]);
    }
    
    // Save values into output array
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_store_type()
            .store(
                output + block_offset,
                values,
                valid_in_last_block,
                storage.store
            );
    }
    else
    {
        block_store_type()
            .store(
                output + block_offset,
                values,
                storage.store
            );
    }
}
    
template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator1,
    class InputIterator2,
    class OutputIterator,
    class BinaryFunction
>
void transform_kernel_impl(InputIterator1 input1,
                           InputIterator2 input2,
                           const size_t input_size,
                           OutputIterator output,
                           BinaryFunction transform_op) [[hc]]
{
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using block_load_type = ::rocprim::block_load<
        output_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose
    >;
    using block_store_type = ::rocprim::block_store<
        output_type, BlockSize, ItemsPerThread,
        ::rocprim::block_store_method::block_store_transpose
    >;
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    
    __shared__ union
    {
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
    } storage;
    
    const unsigned int flat_block_id = ::rocprim::block_id(0);
    const unsigned int block_offset = flat_block_id * BlockSize * ItemsPerThread;
    const unsigned int number_of_blocks = (input_size + items_per_block - 1)/items_per_block;
    auto valid_in_last_block = input_size - items_per_block * (number_of_blocks - 1);

    output_type values1[ItemsPerThread];
    output_type values2[ItemsPerThread];
    
    // load input values into values
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_type()
            .load(
                input1 + block_offset,
                values1,
                valid_in_last_block,
                storage.load
            );
        
        block_load_type()
            .load(
                input2 + block_offset,
                values2,
                valid_in_last_block,
                storage.load
            );
    }
    else
    {
        block_load_type()
            .load(
                input1 + block_offset,
                values1,
                storage.load
            );
        
        block_load_type()
            .load(
                input2 + block_offset,
                values2,
                storage.load
            );
    }
    
    #pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        values2[i] = transform_op(values1[i], values2[i]);
    }
    
    // Save values into output array
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_store_type()
            .store(
                output + block_offset,
                values2,
                valid_in_last_block,
                storage.store
            );
    }
    else
    {
        block_store_type()
            .store(
                output + block_offset,
                values2,
                storage.store
            );
    }
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_TRANSFORM_HPP_
