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

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Helper functions for reducing final value with
// initial value.
template<
    bool WithInitialValue,
    class T,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
auto reduce_with_initial(T output,
                         T initial_value,
                         BinaryFunction reduce_op)
    -> typename std::enable_if<WithInitialValue, T>::type
{
    return reduce_op(initial_value, output);
}

template<
    bool WithInitialValue,
    class T,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
auto reduce_with_initial(T output,
                         T initial_value,
                         BinaryFunction reduce_op)
    -> typename std::enable_if<!WithInitialValue, T>::type
{
    (void) initial_value;
    (void) reduce_op;
    return output;
}

template<
    bool WithInitialValue,
    class Config,
    class ResultType,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
void block_reduce_kernel_impl(InputIterator input,
                              const size_t input_size,
                              OutputIterator output,
                              InitValueType initial_value,
                              BinaryFunction reduce_op)
{
    constexpr unsigned int block_size = Config::block_size;
    constexpr unsigned int items_per_thread = Config::items_per_thread;

    using result_type = ResultType;

    using block_reduce_type = ::rocprim::block_reduce<
        result_type, block_size,
        Config::block_reduce_method
    >;
    constexpr unsigned int items_per_block = block_size * items_per_thread;

    const unsigned int flat_id = ::rocprim::flat_block_thread_id();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * items_per_block;
    const unsigned int number_of_blocks = ::rocprim::detail::grid_size<0>();
    auto valid_in_last_block = input_size - items_per_block * (number_of_blocks - 1);

    result_type values[items_per_thread];
    result_type output_value;
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_direct_striped<block_size>(
            flat_id,
            input + block_offset,
            values,
            valid_in_last_block
        );

        output_value = values[0];
        #pragma unroll
        for(unsigned int i = 1; i < items_per_thread; i++)
        {
            unsigned int offset = i * block_size;
            if(flat_id + offset < valid_in_last_block)
            {
                output_value = reduce_op(output_value, values[i]);
            }
        }

        block_reduce_type()
            .reduce(
                output_value, // input
                output_value, // output
                valid_in_last_block,
                reduce_op
            );
    }
    else
    {
        block_load_direct_striped<block_size>(
            flat_id,
            input + block_offset,
            values
        );

        // load input values into values
        block_reduce_type()
            .reduce(
                values, // input
                output_value, // output
                reduce_op
            );
    }

    // Save value into output
    if(flat_id == 0)
    {
        output[flat_block_id] = input_size == 0
            ? static_cast<result_type>(initial_value)
            : reduce_with_initial<WithInitialValue>(
                output_value,
                static_cast<result_type>(initial_value),
                reduce_op
            );
    }
}

// Returns size of temporary storage in bytes.
template<class T>
size_t reduce_get_temporary_storage_bytes(size_t input_size,
                                          size_t items_per_block)
{
    if(input_size <= items_per_block)
    {
        return 0;
    }
    auto size = (input_size + items_per_block - 1)/(items_per_block);
    return size * sizeof(T) + reduce_get_temporary_storage_bytes<T>(size, items_per_block);
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_HPP_
