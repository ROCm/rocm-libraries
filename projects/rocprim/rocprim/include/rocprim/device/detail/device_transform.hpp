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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_TRANSFORM_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_TRANSFORM_HPP_

#include <iterator>
#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Function, class... Ts>
struct unpack_nary_op
{
    using result_type = typename ::rocprim::invoke_result<Function, Ts...>::type;

    ROCPRIM_HOST_DEVICE inline unpack_nary_op() = default;

    ROCPRIM_HOST_DEVICE inline unpack_nary_op(Function op) : op_(op) {}

    ROCPRIM_HOST_DEVICE inline ~unpack_nary_op() = default;

    ROCPRIM_HOST_DEVICE
    inline result_type
        operator()(const ::rocprim::tuple<Ts...>& t) const
    {
        return apply_impl(t, std::index_sequence_for<Ts...>{});
    }

private:
    Function op_;

    template<std::size_t... Is>
    ROCPRIM_HOST_DEVICE
    inline result_type apply_impl(const ::rocprim::tuple<Ts...>& t,
                                  std::index_sequence<Is...>) const
    {
        return op_(::rocprim::get<Is>(t)...);
    }
};

// Wrapper for unpacking tuple to be used with BinaryFunction.
// See transform function which accepts two input iterators.
template<class T1, class T2, class BinaryFunction>
using unpack_binary_op = unpack_nary_op<BinaryFunction, T1, T2>;

template<typename T, unsigned int ItemsPerThread>
using dynamic_size_type = std::conditional_t<
    (sizeof(T) * ItemsPerThread <= 1),
    uint8_t,
    std::conditional_t<
        (sizeof(T) * ItemsPerThread <= 2),
        uint16_t,
        std::conditional_t<
            (sizeof(T) * ItemsPerThread <= 4),
            uint32_t,
            std::conditional_t<(sizeof(T) * ItemsPerThread <= 8), uint64_t, uint128_t>>>>;

template<bool                VectorLoadStore,
         unsigned int        BlockSize,
         unsigned int        ItemsPerThread,
         cache_load_modifier LoadType,
         class ResultType,
         class InputIterator,
         class OutputIterator,
         class UnaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto transform_kernel_impl(InputIterator  input,
                           const size_t   input_size,
                           OutputIterator output,
                           UnaryFunction  transform_op) ->
    typename std::enable_if<VectorLoadStore, void>::type
{
    using input_type  = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using result_type =
        typename std::conditional<std::is_void<output_type>::value, ResultType, output_type>::type;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id             = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id       = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset        = flat_block_id * items_per_block;
    const unsigned int number_of_blocks    = ::rocprim::detail::grid_size<0>();
    const unsigned int valid_in_last_block = input_size - block_offset;

    input_type  input_values[ItemsPerThread];
    result_type output_values[ItemsPerThread];

    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_direct_striped<BlockSize>(flat_id,
                                             input + block_offset,
                                             input_values,
                                             valid_in_last_block);

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(BlockSize * i + flat_id < valid_in_last_block)
            {
                output_values[i] = transform_op(input_values[i]);
            }
        }

        block_store_direct_striped<BlockSize>(flat_id,
                                              output + block_offset,
                                              output_values,
                                              valid_in_last_block);
    }
    else
    {
        using vec_input_type = dynamic_size_type<input_type, ItemsPerThread>;
        block_load_direct_blocked_cast<vec_input_type, LoadType>(flat_id,
                                                                 input + block_offset,
                                                                 input_values);

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output_values[i] = transform_op(input_values[i]);
        }

        using vec_output_type = dynamic_size_type<output_type, ItemsPerThread>;
        block_store_direct_blocked_cast<vec_output_type>(flat_id,
                                                         output + block_offset,
                                                         output_values);
    }
}

template<bool                VectorLoadStore,
         unsigned int        BlockSize,
         unsigned int        ItemsPerThread,
         cache_load_modifier LoadType,
         class ResultType,
         class InputIterator,
         class OutputIterator,
         class UnaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto transform_kernel_impl(InputIterator  input,
                           const size_t   input_size,
                           OutputIterator output,
                           UnaryFunction  transform_op) ->
    typename std::enable_if<!VectorLoadStore, void>::type
{
    using input_type  = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using result_type =
        typename std::conditional<std::is_void<output_type>::value, ResultType, output_type>::type;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id             = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id       = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset        = flat_block_id * items_per_block;
    const unsigned int number_of_blocks    = ::rocprim::detail::grid_size<0>();
    const unsigned int valid_in_last_block = input_size - block_offset;

    input_type  input_values[ItemsPerThread];
    result_type output_values[ItemsPerThread];

    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_direct_striped<BlockSize>(flat_id,
                                             input + block_offset,
                                             input_values,
                                             valid_in_last_block);

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(BlockSize * i + flat_id < valid_in_last_block)
            {
                output_values[i] = transform_op(input_values[i]);
            }
        }

        block_store_direct_striped<BlockSize>(flat_id,
                                              output + block_offset,
                                              output_values,
                                              valid_in_last_block);
    }
    else
    {
        block_load_direct_striped<BlockSize>(flat_id, input + block_offset, input_values);

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output_values[i] = transform_op(input_values[i]);
        }

        block_store_direct_striped<BlockSize>(flat_id, output + block_offset, output_values);
    }
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_TRANSFORM_HPP_
