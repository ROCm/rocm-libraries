// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_REDUCE_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_REDUCE_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

#include "../../warp/warp_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_reduce_raking_reduce
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
    // Number of items to reduce per thread
    static constexpr unsigned int thread_reduction_size_ =
        (BlockSize + ::rocprim::warp_size() - 1)/ ::rocprim::warp_size();

    // Warp reduce, warp_reduce_crosslane does not require shared memory (storage), but
    // logical warp size must be a power of two.
    static constexpr unsigned int warp_size_ =
        detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
    // BlockSize is multiple of hardware warp
    static constexpr bool block_size_smaller_than_warp_size_ = (BlockSize < warp_size_);
    using warp_reduce_prefix_type = ::rocprim::detail::warp_reduce_crosslane<T, warp_size_, false>;

    struct storage_type_
    {
        T threads[BlockSize];
    };

public:
    using storage_type = detail::raw_storage<storage_type_>;

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T input,
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op)
    {
        this->reduce_impl(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, storage, reduce_op
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T input,
                T& output,
                BinaryFunction reduce_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->reduce(input, output, storage, reduce_op);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op)
    {
        // Reduce thread items
        T thread_input = input[0];
        #pragma unroll
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = reduce_op(thread_input, input[i]);
        }

        // Reduction of reduced values to get partials
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->reduce_impl(
            flat_tid,
            thread_input, output, // input, output
            storage,
            reduce_op
        );
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                BinaryFunction reduce_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->reduce(input, output, storage, reduce_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T input,
                T& output,
                unsigned int valid_items,
                storage_type& storage,
                BinaryFunction reduce_op)
    {
        this->reduce_impl(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, valid_items, storage, reduce_op
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T input,
                T& output,
                unsigned int valid_items,
                BinaryFunction reduce_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->reduce(input, output, valid_items, storage, reduce_op);
    }

private:
    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce_impl(const unsigned int flat_tid,
                     T input,
                     T& output,
                     storage_type& storage,
                     BinaryFunction reduce_op)
    {
        storage_type_& storage_ = storage.get();
        storage_.threads[flat_tid] = input;
        ::rocprim::syncthreads();

        if (flat_tid < warp_size_)
        {
            T thread_reduction = storage_.threads[flat_tid];
            for(unsigned int i = warp_size_ + flat_tid; i < BlockSize; i += warp_size_)
            {
                thread_reduction = reduce_op(
                    thread_reduction, storage_.threads[i]
                );
            }
            warp_reduce<block_size_smaller_than_warp_size_, warp_reduce_prefix_type>(
                thread_reduction, output, BlockSize, reduce_op
            );
        }
    }

    template<bool UseValid, class WarpReduce, class BinaryFunction>
    ROCPRIM_DEVICE inline
    auto warp_reduce(T input,
                     T& output,
                     const unsigned int valid_items,
                     BinaryFunction reduce_op)
        -> typename std::enable_if<UseValid>::type
    {
        WarpReduce().reduce(
            input, output, valid_items, reduce_op
        );
    }

    template<bool UseValid, class WarpReduce, class BinaryFunction>
    ROCPRIM_DEVICE inline
    auto warp_reduce(T input,
                     T& output,
                     const unsigned int valid_items,
                     BinaryFunction reduce_op)
        -> typename std::enable_if<!UseValid>::type
    {
        (void) valid_items;
        WarpReduce().reduce(
            input, output, reduce_op
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce_impl(const unsigned int flat_tid,
                     T input,
                     T& output,
                     const unsigned int valid_items,
                     storage_type& storage,
                     BinaryFunction reduce_op)
    {
        storage_type_& storage_ = storage.get();
        storage_.threads[flat_tid] = input;
        ::rocprim::syncthreads();

        if (flat_tid < warp_size_)
        {
            T thread_reduction = storage_.threads[flat_tid];
            for(unsigned int i = warp_size_ + flat_tid; i < BlockSize; i += warp_size_)
            {
                if(i < valid_items)
                {
                    thread_reduction = reduce_op(thread_reduction, storage_.threads[i]);
                }
            }
            warp_reduce_prefix_type().reduce(thread_reduction, output, valid_items, reduce_op);
        }
    }
};
} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_REDUCE_HPP_
