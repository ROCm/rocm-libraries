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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_WARP_REDUCE_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_WARP_REDUCE_HPP_

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
    unsigned int BlockSize
>
class block_reduce_warp_reduce
{
    // Select warp size
    static constexpr unsigned int warp_size_ =
        detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
    // Number of warps in block
    static constexpr unsigned int warps_no_ = (BlockSize + warp_size_ - 1) / warp_size_;

    // Check if we have to pass number of valid items into warp reduction primitive
    static constexpr bool block_size_is_warp_multiple_ = ((BlockSize % warp_size_) == 0);
    static constexpr bool warps_no_is_pow_of_two_ = detail::is_power_of_two(warps_no_);

    // typedef of warp_reduce primitive that will be used to perform warp-level
    // reduce operation on input values.
    // warp_reduce_crosslane is an implementation of warp_reduce that does not need storage,
    // but requires logical warp size to be a power of two.
    using warp_reduce_input_type = ::rocprim::detail::warp_reduce_crosslane<T, warp_size_, false>;
    // typedef of warp_reduce primitive that will be used to perform reduction
    // of results of warp-level reduction.
    using warp_reduce_output_type = ::rocprim::detail::warp_reduce_crosslane<
        T, detail::next_power_of_two(warps_no_), false
    >;

    struct storage_type_
    {
        T warp_partials[warps_no_];
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
            ::rocprim::flat_block_thread_id(),
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
        const auto flat_tid = ::rocprim::flat_block_thread_id();
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
            ::rocprim::flat_block_thread_id(),
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
        const auto warp_id = ::rocprim::warp_id();
        const auto lane_id = ::rocprim::lane_id();
        const unsigned int warp_offset = warp_id * warp_size_;
        const unsigned int num_valid =
            (warp_offset < BlockSize) ? BlockSize - warp_offset : 0;
        storage_type_& storage_ = storage.get();

        // Perform warp reduce
        warp_reduce<!block_size_is_warp_multiple_, warp_reduce_input_type>(
            input, output, num_valid, reduce_op
        );

        // i-th warp will have its partial stored in storage_.warp_partials[i-1]
        if(lane_id == 0)
        {
            storage_.warp_partials[warp_id] = output;
        }
        ::rocprim::syncthreads();

        if(flat_tid < warps_no_)
        {
            // Use warp partial to calculate the final reduce results for every thread
            auto warp_partial = storage_.warp_partials[lane_id];

            warp_reduce<!warps_no_is_pow_of_two_, warp_reduce_output_type>(
                warp_partial, output, warps_no_, reduce_op
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
        const auto warp_id = ::rocprim::warp_id();
        const auto lane_id = ::rocprim::lane_id();
        const unsigned int warp_offset = warp_id * warp_size_;
        const unsigned int num_valid =
            (warp_offset < valid_items) ? valid_items - warp_offset : 0;
        storage_type_& storage_ = storage.get();

        // Perform warp reduce
        warp_reduce_input_type().reduce(
            input, output, num_valid, reduce_op
        );

        // i-th warp will have its partial stored in storage_.warp_partials[i-1]
        if(lane_id == 0)
        {
            storage_.warp_partials[warp_id] = output;
        }
        ::rocprim::syncthreads();

        if(flat_tid < warps_no_)
        {
            // Use warp partial to calculate the final reduce results for every thread
            auto warp_partial = storage_.warp_partials[lane_id];

            unsigned int valid_warps_no = (valid_items + warp_size_ - 1) / warp_size_;
            warp_reduce_output_type().reduce(
                warp_partial, output, valid_warps_no, reduce_op
            );
        }
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_WARP_REDUCE_HPP_
