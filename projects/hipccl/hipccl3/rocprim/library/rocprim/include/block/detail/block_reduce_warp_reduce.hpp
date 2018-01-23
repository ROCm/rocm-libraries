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

// HC API
#include <hcc/hc.hpp>

#include "../../detail/config.hpp"
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

    // typedef of warp_reduce primitive that will be used to perform warp-level
    // reduce operations on input values.
    // warp_reduce_shuffle is an implementation of warp_reduce that does not need storage,
    // but requires logical warp size to be a power of two.
    using warp_reduce_input_type = ::rocprim::detail::warp_reduce_shuffle<T, warp_size_, false>;

public:
    struct storage_type
    {
        T warp_partials[warps_no_];
    };
    
    template<class BinaryFunction>
    void reduce(T input,
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op) [[hc]]
    {
        this->reduce_impl(
            ::rocprim::flat_block_thread_id(),
            input, output, storage, reduce_op
        );
    }
    
    template<class BinaryFunction>
    void reduce(T input,
                T& output,
                BinaryFunction reduce_op) [[hc]]
    {
        tile_static storage_type storage;
        this->reduce(input, output, storage, reduce_op);
    }
    
    template<unsigned int ItemsPerThread, class BinaryFunction>
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op) [[hc]]
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
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                BinaryFunction reduce_op) [[hc]]
    {
        tile_static storage_type storage;
        this->reduce(input, output, storage, reduce_op);
    }

private:
    template<class BinaryFunction>
    void reduce_impl(const unsigned int flat_tid,
                     T input,
                     T& output,
                     storage_type& storage,
                     BinaryFunction reduce_op) [[hc]]
    {
        typename warp_reduce_input_type::storage_type reduce_storage;
        // Perform warp reduce
        warp_reduce_input_type().reduce(
            // not using shared mem, see note in storage_type
            input, output, reduce_storage, reduce_op
        );

        // i-th warp will have its partial stored in storage.warp_partials[i-1]
        const auto warp_id = ::rocprim::warp_id();
        const auto lane_id = ::rocprim::lane_id();
        if(lane_id == 0)
        {
            storage.warp_partials[warp_id] = output;
        }
        ::rocprim::syncthreads();

        // Use warp partial to calculate the final reduce results for every thread
        auto warp_partial = (flat_tid < warps_no_) ? storage.warp_partials[lane_id] : 0;
        
        if(warp_id == 0)
        {
            warp_reduce_input_type().reduce(
                // not using shared mem, see note in storage_type
                warp_partial, output, reduce_storage, reduce_op
            );
        }
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_WARP_REDUCE_HPP_
