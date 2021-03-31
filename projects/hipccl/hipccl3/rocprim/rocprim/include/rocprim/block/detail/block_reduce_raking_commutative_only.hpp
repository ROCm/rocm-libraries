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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

#include "../../warp/warp_reduce.hpp"
#include "../../thread/thread_reduce.hpp"

#include "block_raking_layout.hpp"
BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_reduce_raking_communtative_only
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

    // Select warp size
    static constexpr unsigned int warp_size_ =
        detail::warp_size_in_class(::rocprim::device_warp_size());
    static constexpr unsigned int raking_threads_ = warp_size_;
    static constexpr bool use_fall_back_ = ((BlockSize % warp_size_ != 0) || (BlockSize <= warp_size_));

    typedef block_reduce_raking_reduce<T,BlockSizeX,BlockSizeY,BlockSizeZ> fall_back;
    // Number of items to reduce per thread

    static constexpr unsigned int sharing_threads_ = ::rocprim::max<int>(1, BlockSize - raking_threads_);
    static constexpr unsigned int segment_length_ = sharing_threads_ / warp_size_;

    // BlockSize is multiple of hardware warp
    typedef warp_reduce<T, raking_threads_> WarpReduce;
    typedef block_raking_layout<T, sharing_threads_> BlockRakingLayout;

    union storage_type_
    {
      struct DefaultStorage
      {
          typename WarpReduce::storage_type        warp_storage;        ///< Storage for warp-synchronous reduction
          typename BlockRakingLayout::storage_type raking_grid;         ///< Padded thread block raking grid
      } default_storage;

      typename fall_back::storage_type              fallback_storage;    ///< Fall-back storage for non-commutative block scan

    };

public:

    using storage_type = detail::raw_storage<storage_type_>;

    template<typename ReductionOp>
    ROCPRIM_DEVICE inline
    T reduce(
        T             partial,
        T&            output,
        int           num_valid,
        ReductionOp   reduction_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        return reduce(partial, output, num_valid, storage, reduction_op);
    }

    template<typename ReductionOp>
    ROCPRIM_DEVICE inline
    T reduce(
        T                   partial,
        T&                  output,
        ReductionOp         reduction_op)
    {
        return this->reduce(partial, output, BlockSize, reduction_op);
    }

    template<typename ReductionOp>
    ROCPRIM_DEVICE inline
    T reduce(
        T             partial,
        T&            output,
        unsigned int  num_valid,
        storage_type& storage,
        ReductionOp   reduction_op)
    {
        return this->reduce(partial, output, num_valid, storage, reduction_op);
    }

    template<typename ReductionOp>
    ROCPRIM_DEVICE inline
    T reduce(
        T             partial,
        T&            output,
        storage_type& storage,
        ReductionOp   reduction_op)
    {
        return this->reduce(partial, output, BlockSize, storage, reduction_op);
    }


    template<typename ReductionOp>
    ROCPRIM_DEVICE inline
    T reduce(
        T             partial,
        T&            output,
        int           num_valid,
        storage_type& storage,
        ReductionOp   reduction_op)
    {
        const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        bool FullTile = (num_valid >= (int)BlockSize);

        if (use_fall_back_ || !FullTile)
        {
            T output_value;
            fall_back().reduce(partial, output_value, num_valid, reduction_op);
            output =  output_value;
            return output;
        }
        else
        {
            storage_type_& storage_ = storage.get();

            // Place partial into shared memory grid
            if (linear_tid >= raking_threads_)
                *BlockRakingLayout::placement_ptr(storage_.default_storage.raking_grid, linear_tid - raking_threads_) = partial;

            ::rocprim::syncthreads();

            // Reduce parallelism to one warp
            if (linear_tid < raking_threads_)
            {
                // Raking reduction in grid
                T *raking_segment = BlockRakingLayout::raking_ptr(storage_.default_storage.raking_grid, linear_tid);
                partial = ::rocprim::thread_reduce<segment_length_>(
                    raking_segment,
                    reduction_op,
                    partial
                );

                // Warpscan
                T output_value;
                WarpReduce().reduce(partial, output_value, storage_.default_storage.warp_storage, reduction_op);
                partial = output_value;
            }
        }
        output =  partial;
        return output;
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY_HPP_
