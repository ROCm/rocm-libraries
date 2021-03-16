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

    typedef block_reduce_raking_reduce<T,BlockSizeX,BlockSizeY,BlockSizeZ> fall_back;
    // Number of items to reduce per thread

    static constexpr unsigned int WarpSize =::rocprim::warp_size();
    static constexpr unsigned int RakingThreads =::rocprim::warp_size();
    static constexpr bool UseFallBack = ((BlockSize % WarpSize != 0) || (BlockSize <= WarpSize));

    static constexpr unsigned int SharingThreads =::rocprim::max<unsigned int>(1u,BlockSize - RakingThreads);
    static constexpr unsigned int SegmentLength = SharingThreads / WarpSize;

    // BlockSize is multiple of hardware warp
    typedef warp_reduce<T,RakingThreads> WarpReduce;
    typedef block_raking_layout<T,SharingThreads> BlockRakingLayout;
    union storage_type_
    {
      struct DefaultStorage
      {
          typename WarpReduce::storage_type        warp_storage;        ///< Storage for warp-synchronous reduction
          typename BlockRakingLayout::storage_type raking_grid;         ///< Padded thread block raking grid
      } default_storage;

      typename fall_back::storage_type              fallback_storage;    ///< Fall-back storage for non-commutative block scan

    };

    storage_type_ *temp_storage;

public:
    ROCPRIM_DEVICE inline
    block_reduce_raking_communtative_only()
    {
      ROCPRIM_SHARED_MEMORY storage_type_ shared_storage;
      temp_storage = &shared_storage;
    }
    using storage_type = detail::raw_storage<storage_type_>;
    /// Computes a thread block-wide reduction using addition (+) as the reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
    ROCPRIM_DEVICE inline
    T sum(
        T                   partial,            ///< [in] Calling thread's input partial reductions
        int                 num_valid)          ///< [in] Number of valid elements (may be less than BlockSize)
    {
      const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
      bool FullTile = (num_valid >= BlockSize);
      if (UseFallBack || !FullTile)
      {
          return fall_back().template Sum<FullTile>(partial, num_valid);
      }
      else
      {
        // Place partial into shared memory grid
        if (linear_tid >= RakingThreads)
            *BlockRakingLayout::placement_ptr(temp_storage->default_storage.raking_grid, linear_tid - RakingThreads) = partial;

        ::rocprim::syncthreads();

        // Reduce parallelism to one warp
        if (linear_tid < RakingThreads)
        {
            // Raking reduction in grid
            T *raking_segment = BlockRakingLayout::raking_ptr(temp_storage->default_storage.raking_grid, linear_tid);
            partial = internal::thread_reduce<SegmentLength>(raking_segment, ::rocprim::plus<T>(), partial);

            // Warpscan
            partial = WarpReduce(temp_storage->default_storage.warp_storage).Sum(partial);
        }
      }

      return partial;
    }


    /// Computes a thread block-wide reduction using the specified reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
    template <
        typename            ReductionOp>
    ROCPRIM_DEVICE inline
    T reduce(
        T                   partial,            ///< [in] Calling thread's input partial reductions
        T&                  output,
        int                 num_valid,          ///< [in] Number of valid elements (may be less than BlockSize)
        ReductionOp         reduction_op)       ///< [in] Binary reduction operator
    {
        const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        bool FullTile = (num_valid >= (int)BlockSize);

        if (UseFallBack || !FullTile)
        {
            T output_value;
            fall_back().reduce(partial, output_value, num_valid, reduction_op);
            output =  output_value;
            return output;
        }
        else
        {

            // Place partial into shared memory grid
            if (linear_tid >= RakingThreads)
                *BlockRakingLayout::placement_ptr(temp_storage->default_storage.raking_grid, linear_tid - RakingThreads) = partial;

            ::rocprim::syncthreads();

            // Reduce parallelism to one warp
            if (linear_tid < RakingThreads)
            {
                // Raking reduction in grid
                T *raking_segment = BlockRakingLayout::raking_ptr(temp_storage->default_storage.raking_grid, linear_tid);
                partial = internal::thread_reduce<SegmentLength>(raking_segment, reduction_op, partial);

                // Warpscan
                T output_value;
                WarpReduce().reduce(partial, output_value, temp_storage->default_storage.warp_storage, reduction_op);
                partial = output_value;
            }
        }

        output =  partial;
        return output;
    }

    /// Computes a thread block-wide reduction using the specified reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
    template <
        typename            ReductionOp>
    ROCPRIM_DEVICE inline
    T reduce(
        T                   partial,            ///< [in] Calling thread's input partial reductions
        T&                  output,
        ReductionOp         reduction_op)       ///< [in] Binary reduction operator
    {
      return this->reduce(partial,output,BlockSize,reduction_op);
    }

    /// Computes a thread block-wide reduction using the specified reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
    template <
        typename            ReductionOp>
    ROCPRIM_DEVICE inline
    T reduce(
        T                   partial,            ///< [in] Calling thread's input partial reductions
        T&                  output,
        unsigned int        num_valid,
        storage_type         ,
        ReductionOp         reduction_op)       ///< [in] Binary reduction operator
    {
      return this->reduce(partial,output,num_valid,reduction_op);
    }

    /// Computes a thread block-wide reduction using the specified reduction operator. The first num_valid threads each contribute one reduction partial.  The return value is only valid for thread<sub>0</sub>.
    template <
        typename            ReductionOp>
    ROCPRIM_DEVICE inline
    T reduce(
        T                   partial,            ///< [in] Calling thread's input partial reductions
        T&                  output,
        storage_type         ,
        ReductionOp         reduction_op)       ///< [in] Binary reduction operator
    {
      return this->reduce(partial,output,BlockSize,reduction_op);
    }

};
} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY_HPP_
