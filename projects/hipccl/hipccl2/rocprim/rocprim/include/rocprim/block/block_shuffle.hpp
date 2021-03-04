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

#ifndef ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_
#define ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_reduce_warp_reduce.hpp"
#include "detail/block_reduce_raking_reduce.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1>
class block_shuffle
{
  static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
  static constexpr unsigned int warp_size = detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
  // Number of warps in block
  static constexpr unsigned int warps_no = (BlockSize + warp_size - 1) / warp_size;

  struct storage_type_
  {
      T prev[BlockSize];
      T next[BlockSize];
  };

public:

  #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
  using storage_type = detail::raw_storage<storage_type_>;
  #else
  using storage_type = storage_type_; // only for Doxygen
  #endif

private:
  ROCPRIM_SHARED_MEMORY storage_type_ storage;

public:
  /**
   * \brief Each <em>thread<sub>i</sub></em> obtains the \p input provided by <em>thread</em><sub><em>i</em>+<tt>distance</tt></sub>. The offset \p distance may be negative.
   *
   * \par
   * - \smemreuse
   */
  ROCPRIM_DEVICE inline
  void offset(
    T input,
    T& output,
    int distance =1
  )
  {
      const size_t flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
      storage.prev[flat_id] = input;
      ::rocprim::syncthreads();

      const int offset_tid = static_cast<int>(flat_id) + distance;
      if ((offset_tid >= 0) && (offset_tid < (int)BlockSize))
      {
          output = storage.prev[static_cast<size_t>(offset_tid)];
      }
  }

  /**
   * \brief Each <em>thread<sub>i</sub></em> obtains the \p input provided by <em>thread</em><sub><em>i</em>+<tt>distance</tt></sub>.
   *
   * \par
   * - \smemreuse
   */
  ROCPRIM_DEVICE inline
  void rotate(
      T   input,                  ///< [in] The calling thread's input item
      T&  output,                 ///< [out] The \p input item from thread <em>thread</em><sub>(<em>i</em>+<tt>distance></tt>)%<tt><BlockSize></tt></sub> (may be aliased to \p input).  This value is not updated for <em>thread</em><sub>BlockSize-1</sub>
      unsigned int distance = 1)  ///< [in] Offset distance (0 < \p distance < <tt>BlockSize</tt>)
  {
      const size_t flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
      storage.prev[flat_id] = input;
      ::rocprim::syncthreads();

      unsigned int offset = threadIdx.x + distance;
      if (offset >= BlockSize)
          offset -= BlockSize;

      output = storage.prev[offset];
  }


  /**
     * \brief The thread block rotates its [<em>blocked arrangement</em>](index.html#sec5sec3) of \p input items, shifting it up by one item
     *
     * \par
     * - \blocked
     * - \granularity
     * - \smemreuse
     */
    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void up(
        T (&input)[ItemsPerThread],   ///< [in] The calling thread's input items
        T (&prev)[ItemsPerThread])    ///< [out] The corresponding predecessor items (may be aliased to \p input).  The item \p prev[0] is not updated for <em>thread</em><sub>0</sub>.
    {
      const size_t flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
      storage.prev[flat_id] = input[ItemsPerThread -1];
      ::rocprim::syncthreads();
      #pragma unroll
      for (unsigned int i = ItemsPerThread - 1; i > 0; --i)
      {
          prev[i] = input[i - 1];
      }
      if (flat_id > 0)
      {
          prev[0] = storage.prev[flat_id - 1];
      }
    }



    /**
     * \brief The thread block rotates its [<em>blocked arrangement</em>](index.html#sec5sec3) of \p input items, shifting it up by one item.  All threads receive the \p input provided by <em>thread</em><sub><tt>BlockSize-1</tt></sub>.
     *
     * \par
     * - \blocked
     * - \granularity
     * - \smemreuse
     */
    template <int ItemsPerThread>
    __device__ __forceinline__ void up(
        T (&input)[ItemsPerThread],   ///< [in] The calling thread's input items
        T (&prev)[ItemsPerThread],    ///< [out] The corresponding predecessor items (may be aliased to \p input).  The item \p prev[0] is not updated for <em>thread</em><sub>0</sub>.
        T &block_suffix)                ///< [out] The item \p input[ItemsPerThread-1] from <em>thread</em><sub><tt>BlockSize-1</tt></sub>, provided to all threads
    {
        up(input, prev);
        block_suffix = storage.prev[BlockSize - 1];
    }

    /**
     * \brief The thread block rotates its [<em>blocked arrangement</em>](index.html#sec5sec3) of \p input items, shifting it down by one item
     *
     * \par
     * - \blocked
     * - \granularity
     * - \smemreuse
     */
    template <unsigned int ItemsPerThread>
    __device__ __forceinline__ void down(
        T (&input)[ItemsPerThread],   ///< [in] The calling thread's input items
        T (&next)[ItemsPerThread])    ///< [out] The corresponding predecessor items (may be aliased to \p input).  The value \p next[0] is not updated for <em>thread</em><sub>BlockSize-1</sub>.
    {
      const size_t flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
      storage.next[flat_id] = input[0];
      ::rocprim::syncthreads();

      #pragma unroll
      for (int i = 0; i < (ItemsPerThread - 1); ++i)
      {
        next[i] = input[i + 1];
      }

      if (flat_id <(BlockSize -1))
      {
        next[0] = storage.next[flat_id + 1];
      }
    }


    /**
     * \brief The thread block rotates its [<em>blocked arrangement</em>](index.html#sec5sec3) of input items, shifting it down by one item.  All threads receive \p input[0] provided by <em>thread</em><sub><tt>0</tt></sub>.
     *
     * \par
     * - \blocked
     * - \granularity
     * - \smemreuse
     */
    template <unsigned int ItemsPerThread>
    __device__ __forceinline__ void Down(
        T (&input)[ItemsPerThread],   ///< [in] The calling thread's input items
        T (&next)[ItemsPerThread],    ///< [out] The corresponding predecessor items (may be aliased to \p input).  The value \p next[0] is not updated for <em>thread</em><sub>BlockSize-1</sub>.
        T &block_suffix)                ///< [out] The item \p input[0] from <em>thread</em><sub><tt>0</tt></sub>, provided to all threads
    {
        Up(input, next);
        block_suffix = storage.next[0];
    }


};


END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_
