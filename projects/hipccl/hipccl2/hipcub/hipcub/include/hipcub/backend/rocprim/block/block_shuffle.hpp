/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_

#include <type_traits>

#include "../../../config.hpp"

#include "../thread/thread_operators.hpp"

#include <rocprim/block/block_shuffle.hpp>

BEGIN_HIPCUB_NAMESPACE



template <
    typename            T,
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y         = 1,
    int                 BLOCK_DIM_Z         = 1,
    int                 ARCH            = HIPCUB_ARCH>
class BlockShuffle : public ::rocprim::block_shuffle<
                    T,
                    BLOCK_DIM_X,
                    BLOCK_DIM_Y,
                    BLOCK_DIM_Z>
{
  static_assert(
      BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
      "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
  );

  using base_type =
      typename ::rocprim::block_shuffle<
          T,
          BLOCK_DIM_X,
          BLOCK_DIM_Y,
          BLOCK_DIM_Z
      >;

  // Reference to temporary storage (usually shared memory)
  typename base_type::storage_type& temp_storage_;

public:
  using TempStorage = typename base_type::storage_type;

  HIPCUB_DEVICE inline
  BlockShuffle()  :      temp_storage_(private_storage())
  {}


  HIPCUB_DEVICE inline
  BlockShuffle(TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
  :      temp_storage_(temp_storage)
  {}

  /**
   * \brief Each <em>thread<sub>i</sub></em> obtains the \p input provided by <em>thread</em><sub><em>i</em>+<tt>distance</tt></sub>. The offset \p distance may be negative.
   */
  HIPCUB_DEVICE inline void Offset(
      T   input,                  ///< [in] The input item from the calling thread (<em>thread<sub>i</sub></em>)
      T&  output,                 ///< [out] The \p input item from the successor (or predecessor) thread <em>thread</em><sub><em>i</em>+<tt>distance</tt></sub> (may be aliased to \p input).  This value is only updated for for <em>thread<sub>i</sub></em> when 0 <= (<em>i</em> + \p distance) < <tt>BLOCK_THREADS-1</tt>
      int distance = 1)           ///< [in] Offset distance (may be negative)
  {
    base_type::offset(input,output,distance);
  }

  /**
 * \brief Each <em>thread<sub>i</sub></em> obtains the \p input provided by <em>thread</em><sub><em>i</em>+<tt>distance</tt></sub>.
 */
  HIPCUB_DEVICE inline void Rotate(
      T   input,                  ///< [in] The calling thread's input item
      T&  output,                 ///< [out] The \p input item from thread <em>thread</em><sub>(<em>i</em>+<tt>distance></tt>)%<tt>\<BLOCK_THREADS\></tt></sub> (may be aliased to \p input).  This value is not updated for <em>thread</em><sub>BLOCK_THREADS-1</sub>
      unsigned int distance = 1)  ///< [in] Offset distance (0 < \p distance < <tt>BLOCK_THREADS</tt>)
  {
    base_type::rotate(input,output,distance);
  }
  /**
  * \brief The thread block rotates its [<em>blocked arrangement</em>](index.html#sec5sec3) of \p input items, shifting it up by one item
  */
  template <int ITEMS_PER_THREAD>
  HIPCUB_DEVICE inline void Up(
    T (&input)[ITEMS_PER_THREAD],   ///< [in] The calling thread's input items
    T (&prev)[ITEMS_PER_THREAD])    ///< [out] The corresponding predecessor items (may be aliased to \p input).  The item \p prev[0] is not updated for <em>thread</em><sub>0</sub>.
  {
    base_type::up(input,prev);
  }


   /**
   * \brief The thread block rotates its [<em>blocked arrangement</em>](index.html#sec5sec3) of \p input items, shifting it up by one item.  All threads receive the \p input provided by <em>thread</em><sub><tt>BLOCK_THREADS-1</tt></sub>.
   */
  template <int ITEMS_PER_THREAD>
  HIPCUB_DEVICE inline void Up(
      T (&input)[ITEMS_PER_THREAD],   ///< [in] The calling thread's input items
      T (&prev)[ITEMS_PER_THREAD],    ///< [out] The corresponding predecessor items (may be aliased to \p input).  The item \p prev[0] is not updated for <em>thread</em><sub>0</sub>.
      T &block_suffix)                ///< [out] The item \p input[ITEMS_PER_THREAD-1] from <em>thread</em><sub><tt>BLOCK_THREADS-1</tt></sub>, provided to all threads
  {
    base_type::up(input,prev,block_suffix);
  }

   /**
   * \brief The thread block rotates its [<em>blocked arrangement</em>](index.html#sec5sec3) of \p input items, shifting it down by one item
   */
  template <int ITEMS_PER_THREAD>
  HIPCUB_DEVICE inline void Down(
      T (&input)[ITEMS_PER_THREAD],   ///< [in] The calling thread's input items
      T (&next)[ITEMS_PER_THREAD])    ///< [out] The corresponding predecessor items (may be aliased to \p input).  The value \p next[0] is not updated for <em>thread</em><sub>BLOCK_THREADS-1</sub>.
  {
    base_type::down(input,next);
  }

   /**
   * \brief The thread block rotates its [<em>blocked arrangement</em>](index.html#sec5sec3) of input items, shifting it down by one item.  All threads receive \p input[0] provided by <em>thread</em><sub><tt>0</tt></sub>.
   */
  template <int ITEMS_PER_THREAD>
  HIPCUB_DEVICE inline void Down(
      T (&input)[ITEMS_PER_THREAD],   ///< [in] The calling thread's input items
      T (&next)[ITEMS_PER_THREAD],    ///< [out] The corresponding predecessor items (may be aliased to \p input).  The value \p next[0] is not updated for <em>thread</em><sub>BLOCK_THREADS-1</sub>.
      T &block_prefix)                ///< [out] The item \p input[0] from <em>thread</em><sub><tt>0</tt></sub>, provided to all threads
  {
    base_type::down(input,next,block_prefix);
  }

private:
    HIPCUB_DEVICE inline
    TempStorage& private_storage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_
