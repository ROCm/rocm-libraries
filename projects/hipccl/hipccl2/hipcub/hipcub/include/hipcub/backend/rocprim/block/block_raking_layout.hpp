/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * \file
 * cub::BlockRakingLayout provides a conflict-free shared memory layout abstraction for warp-raking across thread block data.
 */


#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_RAKING_LAYOUT_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_RAKING_LAYOUT_HPP_

#include <type_traits>


BEGIN_HIPCUB_NAMESPACE

/**
 * \brief BlockRakingLayout provides a conflict-free shared memory layout abstraction for 1D raking across thread block data.    ![](raking.png)
 * \ingroup BlockModule
 *
 * \par Overview
 * This type facilitates a shared memory usage pattern where a block of CUDA
 * threads places elements into shared memory and then reduces the active
 * parallelism to one "raking" warp of threads for serially aggregating consecutive
 * sequences of shared items.  Padding is inserted to eliminate bank conflicts
 * (for most data types).
 *
 * \tparam T                        The data type to be exchanged.
 * \tparam BLOCK_THREADS            The thread block size in threads.
 * \tparam PTX_ARCH                 <b>[optional]</b> \ptxversion
 */
template <
    typename    T,
    int         BLOCK_THREADS,
    int ARCH = HIPCUB_ARCH /* ignored */
>
struct BlockRakingLayout
: private ::rocprim::block_raking_layout
{

  using base_type =
      typename ::rocprim::block_raking_layout<
          T,
          BLOCK_THREADS
      >;


    using TempStorage = typename base_type::storage_type;

    /**
     * \brief Returns the location for the calling thread to place data into the grid
     */
    static HIPCUB_DEVICE inline T* PlacementPtr(
        TempStorage &temp_storage,
        unsigned int linear_tid)
    {
        return base_type::placement_ptr(temp_storage, linear_tid);
    }


    /**
     * \brief Returns the location for the calling thread to begin sequential raking
     */
    static HIPCUB_DEVICE inline T* RakingPtr(
        TempStorage &temp_storage,
        unsigned int linear_tid)
    {
        return base_type::raking_ptr(temp_storage, linear_tid);
    }
};


END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_RAKING_LAYOUT_HPP_
