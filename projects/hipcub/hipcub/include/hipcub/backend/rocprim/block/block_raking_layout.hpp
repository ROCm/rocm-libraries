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

#include "../../../config.hpp"
#include "../util_type.hpp"

#include <rocprim/config.hpp> // IWYU pragma: export
#include <rocprim/detail/various.hpp> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

/**
 * \brief BlockRakingLayout provides a conflict-free shared memory layout abstraction for 1D raking across thread block data.
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
 * \tparam PTX_ARCH                 <b>[optional]</b> ptx version
 */
template<typename T, int BLOCK_THREADS, int ARCH = HIPCUB_ARCH /* ignored */
         >
struct BlockRakingLayout
{
    //---------------------------------------------------------------------
    // Constants and type definitions
    //---------------------------------------------------------------------

    enum
    {
        /// The total number of elements that need to be cooperatively reduced
        SHARED_ELEMENTS = BLOCK_THREADS,

        /// Maximum number of warp-synchronous raking threads
        MAX_RAKING_THREADS = ::rocprim::detail::get_min_warp_size(
            static_cast<unsigned int>(BLOCK_THREADS), HIPCUB_DEVICE_WARP_THREADS),

        /// Number of raking elements per warp-synchronous raking thread (rounded up)
        SEGMENT_LENGTH = (SHARED_ELEMENTS + MAX_RAKING_THREADS - 1) / MAX_RAKING_THREADS,

        /// Never use a raking thread that will have no valid data (e.g., when BLOCK_THREADS is 62 and SEGMENT_LENGTH is 2, we should only use 31 raking threads)
        RAKING_THREADS = (SHARED_ELEMENTS + SEGMENT_LENGTH - 1) / SEGMENT_LENGTH,

        /// Pad each segment length with one element if segment length is not relatively prime to warp size and can't be optimized as a vector load
        USE_SEGMENT_PADDING = ((SEGMENT_LENGTH & 1) == 0) && (SEGMENT_LENGTH > 2),

        /// Total number of elements in the raking grid
        GRID_ELEMENTS = RAKING_THREADS * (SEGMENT_LENGTH + USE_SEGMENT_PADDING),

        /// Whether or not we need bounds checking during raking (the number of reduction elements is not a multiple of the number of raking threads)
        UNGUARDED = (SHARED_ELEMENTS % RAKING_THREADS == 0),
    };

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document
    /**
     * \brief Shared memory storage type
     */
    struct __align__(16) _TempStorage
    {
        T buff[BlockRakingLayout::GRID_ELEMENTS];
    };

#endif

    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};

    /**
     * \brief Returns the location for the calling thread to place data into the grid
     */
    static HIPCUB_DEVICE inline T* PlacementPtr(
        TempStorage &temp_storage,
        unsigned int linear_tid)
    {
        // Offset for partial
        unsigned int offset = linear_tid;

        // Add in one padding element for every segment
        if (USE_SEGMENT_PADDING > 0)
        {
            offset += offset / SEGMENT_LENGTH;
        }

        // Incorporating a block of padding partials every shared memory segment
        return temp_storage.Alias().buff + offset;
    }

    /**
     * \brief Returns the location for the calling thread to begin sequential raking
     */
    static HIPCUB_DEVICE inline T* RakingPtr(
        TempStorage &temp_storage,
        unsigned int linear_tid)
    {
        return temp_storage.Alias().buff + (linear_tid * (SEGMENT_LENGTH + USE_SEGMENT_PADDING));
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_RAKING_LAYOUT_HPP_
