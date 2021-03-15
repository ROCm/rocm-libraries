/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021, Advanced Micro Devices, Inc.  All rights reserved.
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


#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_RAKING_LAYOUT_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_RAKING_LAYOUT_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"


/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

template<
    class T,
    unsigned int BlockThreads
    >
struct block_raking_layout
{
  /// The total number of elements that need to be cooperatively reduced
  static constexpr unsigned int SharedElements = BlockThreads;

  /// Maximum number of warp-synchronous raking threads
  static constexpr unsigned int MaxRakingThreads = min(BlockThreads,::rocprim::warp_size());

  /// Number of raking elements per warp-synchronous raking thread (rounded up)
  static constexpr unsigned int SegmentLength = (SharedElements + MaxRakingThreads - 1)/MaxRakingThreads;

  /// Never use a raking thread that will have no valid data (e.g., when BLOCK_THREADS is 62 and SEGMENT_LENGTH is 2, we should only use 31 raking threads)
  static constexpr unsigned int RakingThreads = (SharedElements + SegmentLength - 1)/SegmentLength;
  /// Pad each segment length with one element if segment length is not relatively prime to warp size and can't be optimized as a vector load
  static constexpr unsigned int UseSegmentPadding = ((SegmentLength & 1) == 0) && (SegmentLength > 2);

  /// Total number of elements in the raking grid
  static constexpr unsigned int GridElements = RakingThreads * (SegmentLength + UseSegmentPadding);

  /// Whether or not we need bounds checking during raking (the number of reduction elements is not a multiple of the number of raking threads)
  static constexpr unsigned int Unguarded = (SharedElements % RakingThreads == 0);

  struct storage_type_
  {
      T buff[GridElements];
  };


  #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
  using storage_type = detail::raw_storage<storage_type_>;
  #else
  using storage_type = storage_type_; // only for Doxygen
  #endif

  static ROCPRIM_DEVICE inline T* placement_ptr(
          storage_type &temp_storage,
          unsigned int linear_tid)
      {
          // Offset for partial
          unsigned int offset = linear_tid;

          // Add in one padding element for every segment
          if (UseSegmentPadding > 0)
          {
              offset += offset / SegmentLength;
          }

          // Incorporating a block of padding partials every shared memory segment
          return temp_storage.get().buff + offset;
      }


      /**
       * \brief Returns the location for the calling thread to begin sequential raking
       */
      static ROCPRIM_DEVICE inline T* raking_ptr(
          storage_type &temp_storage,
          unsigned int linear_tid)
      {
          return temp_storage.get().buff + (linear_tid * (SegmentLength + UseSegmentPadding));
      }

};


END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_RAKING_LAYOUT_HPP_
