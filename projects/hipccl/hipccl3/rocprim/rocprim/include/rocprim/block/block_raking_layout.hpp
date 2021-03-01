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

#ifndef ROCPRIM_BLOCK_BLOCK_RAKING_LAYOUT_HPP_
#define ROCPRIM_BLOCK_BLOCK_RAKING_LAYOUT_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"


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

#endif // ROCPRIM_BLOCK_BLOCK_RAKING_LAYOUT_HPP_
