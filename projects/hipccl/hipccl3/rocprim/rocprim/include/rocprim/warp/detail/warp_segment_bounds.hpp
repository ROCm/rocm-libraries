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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SEGMENT_BOUNDS_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SEGMENT_BOUNDS_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../intrinsics.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Returns logical warp id of the last thread in thread's segment
template<bool HeadSegmented, unsigned int WarpSize, class Flag>
ROCPRIM_DEVICE inline
unsigned int last_in_warp_segment(Flag flag)
{
    // Get flags (now every thread know where the flags are)
    auto warp_flags = ::rocprim::ballot(flag);
    using ballot_type = decltype(warp_flags);

    // In case of head flags change them to tail flags
    if(HeadSegmented)
    {
        warp_flags >>= 1;
    }
    const auto lane_id = ::rocprim::lane_id();
    // Zero bits from thread with lower lane id
    warp_flags &= ballot_type(-1) ^ ((ballot_type(1) << lane_id) - 1U);
    // Ignore bits from thread from other (previous) logical warps
    warp_flags >>= (lane_id / WarpSize) * WarpSize;
    // Make sure last item in logical warp is marked as a tail
    warp_flags |= ballot_type(1) << (WarpSize - 1U);
    // Calculate logical lane id of the last valid value in the segment
    return ::__lastbit_u32_u64(warp_flags);
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SEGMENT_BOUNDS_HPP_
