// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_INTRINSICS_WARP_HPP_
#define ROCPRIM_INTRINSICS_WARP_HPP_

#include "../config.hpp"
#include "../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup intrinsicsmodule
/// @{

/// Evaluate predicate for all active work-items in the warp and return an integer
/// whose <tt>i</tt>-th bit is set if and only if \p predicate is <tt>true</tt>
/// for the <tt>i</tt>-th thread of the warp and the <tt>i</tt>-th thread is active.
///
/// \param predicate - input to be evaluated for all active lanes
ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type ballot(int predicate)
{
    return ::__ballot(predicate);
}

/// \brief Masked bit count
///
/// For each thread, this function returns the number of active threads which
/// have <tt>i</tt>-th bit of \p x set and come before the current thread.
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int masked_bit_count(lane_mask_type x, unsigned int add = 0)
{
    int c;
    #ifndef __HIP_CPU_RT__
        #if __AMDGCN_WAVEFRONT_SIZE == 32
            #ifdef __HIP__
            c = ::__builtin_amdgcn_mbcnt_lo(x, add);
            #else
            c = ::__mbcnt_lo(x, add);
            #endif
        #else
            #ifdef __HIP__
            c = ::__builtin_amdgcn_mbcnt_lo(static_cast<int>(x), add);
            c = ::__builtin_amdgcn_mbcnt_hi(static_cast<int>(x >> 32), c);
            #else
            c = ::__mbcnt_lo(static_cast<int>(x), add);
            c = ::__mbcnt_hi(static_cast<int>(x >> 32), c);
            #endif
        #endif
    #else
        using namespace hip::detail;
        const auto tidx{id(Fiber::this_fiber()) % warpSize};
        std::bitset<warpSize> bits{x >> (warpSize - tidx)};
        c = static_cast<unsigned int>(bits.count()) + add;
    #endif
    return c;
}

namespace detail
{

ROCPRIM_DEVICE ROCPRIM_INLINE
int warp_any(int predicate)
{
#ifndef __HIP_CPU_RT__
    return ::__any(predicate);
#else
    using namespace hip::detail;
    const auto tidx{id(Fiber::this_fiber()) % warpSize};
    auto& lds{Tile::scratchpad<std::bitset<warpSize>, 1>()[0]};

    lds[tidx] = static_cast<bool>(predicate);

    barrier(Tile::this_tile());

    return lds.any();
#endif
}

ROCPRIM_DEVICE ROCPRIM_INLINE
int warp_all(int predicate)
{
#ifndef __HIP_CPU_RT__
    return ::__all(predicate);
#else
    using namespace hip::detail;
    const auto tidx{id(Fiber::this_fiber()) % warpSize};
    auto& lds{Tile::scratchpad<std::bitset<warpSize>, 1>()[0]};

    lds[tidx] = static_cast<bool>(predicate);

    barrier(Tile::this_tile());

    return lds.all();
#endif
}

} // end detail namespace

/// @}
// end of group intrinsicsmodule

/**
 * This function computes a lane mask of active lanes in the warp which which have
 * the same value for <tt>label</tt> as the lane which calls the function. The bit at
 * index \p i in the lane mask is set if the thread of lane \p i calls this function
 * with the same value <tt>label</tt>. Only the least-significant \p LabelBits bits
 * are taken into account when labels are considered to be equal.
 */
template<unsigned int LabelBits>
ROCPRIM_DEVICE ROCPRIM_INLINE lane_mask_type match_any(unsigned int label)
{
    // Obtain a mask with the threads which are currently active.
    lane_mask_type peer_mask = ballot(1);

    // Compute the final value iteratively by testing each bit separately.
    ROCPRIM_UNROLL
    for(unsigned int bit = 0; bit < LabelBits; ++bit)
    {
        const auto bit_set = label & (1u << bit);
        // Create mask of threads which have the same bit set or unset.
        const auto same_mask = ballot(bit_set);
        // Remove bits which do not match from the peer mask.
        peer_mask &= (bit_set ? same_mask : ~same_mask);
    }

    return peer_mask;
}

/**
 * This function computes a lane mask of active lanes in the warp which which have
 * the same value for <tt>label</tt> as the lane which calls the function. The bit at
 * index \p i in the lane mask is set if the thread of lane \p i calls this function
 * with the same value <tt>label</tt>. Only the least-significant \p LabelBits bits
 * are taken into account when labels are considered to be equal.
 */
template<int LabelBits>
[[deprecated("use rocprim::match_any instead")]] ROCPRIM_DEVICE ROCPRIM_INLINE lane_mask_type
    MatchAny(unsigned int label)
{
    return match_any<LabelBits>(label);
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_WARP_HPP_
