// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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
 * Compute a 32b mask of threads having the same least-significant
 * LABEL_BITS of \p label as the calling thread.
 */
template <int LABEL_BITS>
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int MatchAny(unsigned int label)
{
    unsigned int retval;

    // Extract masks of common threads for each bit
    ROCPRIM_UNROLL
    for (int BIT = 0; BIT < LABEL_BITS; ++BIT)
    {
        unsigned long long  mask;
        unsigned long long current_bit = 1 << BIT;
        mask = label & current_bit;
        bool bit_match = (mask==current_bit);
        mask = ballot(bit_match);
        if(!bit_match)
        {
          mask = ! mask;
        }
        // Remove peers who differ
        retval = (BIT == 0) ? mask : retval & mask;
    }

    return retval;

}
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_WARP_HPP_
