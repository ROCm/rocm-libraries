// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "arch.hpp"

#include "../config.hpp"
#include "../types.hpp"

#include <hip/device_functions.h>

#include <type_traits>

/// \addtogroup intrinsicsmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// Evaluate predicate for all active work-items in the warp and return an integer
/// whose <tt>i</tt>-th bit is set if and only if \p predicate is <tt>true</tt>
/// for the <tt>i</tt>-th thread of the warp and the <tt>i</tt>-th thread is active.
///
/// \param predicate input to be evaluated for all active lanes
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
    c = ::__builtin_amdgcn_mbcnt_lo(static_cast<unsigned int>(x), add);
    if constexpr(sizeof(lane_mask_type) == 8)
    {
        // SPIR-V: We assumed 64 threads per wave, but this might not
        // be correct. Do an extra check to only do the upper half, when
        // there actually is an upper half.
        if(::rocprim::arch::wavefront::size() == ROCPRIM_WARP_SIZE_64)
        {
            c = ::__builtin_amdgcn_mbcnt_hi(static_cast<unsigned int>(x >> 32), c);
        }
    }
    return c;
}

namespace detail
{

ROCPRIM_DEVICE ROCPRIM_INLINE
int warp_any(int predicate)
{
    return ::__any(predicate);
}

ROCPRIM_DEVICE ROCPRIM_INLINE
int warp_all(int predicate)
{
    return ::__all(predicate);
}

} // namespace detail

/// \overload
/// \brief Group active lanes having the same bits of \p label
///
/// Threads that have the same least significant \p label_bits bits are grouped into the same group.
/// Every lane in the warp receives a mask of all active lanes participating in its group.
///
/// This overload does not accept a template parameter for label bits. It is passed as a function parameter instead.
///
/// \param [in] label the label for the calling lane
/// \param [in] label_bits number of bits to compare between labels
/// \param [in] valid lanes passing <tt>false</tt> will be ignored for comparisons,
/// such lanes will not be part of any group, and will always return an empty mask (0)
///
/// \return A bit mask of lanes sharing the same bits for \p label. The bit at index
/// lane <tt>i</tt>'s result includes bit <tt>j</tt> in the lane mask if lane <tt>j</tt> is part
/// of the same group as lane <tt>i</tt>, i.e. lane <tt>i</tt> and <tt>j</tt> called with the
/// same value for label.
ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type match_any(unsigned int label, unsigned int label_bits, bool valid = true)
{
    // Obtain a mask with the threads which are currently active.
    lane_mask_type peer_mask = ballot(valid);

    // Zero out all bits above label_bits so that __match_any effectively
    // compares only the low label_bits of the label, grouping lanes that
    // share the same least significant bits.
    const unsigned int bit_mask   = (label_bits >= 32) ? ~0u : ((1u << label_bits) - 1);
    const unsigned int masked_lbl = label & bit_mask;

    // Query the hardware for all lanes with the same masked label,
    // then intersect with the active lanes to exclude any that opted out.
    peer_mask &= static_cast<lane_mask_type>(__match_any(masked_lbl));

    return -lane_mask_type{valid} & peer_mask;
}

/// \brief Group active lanes having the same bits of \p label
///
/// Threads that have the same least significant \p LabelBits bits are grouped into the same group.
/// Every lane in the warp receives a mask of all active lanes participating in its group.
///
/// \tparam LabelBits number of bits to compare between labels
///
/// \param [in] label the label for the calling lane
/// \param [in] valid lanes passing <tt>false</tt> will be ignored for comparisons,
/// such lanes will not be part of any group, and will always return an empty mask (0)
///
/// \return A bit mask of lanes sharing the same bits for \p label. The bit at index
/// lane <tt>i</tt>'s result includes bit <tt>j</tt> in the lane mask if lane <tt>j</tt> is part
/// of the same group as lane <tt>i</tt>, i.e. lane <tt>i</tt> and <tt>j</tt> called with the
/// same value for label.

template<unsigned int LabelBits>
ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type match_any(unsigned int label, bool valid = true)
{
    // Dispatch to runtime version
    return match_any(label, LabelBits, valid);
}

/// \brief Elect a single lane for each group in \p mask
///
/// \param [in] mask bit mask of the lanes in the same group as the calling lane.
/// The <tt>i</tt>-th bit should be set if lane <tt>i</tt> is in the same group
/// as the calling lane.
///
/// \returns <tt>true</tt> for one unspecified lane in the <tt>mask</tt>, false for everyone else.
/// Returns <tt>false</tt> for all lanes not in any group, that is lanes passing 0 as \p mask.
///
/// \pre The relation specified by \p mask must be symmetric and transitive, in other words: the groups
/// should be consistent between threads.
ROCPRIM_DEVICE ROCPRIM_INLINE
bool group_elect(lane_mask_type mask)
{
    const unsigned int prev_same_count = ::rocprim::masked_bit_count(mask);
    return prev_same_count == 0 && mask != 0;
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group intrinsicsmodule

#endif // ROCPRIM_INTRINSICS_WARP_HPP_
