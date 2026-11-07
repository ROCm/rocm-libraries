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

// Zero out all bits above label_bits
ROCPRIM_DEVICE ROCPRIM_INLINE
constexpr unsigned int low_bits_mask(unsigned int label_bits)
{
    return label_bits >= 32 ? ~0u : ((1u << label_bits) - 1u);
}

// Mask of the lanes whose `bit`-th label bit matches the calling lane's.
ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type matching_bit_mask(unsigned int label, unsigned int bit)
{
    static constexpr int lane_width = std::numeric_limits<lane_mask_type>::digits;
    using lane_mask_type_s          = std::make_signed_t<lane_mask_type>;
    const auto label_signed         = static_cast<lane_mask_type_s>(label);

    // Get all zeros or all ones depending on label's i-th bit.
    // Moves the bit into the sign position by left shifting, then shifts it into all the bits
    // by (arithmetic) right shift which does sign-extension.
    const lane_mask_type_s bit_set = (label_signed << (lane_width - 1 - bit)) >> (lane_width - 1);

    // Remove all lanes from the mask with a bit that differs from ours
    // - if we have the bit set we keep the lanes that do too so we mask with the result
    //   of the ballot
    // - if we don't have it, then we keep the lanes that also don't, so we flip all bits
    //   in the mask before and-ing.
    // since bit_set is all ones if we have the bit and all zeros if not, the flipping is
    // the same as xor-ing with its inverse
    const lane_mask_type bit_set_mask = ballot(bit_set);
    return bit_set_mask ^ ~bit_set;
}

template<bool UseIntrinsic>
ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type match_any_mask(unsigned int label, unsigned int label_bits, bool valid)
{
    // Obtain a mask with the threads which are currently active.
    lane_mask_type peer_mask = ballot(valid);

    if constexpr(UseIntrinsic)
    {
        // Zero out all bits above label_bits so that __match_any effectively
        // compares only the low label_bits of the label, grouping lanes that
        // share the same least significant bits.
        const unsigned int masked_lbl = label & low_bits_mask(label_bits);

        // Query the hardware for all lanes with the same masked label,
        // then intersect with the active lanes to exclude any that opted out.
        peer_mask &= static_cast<lane_mask_type>(__match_any(masked_lbl));
    }
    else
    {
        // Compute the final value iteratively by testing each bit separately.
        for(unsigned int bit = 0; bit < label_bits; ++bit)
            peer_mask &= matching_bit_mask(label, bit);
    }

    return -lane_mask_type{valid} & peer_mask;
}

// This is where the constexpr simplification lives: the mask is an immediate and
// the loop is fully unrolled with constant shift amounts.
template<unsigned int LabelBits, bool UseIntrinsic>
ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type match_any_mask(unsigned int label, bool valid)
{
    // Obtain a mask with the threads which are currently active.
    lane_mask_type peer_mask = ballot(valid);

    if constexpr(UseIntrinsic)
    {
        // Zero out all bits above LabelBits so that __match_any effectively
        // compares only the low LabelBits of the label, grouping lanes that
        // share the same least significant bits.
        // LabelBits is known at compile time, so the mask is folded to a constant.
        constexpr unsigned int bit_mask = low_bits_mask(LabelBits);

        // Query the hardware for all lanes with the same masked label,
        // then intersect with the active lanes to exclude any that opted out.
        peer_mask &= static_cast<lane_mask_type>(__match_any(label & bit_mask));
    }
    else
    {
        // Compute the final value iteratively by testing each bit separately.
        for(unsigned int bit = 0; bit < LabelBits; ++bit)
            peer_mask &= matching_bit_mask(label, bit);
    }

    return -lane_mask_type{valid} & peer_mask;
}

template<unsigned int LabelBits, bool UseIntrinsic>
ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type match_any_mask_all_valid(unsigned int label)
{
    if constexpr(UseIntrinsic)
    {
        // LabelBits is known at compile time, so the mask is folded to a constant.
        constexpr unsigned int bit_mask = low_bits_mask(LabelBits);
        return static_cast<lane_mask_type>(__match_any(label & bit_mask));
    }
    else
    {
        // Compute the final value iteratively by testing each bit separately.
        lane_mask_type peer_mask = ballot(true);
        for(unsigned int bit = 0; bit < LabelBits; ++bit)
            peer_mask &= matching_bit_mask(label, bit);
        return peer_mask;
    }
}

} // namespace detail

/// \brief Group active lanes having the same bits of \p label
///
/// Threads that have the same least significant bits of \p label are grouped into the same group.
/// Every lane in the warp receives a mask of all active lanes participating in its group.
///
/// \tparam UseIntrinsic whether to use the hardware \c __match_any instruction
/// (<tt>true</tt>) or emulate the match with ballots (<tt>false</tt>)
///
/// \param [in] label the label for the calling lane
///
/// \param [in] label_bits number of bits to compare between labels
///
/// \param [in] valid lanes passing <tt>false</tt> will be ignored for comparisons,
/// such lanes will not be part of any group, and will always return an empty mask (0)
///
/// \return A bit mask of lanes sharing the same bits for \p label. The bit at index
/// lane <tt>i</tt>'s result includes bit <tt>j</tt> in the lane mask if lane <tt>j</tt> is part
/// of the same group as lane <tt>i</tt>, i.e. lane <tt>i</tt> and lane <tt>j</tt> called with the
/// same value for label.
template<bool UseIntrinsic = false>
ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type match_any(unsigned int label, unsigned int label_bits, bool valid = true)
{
    return detail::match_any_mask<UseIntrinsic>(label, label_bits, valid);
}

/// \brief Group active lanes having the same bits of \p label
///
/// Threads that have the same least significant \p LabelBits bits are grouped into the same group.
/// Every lane in the warp receives a mask of all active lanes participating in its group.
///
/// \tparam LabelBits number of bits to compare between labels
/// \tparam UseIntrinsic whether to use the hardware \c __match_any instruction
/// (<tt>true</tt>) or emulate the match with ballots (<tt>false</tt>)
///
/// \param [in] label the label for the calling lane
///
/// \param [in] valid lanes passing <tt>false</tt> will be ignored for comparisons,
/// such lanes will not be part of any group, and will always return an empty mask (0)
///
/// \return A bit mask of lanes sharing the same bits for \p label. The bit at index
/// lane <tt>i</tt>'s result includes bit <tt>j</tt> in the lane mask if lane <tt>j</tt> is part
/// of the same group as lane <tt>i</tt>, i.e. lane <tt>i</tt> and lane <tt>j</tt> called with the
/// same value for label.
template<unsigned int LabelBits, bool UseIntrinsic = false>
ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type match_any(unsigned int label, bool valid)
{
    return detail::match_any_mask<LabelBits, UseIntrinsic>(label, valid);
}

/// \brief Group active lanes having the same bits of \p label
///
/// Threads that have the same least significant \p LabelBits bits are grouped into the same group.
/// Every lane in the warp receives a mask of all lanes participating in its group.
///
/// This is the all-lanes-valid form of match_any: no lane opts out of the comparison,
/// so the result contains no ballots or zeroing of invalid lanes.
///
/// \tparam LabelBits number of bits to compare between labels
/// \tparam UseIntrinsic whether to use the hardware \c __match_any instruction
/// (<tt>true</tt>) or emulate the match with ballots (<tt>false</tt>)
///
/// \param [in] label the label for the calling lane
///
/// \return A bit mask of lanes sharing the same bits for \p label. The bit at index
/// lane <tt>i</tt>'s result includes bit <tt>j</tt> in the lane mask if lane <tt>j</tt> is part
/// of the same group as lane <tt>i</tt>, i.e. lane <tt>i</tt> and lane <tt>j</tt> called with the
/// same value for label.
template<unsigned int LabelBits, bool UseIntrinsic = false>
ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type match_any(unsigned int label)
{
    return detail::match_any_mask_all_valid<LabelBits, UseIntrinsic>(label);
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
