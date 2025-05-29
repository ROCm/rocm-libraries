// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_
#define ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_

#include <climits>
#include <limits>
#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../intrinsics/bit.hpp"
#include "../intrinsics/warp.hpp"
#include "../types.hpp"
#include "thread.hpp"

/// \addtogroup warpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
template<class T, class ShuffleOp>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::enable_if<std::is_trivially_copyable<T>::value && (sizeof(T) % sizeof(int) == 0), T>::type
warp_shuffle_op(const T& input, ShuffleOp&& op)
{
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    struct V { int words[words_no]; };

    auto a = ::rocprim::detail::bit_cast<V>(input);

    ROCPRIM_UNROLL
    for(int i = 0; i < words_no; i++)
    {
        a.words[i] = op(a.words[i]);
    }

    return ::rocprim::detail::bit_cast<T>(a);
}

template<class T, class ShuffleOp>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::enable_if<!(std::is_trivially_copyable<T>::value && (sizeof(T) % sizeof(int) == 0)), T>::type
warp_shuffle_op(const T& input, ShuffleOp&& op)
{
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    T output;
    ROCPRIM_UNROLL
    for(int i = 0; i < words_no; i++)
    {
        const size_t s = std::min(sizeof(int), sizeof(T) - i * sizeof(int));
        int          word;
        __builtin_memcpy(&word, reinterpret_cast<const char*>(&input) + i * sizeof(int), s);
        word = op(word);
        __builtin_memcpy(reinterpret_cast<char*>(&output) + i * sizeof(int), &word, s);
    }

    return output;

}

template<class T, int dpp_ctrl, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = false>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_move_dpp(const T& input)
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            // TODO: clean-up, this function activates based ROCPRIM_DETAIL_USE_DPP, however inclusion and
            //       parsing of the template happens unconditionally. The condition causing compilation to
            //       fail is ordinary host-compilers looking at the headers. Non-hipcc compilers don't define
            //       __builtin_amdgcn_update_dpp, hence fail to parse the template altogether. (Except MSVC
            //       because even using /permissive- they somehow still do delayed parsing of the body of
            //       function templates, even though they pinky-swear they don't.)
            return ::__builtin_amdgcn_mov_dpp(v, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
        });
}

/// \brief Swizzle for any data type.
///
/// Each thread in warp obtains \p input from <tt>src_lane</tt>-th thread
/// in warp, where <tt>src_lane</tt> is current lane with a <tt>mask</tt> applied.
///
/// \param input input to pass to other threads
template<class T, int mask>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_swizzle(const T& input)
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return ::__builtin_amdgcn_ds_swizzle(v, mask);
        }
    );
}

} // end namespace detail

/// \brief Shuffle for any data type.
///
/// Each thread in warp obtains \p input from <tt>src_lane</tt>-th thread
/// in warp. If \p width is less than arch::wavefront::min_size() then each subsection of the
/// warp behaves as a separate entity with a starting logical lane id of 0.
/// If \p src_lane is not in [0; \p width) range, the returned value is
/// equal to \p input passed by the <tt>src_lane modulo width</tt> thread.
///
/// Note: The optional \p width parameter must be a power of 2; results are
/// undefined if it is not a power of 2, or it is greater than arch::wavefront::min_size().
///
/// \param input input to pass to other threads
/// \param src_lane warp if of a thread whose \p input should be returned
/// \param width logical warp width
template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle(const T& input, const int src_lane, const int width = arch::wavefront::min_size())
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return __shfl(v, src_lane, width);
        }
    );
}

/// \brief Shuffle up for any data type.
///
/// <tt>i</tt>-th thread in warp obtains \p input from <tt>i-delta</tt>-th
/// thread in warp. If \p <tt>i-delta</tt> is not in [0; \p width) range,
/// thread's own \p input is returned.
///
/// Note: The optional \p width parameter must be a power of 2; results are
/// undefined if it is not a power of 2, or it is greater than arch::wavefront::min_size().
///
/// \param input input to pass to other threads
/// \param delta offset for calculating source lane id
/// \param width logical warp width
template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle_up(const T&           input,
                  const unsigned int delta,
                  const int          width = arch::wavefront::min_size())
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return __shfl_up(v, delta, width);
        }
    );
}

/// \brief Shuffle down for any data type.
///
/// <tt>i</tt>-th thread in warp obtains \p input from <tt>i+delta</tt>-th
/// thread in warp. If \p <tt>i+delta</tt> is not in [0; \p width) range,
/// thread's own \p input is returned.
///
/// Note: The optional \p width parameter must be a power of 2; results are
/// undefined if it is not a power of 2, or it is greater than arch::wavefront::min_size().
///
/// \param input input to pass to other threads
/// \param delta offset for calculating source lane id
/// \param width logical warp width
template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle_down(const T&           input,
                    const unsigned int delta,
                    const int          width = arch::wavefront::min_size())
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return __shfl_down(v, delta, width);
        }
    );
}

/// \brief Shuffle XOR for any data type.
///
/// <tt>i</tt>-th thread in warp obtains \p input from <tt>i^lane_mask</tt>-th
/// thread in warp.
///
/// Note: The optional \p width parameter must be a power of 2; results are
/// undefined if it is not a power of 2, or it is greater than arch::wavefront::min_size().
///
/// \param input input to pass to other threads
/// \param lane_mask mask used for calculating source lane id
/// \param width logical warp width
template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle_xor(const T&  input,
                   const int lane_mask,
                   const int width = arch::wavefront::min_size())
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return __shfl_xor(v, lane_mask, width);
        }
    );
}

namespace detail
{

/// \brief Shuffle XOR for any data type using warp_swizzle.
///
/// <tt>i</tt>-th thread in warp obtains \p input from <tt>i^lane_mask</tt>-th
/// thread in warp. Makes use of of the swizzle instruction for powers of 2 till 16.
/// Defaults to warp_shuffle_xor.
///
/// Note: The optional \p width parameter must be a power of 2; results are
/// undefined if it is not a power of 2, or it is greater than arch::wavefront::min_size().
///
/// \param v input to pass to other threads
/// \param mask mask used for calculating source lane id
/// \param width logical warp width
template<class V>
ROCPRIM_DEVICE ROCPRIM_INLINE
V warp_swizzle_shuffle(V& v, const int mask, const int width = arch::wavefront::min_size())
{
    switch(mask)
    {
        case 1: return warp_swizzle<V, 0x041F>(v);
        case 2: return warp_swizzle<V, 0x081F>(v);
        case 4: return warp_swizzle<V, 0x101F>(v);
        case 8: return warp_swizzle<V, 0x201F>(v);
        case 16: return warp_swizzle<V, 0x401F>(v);
        default: return warp_shuffle_xor(v, mask, width);
    }
}

} // namespace detail

/// \brief Permute items across the threads in a warp.
///
/// The value from this thread in the warp is permuted to the <tt>dst_lane</tt>-th
/// thread in the warp. If multiple warps write to the same destination, the result
/// is undefined but will be a value from either of the source values. If no threads
/// write to a particular thread then the value for that thread will be 0.
/// The destination index is taken modulo the logical warp size, so any value larger
/// than the logical warp size will wrap around.
///
/// Note: The optional \p width parameter must be a power of 2; results are
/// undefined if it is not a power of 2, or it is greater than arch::wavefront::min_size().
///
/// \param input input to pass to other threads
/// \param dst_lane the destination lane to which the value from this thread is written.
/// \param width logical warp width
template<typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_permute(const T& input, const int dst_lane, const int width = arch::wavefront::min_size())
{
    // The amdgcn intrinsic does not support virtual warp sizes, so in order to support those, manually
    // wrap around the dst_lane within groups of log2(width) bits.
    const int self  = lane_id();
    // Construct a mask of bits which make up a virtual warp. If the warp size is `width` (a power of 2),
    // then the lower `width - 1` bits indicate the position within a virtual warp, and the remainder
    // indicate the position of the virtual warp within the real warp.
    const unsigned int mask = width - 1;
    // Wrap the `dst_lane` around the virtual warp size by only considering the part within the
    // virtual warp size, defined by the bits in the mask. The hardware warp index is given by adding the
    // virtual warp's destination lane to the virtual warp's base offset, which is given by rounding down the
    // current lane's position by the virtual warp size.
    // Note that the extra right shift by 2 is required for the amdgcn intrinsic.
    const int index = ((dst_lane & mask) + (self & ~mask)) << 2;
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        // __builtin_amdgcn_ds_permute maps to the `ds_permute_b32` instruction.
        // See AMD ISA reference at https://gpuopen.com/amd-isa-documentation/.
        { return __builtin_amdgcn_ds_permute(index, v); });
}

/// \brief Broadcast the first lane to all threads.
///
/// Each thread in the warp obtains \p input from the first active thread in a warp.
/// This function always operates on all <tt>arch::wavefront::min_size()</tt> threads in the warp.
///
/// \remark This operation is significantly faster than \p warp_shuffle.
///
/// \param input the value to broadcast
template<typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE T warp_readfirstlane(const T& input)
{
    return detail::warp_shuffle_op(input,
                                   [](int v) -> int { return __builtin_amdgcn_readfirstlane(v); });
}

/// \brief Broadcast a particular lane to all threads.
///
/// Each thread in the warp obtains \p input from the <tt>src_lane</tt>-th thread
/// in the warp. \p src_lane must be the same value for all threads in the warp.
/// This function does not distinguish between active threads and non-active
/// threads: all threads must participate in the broadcast. This function also
/// always operates on all <tt>arch::wavefront::min_size()</tt> threads in the warp.
///
/// \remark This operation is significantly faster than \p warp_shuffle.
///
/// \param input the value to broadcast
/// \param src_lane the lane whose value to broadcast to other threads in the warp
template<typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE T warp_readlane(const T& input, const int src_lane)
{
    return detail::warp_shuffle_op(input,
                                   [=](int v) -> int
                                   { return __builtin_amdgcn_readlane(v, src_lane); });
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_
