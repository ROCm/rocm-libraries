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

#ifndef ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_
#define ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "thread.hpp"

/// \addtogroup warpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#ifdef __HIP_CPU_RT__
// TODO: consider adding macro checks relaying to std::bit_cast when compiled
//       using C++20.
template <class To, class From>
typename std::enable_if_t<
    sizeof(To) == sizeof(From) &&
    std::is_trivially_copyable_v<From> &&
    std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept
{
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}
#endif

template<class T, class ShuffleOp>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::enable_if<std::is_trivially_copyable<T>::value && (sizeof(T) % sizeof(int) == 0), T>::type
warp_shuffle_op(const T& input, ShuffleOp&& op)
{
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    struct V { int words[words_no]; };
#ifdef __HIP_CPU_RT__
    V a = bit_cast<V>(input);
#else
    V a = __builtin_bit_cast(V, input);
#endif

    ROCPRIM_UNROLL
    for(int i = 0; i < words_no; i++)
    {
        a.words[i] = op(a.words[i]);
    }

#ifdef __HIP_CPU_RT__
    return bit_cast<T>(a);
#else
    return __builtin_bit_cast(T, a);
#endif
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
        int word;
#ifdef __HIP_CPU_RT__
        std::memcpy(&word, reinterpret_cast<const char*>(&input) + i * sizeof(int), s);
#else
        __builtin_memcpy(&word, reinterpret_cast<const char*>(&input) + i * sizeof(int), s);
#endif
        word = op(word);
#ifdef __HIP_CPU_RT__
        std::memcpy(reinterpret_cast<char*>(&output) + i * sizeof(int), &word, s);
#else
        __builtin_memcpy(reinterpret_cast<char*>(&output) + i * sizeof(int), &word, s);
#endif
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
#if !defined(__HIP_CPU_RT__)
            return ::__builtin_amdgcn_mov_dpp(v, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
#else
            return v;
#endif
        }
    );
}

/// \brief Swizzle for any data type.
///
/// Each thread in warp obtains \p input from <tt>src_lane</tt>-th thread
/// in warp, where <tt>src_lane</tt> is current lane with a <tt>mask</tt> applied.
///
/// \param input - input to pass to other threads
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
/// in warp. If \p width is less than device_warp_size() then each subsection of the
/// warp behaves as a separate entity with a starting logical lane id of 0.
/// If \p src_lane is not in [0; \p width) range, the returned value is
/// equal to \p input passed by the <tt>src_lane modulo width</tt> thread.
///
/// Note: The optional \p width parameter must be a power of 2; results are
/// undefined if it is not a power of 2, or it is greater than device_warp_size().
///
/// \param input - input to pass to other threads
/// \param src_lane - warp if of a thread whose \p input should be returned
/// \param width - logical warp width
template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle(const T& input, const int src_lane, const int width = device_warp_size())
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
/// undefined if it is not a power of 2, or it is greater than device_warp_size().
///
/// \param input - input to pass to other threads
/// \param delta - offset for calculating source lane id
/// \param width - logical warp width
template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle_up(const T& input, const unsigned int delta, const int width = device_warp_size())
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
/// undefined if it is not a power of 2, or it is greater than device_warp_size().
///
/// \param input - input to pass to other threads
/// \param delta - offset for calculating source lane id
/// \param width - logical warp width
template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle_down(const T& input, const unsigned int delta, const int width = device_warp_size())
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
/// undefined if it is not a power of 2, or it is greater than device_warp_size().
///
/// \param input - input to pass to other threads
/// \param lane_mask - mask used for calculating source lane id
/// \param width - logical warp width
template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle_xor(const T& input, const int lane_mask, const int width = device_warp_size())
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return __shfl_xor(v, lane_mask, width);
        }
    );
}

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
/// undefined if it is not a power of 2, or it is greater than device_warp_size().
///
/// \param input - input to pass to other threads
/// \param dst_lane - the destination lane to which the value from this thread is written.
/// \param width - logical warp width
template<typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE T warp_permute(const T&  input,
                                             const int dst_lane,
                                             const int width = device_warp_size())
{
    const int self  = lane_id();
    const int index = (dst_lane + (self & ~(width - 1))) << 2;
    return detail::warp_shuffle_op(input,
                                   [=](int v) -> int
                                   { return __builtin_amdgcn_ds_permute(index, v); });
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_

/// @}
// end of group warpmodule
