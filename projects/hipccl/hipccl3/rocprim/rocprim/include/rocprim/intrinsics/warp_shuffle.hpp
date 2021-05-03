// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

template<class T, class ShuffleOp>
ROCPRIM_DEVICE inline
typename std::enable_if<std::is_trivially_copyable<T>::value && (sizeof(T) % sizeof(int) == 0), T>::type
warp_shuffle_op(const T& input, ShuffleOp&& op)
{
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    struct V { int words[words_no]; };
    V a = __builtin_bit_cast(V, input);

    #pragma unroll
    for(int i = 0; i < words_no; i++)
    {
        a.words[i] = op(a.words[i]);
    }

    return __builtin_bit_cast(T, a);
}

template<class T, class ShuffleOp>
ROCPRIM_DEVICE inline
typename std::enable_if<!(std::is_trivially_copyable<T>::value && (sizeof(T) % sizeof(int) == 0)), T>::type
warp_shuffle_op(const T& input, ShuffleOp&& op)
{
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    T output;
    #pragma unroll
    for(int i = 0; i < words_no; i++)
    {
        const size_t s = std::min(sizeof(int), sizeof(T) - i * sizeof(int));
        int word;
        __builtin_memcpy(&word, reinterpret_cast<const char*>(&input) + i * sizeof(int), s);
        word = op(word);
        __builtin_memcpy(reinterpret_cast<char*>(&output) + i * sizeof(int), &word, s);
    }

    return output;
}

template<class T, int dpp_ctrl, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = false>
ROCPRIM_DEVICE inline
T warp_move_dpp(const T& input)
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return ::__builtin_amdgcn_update_dpp(0, v, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
        }
    );
}

} // end namespace detail

/// \brief Shuffle for any data type.
///
/// Each thread in warp obtains \p input from <tt>src_lane</tt>-th thread
/// in warp. If \p width is less than warp_size() then each subsection of the
/// warp behaves as a separate entity with a starting logical lane id of 0.
/// If \p src_lane is not in [0; \p width) range, the returned value is
/// equal to \p input passed by the <tt>src_lane modulo width</tt> thread.
///
/// Note: The optional \p width parameter must be a power of 2; results are
/// undefined if it is not a power of 2, or it is greater than warp_size().
///
/// \param input - input to pass to other threads
/// \param src_lane - warp if of a thread whose \p input should be returned
/// \param width - logical warp width
template<class T>
ROCPRIM_DEVICE inline
T warp_shuffle(const T& input, const int src_lane, const int width = warp_size())
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
/// undefined if it is not a power of 2, or it is greater than warp_size().
///
/// \param input - input to pass to other threads
/// \param delta - offset for calculating source lane id
/// \param width - logical warp width
template<class T>
ROCPRIM_DEVICE inline
T warp_shuffle_up(const T& input, const unsigned int delta, const int width = warp_size())
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
/// undefined if it is not a power of 2, or it is greater than warp_size().
///
/// \param input - input to pass to other threads
/// \param delta - offset for calculating source lane id
/// \param width - logical warp width
template<class T>
ROCPRIM_DEVICE inline
T warp_shuffle_down(const T& input, const unsigned int delta, const int width = warp_size())
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
/// undefined if it is not a power of 2, or it is greater than warp_size().
///
/// \param input - input to pass to other threads
/// \param lane_mask - mask used for calculating source lane id
/// \param width - logical warp width
template<class T>
ROCPRIM_DEVICE inline
T warp_shuffle_xor(const T& input, const int lane_mask, const int width = warp_size())
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return __shfl_xor(v, lane_mask, width);
        }
    );
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_

/// @}
// end of group warpmodule
