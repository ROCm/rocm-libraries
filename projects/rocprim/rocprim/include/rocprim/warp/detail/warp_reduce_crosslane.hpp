// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_WARP_DETAIL_WARP_REDUCE_CROSSLANE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_REDUCE_CROSSLANE_HPP_

#include <type_traits>

#include "../../config.hpp"

#include "warp_reduce_dpp.hpp"
#include "warp_reduce_shuffle.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class T,
         unsigned int VirtualWaveSize,
         bool         UseAllReduce,
         bool         UseDPP = ROCPRIM_DETAIL_USE_DPP>
class warp_reduce_crosslane
{
private:
    template<class F>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void dispatch(F&& f)
    {
        // We use a dispatch here because when we target SPIR-V, we have to know after
        // compiling SPIR-V whether DPP is available. Therefore, this check cannot
        // be done at the C++ constexpr-level.
        if constexpr(UseDPP)
        {
            if ROCPRIM_AMDGCN_CONSTEXPR(ROCPRIM_HAS_DPP())
            {
                f(warp_reduce_dpp<T, VirtualWaveSize, UseAllReduce>{});
            }
            else
            {
                f(warp_reduce_shuffle<T, VirtualWaveSize, UseAllReduce>{});
            }
        }
        else
        {
            f(warp_reduce_shuffle<T, VirtualWaveSize, UseAllReduce>{});
        }
    }

public:
    using storage_type = detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, BinaryFunction reduce_op)
    {
        dispatch([&](auto warp_reduce_impl) { warp_reduce_impl.reduce(input, output, reduce_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, storage_type& storage, BinaryFunction reduce_op)
    {
        (void)storage; // disables unused parameter warning
        this->reduce(input, output, reduce_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, unsigned int valid_items, BinaryFunction reduce_op)
    {
        dispatch([&](auto warp_reduce_impl)
                 { warp_reduce_impl.reduce(input, output, valid_items, reduce_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T              input,
                T&             output,
                unsigned int   valid_items,
                storage_type&  storage,
                BinaryFunction reduce_op)
    {
        (void)storage; // disables unused parameter warning
        this->reduce(input, output, valid_items, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void head_segmented_reduce(T input, T& output, Flag flag, BinaryFunction reduce_op)
    {
        dispatch([&](auto warp_reduce_impl)
                 { warp_reduce_impl.head_segmented_reduce(input, output, flag, reduce_op); });
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void head_segmented_reduce(
        T input, T& output, Flag flag, storage_type& storage, BinaryFunction reduce_op)
    {
        dispatch(
            [&](auto warp_reduce_impl)
            { warp_reduce_impl.head_segmented_reduce(input, output, flag, storage, reduce_op); });
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tail_segmented_reduce(T input, T& output, Flag flag, BinaryFunction reduce_op)
    {
        dispatch([&](auto warp_reduce_impl)
                 { warp_reduce_impl.tail_segmented_reduce(input, output, flag, reduce_op); });
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tail_segmented_reduce(
        T input, T& output, Flag flag, storage_type& storage, BinaryFunction reduce_op)
    {
        dispatch(
            [&](auto warp_reduce_impl)
            { warp_reduce_impl.tail_segmented_reduce(input, output, flag, storage, reduce_op); });
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_REDUCE_CROSSLANE_HPP_
