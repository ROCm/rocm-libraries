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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SCAN_CROSSLANE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SCAN_CROSSLANE_HPP_

#include <type_traits>

#include "../../config.hpp"

#include "warp_scan_dpp.hpp"
#include "warp_scan_shuffle.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class T, unsigned int VirtualWaveSize, bool UseDPP = ROCPRIM_DETAIL_USE_DPP>
class warp_scan_crosslane
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
                f(warp_scan_dpp<T, VirtualWaveSize>{});
            }
            else
            {
                f(warp_scan_shuffle<T, VirtualWaveSize>{});
            }
        }
        else
        {
            f(warp_scan_shuffle<T, VirtualWaveSize>{});
        }
    }

public:
    using storage_type = detail::empty_storage_type;

    // Inclusive scan
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, BinaryFunction scan_op)
    {
        dispatch([&](auto impl) { impl.inclusive_scan(input, output, scan_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, storage_type& storage, BinaryFunction scan_op)
    {
        (void)storage;
        inclusive_scan(input, output, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, BinaryFunction scan_op, T init)
    {
        dispatch([&](auto impl) { impl.inclusive_scan(input, output, scan_op, init); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, storage_type& storage, BinaryFunction scan_op, T init)
    {
        (void)storage;
        inclusive_scan(input, output, scan_op, init);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, T& reduction, BinaryFunction scan_op)
    {
        dispatch([&](auto impl) { impl.inclusive_scan(input, output, reduction, scan_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(
        T input, T& output, T& reduction, storage_type& storage, BinaryFunction scan_op)
    {
        (void)storage;
        inclusive_scan(input, output, reduction, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, T& reduction, BinaryFunction scan_op, T init)
    {
        dispatch([&](auto impl) { impl.inclusive_scan(input, output, reduction, scan_op, init); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(
        T input, T& output, T& reduction, storage_type& storage, BinaryFunction scan_op, T init)
    {
        (void)storage;
        inclusive_scan(input, output, reduction, scan_op, init);
    }

    // Exclusive scan
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, T init, BinaryFunction scan_op)
    {
        dispatch([&](auto impl) { impl.exclusive_scan(input, output, init, scan_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, T init, storage_type& storage, BinaryFunction scan_op)
    {
        (void)storage;
        exclusive_scan(input, output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, storage_type& storage, BinaryFunction scan_op)
    {
        (void)storage;
        dispatch([&](auto impl) { impl.exclusive_scan(input, output, storage, scan_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(
        T input, T& output, storage_type& storage, T& reduction, BinaryFunction scan_op)
    {
        (void)storage;
        dispatch([&](auto impl)
                 { impl.exclusive_scan(input, output, storage, reduction, scan_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, T init, T& reduction, BinaryFunction scan_op)
    {
        dispatch([&](auto impl) { impl.exclusive_scan(input, output, init, reduction, scan_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(
        T input, T& output, T init, T& reduction, storage_type& storage, BinaryFunction scan_op)
    {
        (void)storage;
        exclusive_scan(input, output, init, reduction, scan_op);
    }

    // Scan (inclusive + exclusive)
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scan(T input, T& inclusive_output, T& exclusive_output, T init, BinaryFunction scan_op)
    {
        dispatch([&](auto impl)
                 { impl.scan(input, inclusive_output, exclusive_output, init, scan_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scan(T              input,
              T&             inclusive_output,
              T&             exclusive_output,
              T              init,
              storage_type&  storage,
              BinaryFunction scan_op)
    {
        (void)storage;
        scan(input, inclusive_output, exclusive_output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scan(T              input,
              T&             inclusive_output,
              T&             exclusive_output,
              storage_type&  storage,
              BinaryFunction scan_op)
    {
        (void)storage;
        dispatch([&](auto impl)
                 { impl.scan(input, inclusive_output, exclusive_output, storage, scan_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scan(T              input,
              T&             inclusive_output,
              T&             exclusive_output,
              T              init,
              T&             reduction,
              BinaryFunction scan_op)
    {
        dispatch(
            [&](auto impl)
            { impl.scan(input, inclusive_output, exclusive_output, init, reduction, scan_op); });
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scan(T              input,
              T&             inclusive_output,
              T&             exclusive_output,
              T              init,
              T&             reduction,
              storage_type&  storage,
              BinaryFunction scan_op)
    {
        (void)storage;
        scan(input, inclusive_output, exclusive_output, init, reduction, scan_op);
    }

    // Broadcast
    ROCPRIM_DEVICE ROCPRIM_INLINE
    T broadcast(T input, const unsigned int src_lane, storage_type& storage)
    {
        T result{};
        dispatch([&](auto impl) { result = impl.broadcast(input, src_lane, storage); });
        return result;
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SCAN_CROSSLANE_HPP_
