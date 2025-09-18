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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SCAN_DPP_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SCAN_DPP_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../intrinsics/arch.hpp"
#include "../../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
template<class T, unsigned int VirtualWaveSize>
class warp_scan_dpp
{
public:
    static_assert(detail::is_power_of_two(VirtualWaveSize), "VirtualWaveSize must be power of 2");

    using storage_type = detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, BinaryFunction scan_op)
    {
        const unsigned int lane_id     = ::rocprim::lane_id();
        const unsigned int row_lane_id = lane_id % ::rocprim::min(16u, VirtualWaveSize);

        output = input;

        if(VirtualWaveSize > 1)
        {
            T t = warp_move_dpp<T, 0x111>(output); // row_shr:1
            if(row_lane_id >= 1)
                output = scan_op(t, output);
        }
        if(VirtualWaveSize > 2)
        {
            T t = warp_move_dpp<T, 0x112>(output); // row_shr:2
            if(row_lane_id >= 2)
                output = scan_op(t, output);
        }
        if(VirtualWaveSize > 4)
        {
            T t = warp_move_dpp<T, 0x114>(output); // row_shr:4
            if(row_lane_id >= 4)
                output = scan_op(t, output);
        }
        if(VirtualWaveSize > 8)
        {
            T t = warp_move_dpp<T, 0x118>(output); // row_shr:8
            if(row_lane_id >= 8)
                output = scan_op(t, output);
        }
#ifdef ROCPRIM_DETAIL_HAS_DPP_BROADCAST
        if(VirtualWaveSize > 16)
        {
            T t = warp_move_dpp<T, 0x142>(output); // row_bcast:15
            if(lane_id % 32 >= 16)
                output = scan_op(t, output);
        }
        if(VirtualWaveSize > 32)
        {
            T t = warp_move_dpp<T, 0x143>(output); // row_bcast:31
            if(lane_id >= 32)
                output = scan_op(t, output);
        }
        static_assert(VirtualWaveSize <= 64, "VirtualWaveSize > 64 is not supported");
#else
        if(VirtualWaveSize > 16)
        {
            T t = warp_swizzle<T, 0x1e0>(output); // row_bcast:15
            if(lane_id % 32 >= 16)
                output = scan_op(t, output);
        }
        static_assert(VirtualWaveSize <= 32,
                      "VirtualWaveSize > 32 is not supported without DPP broadcasts");
#endif
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, storage_type& storage, BinaryFunction scan_op)
    {
        (void)storage; // disables unused parameter warning
        inclusive_scan(input, output, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, BinaryFunction scan_op, T init)
    {
        inclusive_scan(input, output, scan_op);
        // Include init value in scan results
        output = scan_op(init, output);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, storage_type& storage, BinaryFunction scan_op, T init)
    {
        (void)storage; // disables unused parameter warning
        inclusive_scan(input, output, scan_op, init);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, T& reduction, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, scan_op);
        // Broadcast value from the last thread in the warp
        reduction = warp_shuffle(output, VirtualWaveSize - 1, VirtualWaveSize);
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
        inclusive_scan(input, output, scan_op);
        // Broadcast value from the last thread in the warp
        reduction = warp_shuffle(output, VirtualWaveSize - 1, VirtualWaveSize);
        // Include init value in scan results
        output = scan_op(init, output);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(
        T input, T& output, T& reduction, storage_type& storage, BinaryFunction scan_op, T init)
    {
        (void)storage;
        inclusive_scan(input, output, reduction, scan_op, init);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, T init, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, scan_op);
        // Convert inclusive scan result to exclusive
        to_exclusive(output, output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, T init, storage_type& storage, BinaryFunction scan_op)
    {
        (void)storage; // disables unused parameter warning
        exclusive_scan(input, output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, storage_type& storage, BinaryFunction scan_op)
    {
        (void)storage; // disables unused parameter warning
        inclusive_scan(input, output, scan_op);
        // Convert inclusive scan result to exclusive
        to_exclusive(output, output);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(
        T input, T& output, storage_type& /*storage*/, T& reduction, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, scan_op);
        // Broadcast value from the last thread in the warp
        reduction = warp_shuffle(output, VirtualWaveSize - 1, VirtualWaveSize);
        // Convert inclusive scan result to exclusive
        to_exclusive(output, output);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, T init, T& reduction, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, scan_op);
        // Broadcast value from the last thread in the warp
        reduction = warp_shuffle(output, VirtualWaveSize - 1, VirtualWaveSize);
        // Convert inclusive scan result to exclusive
        to_exclusive(output, output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(
        T input, T& output, T init, T& reduction, storage_type& storage, BinaryFunction scan_op)
    {
        (void)storage;
        exclusive_scan(input, output, init, reduction, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scan(T input, T& inclusive_output, T& exclusive_output, T init, BinaryFunction scan_op)
    {
        inclusive_scan(input, inclusive_output, scan_op);
        // Convert inclusive scan result to exclusive
        to_exclusive(inclusive_output, exclusive_output, init, scan_op);
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
        (void)storage; // disables unused parameter warning
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
        (void)storage; // disables unused parameter warning
        inclusive_scan(input, inclusive_output, scan_op);
        // Convert inclusive scan result to exclusive
        to_exclusive(inclusive_output, exclusive_output);
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
        inclusive_scan(input, inclusive_output, scan_op);
        // Broadcast value from the last thread in the warp
        reduction = warp_shuffle(inclusive_output, VirtualWaveSize - 1, VirtualWaveSize);
        // Convert inclusive scan result to exclusive
        to_exclusive(inclusive_output, exclusive_output, init, scan_op);
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

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T broadcast(T input, const unsigned int src_lane, storage_type& storage)
    {
        (void)storage;

        if(VirtualWaveSize == ::rocprim::arch::wavefront::size())
        {
            return warp_readlane(input, warp_readfirstlane(src_lane));
        }

        return warp_shuffle(input, src_lane, VirtualWaveSize);
    }

private:
    // Changes inclusive scan results to exclusive scan results
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T inclusive_input, T& exclusive_output, T init, BinaryFunction scan_op)
    {
        // include init value in scan results
        exclusive_output = scan_op(init, inclusive_input);
        // get exclusive results
        exclusive_output = warp_shuffle_up(exclusive_output, 1, VirtualWaveSize);
        if(detail::logical_lane_id<VirtualWaveSize>() == 0)
        {
            exclusive_output = init;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T inclusive_input, T& exclusive_output)
    {
        // shift to get exclusive results
        exclusive_output = warp_shuffle_up(inclusive_input, 1, VirtualWaveSize);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SCAN_DPP_HPP_
