// Copyright (c) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
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
        output = input;
        common_inclusive_scan(output, scan_op);
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
    void partial_inclusive_scan(T input, T& output, int valid_items, BinaryFunction scan_op)
    {
        const unsigned int lane_id  = detail::logical_lane_id<VirtualWaveSize>();
        const bool         is_valid = (static_cast<int>(lane_id) < valid_items);

        if(is_valid)
        {
            output = input;
        }

        common_inclusive_scan(output, scan_op, is_valid);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_inclusive_scan(
        T input, T& output, int valid_items, storage_type& storage, BinaryFunction scan_op)
    {
        (void)storage; // disables unused parameter warning
        partial_inclusive_scan(input, output, valid_items, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, BinaryFunction scan_op, T init)
    {
        inclusive_scan(input, output, scan_op);
        include_init(output, output, scan_op, init);
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
    void partial_inclusive_scan(T input, T& output, int valid_items, BinaryFunction scan_op, T init)
    {
        partial_inclusive_scan(input, output, valid_items, scan_op);
        include_init(output, output, valid_items, scan_op, init);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_inclusive_scan(
        T input, T& output, int valid_items, storage_type& storage, BinaryFunction scan_op, T init)
    {
        (void)storage; // disables unused parameter warning
        partial_inclusive_scan(input, output, valid_items, scan_op, init);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, T& reduction, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, scan_op);
        compute_reduction(output, reduction);
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
    void partial_inclusive_scan(
        T input, T& output, int valid_items, T& reduction, BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, output, valid_items, scan_op);
        compute_reduction(output, valid_items, reduction);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_inclusive_scan(T              input,
                                T&             output,
                                int            valid_items,
                                T&             reduction,
                                storage_type&  storage,
                                BinaryFunction scan_op)
    {
        (void)storage; // disables unused parameter warning
        partial_inclusive_scan(input, output, valid_items, reduction, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, T& reduction, BinaryFunction scan_op, T init)
    {
        inclusive_scan(input, output, scan_op);
        compute_reduction(output, reduction);
        include_init(output, output, scan_op, init);
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
    void partial_inclusive_scan(
        T input, T& output, int valid_items, T& reduction, BinaryFunction scan_op, T init)
    {
        partial_inclusive_scan(input, output, valid_items, scan_op);
        compute_reduction(output, valid_items, reduction);
        include_init(output, output, valid_items, scan_op, init);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_inclusive_scan(T              input,
                                T&             output,
                                int            valid_items,
                                T&             reduction,
                                storage_type&  storage,
                                BinaryFunction scan_op,
                                T              init)
    {
        (void)storage; // disables unused parameter warning
        partial_inclusive_scan(input, output, valid_items, reduction, scan_op, init);
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
    void partial_exclusive_scan(T input, T& output, int valid_items, T init, BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, output, valid_items, scan_op);
        to_exclusive(output, output, valid_items, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_exclusive_scan(T   input,
                                T&  output,
                                int valid_items,
                                T   init,
                                storage_type& /*storage*/,
                                BinaryFunction scan_op)
    {
        partial_exclusive_scan(input, output, valid_items, init, scan_op);
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
    void partial_exclusive_scan(
        T input, T& output, int valid_items, storage_type& /*storage*/, BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, output, valid_items, scan_op);
        to_exclusive(output, output, valid_items);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(
        T input, T& output, storage_type& /*storage*/, T& reduction, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, scan_op);
        compute_reduction(output, reduction);
        // Convert inclusive scan result to exclusive
        to_exclusive(output, output);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_exclusive_scan(T   input,
                                T&  output,
                                int valid_items,
                                storage_type& /*storage*/,
                                T&             reduction,
                                BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, output, valid_items, scan_op);
        compute_reduction(output, valid_items, reduction);
        to_exclusive(output, output, valid_items);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, T init, T& reduction, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, scan_op);
        compute_reduction(output, reduction);
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
    void partial_exclusive_scan(
        T input, T& output, int valid_items, T init, T& reduction, BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, output, valid_items, scan_op);
        compute_reduction(output, valid_items, reduction);
        to_exclusive(output, output, valid_items, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_exclusive_scan(T   input,
                                T&  output,
                                int valid_items,
                                T   init,
                                T&  reduction,
                                storage_type& /*storage*/,
                                BinaryFunction scan_op)
    {
        partial_exclusive_scan(input, output, valid_items, init, reduction, scan_op);
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
    void partial_scan(T              input,
                      T&             inclusive_output,
                      T&             exclusive_output,
                      int            valid_items,
                      T              init,
                      BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, inclusive_output, valid_items, scan_op);
        to_exclusive(inclusive_output, exclusive_output, valid_items, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_scan(T   input,
                      T&  inclusive_output,
                      T&  exclusive_output,
                      int valid_items,
                      T   init,
                      storage_type& /*storage*/,
                      BinaryFunction scan_op)
    {
        partial_scan(input, inclusive_output, exclusive_output, valid_items, init, scan_op);
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
    void partial_scan(T   input,
                      T&  inclusive_output,
                      T&  exclusive_output,
                      int valid_items,
                      storage_type& /*storage*/,
                      BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, inclusive_output, valid_items, scan_op);
        to_exclusive(inclusive_output, exclusive_output, valid_items);
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
        inclusive_scan(input, inclusive_output, reduction, scan_op);
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

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_scan(T              input,
                      T&             inclusive_output,
                      T&             exclusive_output,
                      int            valid_items,
                      T              init,
                      T&             reduction,
                      BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, inclusive_output, valid_items, reduction, scan_op);
        to_exclusive(inclusive_output, exclusive_output, valid_items, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_scan(T   input,
                      T&  inclusive_output,
                      T&  exclusive_output,
                      int valid_items,
                      T   init,
                      T&  reduction,
                      storage_type& /*storage*/,
                      BinaryFunction scan_op)
    {
        partial_scan(input,
                     inclusive_output,
                     exclusive_output,
                     valid_items,
                     init,
                     reduction,
                     scan_op);
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
    // Include init value in scan results
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void include_init(T input, T& output, BinaryFunction scan_op, T init)
    {
        output = scan_op(init, input);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void include_init(T input, T& output, int valid_items, BinaryFunction scan_op, T init)
    {
        const unsigned int lid = detail::logical_lane_id<VirtualWaveSize>();
        if(static_cast<int>(lid) < valid_items)
        {
            output = scan_op(init, input);
        }
    }

    // Changes inclusive scan results to exclusive scan results
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T inclusive_input, T& exclusive_output, T init, BinaryFunction scan_op)
    {
        include_init(inclusive_input, exclusive_output, scan_op, init);
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

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T inclusive_input, T& exclusive_output, int valid_items)
    {
        const unsigned int id = detail::logical_lane_id<VirtualWaveSize>();
        T                  me = warp_shuffle_up(inclusive_input, 1, VirtualWaveSize);
        if(static_cast<int>(id) < valid_items)
        {
            exclusive_output = me;
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(
        T inclusive_input, T& exclusive_output, int valid_items, T init, BinaryFunction scan_op)
    {
        const unsigned int id = detail::logical_lane_id<VirtualWaveSize>();
        include_init(inclusive_input, exclusive_output, valid_items, scan_op, init);
        T me = warp_shuffle_up(exclusive_output, 1, VirtualWaveSize);
        if(static_cast<int>(id) < valid_items)
        {
            if(id != 0)
            {
                exclusive_output = me;
            }
            else
            {
                exclusive_output = init;
            }
        }
    }

    // Common inclusive scan code
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void common_inclusive_scan(T& output, BinaryFunction scan_op, bool extra_cond = true)
    {
        const unsigned int     lane_id              = ::rocprim::lane_id();
        constexpr unsigned int BoundVirtualWaveSize = rocprim::min(16u, VirtualWaveSize);
        const unsigned int     row_lane_id = detail::logical_lane_id<BoundVirtualWaveSize>(lane_id);

        if(VirtualWaveSize > 1)
        {
            T t = warp_move_dpp<T, 0x111>(output); // row_shr:1
            if(row_lane_id >= 1 && extra_cond)
                output = scan_op(t, output);
        }
        if(VirtualWaveSize > 2)
        {
            T t = warp_move_dpp<T, 0x112>(output); // row_shr:2
            if(row_lane_id >= 2 && extra_cond)
                output = scan_op(t, output);
        }
        if(VirtualWaveSize > 4)
        {
            T t = warp_move_dpp<T, 0x114>(output); // row_shr:4
            if(row_lane_id >= 4 && extra_cond)
                output = scan_op(t, output);
        }
        if(VirtualWaveSize > 8)
        {
            T t = warp_move_dpp<T, 0x118>(output); // row_shr:8
            if(row_lane_id >= 8 && extra_cond)
                output = scan_op(t, output);
        }

        // Check for __builtin_amdgcn_permlane16; if it exists, the DPP equivalent is not available.
        // Swizzle is kept instead of __builtin_amdgcn_permlanex16, as the latter can be slower in some cases.
        if(ROCPRIM_HAS_PERMLANE())
        {
            if(VirtualWaveSize > 16)
            {
                T t = warp_swizzle<T, 0x1e0>(output); // row_bcast:15
                if(lane_id % 32 >= 16 && extra_cond)
                    output = scan_op(t, output);
            }

            if constexpr(VirtualWaveSize > 32)
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "VirtualWaveSize > 32 is not supported without DPP broadcasts");
                return;
            }
        }
        else
        {
            if(VirtualWaveSize > 16)
            {
                T t = warp_move_dpp<T, 0x142>(output); // row_bcast:15
                if(lane_id % 32 >= 16 && extra_cond)
                    output = scan_op(t, output);
            }
            if(VirtualWaveSize > 32)
            {
                T t = warp_move_dpp<T, 0x143>(output); // row_bcast:31
                if(lane_id >= 32 && extra_cond)
                    output = scan_op(t, output);
            }
#if !ROCPRIM_TARGET_SPIRV
            static_assert(VirtualWaveSize <= 64, "VirtualWaveSize > 64 is not supported");
#endif
        }
    }

    // Broadcast reduction from the last valid thread in the warp
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void compute_reduction(T& output, T& reduction)
    {
        reduction = warp_shuffle(output, VirtualWaveSize - 1, VirtualWaveSize);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void compute_reduction(T& output, int valid_items, T& reduction)
    {
        const int last_valid_src_lane
            = ::rocprim::clamp(valid_items - 1, 0, static_cast<int>(VirtualWaveSize - 1));
        reduction = warp_shuffle(output, last_valid_src_lane, VirtualWaveSize);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SCAN_DPP_HPP_
