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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SCAN_SHUFFLE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SCAN_SHUFFLE_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../intrinsics/warp_shuffle.hpp"
#include "../../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class T, unsigned int VirtualWaveSize>
class warp_scan_shuffle
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
        const unsigned int id       = detail::logical_lane_id<VirtualWaveSize>();
        const bool         is_valid = (static_cast<int>(id) < valid_items);

        if(is_valid)
        {
            output = input;
        }

        common_inclusive_scan(output, scan_op, is_valid);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_inclusive_scan(
        T input, T& output, int valid_items, storage_type& /*storage*/, BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, output, valid_items, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, BinaryFunction scan_op, T init)
    {
        output = input;

        common_inclusive_scan(output, scan_op);
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
        const unsigned int id       = detail::logical_lane_id<VirtualWaveSize>();
        const bool         is_valid = (static_cast<int>(id) < valid_items);

        if(is_valid)
        {
            output = input;
        }

        common_inclusive_scan(output, scan_op, is_valid);
        include_init(output, output, valid_items, scan_op, init);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_inclusive_scan(T   input,
                                T&  output,
                                int valid_items,
                                storage_type& /*storage*/,
                                BinaryFunction scan_op,
                                T              init)
    {
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
    void partial_inclusive_scan(T   input,
                                T&  output,
                                int valid_items,
                                T&  reduction,
                                storage_type& /*storage*/,
                                BinaryFunction scan_op)
    {
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
    void partial_inclusive_scan(T   input,
                                T&  output,
                                int valid_items,
                                T&  reduction,
                                storage_type& /*storage*/,
                                BinaryFunction scan_op,
                                T              init)
    {
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
        to_exclusive(exclusive_output, exclusive_output);
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

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T inclusive_input, T& exclusive_output, int valid_items)
    {
        const unsigned int id = detail::logical_lane_id<VirtualWaveSize>();
        // shift to get exclusive results
        T me = warp_shuffle_up(inclusive_input, 1, VirtualWaveSize);
        if(static_cast<int>(id) < valid_items)
        {
            exclusive_output = me;
        }
    }

    // Common inclusive scan code
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void common_inclusive_scan(T& output, BinaryFunction scan_op, bool extra_cond = true)
    {
        T                  value;
        const unsigned int id = detail::logical_lane_id<VirtualWaveSize>();
        ROCPRIM_UNROLL
        for(unsigned int offset = 1; offset < VirtualWaveSize; offset *= 2)
        {
            value = warp_shuffle_up(output, offset, VirtualWaveSize);
            if(id >= offset && extra_cond)
                output = scan_op(value, output);
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

#endif // ROCPRIM_WARP_DETAIL_WARP_SCAN_SHUFFLE_HPP_
