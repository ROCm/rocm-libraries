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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SCAN_SHARED_MEM_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SCAN_SHARED_MEM_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class T, unsigned int VirtualWaveSize>
class warp_scan_shared_mem
{
    struct storage_type_
    {
        T threads[VirtualWaveSize];
    };

public:
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<storage_type_>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, storage_type& storage, BinaryFunction scan_op)
    {
        output = common_inclusive_scan(scan_op, storage, input);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_inclusive_scan(
        T input, T& output, int valid_items, storage_type& storage, BinaryFunction scan_op)
    {
        const unsigned int lid      = detail::logical_lane_id<VirtualWaveSize>();
        const bool         is_valid = (static_cast<int>(lid) < valid_items);

        const T me = common_inclusive_scan(scan_op, storage, is_valid ? input : output, is_valid);
        if(is_valid)
        {
            output = me;
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, storage_type& storage, BinaryFunction scan_op, T init)
    {
        T me = common_inclusive_scan(scan_op, storage, input);
        include_init(me, output, scan_op, init);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_inclusive_scan(
        T input, T& output, int valid_items, storage_type& storage, BinaryFunction scan_op, T init)
    {
        T me = common_inclusive_scan(scan_op, storage, input);
        include_init(me, output, valid_items, scan_op, init);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(
        T input, T& output, T& reduction, storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, storage, scan_op);
        compute_reduction(reduction, storage);
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
        partial_inclusive_scan(input, output, valid_items, storage, scan_op);
        compute_reduction(valid_items, reduction, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(
        T input, T& output, T& reduction, storage_type& storage, BinaryFunction scan_op, T init)
    {
        inclusive_scan(input, output, storage, scan_op, init);
        ::rocprim::wave_barrier();
        compute_reduction(reduction, storage);
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
        partial_inclusive_scan(input, output, valid_items, storage, scan_op, init);
        ::rocprim::wave_barrier();
        compute_reduction(valid_items, reduction, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, T init, storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, storage, scan_op);
        to_exclusive(output, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_exclusive_scan(
        T input, T& output, int valid_items, T init, storage_type& storage, BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, output, valid_items, storage, scan_op);
        to_exclusive(output, valid_items, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, storage, scan_op);
        to_exclusive(output, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_exclusive_scan(
        T input, T& output, int valid_items, storage_type& storage, BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, output, valid_items, storage, scan_op);
        to_exclusive(output, valid_items, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(
        T input, T& output, storage_type& storage, T& reduction, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, storage, scan_op);
        compute_reduction(reduction, storage);
        to_exclusive(output, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_exclusive_scan(T              input,
                                T&             output,
                                int            valid_items,
                                storage_type&  storage,
                                T&             reduction,
                                BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, output, valid_items, storage, scan_op);
        compute_reduction(valid_items, reduction, storage);
        to_exclusive(output, valid_items, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(
        T input, T& output, T init, T& reduction, storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, storage, scan_op);
        compute_reduction(reduction, storage);
        to_exclusive(output, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_exclusive_scan(T              input,
                                T&             output,
                                int            valid_items,
                                T              init,
                                T&             reduction,
                                storage_type&  storage,
                                BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, output, valid_items, storage, scan_op);
        compute_reduction(valid_items, reduction, storage);
        to_exclusive(output, valid_items, init, storage, scan_op);
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
        inclusive_scan(input, inclusive_output, storage, scan_op);
        to_exclusive(exclusive_output, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_scan(T              input,
                      T&             inclusive_output,
                      T&             exclusive_output,
                      int            valid_items,
                      T              init,
                      storage_type&  storage,
                      BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, inclusive_output, valid_items, storage, scan_op);
        to_exclusive(exclusive_output, valid_items, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scan(T              input,
              T&             inclusive_output,
              T&             exclusive_output,
              storage_type&  storage,
              BinaryFunction scan_op)
    {
        inclusive_scan(input, inclusive_output, storage, scan_op);
        to_exclusive(exclusive_output, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_scan(T              input,
                      T&             inclusive_output,
                      T&             exclusive_output,
                      int            valid_items,
                      storage_type&  storage,
                      BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, inclusive_output, valid_items, storage, scan_op);
        to_exclusive(exclusive_output, valid_items, storage);
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
        inclusive_scan(input, inclusive_output, reduction, storage, scan_op);
        ::rocprim::wave_barrier();
        to_exclusive(exclusive_output, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void partial_scan(T              input,
                      T&             inclusive_output,
                      T&             exclusive_output,
                      int            valid_items,
                      T              init,
                      T&             reduction,
                      storage_type&  storage,
                      BinaryFunction scan_op)
    {
        partial_inclusive_scan(input, inclusive_output, valid_items, reduction, storage, scan_op);
        ::rocprim::wave_barrier();
        to_exclusive(exclusive_output, valid_items, init, storage, scan_op);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T broadcast(T input, const unsigned int src_lane, storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        if(src_lane == detail::logical_lane_id<VirtualWaveSize>())
        {
            storage_.threads[src_lane] = input;
        }
        ::rocprim::wave_barrier();
        return storage_.threads[src_lane];
    }

private:
    // Include init value in scan results
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void include_init(T input, T& output, BinaryFunction scan_op, T init)
    {
        // Apply the initial value. Do not write the result
        // of applying the initial value to memory.
        output = scan_op(init, input);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void include_init(T input, T& output, int valid_items, BinaryFunction scan_op, T init)
    {
        // Apply the initial value. Do not write the result
        // of applying the initial value to memory.
        const unsigned int lid = detail::logical_lane_id<VirtualWaveSize>();
        if(static_cast<int>(lid) < valid_items)
        {
            output = scan_op(init, input);
        }
    }

    // Calculate exclusive results base on inclusive scan results in storage.threads[].
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T& exclusive_output, T init, storage_type& storage, BinaryFunction scan_op)
    {
        const unsigned int lid      = detail::logical_lane_id<VirtualWaveSize>();
        storage_type_&     storage_ = storage.get();
        exclusive_output            = init;
        if(lid != 0)
        {
            exclusive_output = scan_op(init, storage_.threads[lid - 1]);
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T& exclusive_output, storage_type& storage)
    {
        const unsigned int lid      = detail::logical_lane_id<VirtualWaveSize>();
        storage_type_&     storage_ = storage.get();
        if(lid != 0)
        {
            exclusive_output = storage_.threads[lid - 1];
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(
        T& exclusive_output, int valid_items, T init, storage_type& storage, BinaryFunction scan_op)
    {
        const unsigned int lid      = detail::logical_lane_id<VirtualWaveSize>();
        storage_type_&     storage_ = storage.get();
        if(static_cast<int>(lid) < valid_items)
        {
            if(lid != 0)
            {
                exclusive_output = scan_op(init, storage_.threads[lid - 1]);
            }
            else
            {
                exclusive_output = init;
            }
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T& exclusive_output, int valid_items, storage_type& storage)
    {
        const unsigned int lid      = detail::logical_lane_id<VirtualWaveSize>();
        storage_type_&     storage_ = storage.get();
        if((lid != 0u) && (static_cast<int>(lid) < valid_items))
        {
            exclusive_output = storage_.threads[lid - 1];
        }
    }

    // Common inclusive scan code
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    T common_inclusive_scan(BinaryFunction scan_op,
                            storage_type&  storage,
                            T              me_init,
                            bool           extra_cond = true)
    {
        const unsigned int lid      = detail::logical_lane_id<VirtualWaveSize>();
        storage_type_&     storage_ = storage.get();

        T me                  = me_init;
        storage_.threads[lid] = me_init;
        ::rocprim::wave_barrier();
        for(unsigned int i = 1; i < VirtualWaveSize; i *= 2)
        {
            const bool do_op = (lid >= i) && extra_cond;
            if(do_op)
            {
                T other = storage_.threads[lid - i];
                me      = scan_op(other, me);
            }
            ::rocprim::wave_barrier();
            if(do_op)
            {
                storage_.threads[lid] = me;
            }
            ::rocprim::wave_barrier();
        }
        return me;
    }

    // Broadcast reduction from the last valid thread in the warp
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void compute_reduction(T& reduction, storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        reduction               = storage_.threads[VirtualWaveSize - 1];
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void compute_reduction(int valid_items, T& reduction, storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        const int      last_valid_src_lane
            = ::rocprim::clamp(valid_items - 1, 0, static_cast<int>(VirtualWaveSize - 1));
        reduction = storage_.threads[last_valid_src_lane];
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SCAN_SHARED_MEM_HPP_
