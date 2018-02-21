// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

template<
    class T,
    unsigned int WarpSize
>
class warp_scan_shared_mem
{
public:
    struct storage_type
    {
        T threads[WarpSize];
    };

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input, T& output,
                        storage_type& storage, BinaryFunction scan_op)
    {
        const unsigned int lid = detail::logical_lane_id<WarpSize>();

        T me = input;
        store_volatile(&storage.threads[lid], me);
        for(unsigned int i = 1; i < WarpSize; i *= 2)
        {
            if(lid >= i)
            {
                T other = load_volatile(&storage.threads[lid - i]);
                me = scan_op(other, me);
                store_volatile(&storage.threads[lid], me);
            }
        }
        output = me;
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input, T& output, T& reduction,
                        storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, storage, scan_op);
        reduction = load_volatile(&storage.threads[WarpSize - 1]);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input, T& output, T init,
                        storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, storage, scan_op);
        to_exclusive(output, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input, T& output, T init, T& reduction,
                        storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, storage, scan_op);
        reduction = load_volatile(&storage.threads[WarpSize - 1]);
        to_exclusive(output, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void scan(T input, T& inclusive_output, T& exclusive_output, T init,
              storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, inclusive_output, storage, scan_op);
        to_exclusive(exclusive_output, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void scan(T input, T& inclusive_output, T& exclusive_output,
              storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, inclusive_output, storage, scan_op);
        to_exclusive(exclusive_output, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void scan(T input, T& inclusive_output, T& exclusive_output, T init, T& reduction,
              storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, inclusive_output, storage, scan_op);
        reduction = load_volatile(&storage.threads[WarpSize - 1]);
        to_exclusive(exclusive_output, init, storage, scan_op);
    }

private:

    // Calculate exclusive results base on inclusive scan results in storage.threads[].
    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void to_exclusive(T& exclusive_output, T init,
                      storage_type& storage, BinaryFunction scan_op)
    {
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        exclusive_output = init;
        if(lid != 0)
        {
            exclusive_output = scan_op(init, load_volatile(&storage.threads[lid-1]));
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void to_exclusive(T& exclusive_output, storage_type& storage)
    {
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        if(lid != 0)
        {
            exclusive_output = load_volatile(&storage.threads[lid-1]);
        }
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SCAN_SHARED_MEM_HPP_
