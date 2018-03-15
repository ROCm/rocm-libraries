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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SCAN_SHUFFLE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SCAN_SHUFFLE_HPP_

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
class warp_scan_shuffle
{
public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    using storage_type = detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input, T& output, BinaryFunction scan_op)
    {
        output = input;

        T value;
        const unsigned int id = detail::logical_lane_id<WarpSize>();
        #pragma unroll
        for(unsigned int offset = 1; offset < WarpSize; offset *= 2)
        {
            value = warp_shuffle_up(output, offset, WarpSize);
            if(id >= offset) output = scan_op(value, output);
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input, T& output,
                        storage_type& storage, BinaryFunction scan_op)
    {
        (void) storage; // disables unused parameter warning
        inclusive_scan(input, output, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input, T& output, T& reduction,
                        BinaryFunction scan_op)
    {
        inclusive_scan(input, output, scan_op);
        // Broadcast value from the last thread in warp
        reduction = warp_shuffle(output, WarpSize-1, WarpSize);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input, T& output, T& reduction,
                        storage_type& storage, BinaryFunction scan_op)
    {
        (void) storage;
        inclusive_scan(input, output, reduction, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input, T& output, T init, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, scan_op);
        // Convert inclusive scan result to exclusive
        to_exclusive(output, output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input, T& output, T init,
                        storage_type& storage, BinaryFunction scan_op)
    {
        (void) storage; // disables unused parameter warning
        exclusive_scan(input, output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input, T& output,
                        storage_type& storage, BinaryFunction scan_op)
    {
        (void) storage; // disables unused parameter warning
        inclusive_scan(input, output, scan_op);
        // Convert inclusive scan result to exclusive
        to_exclusive(output, output);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input, T& output, T init, T& reduction,
                        BinaryFunction scan_op)
    {
        inclusive_scan(input, output, scan_op);
        // Broadcast value from the last thread in warp
        reduction = warp_shuffle(output, WarpSize-1, WarpSize);
        // Convert inclusive scan result to exclusive
        to_exclusive(output, output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input, T& output, T init, T& reduction,
                        storage_type& storage, BinaryFunction scan_op)
    {
        (void) storage;
        exclusive_scan(input, output, init, reduction, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void scan(T input, T& inclusive_output, T& exclusive_output, T init,
              BinaryFunction scan_op)
    {
        inclusive_scan(input, inclusive_output, scan_op);
        // Convert inclusive scan result to exclusive
        to_exclusive(inclusive_output, exclusive_output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void scan(T input, T& inclusive_output, T& exclusive_output, T init,
              storage_type& storage, BinaryFunction scan_op)
    {
        (void) storage; // disables unused parameter warning
        scan(input, inclusive_output, exclusive_output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void scan(T input, T& inclusive_output, T& exclusive_output,
              storage_type& storage, BinaryFunction scan_op)
    {
        (void) storage; // disables unused parameter warning
        inclusive_scan(input, inclusive_output, scan_op);
        // Convert inclusive scan result to exclusive
        to_exclusive(inclusive_output, exclusive_output);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void scan(T input, T& inclusive_output, T& exclusive_output, T init, T& reduction,
              BinaryFunction scan_op)
    {
        inclusive_scan(input, inclusive_output, scan_op);
        // Broadcast value from the last thread in warp
        reduction = warp_shuffle(inclusive_output, WarpSize-1, WarpSize);
        // Convert inclusive scan result to exclusive
        to_exclusive(inclusive_output, exclusive_output, init, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void scan(T input, T& inclusive_output, T& exclusive_output, T init, T& reduction,
              storage_type& storage, BinaryFunction scan_op)
    {
        (void) storage;
        scan(input, inclusive_output, exclusive_output, init, reduction, scan_op);
    }

    ROCPRIM_DEVICE inline
    T broadcast(T input, const unsigned int src_lane, storage_type& storage)
    {
        (void) storage;
        return warp_shuffle(input, src_lane, WarpSize);
    }

protected:
    ROCPRIM_DEVICE inline
    void to_exclusive(T inclusive_input, T& exclusive_output, storage_type& storage)
    {
        (void) storage;
        return to_exclusive(inclusive_input, exclusive_output);
    }

private:
    // Changes inclusive scan results to exclusive scan results
    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void to_exclusive(T inclusive_input, T& exclusive_output, T init,
                      BinaryFunction scan_op)
    {
        // include init value in scan results
        exclusive_output = scan_op(init, inclusive_input);
        // get exclusive results
        exclusive_output = warp_shuffle_up(exclusive_output, 1, WarpSize);
        if(detail::logical_lane_id<WarpSize>() == 0)
        {
            exclusive_output = init;
        }
    }

    ROCPRIM_DEVICE inline
    void to_exclusive(T inclusive_input, T& exclusive_output)
    {
        // shift to get exclusive results
        exclusive_output = warp_shuffle_up(inclusive_input, 1, WarpSize);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SCAN_SHUFFLE_HPP_
