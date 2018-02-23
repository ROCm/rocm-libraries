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

#ifndef ROCPRIM_HIPCUB_WARP_WARP_SCAN_HPP_
#define ROCPRIM_HIPCUB_WARP_WARP_SCAN_HPP_

#include "../../config.hpp"

#include "../intrinsics.hpp"
#include "../thread/thread_operators.hpp"

BEGIN_HIPCUB_NAMESPACE

template<
    typename T,
    int LOGICAL_WARP_THREADS = HIPCUB_WARP_THREADS,
    int ARCH = HIPCUB_ARCH>
class WarpScan : private ::rocprim::warp_scan<T, LOGICAL_WARP_THREADS>
{
    static_assert(LOGICAL_WARP_THREADS > 0, "LOGICAL_WARP_THREADS must be greater than 0");
    using base_type = typename ::rocprim::warp_scan<T, LOGICAL_WARP_THREADS>;

    typename base_type::storage_type &temp_storage;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    WarpScan(TempStorage& temp_storage) : temp_storage(temp_storage)
    {
    }

    HIPCUB_DEVICE inline
    void InclusiveSum(T input, T& inclusive_output)
    {
        base_type::inclusive_scan(input, inclusive_output, temp_storage);
    }

    HIPCUB_DEVICE inline
    void InclusiveSum(T input, T& inclusive_output, T& warp_aggregate)
    {
        base_type::inclusive_scan(input, inclusive_output, warp_aggregate, temp_storage);
    }

    HIPCUB_DEVICE inline
    void ExclusiveSum(T input, T& exclusive_output)
    {
        base_type::exclusive_scan(input, exclusive_output, T(0), temp_storage);
    }

    HIPCUB_DEVICE inline
    void ExclusiveSum(T input, T& exclusive_output, T& warp_aggregate)
    {
        base_type::exclusive_scan(input, exclusive_output, T(0), warp_aggregate, temp_storage);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op)
    {
        base_type::inclusive_scan(input, inclusive_output, temp_storage, scan_op);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op, T& warp_aggregate)
    {
        base_type::inclusive_scan(
            input, inclusive_output, warp_aggregate,
            temp_storage, scan_op
        );
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& exclusive_output, ScanOp scan_op)
    {
        base_type::inclusive_scan(input, exclusive_output, temp_storage, scan_op);
        base_type::to_exclusive(exclusive_output, exclusive_output, temp_storage);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& exclusive_output, T initial_value, ScanOp scan_op)
    {
        base_type::exclusive_scan(
            input, exclusive_output, initial_value,
            temp_storage, scan_op
        );
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& exclusive_output, ScanOp scan_op, T& warp_aggregate)
    {
        base_type::inclusive_scan(
            input, exclusive_output, warp_aggregate, temp_storage, scan_op
        );
        base_type::to_exclusive(exclusive_output, exclusive_output, temp_storage);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& exclusive_output, T initial_value, ScanOp scan_op, T& warp_aggregate)
    {
        base_type::exclusive_scan(
            input, exclusive_output, initial_value, warp_aggregate,
            temp_storage, scan_op
        );
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void Scan(T input, T& inclusive_output, T& exclusive_output, T initial_value, ScanOp scan_op)
    {
        base_type::scan(
            input, inclusive_output, exclusive_output, initial_value,
            temp_storage, scan_op
        );
    }
};

END_HIPCUB_NAMESPACE

#endif // ROCPRIM_HIPCUB_WARP_WARP_SCAN_HPP_
