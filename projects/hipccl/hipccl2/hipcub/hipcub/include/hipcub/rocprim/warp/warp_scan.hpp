/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_WARP_WARP_SCAN_HPP_
#define HIPCUB_ROCPRIM_WARP_WARP_SCAN_HPP_

#include "../../config.hpp"

#include "../util_ptx.hpp"
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

    typename base_type::storage_type &temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    WarpScan(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    HIPCUB_DEVICE inline
    void InclusiveSum(T input, T& inclusive_output)
    {
        base_type::inclusive_scan(input, inclusive_output, temp_storage_);
    }

    HIPCUB_DEVICE inline
    void InclusiveSum(T input, T& inclusive_output, T& warp_aggregate)
    {
        base_type::inclusive_scan(input, inclusive_output, warp_aggregate, temp_storage_);
    }

    HIPCUB_DEVICE inline
    void ExclusiveSum(T input, T& exclusive_output)
    {
        base_type::exclusive_scan(input, exclusive_output, T(0), temp_storage_);
    }

    HIPCUB_DEVICE inline
    void ExclusiveSum(T input, T& exclusive_output, T& warp_aggregate)
    {
        base_type::exclusive_scan(input, exclusive_output, T(0), warp_aggregate, temp_storage_);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op)
    {
        base_type::inclusive_scan(input, inclusive_output, temp_storage_, scan_op);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op, T& warp_aggregate)
    {
        base_type::inclusive_scan(
            input, inclusive_output, warp_aggregate,
            temp_storage_, scan_op
        );
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& exclusive_output, ScanOp scan_op)
    {
        base_type::inclusive_scan(input, exclusive_output, temp_storage_, scan_op);
        base_type::to_exclusive(exclusive_output, exclusive_output, temp_storage_);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& exclusive_output, T initial_value, ScanOp scan_op)
    {
        base_type::exclusive_scan(
            input, exclusive_output, initial_value,
            temp_storage_, scan_op
        );
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& exclusive_output, ScanOp scan_op, T& warp_aggregate)
    {
        base_type::inclusive_scan(
            input, exclusive_output, warp_aggregate, temp_storage_, scan_op
        );
        base_type::to_exclusive(exclusive_output, exclusive_output, temp_storage_);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& exclusive_output, T initial_value, ScanOp scan_op, T& warp_aggregate)
    {
        base_type::exclusive_scan(
            input, exclusive_output, initial_value, warp_aggregate,
            temp_storage_, scan_op
        );
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void Scan(T input, T& inclusive_output, T& exclusive_output, ScanOp scan_op)
    {
        base_type::inclusive_scan(input, inclusive_output, temp_storage_, scan_op);
        base_type::to_exclusive(inclusive_output, exclusive_output, temp_storage_);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void Scan(T input, T& inclusive_output, T& exclusive_output, T initial_value, ScanOp scan_op)
    {
        base_type::scan(
            input, inclusive_output, exclusive_output, initial_value,
            temp_storage_, scan_op
        );
        // In CUB documentation it's unclear if inclusive_output should include initial_value,
        // however,the implementation includes initial_value in inclusive_output in WarpScan::Scan().
        // In rocPRIM it's not included, and this is a fix to match CUB implementation.
        // After confirmation from CUB's developers we will most probably change rocPRIM too.
        inclusive_output = scan_op(initial_value, inclusive_output);
    }

    HIPCUB_DEVICE inline
    T Broadcast(T input, unsigned int src_lane)
    {
        return base_type::broadcast(input, src_lane, temp_storage_);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_WARP_WARP_SCAN_HPP_
