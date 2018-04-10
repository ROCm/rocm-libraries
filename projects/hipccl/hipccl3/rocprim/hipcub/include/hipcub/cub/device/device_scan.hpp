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

#ifndef HIPCUB_CUB_DEVICE_DEVICE_SCAN_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_SCAN_HPP_

#include "../../config.hpp"

#include <cub/device/device_scan.cuh>

BEGIN_HIPCUB_NAMESPACE

class DeviceScan
{
public:
    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t InclusiveSum(void *d_temp_storage,
                            size_t &temp_storage_bytes,
                            InputIteratorT d_in,
                            OutputIteratorT d_out,
                            int num_items,
                            hipStream_t stream = 0,
                            bool debug_synchronous = false)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceScan::InclusiveSum(
                d_temp_storage, temp_storage_bytes,
                d_in, d_out, num_items,
                stream, debug_synchronous
            )
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ScanOpT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t InclusiveScan(void *d_temp_storage,
                             size_t &temp_storage_bytes,
                             InputIteratorT d_in,
                             OutputIteratorT d_out,
                             ScanOpT scan_op,
                             int num_items,
                             hipStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceScan::InclusiveScan(
                d_temp_storage, temp_storage_bytes,
                d_in, d_out, scan_op, num_items,
                stream, debug_synchronous
            )
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ExclusiveSum(void *d_temp_storage,
                            size_t &temp_storage_bytes,
                            InputIteratorT d_in,
                            OutputIteratorT d_out,
                            int num_items,
                            hipStream_t stream = 0,
                            bool debug_synchronous = false)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceScan::ExclusiveSum(
                d_temp_storage, temp_storage_bytes,
                d_in, d_out, num_items,
                stream, debug_synchronous
            )
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ScanOpT,
        typename InitValueT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ExclusiveScan(void *d_temp_storage,
                             size_t &temp_storage_bytes,
                             InputIteratorT d_in,
                             OutputIteratorT d_out,
                             ScanOpT scan_op,
                             InitValueT init_value,
                             int num_items,
                             hipStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceScan::ExclusiveScan(
                d_temp_storage, temp_storage_bytes,
                d_in, d_out, scan_op, init_value, num_items,
                stream, debug_synchronous
            )
        );
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_SCAN_HPP_
