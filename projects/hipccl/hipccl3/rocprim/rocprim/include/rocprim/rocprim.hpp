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

#ifndef ROCPRIM_HPP_
#define ROCPRIM_HPP_

/// \file
///
/// Meta-header to include rocPRIM HC or HIP APIs.

// Meta configuration for rocPRIM
#include "config.hpp"

#include "rocprim_version.hpp"

#include "intrinsics.hpp"
#include "functional.hpp"
#include "types.hpp"
#include "type_traits.hpp"
#include "iterator.hpp"

#include "warp/warp_reduce.hpp"
#include "warp/warp_scan.hpp"
#include "warp/warp_sort.hpp"

#include "block/block_discontinuity.hpp"
#include "block/block_exchange.hpp"
#include "block/block_histogram.hpp"
#include "block/block_load.hpp"
#include "block/block_radix_sort.hpp"
#include "block/block_scan.hpp"
#include "block/block_sort.hpp"
#include "block/block_store.hpp"

#ifdef ROCPRIM_HC_API
    #include "device/device_histogram_hc.hpp"
    #include "device/device_merge_hc.hpp"
    #include "device/device_merge_sort_hc.hpp"
    #include "device/device_partition_hc.hpp"
    #include "device/device_radix_sort_hc.hpp"
    #include "device/device_reduce_by_key_hc.hpp"
    #include "device/device_reduce_hc.hpp"
    #include "device/device_run_length_encode_hc.hpp"
    #include "device/device_scan_by_key_hc.hpp"
    #include "device/device_scan_hc.hpp"
    #include "device/device_segmented_radix_sort_hc.hpp"
    #include "device/device_segmented_reduce_hc.hpp"
    #include "device/device_segmented_scan_hc.hpp"
    #include "device/device_select_hc.hpp"
    #include "device/device_transform_hc.hpp"
#else
    #include "device/device_histogram_hip.hpp"
    #include "device/device_merge_hip.hpp"
    #include "device/device_merge_sort_hip.hpp"
    #include "device/device_partition_hip.hpp"
    #include "device/device_radix_sort_hip.hpp"
    #include "device/device_reduce_by_key_hip.hpp"
    #include "device/device_reduce_hip.hpp"
    #include "device/device_run_length_encode_hip.hpp"
    #include "device/device_scan_by_key_hip.hpp"
    #include "device/device_scan_hip.hpp"
    #include "device/device_segmented_radix_sort_hip.hpp"
    #include "device/device_segmented_reduce_hip.hpp"
    #include "device/device_segmented_scan_hip.hpp"
    #include "device/device_select_hip.hpp"
    #include "device/device_transform_hip.hpp"
#endif

BEGIN_ROCPRIM_NAMESPACE

/// \brief Returns version of rocPRIM library.
/// \return version of rocPRIM library
ROCPRIM_HOST_DEVICE inline
unsigned int version()
{
    return ROCPRIM_VERSION;
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_HPP_
