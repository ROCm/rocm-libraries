/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

// Device-side comparison of a buffer with a reference copy, for hipblaslt-bench
// --post_bench_verify_iters. Each relaunch gets one RelaunchSlot in device memory, and nothing
// waits on the host between relaunches, so the relaunches run back to back.
//
// A slot can cover several buffers (the gemms of a grouped GEMM, or the batches of a
// pointer-array GEMM): each buffer is compared with an offset, the number of elements in the
// buffers before it, and `first` is an index into all of them in that order.
struct RelaunchSlot
{
    unsigned long long count; // elements whose bits differ from the reference
    unsigned long long first; // index of the first one; UINT64_MAX while count is 0
    unsigned long long bits[2]; // that element's bits, for elements up to 16 bytes
};

// Sets n slots to "no difference". Returns the launch error.
hipError_t relaunch_slots_reset_device(RelaunchSlot* slots, size_t n, hipStream_t stream);

// Adds the elements of d whose bits differ from ref to *slot. element_size is 1, 2, 4, 8 or 16;
// any other size returns hipErrorInvalidValue.
hipError_t relaunch_compare_device(const void*   d,
                                   const void*   ref,
                                   size_t        elements,
                                   size_t        element_size,
                                   uint64_t      offset,
                                   RelaunchSlot* slot,
                                   hipStream_t   stream);

// Copies the bits of the slot's first differing element into slot->bits when that element is
// in d, the buffer just compared with this offset. Call it after each relaunch_compare_device.
hipError_t relaunch_capture_first_device(const void*   d,
                                         size_t        elements,
                                         size_t        element_size,
                                         uint64_t      offset,
                                         RelaunchSlot* slot,
                                         hipStream_t   stream);
