/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/
#include "relaunch_compare.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <limits>

namespace
{
    constexpr unsigned long long kNoDifference = std::numeric_limits<unsigned long long>::max();

    __global__ void reset_kernel(RelaunchSlot* slots, size_t n)
    {
        const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
        if(i < n)
            slots[i] = {0, kNoDifference, {0, 0}};
    }

    // W is the word compared; an element is WORDS words.
    template <typename W, int WORDS>
    __global__ void compare_kernel(
        const W* d, const W* ref, uint64_t elements, uint64_t offset, RelaunchSlot* slot)
    {
        unsigned long long count = 0;
        unsigned long long first = kNoDifference;
        const uint64_t     step  = uint64_t(gridDim.x) * blockDim.x;
        for(uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < elements; i += step)
        {
            bool same = true;
            for(int w = 0; w < WORDS; w++)
                same = same && d[i * WORDS + w] == ref[i * WORDS + w];
            if(!same)
            {
                if(count == 0)
                    first = offset + i;
                count++;
            }
        }
        if(count)
        {
            atomicAdd(&slot->count, count);
            atomicMin(&slot->first, first);
        }
    }

    __global__ void capture_kernel(const unsigned char* d,
                                   uint64_t             elements,
                                   size_t               element_size,
                                   uint64_t             offset,
                                   RelaunchSlot*        slot)
    {
        const unsigned long long first = slot->first;
        if(first == kNoDifference || first < offset || first - offset >= elements)
            return;
        const unsigned char* p   = d + (first - offset) * element_size;
        unsigned char*       out = reinterpret_cast<unsigned char*>(slot->bits);
        for(size_t b = 0; b < element_size; b++)
            out[b] = p[b];
    }

    template <typename W, int WORDS>
    hipError_t launch_compare(const void*   d,
                              const void*   ref,
                              size_t        elements,
                              uint64_t      offset,
                              RelaunchSlot* slot,
                              hipStream_t   stream)
    {
        if(elements == 0)
            return hipSuccess;
        constexpr int threads = 256;
        const auto    blocks = unsigned(std::min<size_t>((elements + threads - 1) / threads, 4096));
        hipLaunchKernelGGL((compare_kernel<W, WORDS>),
                           dim3(blocks),
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const W*>(d),
                           static_cast<const W*>(ref),
                           uint64_t(elements),
                           offset,
                           slot);
        return hipGetLastError();
    }
}

hipError_t relaunch_slots_reset_device(RelaunchSlot* slots, size_t n, hipStream_t stream)
{
    if(n == 0)
        return hipSuccess;
    constexpr int threads = 256;
    hipLaunchKernelGGL(reset_kernel,
                       dim3(unsigned((n + threads - 1) / threads)),
                       dim3(threads),
                       0,
                       stream,
                       slots,
                       n);
    return hipGetLastError();
}

hipError_t relaunch_compare_device(const void*   d,
                                   const void*   ref,
                                   size_t        elements,
                                   size_t        element_size,
                                   uint64_t      offset,
                                   RelaunchSlot* slot,
                                   hipStream_t   stream)
{
    switch(element_size)
    {
    case 1:
        return launch_compare<uint8_t, 1>(d, ref, elements, offset, slot, stream);
    case 2:
        return launch_compare<uint16_t, 1>(d, ref, elements, offset, slot, stream);
    case 4:
        return launch_compare<uint32_t, 1>(d, ref, elements, offset, slot, stream);
    case 8:
        return launch_compare<uint64_t, 1>(d, ref, elements, offset, slot, stream);
    case 16:
        return launch_compare<uint64_t, 2>(d, ref, elements, offset, slot, stream);
    default:
        return hipErrorInvalidValue;
    }
}

hipError_t relaunch_capture_first_device(const void*   d,
                                         size_t        elements,
                                         size_t        element_size,
                                         uint64_t      offset,
                                         RelaunchSlot* slot,
                                         hipStream_t   stream)
{
    if(element_size > sizeof(RelaunchSlot::bits))
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(capture_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       static_cast<const unsigned char*>(d),
                       uint64_t(elements),
                       element_size,
                       offset,
                       slot);
    return hipGetLastError();
}
