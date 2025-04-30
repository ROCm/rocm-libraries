// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_DEVICE_PROPS_H
#define ROCFFT_DEVICE_PROPS_H

#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <stdexcept>

// get device properties
static hipDeviceProp_t get_curr_device_prop()
{
    hipDeviceProp_t prop;
    int             deviceId = 0;
    if(hipGetDevice(&deviceId) != hipSuccess)
        throw std::runtime_error("hipGetDevice failed.");

    if(hipGetDeviceProperties(&prop, deviceId) != hipSuccess)
        throw std::runtime_error("hipGetDeviceProperties failed for deviceId "
                                 + std::to_string(deviceId));

    return prop;
}

// check that the given grid/block dims will fit into the limits in
// the device properties.  throws std::runtime_error if the limits
// are exceeded.
static void launch_limits_check(const std::string&     kernel_name,
                                const dim3             gridDim,
                                const dim3             blockDim,
                                const hipDeviceProp_t& deviceProp)
{
    // Need lots of casting here because dim3 is unsigned but device
    // props are signed.  Cast direct comparisons to fix signedness
    // issues.  Promote types to 64-bit when multiplying to try to
    // avoid overflow.

    // Block limits along each dimension
    if(blockDim.x > static_cast<uint32_t>(deviceProp.maxThreadsDim[0])
       || blockDim.y > static_cast<uint32_t>(deviceProp.maxThreadsDim[1])
       || blockDim.z > static_cast<uint32_t>(deviceProp.maxThreadsDim[2]))
        throw std::runtime_error("max threads per dim exceeded: " + kernel_name);

    // Total threads for the whole block
    if(static_cast<uint64_t>(blockDim.x) * blockDim.y * blockDim.z
       > static_cast<uint64_t>(deviceProp.maxThreadsPerBlock))
        throw std::runtime_error("max threads per block exceeded: " + kernel_name);

    // Grid dimension limits
    if(gridDim.x > static_cast<uint32_t>(deviceProp.maxGridSize[0])
       || gridDim.y > static_cast<uint32_t>(deviceProp.maxGridSize[1])
       || gridDim.z > static_cast<uint32_t>(deviceProp.maxGridSize[2]))
        throw std::runtime_error("max grid size exceeded: " + kernel_name);
}

#endif
