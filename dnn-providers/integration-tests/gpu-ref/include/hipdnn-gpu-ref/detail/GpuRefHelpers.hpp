// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn-gpu-ref/detail/GpuRefHipError.hpp"

namespace hipdnn_gpu_ref::detail
{

inline void assertValidGridSize(int64_t xGridSize, int64_t yGridSize, int64_t zGridSize)
{
    int deviceId;
    detail::throwOnHipError(hipGetDevice(&deviceId), "hipGetDevice failed");
    hipDeviceProp_t deviceProps;
    detail::throwOnHipError(hipGetDeviceProperties(&deviceProps, deviceId),
                            "hipGetDeviceProperties failed");

    if(xGridSize <= 0 || yGridSize <= 0 || zGridSize <= 0)
    {
        throw std::runtime_error("grid sizes must be positive");
    }

    const auto maxXGridSize = static_cast<int64_t>(deviceProps.maxGridSize[0]);
    const auto maxYGridSize = static_cast<int64_t>(deviceProps.maxGridSize[1]);
    const auto maxZGridSize = static_cast<int64_t>(deviceProps.maxGridSize[2]);
    if(xGridSize > maxXGridSize)
    {
        throw std::runtime_error("X grid size exceeds device limit: " + std::to_string(xGridSize)
                                 + " > " + std::to_string(maxXGridSize));
    }
    if(yGridSize > maxYGridSize)
    {
        throw std::runtime_error("Y grid size exceeds device limit: " + std::to_string(yGridSize)
                                 + " > " + std::to_string(maxYGridSize));
    }
    if(zGridSize > maxZGridSize)
    {
        throw std::runtime_error("Z grid size exceeds device limit: " + std::to_string(zGridSize)
                                 + " > " + std::to_string(maxZGridSize));
    }
}

} // namespace hipdnn_gpu_ref::detail
