// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn-gpu-ref/detail/GpuRefHipError.hpp"

namespace hipdnn_gpu_ref::detail
{

inline void assertValidGridSize(int64_t xGridSize,
                                int64_t yGridSize,
                                int64_t zGridSize,
                                int64_t xMaxGridSizeDivisor = 1,
                                int64_t yMaxGridSizeDivisor = 1,
                                int64_t zMaxGridSizeDivisor = 1)
{
    int deviceId;
    detail::throwOnHipError(hipGetDevice(&deviceId), "hipGetDevice failed");
    hipDeviceProp_t deviceProps;
    detail::throwOnHipError(hipGetDeviceProperties(&deviceProps, deviceId),
                            "hipGetDeviceProperties failed");
    const auto maxXGridSize
        = static_cast<int64_t>(deviceProps.maxGridSize[0]) / xMaxGridSizeDivisor;
    const auto maxYGridSize
        = static_cast<int64_t>(deviceProps.maxGridSize[1]) / yMaxGridSizeDivisor;
    const auto maxZGridSize
        = static_cast<int64_t>(deviceProps.maxGridSize[2]) / zMaxGridSizeDivisor;
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
