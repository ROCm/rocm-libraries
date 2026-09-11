// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <mutex>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>

namespace hipdnn_plugin_sdk::heuristics
{

/// @brief Resolves the stream's device and memoizes immutable hardware properties.
inline const hipDeviceProp_t& predictionDevice(hipStream_t stream)
{
    int device = 0;
    const auto status
        = stream == nullptr ? hipGetDevice(&device) : hipStreamGetDevice(stream, &device);
    if(status != hipSuccess)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "Cannot resolve prediction device from the execution stream");
    }
    static std::mutex mutex;
    static std::map<int, hipDeviceProp_t> properties;
    const std::lock_guard<std::mutex> lock(mutex);
    if(const auto found = properties.find(device); found != properties.end())
    {
        return found->second;
    }
    hipDeviceProp_t queried{};
    if(hipGetDeviceProperties(&queried, device) != hipSuccess)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "Cannot query prediction device properties");
    }
    return properties.emplace(device, queried).first->second;
}

} // namespace hipdnn_plugin_sdk::heuristics
