// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <map>
#include <mutex>
#include <string>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>

namespace hipdnn_plugin_sdk::heuristics
{

/// @brief Resolves the stream's device and memoizes immutable hardware properties.
///
/// Default-stream tokens (null, hipStreamLegacy, hipStreamPerThread) name the calling
/// thread's current device, not a stream object; getDeviceFromStream() is the one place
/// that knows that. Handing hipStreamLegacy to hipStreamGetDevice() instead makes HIP
/// dereference the token as a stream.
inline const hipDeviceProp_t& predictionDevice(hipStream_t stream)
{
    // -1, not 0: a query that reports success without writing the ordinal must be caught
    // below, not silently become device 0's properties.
    hipDevice_t device = -1;
    const auto status = getDeviceFromStream(stream, &device);
    if(status != hipSuccess || device < 0)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "Cannot resolve prediction device from the execution stream: "
                                        + std::to_string(status) + ", device "
                                        + std::to_string(device));
    }
    static std::mutex s_mutex;
    static std::map<int, hipDeviceProp_t> s_properties;
    const std::lock_guard<std::mutex> lock(s_mutex);
    if(const auto found = s_properties.find(device); found != s_properties.end())
    {
        return found->second;
    }
    hipDeviceProp_t queried{};
    if(hipGetDeviceProperties(&queried, device) != hipSuccess)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "Cannot query prediction device properties");
    }
    return s_properties.emplace(device, queried).first->second;
}

} // namespace hipdnn_plugin_sdk::heuristics

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
