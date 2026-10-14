// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file DeviceQuery.hpp
 * @brief Plugin-side HIP device queries shared across plugins so each does not
 * maintain its own implementation.
 *
 * getDeviceArch returns the RAW gcnArchName (e.g. "gfx942:sramecc+:xnack-"),
 * suffix intact. Match against it with hipdnn_plugin_sdk::archMatches
 * (ArchMatch.hpp) rather than string-comparing directly, so the ':' feature
 * suffix is handled consistently.
 */

#include <string>

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hipdnn_plugin_sdk
{

/// @brief Whether a stream token is relative to the calling thread's current device.
inline bool isDefaultStream(hipStream_t stream)
{
    return stream == nullptr || stream == hipStreamLegacy || stream == hipStreamPerThread;
}

/// @brief Query the live current device for default tokens, or a concrete stream's owner.
/// Returns the HIP query status without changing the caller's output policy.
inline hipError_t getDeviceFromStream(hipStream_t stream, hipDevice_t* deviceId)
{
    if(isDefaultStream(stream))
    {
        return hipGetDevice(deviceId);
    }
    return hipStreamGetDevice(stream, deviceId);
}

/// @brief Raw gcnArchName (suffix intact) of the device backing the given HIP
/// stream. Default tokens resolve the calling thread's live current device;
/// concrete streams resolve their owning device. Match the result with archMatches().
/// @throws HipdnnPluginException on HIP failure.
inline std::string getDeviceArch(hipStream_t stream)
{
    hipDevice_t deviceId = -1;
    auto status = getDeviceFromStream(stream, &deviceId);
    if(status != hipSuccess)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "Stream device query failed: " + std::to_string(status));
    }
    hipDeviceProp_t props;
    status = hipGetDeviceProperties(&props, deviceId);
    if(status != hipSuccess)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "hipGetDeviceProperties failed: " + std::to_string(status));
    }
    return {props.gcnArchName};
}

} // namespace hipdnn_plugin_sdk
