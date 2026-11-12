// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::device
{

/// The ordinal a launch on @p stream will run on.
///
/// Deliberately the same rule HandleDeviceResolver::deviceId applies at plan build: a
/// concrete stream names its own device, and anything else -- a default token, a query
/// that fails, a runtime that reports hipSuccess without writing the out-parameter --
/// falls through to the current device. Resolving a launch by a stricter rule than the
/// one that built the plan would let a plan build and then refuse every execute.
///
/// Seeded to -1 and negatives rejected: at 0 an unwritten out-parameter reads as device
/// zero and goes unseen on a single-device host.
inline int launchDeviceOrdinal(hipStream_t stream)
{
    int deviceOrdinal = -1;
    if(!hipdnn_plugin_sdk::isDefaultStream(stream)
       && hipdnn_plugin_sdk::getDeviceFromStream(stream, &deviceOrdinal) == hipSuccess
       && deviceOrdinal >= 0)
    {
        return deviceOrdinal;
    }

    deviceOrdinal = -1;
    const hipError_t status = hipGetDevice(&deviceOrdinal);
    if(status != hipSuccess || deviceOrdinal < 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "cannot resolve the device to launch on from the handle's stream: "
                + std::string(hipGetErrorString(status)) + ", ordinal "
                + std::to_string(deviceOrdinal));
    }
    return deviceOrdinal;
}

} // namespace hip_kernel_provider::device
