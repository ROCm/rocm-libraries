// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE ADAPTATION: Copy as-is and update namespace. This utility function looks up device
// buffers by tensor UID and is used by all Plan::execute() implementations.

#pragma once

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace example_provider
{

/// Find a device buffer by tensor UID in the provided array.
/// Throws HipdnnPluginException if the UID is not found.
inline hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                                   const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                                   uint32_t numDeviceBuffers)
{
    for(uint32_t i = 0; i < numDeviceBuffers; i++)
    {
        if(uid == deviceBuffers[i].uid)
        {
            return deviceBuffers[i];
        }
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
        "Device buffer with the uid: " + std::to_string(uid)
            + " not found in the provided device buffers.");
}

} // namespace example_provider
