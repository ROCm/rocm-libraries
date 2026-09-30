// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file PluginDeviceBuffers.hpp
 * @brief Shared plugin-side helpers for the execute-time device-buffer array.
 *
 * The host hands a plugin its execute-time buffers as a flat
 * `hipdnnPluginDeviceBuffer_t[]` keyed by tensor uid. `findDeviceBuffer` is the
 * generic uid -> buffer lookup every provider needs; it is not specific to any
 * one feature.
 */

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hipdnn_plugin_sdk
{

/// @brief Linear-scans `deviceBuffers` for the entry matching `uid`.
/// @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) if absent.
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

    throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                "Device buffer with the uid: " + std::to_string(uid)
                                    + " not found in the provided device buffers.");
}

} // namespace hipdnn_plugin_sdk
