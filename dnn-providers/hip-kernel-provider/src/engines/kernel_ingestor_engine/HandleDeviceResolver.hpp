// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <mutex>
#include <string>
#include <unordered_map>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>

#include "core/Handle.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Resolves a call's device from the stream its handle carries; a handle can be
/// rebound between calls, so this reads per call rather than caching per-thread.
/// Successful device-property queries are cached for this resolver's lifetime.
class HandleDeviceResolver : public hipdnn_plugin_sdk::ingestor::IDeviceResolver<Handle>
{
public:
    hipdnn_plugin_sdk::ingestor::DeviceId deviceId(const Handle& handle) const override
    {
        int deviceId = 0;

        // Null stream: default stream belongs to the current device.
        if(handle.getStream() != nullptr)
        {
            if(hipStreamGetDevice(handle.getStream(), &deviceId) == hipSuccess)
            {
                return deviceId;
            }
        }

        if(hipGetDevice(&deviceId) != hipSuccess)
        {
            return hipdnn_plugin_sdk::ingestor::NO_DEVICE;
        }
        return deviceId;
    }

    const hipdnn_plugin_sdk::ingestor::DeviceProperties&
        deviceProperties(hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const override
    {
        // MatchContext binds properties before any matcher runs.
        if(deviceId == hipdnn_plugin_sdk::ingestor::NO_DEVICE)
        {
            static const hipdnn_plugin_sdk::ingestor::DeviceProperties s_noDevice{};
            return s_noDevice;
        }

        const std::lock_guard<std::mutex> lock(_mutex);

        auto it = _properties.find(deviceId);
        if(it != _properties.end())
        {
            return it->second;
        }

        hipDeviceProp_t properties{};
        const auto status = queryDeviceProperties(&properties, deviceId);
        if(status != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hipGetDeviceProperties failed for device " + std::to_string(deviceId) + ": "
                    + hipGetErrorString(status));
        }

        // HIP's fields narrow to the ingestor's `$device.*` namespace.
        hipdnn_plugin_sdk::ingestor::DeviceProperties resolved;
        resolved.gcnArchName = properties.gcnArchName;
        resolved.warpSize = properties.warpSize;
        resolved.multiProcessorCount = properties.multiProcessorCount;

        return _properties.emplace(deviceId, std::move(resolved)).first->second;
    }

protected:
    /// Test seam: lets a test supply properties for devices this machine lacks.
    virtual hipError_t queryDeviceProperties(hipDeviceProp_t* properties,
                                             hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const
    {
        return hipGetDeviceProperties(properties, deviceId);
    }

private:
    mutable std::mutex _mutex;
    mutable std::unordered_map<hipdnn_plugin_sdk::ingestor::DeviceId,
                               hipdnn_plugin_sdk::ingestor::DeviceProperties>
        _properties;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
