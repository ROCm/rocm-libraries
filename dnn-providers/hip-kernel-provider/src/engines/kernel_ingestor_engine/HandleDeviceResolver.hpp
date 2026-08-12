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

/**
 * @brief Resolves a call's device from the stream its handle carries.
 *
 * A handle can be rebound between calls, so the device is read per call from the
 * handle's stream rather than the calling thread's current device.
 *
 * Device properties are cached for this resolver's lifetime; the cache is never
 * invalidated. Only successful queries are cached.
 *
 * One instance per process (see KernelIngestorEngine.cpp's deviceResolver()).
 */
class HandleDeviceResolver : public hipdnn_plugin_sdk::ingestor::IDeviceResolver<Handle>
{
public:
    hipdnn_plugin_sdk::ingestor::DeviceId deviceId(const Handle& handle) const override
    {
        int deviceId = 0;

        // A null stream means the default stream, which belongs to the current device.
        if(handle.getStream() != nullptr)
        {
            if(hipStreamGetDevice(handle.getStream(), &deviceId) == hipSuccess)
            {
                return deviceId;
            }
        }

        if(hipGetDevice(&deviceId) != hipSuccess)
        {
            // Neither 0 nor a throw: deviceId() runs under isApplicable(), which
            // EngineManager walks with no try/catch. Matchers decline on NO_DEVICE.
            return hipdnn_plugin_sdk::ingestor::NO_DEVICE;
        }
        return deviceId;
    }

    const hipDeviceProp_t&
        deviceProperties(hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const override
    {
        // Inert values for a call with no resolvable device; MatchContext binds
        // properties before any matcher runs, so the caller receives this regardless.
        if(deviceId == hipdnn_plugin_sdk::ingestor::NO_DEVICE)
        {
            static const hipDeviceProp_t s_noDevice{};
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
            // Never cached; see the class doc.
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hipGetDeviceProperties failed for device " + std::to_string(deviceId) + ": "
                    + hipGetErrorString(status));
        }
        return _properties.emplace(deviceId, properties).first->second;
    }

protected:
    /// Seam for tests that need to grow the cache without that many real devices;
    /// overriding it lets a test supply successful answers for ids this machine
    /// does not have.
    virtual hipError_t queryDeviceProperties(hipDeviceProp_t* properties,
                                             hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const
    {
        return hipGetDeviceProperties(properties, deviceId);
    }

private:
    mutable std::mutex _mutex;
    mutable std::unordered_map<hipdnn_plugin_sdk::ingestor::DeviceId, hipDeviceProp_t> _properties;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
