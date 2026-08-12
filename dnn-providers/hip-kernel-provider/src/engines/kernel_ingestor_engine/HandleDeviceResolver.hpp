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
 * A handle can be rebound between calls, so the device is read per call. Reading it from
 * the handle's stream rather than the calling thread's current device is what keeps the
 * answer right when several threads drive different handles.
 *
 * Device properties are cached and the cache is never invalidated, so the references it
 * hands out stay valid for this resolver's lifetime. Only successful queries are cached:
 * a zeroed hipDeviceProp_t is not an answer, and this resolver is process-lifetime, so
 * caching one failure would answer wrongly for every later caller.
 *
 * One instance per process (see KernelIngestorEngine.cpp's deviceResolver()). It holds no
 * engine state, and two engines asking about one device must agree, so a shared cache is
 * the correct scope rather than a convenient one.
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
            // Neither 0 nor a throw: 0 is a real ordinal this call never asked about,
            // and deviceId() runs under isApplicable(), which EngineManager walks with
            // no try/catch -- throwing would deny sibling engines their answer. Matchers
            // decline on NO_DEVICE, so the engine simply does not apply.
            return hipdnn_plugin_sdk::ingestor::NO_DEVICE;
        }
        return deviceId;
    }

    const hipDeviceProp_t&
        deviceProperties(hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const override
    {
        // Reached whenever deviceId() could not name a device, because MatchContext binds
        // properties before any matcher runs. Inert values, since no kernel survives.
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
    /// Seam for tests that need to grow the cache without that many real devices. The
    /// production path is the HIP call; overriding it lets a test supply successful
    /// answers for ids this machine does not have, which is the only way to exercise
    /// the reference-stability-across-growth invariant now that failures are refused
    /// rather than cached.
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
