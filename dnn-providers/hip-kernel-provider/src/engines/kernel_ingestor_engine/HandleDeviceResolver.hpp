// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <mutex>
#include <unordered_map>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>

#include "core/Handle.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/**
 * @brief Resolves a call's device from the stream its handle carries.
 *
 * A handle can be bound to any device and rebound between calls, so the device is read
 * per call rather than captured once. Resolving it from the handle's stream — rather
 * than from whichever device happens to be current on the calling thread — is what makes
 * the answer correct when several threads drive different handles concurrently.
 *
 * Device properties are cached, because they are asked on the applicability path and
 * never change for a given device. The cache hands out references that stay valid for
 * this resolver's lifetime, so entries are never erased or rehashed away: node handles
 * in std::unordered_map keep referenced values pinned across growth.
 *
 * Instantiated once, at process lifetime (see KernelIngestorEngine.cpp's deviceResolver()),
 * rather than once per engine or once per Container. This class carries no engine state
 * -- only device properties, which are a fact about the machine, not about which engine
 * or container is asking -- so nothing about it needs to be reconstructed when a
 * container is destroyed and rebuilt (SharedContainerManager's weak_ptr does exactly
 * that across handle churn). A per-engine instance would duplicate the same cache once
 * per engine and lose its warm entries every time a container cycled, for no isolation
 * benefit: two engines resolving the same physical device must agree on that device's
 * properties by construction, so a shared cache is the correct scope, not merely a
 * convenient one. That is also why the cache the class holds is never invalidated
 * ("references stay valid for this resolver's lifetime" above is a promise this
 * lifetime choice keeps easy to honor).
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
            // Only reachable with no usable HIP context, where nothing can be matched
            // or launched anyway. Device 0 keys the cache consistently so the failure
            // surfaces at compile or launch, with a message about the real problem,
            // rather than here as a cache-key error.
            return 0;
        }
        return deviceId;
    }

    const hipDeviceProp_t&
        deviceProperties(hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const override
    {
        const std::lock_guard<std::mutex> lock(_mutex);

        auto it = _properties.find(deviceId);
        if(it != _properties.end())
        {
            return it->second;
        }

        hipDeviceProp_t properties{};
        static_cast<void>(hipGetDeviceProperties(&properties, deviceId));
        return _properties.emplace(deviceId, properties).first->second;
    }

private:
    mutable std::mutex _mutex;
    mutable std::unordered_map<hipdnn_plugin_sdk::ingestor::DeviceId, hipDeviceProp_t> _properties;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
