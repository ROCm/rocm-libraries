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
 * Only successful queries are cached. A failed hipGetDeviceProperties throws rather
 * than caching the zeroed struct it left behind: this cache is process-lifetime and is
 * never invalidated, so one transient failure would otherwise pin unusable properties
 * for every later caller asking about that device -- and those zeroed values do not
 * fail loudly, they reach KernelCompileOptions as an empty --offload-arch and surface
 * as an hiprtc error naming neither the device nor the property query. Before this
 * class became process-lifetime a container cycle cleared such an entry; now nothing
 * does, which is exactly why the failure has to be refused rather than remembered.
 * The provider's own convention for this call is the same (see
 * CurrentDevicePropertyProvider), as is the module cache's refusal to cache a failed
 * load.
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
 * convenient one.
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
            // or launched anyway.
            //
            // Not device 0, and not a throw. Not 0 because 0 is a valid ordinal other
            // healthy calls resolve, so answering it would key this call's catalog and
            // property cache to a real device nobody asked about -- and those entries
            // outlive the container, so one failure would answer for every later
            // caller. Not a throw because deviceId() is reached from isApplicable(),
            // and EngineManager::getApplicableEngineIds() walks every engine with no
            // try/catch: an exception here would deny a healthy sibling engine its
            // answer rather than just declining for this one.
            //
            // NO_DEVICE is a key nothing matches and nothing caches against, so the
            // engine simply finds no applicable kernel and declines, which is the
            // correct outcome when there is no device to launch on.
            return hipdnn_plugin_sdk::ingestor::NO_DEVICE;
        }
        return deviceId;
    }

    const hipDeviceProp_t&
        deviceProperties(hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const override
    {
        // NO_DEVICE is not a device to query: MatchContext binds properties eagerly,
        // before any matcher runs, so this is reached whenever deviceId() could not
        // name a device. Answering with an inert zeroed struct lets matching proceed to
        // the matchers, which decline on NO_DEVICE -- throwing here instead would deny
        // every sibling engine its applicability answer (EngineManager walks them with
        // no try/catch). Nothing reads these values, because no kernel survives.
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
            // Not cached: see the class doc. A zeroed hipDeviceProp_t is not a usable
            // answer, and this cache is never invalidated, so remembering one failure
            // would answer wrongly for the life of the process.
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
