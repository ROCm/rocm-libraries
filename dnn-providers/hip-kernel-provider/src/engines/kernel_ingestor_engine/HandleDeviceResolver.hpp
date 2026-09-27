// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
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
        // Seeded to -1, not 0: some HIP runtimes return hipSuccess from hipStreamGetDevice
        // without writing the out-parameter. At 0 that goes unseen -- every stream resolves
        // to device 0 and looks right on a single-device machine.
        int deviceId = -1;

        // Default stream tokens are relative to the live current device.
        const auto stream = handle.getStream();
        if(!hipdnn_plugin_sdk::isDefaultStream(stream))
        {
            if(queryStreamDevice(stream, &deviceId) == hipSuccess && deviceId >= 0)
            {
                return deviceId;
            }
        }

        if(queryCurrentDevice(&deviceId) != hipSuccess)
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
        // Zero is valid, so use a sentinel to detect an unwritten capacity.
        properties.sharedMemPerBlock
            = std::numeric_limits<decltype(properties.sharedMemPerBlock)>::max();
        const auto status = queryDeviceProperties(&properties, deviceId);
        if(status != hipSuccess)
        {
            failDeviceQuery("hipGetDeviceProperties failed for device " + std::to_string(deviceId)
                            + ": " + hipGetErrorString(status));
        }

        // Each fact is checked on its own so the message names the offending field and the
        // value behind it. This fires on a machine the reporter cannot rebuild, so the message
        // is the whole diagnosis. A new fact adds a check here, not a term to a condition.
        const auto rejectFact = [deviceId](const std::string& fact) {
            failDeviceQuery("hipGetDeviceProperties returned an invalid device fact for device "
                            + std::to_string(deviceId) + ": " + fact);
        };

        const auto archEnd
            = std::find(std::begin(properties.gcnArchName), std::end(properties.gcnArchName), '\0');
        if(archEnd == std::begin(properties.gcnArchName))
        {
            rejectFact("gcnArchName is empty");
        }
        if(archEnd == std::end(properties.gcnArchName))
        {
            rejectFact("gcnArchName has no NUL terminator in its "
                       + std::to_string(sizeof(properties.gcnArchName)) + " byte buffer");
        }
        if(properties.warpSize <= 0)
        {
            rejectFact("warpSize is " + std::to_string(properties.warpSize)
                       + ", expected a positive thread count");
        }
        if(properties.multiProcessorCount <= 0)
        {
            rejectFact("multiProcessorCount is " + std::to_string(properties.multiProcessorCount)
                       + ", expected a positive count");
        }
        if(properties.sharedMemPerBlock
           > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        {
            rejectFact("sharedMemPerBlock is " + std::to_string(properties.sharedMemPerBlock)
                       + " bytes, above the " + std::to_string(std::numeric_limits<int64_t>::max())
                       + " byte limit");
        }

        // Cache only complete, validated properties.
        hipdnn_plugin_sdk::ingestor::DeviceProperties resolved{
            std::string(std::begin(properties.gcnArchName), archEnd),
            properties.warpSize,
            properties.multiProcessorCount,
            static_cast<int64_t>(properties.sharedMemPerBlock)};

        return _properties.emplace(deviceId, std::move(resolved)).first->second;
    }

protected:
    /// Test seam: lets a test model a runtime that reports success without an ordinal.
    virtual hipError_t queryStreamDevice(hipStream_t stream, int* deviceId) const
    {
        return hipdnn_plugin_sdk::getDeviceFromStream(stream, deviceId);
    }

    /// Test seam: lets a test pin the fallthrough ordinal without owning a device.
    virtual hipError_t queryCurrentDevice(int* deviceId) const
    {
        return hipGetDevice(deviceId);
    }

    /// Test seam: lets a test supply properties for devices this machine lacks.
    virtual hipError_t queryDeviceProperties(hipDeviceProp_t* properties,
                                             hipdnn_plugin_sdk::ingestor::DeviceId deviceId) const
    {
        return hipGetDeviceProperties(properties, deviceId);
    }

private:
    /// Logs before throwing so the reason survives in the plugin log even when a caller
    /// turns the exception into a status code and drops its message.
    [[noreturn]] static void failDeviceQuery(const std::string& message)
    {
        HIPDNN_PLUGIN_LOG_ERROR("ingestor: " << message);
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       message);
    }

    mutable std::mutex _mutex;
    mutable std::unordered_map<hipdnn_plugin_sdk::ingestor::DeviceId,
                               hipdnn_plugin_sdk::ingestor::DeviceProperties>
        _properties;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
