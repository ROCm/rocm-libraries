// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>

#include "core/Handle.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Resolves a call's device from the stream its handle carries; a handle can be
/// rebound between calls, so this reads per call rather than caching per-thread.
/// Device-property lookups are remembered per device id for this resolver's lifetime:
/// valid facts are cached and returned by reference, and a device whose query succeeded
/// but reported invalid facts is refused again with the same message without asking HIP.
/// A failed HIP query is not remembered, so the next lookup asks HIP again.
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

        auto refusal = _refusals.find(deviceId);
        if(refusal != _refusals.end())
        {
            failDeviceQuery(refusal->second);
        }

        hipDeviceProp_t properties{};
        const auto status = queryDeviceProperties(&properties, deviceId);
        if(status != hipSuccess)
        {
            failDeviceQuery("hipGetDeviceProperties failed for device " + std::to_string(deviceId)
                            + ": " + hipGetErrorString(status));
        }

        // HIP answered, so invalid facts are a property of the device: remember the refusal.
        const auto refuse = [this, deviceId](std::string message) {
            failDeviceQuery(_refusals.emplace(deviceId, std::move(message)).first->second);
        };
        // Check each fact separately so the error names the bad field and its value.
        const auto rejectFact = [&refuse, deviceId](const std::string& fact) {
            refuse("hipGetDeviceProperties returned an invalid device fact for device "
                   + std::to_string(deviceId) + ": " + fact);
        };

        // Byte counts reach the winner cache and the UHD as int64_t.
        constexpr auto MAX_BYTE_COUNT
            = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());

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
        if(properties.sharedMemPerBlock == 0 || properties.sharedMemPerBlock > MAX_BYTE_COUNT)
        {
            rejectFact("sharedMemPerBlock is " + std::to_string(properties.sharedMemPerBlock)
                       + ", expected a positive byte count no larger than "
                       + std::to_string(MAX_BYTE_COUNT));
        }
        // Memory facts tell apart boards that share an arch and CU count (e.g. MI300X,
        // MI325X, MI308X), which a UHD trained on a merged corpus needs. A zero only means
        // "unknown" to the UHD, so zero is accepted; a negative or out-of-range value is not.
        if(properties.totalGlobalMem > MAX_BYTE_COUNT)
        {
            rejectFact("totalGlobalMem is " + std::to_string(properties.totalGlobalMem)
                       + ", expected a byte count no larger than "
                       + std::to_string(MAX_BYTE_COUNT));
        }
        if(properties.memoryBusWidth < 0)
        {
            rejectFact("memoryBusWidth is " + std::to_string(properties.memoryBusWidth)
                       + ", expected a non-negative bit count");
        }
        if(properties.memoryClockRate < 0)
        {
            rejectFact("memoryClockRate is " + std::to_string(properties.memoryClockRate)
                       + ", expected a non-negative kHz rate");
        }

        hipdnn_plugin_sdk::ingestor::DeviceProperties resolved;
        resolved.gcnArchName = std::string(std::begin(properties.gcnArchName), archEnd);
        resolved.warpSize = properties.warpSize;
        resolved.multiProcessorCount = properties.multiProcessorCount;
        resolved.totalGlobalMem = properties.totalGlobalMem;
        resolved.memoryBusWidth = properties.memoryBusWidth;
        resolved.memoryClockRate = properties.memoryClockRate;
        resolved.sharedMemPerBlock = properties.sharedMemPerBlock;

        // The ingestor's own rule is the authority; the checks above only name the field.
        if(!hipdnn_plugin_sdk::ingestor::isResolved(resolved))
        {
            refuse("hipGetDeviceProperties returned facts for device " + std::to_string(deviceId)
                   + " that fail the ingestor's device validity rule");
        }

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
    /// Callers log the exception: plan building through its applicability-failure log,
    /// and the plugin boundary through the plugin's last-error record.
    [[noreturn]] static void failDeviceQuery(const std::string& message)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       message);
    }

    mutable std::mutex _mutex;
    mutable std::unordered_map<hipdnn_plugin_sdk::ingestor::DeviceId,
                               hipdnn_plugin_sdk::ingestor::DeviceProperties>
        _properties;
    mutable std::unordered_map<hipdnn_plugin_sdk::ingestor::DeviceId, std::string> _refusals;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
