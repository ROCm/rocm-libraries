// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::device
{

/// What a plan loaded for each device it has executed on, answered per device.
///
/// A hipModule_t belongs to the device that was current when it was loaded, and a plan
/// is immutable and re-usable across handles and threads, so the device a dispatch runs
/// on is known only at execute(), from the handle's stream. This applies the rule the
/// kpack path (IngestorKernelCode::kernelFor) established:
///
///  - an ordinal already resolved is answered from the memo, with no device query and
///    no module load;
///  - a new ordinal whose feature-stripped architecture is the one the plan was built
///    for is resolved through resolveForDevice() and remembered;
///  - a new ordinal of another architecture is refused with INVALID_VALUE: the kernel
///    was selected for an architecture, and silently selecting another would substitute
///    one the caller never asked for.
///
/// Plan validity keys on the architecture, never on the ordinal: the ordinal identifies
/// a slot.
///
/// @tparam Value What one device needs to launch: cheap to copy, typically shared_ptrs
///               to the loaded modules plus the functions resolved out of them.
template <typename Value>
class PerDeviceResolver
{
public:
    /// @param builtArch The architecture the plan was built for; features are stripped
    ///                  here, so a decorated gcnArchName is accepted.
    /// @param subject   Names the plan or kernel in failure messages.
    PerDeviceResolver(const std::string& builtArch, std::string subject)
        : _builtArch(hipdnn_plugin_sdk::stripArchFeatures(builtArch))
        , _subject(std::move(subject))
    {
    }

    virtual ~PerDeviceResolver() = default;

    // Held behind a pointer by its owners, so a plan stays movable without moving a
    // locked mutex.
    PerDeviceResolver(const PerDeviceResolver&) = delete;
    PerDeviceResolver& operator=(const PerDeviceResolver&) = delete;
    PerDeviceResolver(PerDeviceResolver&&) = delete;
    PerDeviceResolver& operator=(PerDeviceResolver&&) = delete;

    /// Records what plan build already loaded for @p deviceOrdinal, so the first execute
    /// on the build device is a memo hit.
    void seed(int deviceOrdinal, Value value)
    {
        requireRealOrdinal(deviceOrdinal);
        const std::lock_guard<std::mutex> lock(_mutex);
        _byOrdinal.insert_or_assign(deviceOrdinal, std::move(value));
    }

    /// What a dispatch on @p deviceOrdinal must launch. Callers resolve the ordinal from
    /// the executing handle's stream with launchDeviceOrdinal().
    ///
    /// @throws HipdnnPluginException with INVALID_VALUE when the device is of another
    ///         architecture, and with INTERNAL_ERROR when it cannot be queried.
    Value forOrdinal(int deviceOrdinal) const
    {
        requireRealOrdinal(deviceOrdinal);

        // Held across the resolution, so two threads reaching one new ordinal load once.
        const std::lock_guard<std::mutex> lock(_mutex);
        const auto seen = _byOrdinal.find(deviceOrdinal);
        if(seen != _byOrdinal.end())
        {
            return seen->second;
        }

        // Queried rather than taken from the plan: the plan's properties describe the
        // device it was built under, and this branch is reached only for another one.
        std::string reportedArch;
        const hipError_t status = queryDeviceArch(deviceOrdinal, reportedArch);
        if(status != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                _subject + ": cannot query device " + std::to_string(deviceOrdinal)
                    + " to resolve its kernel: " + hipGetErrorString(status));
        }

        const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(reportedArch));
        if(arch != _builtArch)
        {
            // INVALID_VALUE, not INTERNAL_ERROR: nothing inside the provider is
            // inconsistent. The caller paired a plan with a handle on an architecture it
            // was never built for, and that pairing is the caller's to correct.
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                _subject + ": the plan was built for architecture '" + _builtArch
                    + "' but is being executed on device " + std::to_string(deviceOrdinal)
                    + ", which reports '" + arch
                    + "'; a plan is being executed under a handle from a device of another "
                      "architecture");
        }

        return _byOrdinal.emplace(deviceOrdinal, resolveForDevice(deviceOrdinal, reportedArch))
            .first->second;
    }

    /// The feature-stripped architecture the plan was built for.
    const std::string& builtArch() const
    {
        return _builtArch;
    }

    /// How many ordinals have been resolved or seeded.
    std::size_t resolvedCount() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _byOrdinal.size();
    }

protected:
    // NOLINTBEGIN(portability-template-virtual-member-function)
    // The steps for a device this object has not seen before. Each is virtual so a test
    // can stand in for the hardware; production uses the defaults below and the
    // subclass's resolveForDevice().

    /// The architecture @p deviceOrdinal reports, undecorated as HIP gives it.
    virtual hipError_t queryDeviceArch(int deviceOrdinal, std::string& reportedArch) const
    {
        hipDeviceProp_t properties{};
        const hipError_t status = hipGetDeviceProperties(&properties, deviceOrdinal);
        if(status == hipSuccess)
        {
            reportedArch = properties.gcnArchName;
        }
        return status;
    }

    /// Loads what @p deviceOrdinal needs. Reached only once its architecture has been
    /// accepted, and at most once per ordinal.
    ///
    /// @param reportedArch The device's gcnArchName, features intact.
    virtual Value resolveForDevice(int deviceOrdinal, const std::string& reportedArch) const = 0;
    // NOLINTEND(portability-template-virtual-member-function)

    const std::string& subject() const
    {
        return _subject;
    }

private:
    void requireRealOrdinal(int deviceOrdinal) const
    {
        if(deviceOrdinal < 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           _subject + ": device ordinal "
                                                               + std::to_string(deviceOrdinal)
                                                               + " names no device");
        }
    }

    std::string _builtArch;
    std::string _subject;
    mutable std::mutex _mutex;
    mutable std::unordered_map<int, Value> _byOrdinal;
};

} // namespace hip_kernel_provider::device
