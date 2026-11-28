// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// Answers which device a call is for, from the handle that carries it. Resolved per
/// call, not at engine construction, since a handle can rebind between calls.
template <typename THandle>
class IDeviceResolver
{
public:
    virtual ~IDeviceResolver() = default;

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual DeviceId deviceId(const THandle& handle) const = 0;

    /// Facts for @p deviceId. For `NO_DEVICE`, return unresolved (default) properties.
    /// For a real device, return properties that satisfy `isResolved()` or throw; never
    /// substitute zero or a default for a fact the runtime did not report. The ingestor
    /// treats a device whose properties fail `isResolved()` as unusable: it neither
    /// matches kernels for it nor caches rankings under it. The returned reference must
    /// stay valid for the resolver's lifetime.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual const DeviceProperties& deviceProperties(DeviceId deviceId) const = 0;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
