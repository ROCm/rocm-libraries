// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/**
 * @brief Answers which device a call is for, from the handle that carries it.
 *
 * An engine is shared by every handle in the process, and a handle can be rebound to a
 * different device between calls, so the device a query concerns is resolved per call
 * rather than captured once at engine construction.
 *
 * How a handle names its device is provider-specific, which is why this is an
 * interface rather than a call the SDK makes directly.
 */
template <typename THandle>
class IDeviceResolver
{
public:
    virtual ~IDeviceResolver() = default;

    /// @brief The device @p handle currently targets.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual DeviceId deviceId(const THandle& handle) const = 0;

    /**
     * @brief Properties of @p deviceId, for the `$device.*` expression namespace.
     *
     * Returns a reference that stays valid for the resolver's lifetime, because a
     * MatchContext binds it rather than copying it. Implementations are expected to
     * cache: this is asked on the applicability path, which runs once per engine per
     * graph.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual const hipDeviceProp_t& deviceProperties(DeviceId deviceId) const = 0;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
