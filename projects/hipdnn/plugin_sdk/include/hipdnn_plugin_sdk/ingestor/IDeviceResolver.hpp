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
 * Resolved per call, not at engine construction: an engine is shared across every
 * handle in the process, and a handle can rebind to a different device between calls.
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
     * Returns a reference valid for the resolver's lifetime; a MatchContext binds it
     * rather than copying. Implementations should cache.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual const hipDeviceProp_t& deviceProperties(DeviceId deviceId) const = 0;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
