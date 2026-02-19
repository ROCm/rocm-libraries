// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Interface for an executable plan.
 *
 * An IPlan represents a ready-to-execute plan that can take device data and then
 * execute the desired operations on it. Plans are typically created by an IPlanBuilder
 * and are attached to an execution context.
 *
 * Plans should be immutable after creation - their lifecycle and execution can be
 * safely handled by the EnginePluginContainer.
 *
 * @tparam THandle The plugin-specific handle type (e.g., HipdnnMiopenHandle).
 *
 * @note Implementations should be thread-safe as plans may be executed from multiple
 *       threads concurrently with different device buffers.
 */
template <typename THandle>
class IPlan
{
public:
    virtual ~IPlan() = default;

    /**
     * @brief Returns the workspace size required for this plan.
     *
     * @param handle The engine plugin handle.
     * @return The workspace size in bytes required for execution.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual size_t getWorkspaceSize(const THandle& handle) const = 0;

    /**
     * @brief Executes the plan with the provided device buffers.
     *
     * @param handle The engine plugin handle.
     * @param deviceBuffers Array of device buffers containing input/output data.
     * @param numDeviceBuffers Number of device buffers in the array.
     * @param workspace Optional workspace memory. May be nullptr if no workspace is required.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual void execute(const THandle& handle,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         void* workspace = nullptr) const
        = 0;
};

} // namespace hipdnn_plugin_sdk
