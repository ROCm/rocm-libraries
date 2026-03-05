// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Interface for a compilable plan.
 *
 * An ICompilablePlan extends IPlan with a compile step that prepares the plan
 * for execution on a specific device. This is used by providers that generate
 * HIP kernels which need to be compiled for the target device's architecture
 * and capabilities.
 *
 * The typical lifecycle is:
 * 1. Plan builder creates an ICompilablePlan
 * 2. compile() is called with the target device properties
 * 3. The plan is then ready for execute() calls
 *
 * @tparam THandle The plugin-specific handle type.
 *
 * @note compile() must be called before execute(). Calling execute() on an
 *       uncompiled plan is undefined behavior.
 */
template <typename THandle>
class ICompilablePlan : public IPlan<THandle>
{
public:
    ~ICompilablePlan() override = default;

    /**
     * @brief Compiles the plan for a specific device.
     *
     * Prepares the plan for execution by compiling kernels or performing
     * other device-specific setup based on the target device's properties.
     *
     * @param deviceProperties The HIP device properties of the target device.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual void compile(const hipDeviceProp_t& deviceProperties) = 0;
};

} // namespace hipdnn_plugin_sdk
