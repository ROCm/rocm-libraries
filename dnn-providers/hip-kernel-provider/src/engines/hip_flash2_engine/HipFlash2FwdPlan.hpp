// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HipFlash2FwdPlan: IPlan implementation that executes Flash-Attention 2 V7.
// Holds the per-device kernel modules (loaded .co) and the parameters for kernel
// dispatch.
// Mirrors the asm_sdpa_engine::SdpaFwdPlan pattern.

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipFlash2FwdPlanBuilder.hpp" // Flash2FwdParams
#include "HipFlash2KernelResolver.hpp" // Flash2Kernels
#include "HipFlash2KernelUtils.hpp" // Flash2KernelArgs
#include "core/Handle.hpp"

#include <memory>

namespace hip_flash2_engine
{

/**
 * @brief Flash-Attention 2 V7 forward kernel plan.
 *
 * Holds the selected variant's module for each device it has executed on
 * (buildPlan() loads the build device's) and the Flash2FwdParams extracted
 * from the hipDNN graph at buildPlan() time. execute() maps device buffers by
 * UID, fills Flash2KernelArgs, and dispatches via hipModuleLaunchKernel on the
 * device the handle's stream names.
 */
class HipFlash2FwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    HipFlash2FwdPlan(std::unique_ptr<Flash2Kernels> kernels, Flash2FwdParams params);

    ~HipFlash2FwdPlan() override = default;

    HipFlash2FwdPlan(const HipFlash2FwdPlan&) = delete;
    HipFlash2FwdPlan& operator=(const HipFlash2FwdPlan&) = delete;
    HipFlash2FwdPlan(HipFlash2FwdPlan&&) noexcept = default;
    HipFlash2FwdPlan& operator=(HipFlash2FwdPlan&&) noexcept = default;

    /// Flash-Attention 2 V7 uses only registers and LDS -- no external workspace.
    size_t getWorkspaceSize(const Handle& handle) const override;

    /// Map device buffers by UID -> pointer, fill Flash2KernelArgs, launch kernel.
    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    std::unique_ptr<Flash2Kernels> _kernels;
    Flash2FwdParams _params;
};

} // namespace hip_flash2_engine
