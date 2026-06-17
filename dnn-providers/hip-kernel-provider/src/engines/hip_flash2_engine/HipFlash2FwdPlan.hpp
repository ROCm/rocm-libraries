// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HipFlash2FwdPlan: IPlan implementation that executes Flash-Attention 2 V7.
// Holds the HipModuleGuard (loaded .co) and the parameters for kernel dispatch.
// Mirrors the asm_sdpa_engine::SdpaFwdPlan pattern.

#pragma once

#include "HipFlash2FwdPlanBuilder.hpp"  // Flash2FwdParams
#include "HipFlash2KernelUtils.hpp"     // HipModuleGuard, Flash2KernelArgs
#include "core/Handle.hpp"

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hip_flash2_engine
{

/**
 * @brief Flash-Attention 2 V7 forward kernel plan.
 *
 * Holds a preloaded HipModuleGuard (the arch-specific .co file) and the
 * Flash2FwdParams extracted from the hipDNN graph at buildPlan() time.
 * execute() maps device buffers by UID, fills Flash2KernelArgs, and
 * dispatches via hipModuleLaunchKernel.
 */
class HipFlash2FwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    explicit HipFlash2FwdPlan(HipModuleGuard kernel, Flash2FwdParams params);

    ~HipFlash2FwdPlan() override = default;

    HipFlash2FwdPlan(const HipFlash2FwdPlan&)            = delete;
    HipFlash2FwdPlan& operator=(const HipFlash2FwdPlan&) = delete;
    HipFlash2FwdPlan(HipFlash2FwdPlan&&) noexcept        = default;
    HipFlash2FwdPlan& operator=(HipFlash2FwdPlan&&) noexcept = default;

    /// Flash-Attention 2 V7 uses only registers and LDS — no external workspace.
    size_t getWorkspaceSize(const Handle& handle) const override;

    /// Map device buffers by UID → pointer, fill Flash2KernelArgs, launch kernel.
    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    HipModuleGuard  _kernel;
    Flash2FwdParams _params;
};

} // namespace hip_flash2_engine
