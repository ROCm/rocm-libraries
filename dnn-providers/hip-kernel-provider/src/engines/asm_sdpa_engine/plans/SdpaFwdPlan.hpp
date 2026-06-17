// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "SdpaFwdParams.hpp"
#include "SdpaKernelUtils.hpp"
#include "core/Handle.hpp"

namespace asm_sdpa_engine
{

/**
* @brief SDPA forward kernel plan.
*/
class SdpaFwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    /**
     * @brief Construct a plan with kernel module and precomputed metadata.
     */
    SdpaFwdPlan(HipModuleGuard kernel, SdpaFwdParams params);

    ~SdpaFwdPlan() override = default;

    // Delete copy operations (resource ownership)
    SdpaFwdPlan(const SdpaFwdPlan&) = delete;
    SdpaFwdPlan& operator=(const SdpaFwdPlan&) = delete;

    // Move operations (defaulted — HipModuleGuard handles resource cleanup)
    SdpaFwdPlan(SdpaFwdPlan&&) noexcept = default;
    SdpaFwdPlan& operator=(SdpaFwdPlan&&) noexcept = default;

    size_t getWorkspaceSize(const Handle& handle) const override;

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    HipModuleGuard _kernel;
    SdpaFwdParams _params;
};

} // namespace asm_sdpa_engine
