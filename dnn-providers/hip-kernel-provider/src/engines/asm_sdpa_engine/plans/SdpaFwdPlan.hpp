// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "SdpaFwdParams.hpp"
#include "SdpaKernelResolvers.hpp"
#include "core/Handle.hpp"

#include <memory>

namespace asm_sdpa_engine
{

/**
* @brief SDPA forward kernel plan.
*/
class SdpaFwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    /**
     * @brief Construct a plan with its per-device kernels and precomputed metadata.
     *
     * @p kernels answers for the device each execute() runs on; plan build seeds it
     * with the build device's module.
     */
    SdpaFwdPlan(std::unique_ptr<SdpaFwdKernels> kernels, SdpaFwdParams params);

    ~SdpaFwdPlan() override = default;

    // Delete copy operations (resource ownership)
    SdpaFwdPlan(const SdpaFwdPlan&) = delete;
    SdpaFwdPlan& operator=(const SdpaFwdPlan&) = delete;

    // Move operations (defaulted — the resolver is held by pointer, HipModuleGuard
    // handles resource cleanup)
    SdpaFwdPlan(SdpaFwdPlan&&) noexcept = default;
    SdpaFwdPlan& operator=(SdpaFwdPlan&&) noexcept = default;

    size_t getWorkspaceSize(const Handle& handle) const override;

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    std::unique_ptr<SdpaFwdKernels> _kernels;
    SdpaFwdParams _params;
};

} // namespace asm_sdpa_engine
