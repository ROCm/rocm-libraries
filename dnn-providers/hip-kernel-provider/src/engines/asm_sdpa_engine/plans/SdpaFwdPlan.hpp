// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"
#include "SdpaFwdParams.hpp"

namespace asm_sdpa_engine
{

/**
* @brief SDPA forward kernel plan.
*/
class SdpaFwdPlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    /**
     * @brief Construct a plan with kernel module and precomputed metadata.
     */
    SdpaFwdPlan(hipModule_t kernelModule, hipFunction_t function, SdpaFwdParams params);

    ~SdpaFwdPlan() override;

    // Delete copy operations (resource ownership)
    SdpaFwdPlan(const SdpaFwdPlan&) = delete;
    SdpaFwdPlan& operator=(const SdpaFwdPlan&) = delete;

    // Move operations
    SdpaFwdPlan(SdpaFwdPlan&& other) noexcept;
    SdpaFwdPlan& operator=(SdpaFwdPlan&& other) noexcept;

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    hipModule_t _module;
    hipFunction_t _function;
    SdpaFwdParams _params;
};

} // namespace asm_sdpa_engine
