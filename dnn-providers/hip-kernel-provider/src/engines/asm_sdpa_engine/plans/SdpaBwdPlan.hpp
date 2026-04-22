// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"
#include "SdpaBwdParams.hpp"

namespace asm_sdpa_engine
{

/**
 * @brief SDPA backward kernel plan.
 *
 * Orchestrates 3 ASM kernels for the backward pass:
 *   1. ODO       — D reduction: D[b,h,i] = sum_j(O * dO)
 *   2. DQDKDV    — Main gradients: dQ (FP32), dK, dV
 *   3. DQ_CONVERT — Post-processing: FP32 dQ → BF16
 */
class SdpaBwdPlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    SdpaBwdPlan(hipModule_t odoModule,
                hipFunction_t odoFunc,
                hipModule_t dqdkdvModule,
                hipFunction_t dqdkdvFunc,
                hipModule_t postModule,
                hipFunction_t postFunc,
                SdpaBwdParams params);

    ~SdpaBwdPlan() override;

    // Delete copy operations (resource ownership)
    SdpaBwdPlan(const SdpaBwdPlan&) = delete;
    SdpaBwdPlan& operator=(const SdpaBwdPlan&) = delete;

    // Move operations
    SdpaBwdPlan(SdpaBwdPlan&& other) noexcept;
    SdpaBwdPlan& operator=(SdpaBwdPlan&& other) noexcept;

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    hipModule_t _odoModule;
    hipModule_t _dqdkdvModule;
    hipModule_t _postModule;
    hipFunction_t _odoFunc;
    hipFunction_t _dqdkdvFunc;
    hipFunction_t _postFunc;
    SdpaBwdParams _params;
};

} // namespace asm_sdpa_engine
