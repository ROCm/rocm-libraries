// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"
#include "SdpaBwdParams.hpp"
#include "SdpaKernelUtils.hpp"

#include <optional>

namespace asm_sdpa_engine
{

/**
 * @brief SDPA backward kernel plan.
 *
 * Orchestrates ASM kernels for the backward pass:
 *   - A32 (3-kernel path): ODO → DQDKDV → DQ_CONVERT
 *   - A16 (2-kernel path): ODO → DQDKDV (dQ written directly in BF16)
 */
class SdpaBwdPlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    /// A32 constructor: requires all 3 kernels (ODO, DQDKDV, DQ_CONVERT).
    SdpaBwdPlan(HipModuleGuard odoKernel,
                HipModuleGuard dqdkdvKernel,
                HipModuleGuard postKernel,
                SdpaBwdParams params);

    /// A16 constructor: requires only 2 kernels (ODO, DQDKDV).
    SdpaBwdPlan(HipModuleGuard odoKernel, HipModuleGuard dqdkdvKernel, SdpaBwdParams params);

    ~SdpaBwdPlan() override = default;

    SdpaBwdPlan(const SdpaBwdPlan&) = delete;
    SdpaBwdPlan& operator=(const SdpaBwdPlan&) = delete;
    SdpaBwdPlan(SdpaBwdPlan&&) noexcept = default;
    SdpaBwdPlan& operator=(SdpaBwdPlan&&) noexcept = default;

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    HipModuleGuard _odoKernel;
    HipModuleGuard _dqdkdvKernel;
    std::optional<HipModuleGuard> _postKernel;
    SdpaBwdParams _params;
};

} // namespace asm_sdpa_engine
