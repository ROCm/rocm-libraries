// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "SdpaBwdParams.hpp"
#include "SdpaKernelUtils.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <optional>

namespace asm_sdpa_engine
{

/**
 * @brief SDPA backward kernel plan.
 *
 * Orchestrates 2 or 3 ASM kernels for the backward pass:
 *   1. ODO        — D reduction: D[b,h,i] = sum_j(O * dO)
 *   2. DQDKDV     — Main gradients: dQ (FP32 via A32, or direct BF16 via A16), dK, dV
 *   3. DQ_CONVERT — Post-processing: FP32 dQ → BF16 (A32 path only)
 *
 * Accumulator type selects the path:
 *   - A32 (3-kernel path): ODO → DQDKDV → DQ_CONVERT
 *   - A16 (2-kernel path): ODO → DQDKDV (dQ written directly in BF16)
 *
 * When A16, `_postKernel` is std::nullopt and the DQ_CONVERT launch + dq_acc
 * workspace are both skipped.
 */
class SdpaBwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    /// A32 constructor: requires all 3 kernels (ODO, DQDKDV, DQ_CONVERT).
    SdpaBwdPlan(HipModuleGuard odoKernel,
                HipModuleGuard dqdkdvKernel,
                std::optional<HipModuleGuard> postKernel,
                SdpaBwdParams params);

    /// A16 constructor: requires only 2 kernels (ODO, DQDKDV).
    SdpaBwdPlan(HipModuleGuard odoKernel, HipModuleGuard dqdkdvKernel, SdpaBwdParams params);

    ~SdpaBwdPlan() override = default;

    SdpaBwdPlan(const SdpaBwdPlan&) = delete;
    SdpaBwdPlan& operator=(const SdpaBwdPlan&) = delete;
    SdpaBwdPlan(SdpaBwdPlan&&) noexcept = default;
    SdpaBwdPlan& operator=(SdpaBwdPlan&&) noexcept = default;

    size_t getWorkspaceSize(const Handle& handle) const override;

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    HipModuleGuard _odoKernel;
    HipModuleGuard _dqdkdvKernel;
    std::optional<HipModuleGuard> _postKernel; // nullopt when A16 (dq_convert not needed)
    SdpaBwdParams _params;
};

} // namespace asm_sdpa_engine
