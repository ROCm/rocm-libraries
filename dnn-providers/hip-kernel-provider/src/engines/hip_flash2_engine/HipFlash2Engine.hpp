// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HipFlash2Engine: hipDNN IEngine plugin wrapping our V7 Flash-Attention 2 kernel
// (rocWMMA full-MFMA + causal tile skip) for FP16 SDPA on gfx942.
//
// Registered under HIPDNN_ENGINE_HIP_FLASH2 compile flag.
// Handles FP16 inputs where the existing ASM SDPA engine only handles BF16.

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace hip_flash2_engine
{

// Re-use the same Handle/Settings/Context types as other hip-kernel-provider engines
// (included via the plugin SDK)
struct HipFlash2Handle
{
}; // placeholder — real handle from plugin SDK
struct HipFlash2Settings
{
}; // placeholder
struct HipFlash2Context
{
}; // placeholder

class HipFlash2Engine
{
public:
    HipFlash2Engine() = default;

    static const char* engineName()
    {
        return "HIP_FLASH2_SDPA_ENGINE";
    }
    static int64_t staticId()
    {
        return 0x4841544E32ULL;
    } // "HATN2" — Flash2 engine

    // Returns true if:
    //   - Device is gfx942 (MI300X, MI325X) or gfx950
    //   - Input dtype is FP16 (half)
    //   - Single SDPA node graph
    //   - No dropout, no alibi, no padding mask, no attn bias tensor
    //   - head_dim in {64, 128}
    //   - Causal or non-causal mask
    //
    // isApplicable logic: see HipFlash2FwdPlanBuilder.cpp
    bool isApplicable(const void* handle, const void* opGraph) const;

    // Workspace: 0 bytes — Flash2 uses registers and LDS only
    size_t getMaxWorkspaceSize() const
    {
        return 0;
    }
};

} // namespace hip_flash2_engine
