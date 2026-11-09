// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "SdpaBwdParams.hpp"
#include "SdpaModuleCache.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace asm_sdpa_engine
{

/// The window coordinates (mask_y, mask_x) the DQDKDV kernel takes for the
/// mask in `params`, or (0, 0) for a mask the kernel bakes in. Mirrors AITER
/// v0.1.29 mha_bwd.cu fmha_v3_bwd(): the gfx950 window kernels (masks 3 and 4)
/// take the FlashAttention window sizes as they are, negative for unbounded,
/// and align the window to each batch themselves; the gfx942 window kernel
/// (mask 3) takes the coordinates computeMaskCoordinates() derives from the
/// window sizes, the sequence lengths and the alignment.
std::pair<int32_t, int32_t> dqdkdvMaskCoordinates(const SdpaBwdParams& params);

/// True when the DQDKDV kernel for dqdkdv CSV mask `maskOrdinal` runs on half
/// the K/V tiles, rounded up. Mirrors AITER v0.1.29 mha_bwd.cu fmha_v3_bwd()
/// (:713-716): only the causal kernels (masks 1 and 2) halve the grid; the
/// sliding-window kernels (masks 3 and 4) take the full grid.
bool halvesDqdkdvGrid(int32_t maskOrdinal);

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
    SdpaBwdPlan(CachedModule odoKernel,
                CachedModule dqdkdvKernel,
                std::optional<CachedModule> postKernel,
                SdpaBwdParams params);

    /// A16 constructor: requires only 2 kernels (ODO, DQDKDV).
    SdpaBwdPlan(CachedModule odoKernel, CachedModule dqdkdvKernel, SdpaBwdParams params);

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
    CachedModule _odoKernel;
    CachedModule _dqdkdvKernel;
    std::optional<CachedModule> _postKernel; // std::nullopt when A16 (dq_convert not needed)
    SdpaBwdParams _params;
};

} // namespace asm_sdpa_engine
