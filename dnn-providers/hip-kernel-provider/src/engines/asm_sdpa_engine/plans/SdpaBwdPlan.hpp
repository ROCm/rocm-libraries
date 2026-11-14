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
 * Runs 2 or 3 ASM kernels, as AITER v0.1.29 mha_bwd.cu fmha_v3_bwd() does:
 *   1. ODO    — D reduction: D[b,h,i] = sum_j(O * dO)
 *   2. DQDKDV — dK, dV, and dQ, either accumulated into a dq_acc workspace or
 *               written directly
 *   3. Post   — dQ from dq_acc: DQ_CONVERT (A32) or DQ_SHUFFLE (gfx950 A16)
 *
 * With a post kernel the workspace holds a zeroed dq_acc after the D buffer
 * (sdpaBwdDqAccLayout()). Without one (gfx942 A16) DQDKDV accumulates into the
 * dQ output, which execute() zeroes first.
 */
class SdpaBwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    /// `postKernel` is std::nullopt when DQDKDV writes dQ directly.
    SdpaBwdPlan(CachedModule odoKernel,
                CachedModule dqdkdvKernel,
                std::optional<CachedModule> postKernel,
                SdpaBwdParams params);

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
    std::optional<CachedModule> _postKernel;
    SdpaBwdParams _params;
};

} // namespace asm_sdpa_engine
