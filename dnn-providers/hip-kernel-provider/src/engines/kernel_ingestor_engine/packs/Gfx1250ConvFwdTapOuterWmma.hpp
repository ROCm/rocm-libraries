// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdGstreamWmma.hpp"

/**
 * @file Gfx1250ConvFwdTapOuterWmma.hpp
 * @brief Admission and launch of the gfx1250 `direct_fwd_tapouter_wmma` instances.
 *
 * Ports kernels/common/conv_direct_fwd_tapouter_wmma.py: DirectFwdTapOuterSpec (its derived
 * quantities, grid, threads_per_block), tapouter_reason and tapouter_tile_efficiency, with
 * direct_spec_for_problem's tile-efficiency floor after the validator. The WMMA forward
 * gate, the validator tail and the Python arithmetic come from Gfx1250ConvFwdGstreamWmma.hpp.
 *
 * Unchecked agreements (restated by hand): DirectFwdTapOuterSpec's knob defaults,
 * TAPOUTER_VGPR_BUDGET = 512, and the gfx1250 limits listed in Gfx1250ConvFwdGstreamWmma.hpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

constexpr int64_t TAPOUTER_VGPR_BUDGET = 512;

struct TapOuterWmmaKernel
{
    DirectBaked baked;
    int64_t wavesK, wavesP, tilesK, tilesP, tileW, ck, stages;
    bool prefetch;
    std::optional<std::string> sched;
    int64_t waveSize, tdmSplit;

    static TapOuterWmmaKernel parse(const InstanceGeometry& g)
    {
        return {DirectBaked::parse(g),
                g.knob("waves_k", 2),
                g.knob("waves_p", 4),
                g.knob("tiles_k", 4),
                g.knob("tiles_p", 2),
                g.knob("tile_w", 16),
                g.knob("ck", 64),
                g.knob("stages", 3),
                g.boolKnob("prefetch", false),
                schedKnob(g),
                WMMA_SPEC_WAVE_SIZE,
                g.knob("tdm_split", 1)};
    }
};

/// DirectFwdTapOuterSpec's derived quantities on one problem.
struct TapOuterWmmaSpec
{
    const TapOuterWmmaKernel& k;
    const DirectProblem& p;

    int64_t blockK() const
    {
        return k.wavesK * k.tilesK * 16;
    }
    int64_t blockP() const
    {
        return k.wavesP * k.tilesP * 16;
    }
    int64_t tileH() const
    {
        return k.tileW > 0 ? floorDiv(blockP(), k.tileW) : 0;
    }
    int64_t threadsPerBlock() const
    {
        return k.wavesK * k.wavesP * k.waveSize;
    }
    int64_t xPitch() const
    {
        return k.ck + CH_PAD;
    }
    int64_t epiPitch() const
    {
        return blockK() + CH_PAD;
    }
    /// The window and the tile_w cells its shifted last row wraps into.
    int64_t winElems() const
    {
        return (blockP() + k.tileW) * xPitch();
    }
    int64_t weiElems() const
    {
        return blockK() * xPitch();
    }
    int64_t ldsBytes() const
    {
        const auto ring = k.stages * (winElems() + weiElems());
        const auto epi = blockP() * epiPitch();
        return 2 * std::max(ring, epi);
    }
    int64_t gridX() const
    {
        return p.n * p.groups * ceilDiv(p.ho(), tileH()) * ceilDiv(p.wo(), k.tileW)
               * ceilDiv(p.kpg, blockK());
    }
};

/// tapouter_reason.
inline Refusal tapOuterReason(const TapOuterWmmaSpec& s)
{
    const auto& k = s.k;
    const auto& p = s.p;
    if(auto why = wmmaFwdGate(p, k.waveSize, false))
    {
        return why;
    }
    if(p.stride != 1)
    {
        return "stride " + std::to_string(p.stride) + ": the tap windows are dense (stride 1 only)";
    }
    if(p.dilH < 1 || p.dilW < 1)
    {
        return "dilation < 1";
    }
    if(p.pad < 0)
    {
        return "PAD < 0";
    }
    if(p.ho() < 1 || p.wo() < 1)
    {
        return "empty output";
    }
    if(p.cpg % 8 != 0 || p.kpg % 8 != 0)
    {
        return "cpg and kpg must be multiples of 8 (16-byte rows)";
    }
    if(k.ck % WMMA_K != 0 || k.ck <= 0)
    {
        return "ck must be a positive multiple of 32";
    }
    if(std::min({k.wavesK, k.wavesP, k.tilesK, k.tilesP}) < 1)
    {
        return "waves and tiles must be >= 1";
    }
    if(k.wavesK * k.wavesP < 2)
    {
        return "needs at least two waves (one TDM issuer per operand)";
    }
    if(k.stages < 2)
    {
        return "stages must be >= 2";
    }
    const auto ts = k.tdmSplit;
    if(ts < 1 || 2 * ts > k.wavesK * k.wavesP)
    {
        return "tdm_split needs 2 * tdm_split waves";
    }
    if(s.tileH() % ts != 0 || s.blockK() % ts != 0)
    {
        return "tdm_split must divide tile_h and block_k";
    }
    if(ts > 1 && k.stages > 2)
    {
        return "tdm_split needs 2 stages (a skipped slab breaks the wait)";
    }
    if(k.tileW < 1 || s.blockP() % k.tileW != 0)
    {
        return "tile_w must divide block_p";
    }
    if(s.threadsPerBlock() > GFX1250_MAX_THREADS_PER_BLOCK)
    {
        return std::to_string(s.threadsPerBlock()) + " threads per block";
    }
    if(std::max({s.xPitch(), s.epiPitch(), k.tileW, s.blockK()}) > TDM_MAX_EXTENT)
    {
        return "tile too wide for a TDM descriptor";
    }
    if(s.ldsBytes() > GFX1250_LDS_BYTES)
    {
        return "LDS " + std::to_string(s.ldsBytes()) + " B > " + std::to_string(GFX1250_LDS_BYTES);
    }
    const auto vgprs = 8 * k.tilesK * k.tilesP + 8 * (k.tilesK + k.tilesP) * (k.prefetch ? 2 : 1);
    if(vgprs > TAPOUTER_VGPR_BUDGET)
    {
        return "~" + std::to_string(vgprs) + " VGPRs of accumulators and fragments";
    }
    return wmmaFwdLimitsReason(p, k.sched, s.gridX());
}

inline Refusal refuse(const TapOuterWmmaKernel& kernel, const Problem& problem)
{
    DirectProblem p{};
    if(auto why = directProblem(problem, p))
    {
        return why;
    }
    if(auto why = directCapability(kernel.baked, p))
    {
        return why;
    }
    const TapOuterWmmaSpec s{kernel, p};
    if(auto why = tapOuterReason(s))
    {
        return why;
    }
    // tapouter_tile_efficiency: real (Ho, Wo, kpg) over the padded tile grid.
    const auto ho = p.ho();
    const auto wo = p.wo();
    return tileEfficiencyReason(ho * wo * p.kpg,
                                alignUp(ho, s.tileH()) * alignUp(wo, kernel.tileW)
                                    * alignUp(p.kpg, s.blockK()));
}

inline Plan plan(const TapOuterWmmaKernel& kernel, const Problem& problem)
{
    DirectProblem p{};
    directProblem(problem, p);
    const TapOuterWmmaSpec s{kernel, p};
    return directPlan(kernel.baked.label, problem, {s.gridX(), 1, 1}, s.threadsPerBlock());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
