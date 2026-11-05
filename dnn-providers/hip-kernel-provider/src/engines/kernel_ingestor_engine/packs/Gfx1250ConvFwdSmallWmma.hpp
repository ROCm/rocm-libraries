// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdGstreamWmma.hpp"

/**
 * @file Gfx1250ConvFwdSmallWmma.hpp
 * @brief Admission and launch of the gfx1250 `direct_fwd_smallc_wmma` and
 *        `direct_fwd_smallk_wmma` instances.
 *
 * Ports kernels/common/conv_fwd_small_wmma.py: _common_reason / _common_shape_reason,
 * FwdSmallcWmmaSpec and FwdSmallkWmmaSpec (derived quantities, grid, threads_per_block),
 * fwd_smallc_reason, fwd_smallk_reason and their tile efficiencies, with
 * direct_spec_for_problem's rules after the validator: the tile-efficiency floor, and for
 * small C the refusal of a block_k over twice kpg. The WMMA forward gate and the Python
 * arithmetic come from Gfx1250ConvFwdGstreamWmma.hpp.
 *
 * Neither validator checks for an empty output; a zero padded tile grid is the
 * efficiency's ZeroDivisionError, a refusal too.
 *
 * Unchecked agreements (restated by hand): the knob defaults of FwdSmallcWmmaSpec and
 * FwdSmallkWmmaSpec, SMALL_WMMA_VGPR_BUDGET = 448, SMALLC_MAX_C = 4, SMALLK_MAX_K = 8, and the
 * gfx1250 limits listed in Gfx1250ConvFwdGstreamWmma.hpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

constexpr int64_t SMALL_WMMA_VGPR_BUDGET = 448;
constexpr int64_t SMALLC_MAX_C = 4;
constexpr int64_t SMALLK_MAX_K = 8;

/// _common_reason: the WMMA forward gate (a column padding of its own only when not
/// @p square), then the shapes both small kernels serve.
inline Refusal smallCommonReason(const DirectProblem& p, int64_t waveSize, bool square)
{
    if(auto why = wmmaFwdGate(p, waveSize, !square))
    {
        return why;
    }
    if(p.groups != 1 || p.dilated())
    {
        return "ungrouped, undilated convolutions only";
    }
    if(square && p.kh != p.kw)
    {
        return "square filters only";
    }
    if(p.pad < 0 || p.pad > p.kh - 1)
    {
        return "PAD outside 0..K-1";
    }
    if(p.padColumns() < 0 || p.padColumns() > p.kw - 1)
    {
        return "column PAD outside 0..KW-1";
    }
    if(p.stride < 1)
    {
        return "stride < 1";
    }
    return wmmaFwdTensorBytesReason(p);
}

/// fwd_smallc_tile_efficiency / fwd_smallk_tile_efficiency: real pixels over the padded
/// out_h x out_w tile grid.
inline Refusal smallTileEfficiencyReason(const DirectProblem& p, int64_t outH, int64_t outW)
{
    const auto ho = p.ho();
    const auto wo = p.wo();
    return tileEfficiencyReason(ho * wo, alignUp(ho, outH) * alignUp(wo, outW));
}

// ---------------------------------------------------------------------------------------
// direct_fwd_smallc_wmma (FwdSmallcWmmaSpec).
// ---------------------------------------------------------------------------------------

struct SmallcWmmaKernel
{
    DirectBaked baked;
    int64_t wavesK, wavesP, tilesK, tilesP, outH, outW, tilesPerBlock;
    bool tdmStore;
    std::optional<std::string> sched;
    int64_t waveSize;

    static SmallcWmmaKernel parse(const InstanceGeometry& g)
    {
        return {DirectBaked::parse(g),
                g.knob("waves_k", 1),
                g.knob("waves_p", 4),
                g.knob("tiles_k", 2),
                g.knob("tiles_p", 4),
                g.knob("out_h", 8),
                g.knob("out_w", 32),
                g.knob("tiles_per_block", 1),
                g.boolKnob("tdm_store", false),
                schedKnob(g),
                WMMA_SPEC_WAVE_SIZE};
    }
};

/// FwdSmallcWmmaSpec's derived quantities on one problem.
struct SmallcWmmaSpec
{
    const SmallcWmmaKernel& k;
    const DirectProblem& p;

    /// E0: window elements per filter row and lane half.
    int64_t run() const
    {
        return ceilDiv(p.kw * p.cpg, 2);
    }
    int64_t steps() const
    {
        return ceilDiv(p.kh * run(), 16);
    }
    int64_t winH() const
    {
        return (k.outH - 1) * p.stride + p.kh;
    }
    /// One column more for the odd-run slot.
    int64_t winW() const
    {
        return (k.outW - 1) * p.stride + p.kw + 1;
    }
    int64_t rowPitch() const
    {
        return alignUp(winW() * p.cpg, 8) + 8;
    }
    int64_t bufElems() const
    {
        return (winH() + p.pad + 1) * rowPitch();
    }
    int64_t nbuf() const
    {
        return k.tilesPerBlock > 1 ? 2 : 1;
    }
    int64_t blockK() const
    {
        return k.wavesK * k.tilesK * 16;
    }
    int64_t blockP() const
    {
        return k.wavesP * k.tilesP * 16;
    }
    int64_t threadsPerBlock() const
    {
        return k.wavesK * k.wavesP * k.waveSize;
    }
    int64_t outElems() const
    {
        return k.tdmStore ? blockP() * blockK() : 0;
    }
    int64_t ldsBytes() const
    {
        return 2 * (nbuf() * bufElems() + outElems());
    }
    int64_t nTiles() const
    {
        return p.n * ceilDiv(p.ho(), k.outH) * ceilDiv(p.wo(), k.outW);
    }
    Dim3 grid() const
    {
        return {ceilDiv(nTiles(), k.tilesPerBlock), ceilDiv(p.totalK(), blockK()), 1};
    }
};

/// fwd_smallc_reason.
inline Refusal smallcReason(const SmallcWmmaSpec& s)
{
    const auto& k = s.k;
    const auto& p = s.p;
    if(auto why = smallCommonReason(p, k.waveSize, true))
    {
        return why;
    }
    if(p.cpg > SMALLC_MAX_C)
    {
        return "C " + std::to_string(p.cpg) + " > " + std::to_string(SMALLC_MAX_C);
    }
    if(p.kpg % 8 != 0)
    {
        return "K must be a multiple of 8 (16-byte output rows)";
    }
    if(std::min({k.wavesK, k.wavesP, k.tilesK, k.tilesP}) < 1)
    {
        return "waves and tiles must be >= 1";
    }
    if(k.tilesPerBlock < 1)
    {
        return "tiles_per_block must be >= 1";
    }
    if(k.outH * k.outW != s.blockP())
    {
        return "out tile must hold exactly the GEMM pixels";
    }
    if(s.rowPitch() > TDM_MAX_EXTENT || s.winH() > TDM_MAX_EXTENT)
    {
        return "window too large for one TDM tile";
    }
    if(s.threadsPerBlock() > GFX1250_MAX_THREADS_PER_BLOCK)
    {
        return std::to_string(s.threadsPerBlock()) + " threads per block";
    }
    if(s.ldsBytes() > GFX1250_LDS_BYTES)
    {
        return "LDS " + std::to_string(s.ldsBytes()) + " B > " + std::to_string(GFX1250_LDS_BYTES);
    }
    const auto vgprs = 8 * k.tilesK * (k.tilesP + s.steps()) + 8 * k.tilesP;
    if(vgprs > SMALL_WMMA_VGPR_BUDGET)
    {
        return "~" + std::to_string(vgprs) + " VGPRs of accumulators and fragments";
    }
    return schedulerReason(k.sched);
}

inline Refusal refuse(const SmallcWmmaKernel& kernel, const Problem& problem)
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
    const SmallcWmmaSpec s{kernel, p};
    if(auto why = smallcReason(s))
    {
        return why;
    }
    if(auto why = smallTileEfficiencyReason(p, kernel.outH, kernel.outW))
    {
        return why;
    }
    if(s.blockK() > 2 * p.kpg)
    {
        return "block_k " + std::to_string(s.blockK()) + " is over twice the "
               + std::to_string(p.kpg) + " output channels";
    }
    return std::nullopt;
}

inline Plan plan(const SmallcWmmaKernel& kernel, const Problem& problem)
{
    DirectProblem p{};
    directProblem(problem, p);
    const SmallcWmmaSpec s{kernel, p};
    return directPlan(kernel.baked.label, problem, s.grid(), s.threadsPerBlock());
}

// ---------------------------------------------------------------------------------------
// direct_fwd_smallk_wmma (FwdSmallkWmmaSpec).
// ---------------------------------------------------------------------------------------

struct SmallkWmmaKernel
{
    DirectBaked baked;
    int64_t wavesC, wavesP, tilesC, tilesP, outH, outW, ck, stages;
    bool prefetch;
    std::optional<std::string> sched;
    int64_t waveSize;

    static SmallkWmmaKernel parse(const InstanceGeometry& g)
    {
        return {DirectBaked::parse(g),
                g.knob("waves_c", 1),
                g.knob("waves_p", 4),
                g.knob("tiles_c", 3),
                g.knob("tiles_p", 6),
                g.knob("out_h", 16),
                g.knob("out_w", 16),
                g.knob("ck", 32),
                g.knob("stages", 2),
                g.boolKnob("prefetch", false),
                schedKnob(g),
                WMMA_SPEC_WAVE_SIZE};
    }
};

/// FwdSmallkWmmaSpec's derived quantities on one problem.
struct SmallkWmmaSpec
{
    const SmallkWmmaKernel& k;
    const DirectProblem& p;

    /// Real GEMM columns: R * S * K.
    int64_t ncols() const
    {
        return p.kh * p.kw * p.kpg;
    }
    int64_t winH() const
    {
        return (k.outH - 1) * p.stride + p.kh;
    }
    int64_t winW() const
    {
        return (k.outW - 1) * p.stride + p.kw;
    }
    int64_t blockC() const
    {
        return k.wavesC * k.tilesC * 16;
    }
    int64_t blockP() const
    {
        return k.wavesP * k.tilesP * 16;
    }
    int64_t threadsPerBlock() const
    {
        return k.wavesC * k.wavesP * k.waveSize;
    }
    int64_t xPitch() const
    {
        return k.ck + CH_PAD;
    }
    int64_t wPitch() const
    {
        return k.ck + CH_PAD;
    }
    int64_t zPitch() const
    {
        return blockC() + 8;
    }
    /// The GEMM pixels plus the last window row's wrap past the window.
    int64_t xElems() const
    {
        return (blockP() + p.padColumns() + 1) * xPitch();
    }
    int64_t weiElems() const
    {
        return blockC() * wPitch();
    }
    /// The bf16/fp16 TDM ring, or the fp32 overlap-add buffer Z, whichever is larger.
    int64_t ldsBytes() const
    {
        const auto ring = 2 * k.stages * (xElems() + weiElems());
        const auto z = 4 * blockP() * zPitch();
        return std::max(ring, z);
    }
    Dim3 grid() const
    {
        return {p.n * ceilDiv(p.ho(), k.outH) * ceilDiv(p.wo(), k.outW), 1, 1};
    }
};

/// fwd_smallk_reason.
inline Refusal smallkReason(const SmallkWmmaSpec& s)
{
    const auto& k = s.k;
    const auto& p = s.p;
    if(auto why = smallCommonReason(p, k.waveSize, false))
    {
        return why;
    }
    if(p.kpg > SMALLK_MAX_K)
    {
        return "K " + std::to_string(p.kpg) + " > " + std::to_string(SMALLK_MAX_K);
    }
    if(p.cpg % 8 != 0)
    {
        return "C must be a multiple of 8 (16-byte input rows)";
    }
    if(k.ck % WMMA_K != 0 || k.ck <= 0)
    {
        return "ck must be a positive multiple of 32";
    }
    if(std::min({k.wavesC, k.wavesP, k.tilesC, k.tilesP}) < 1)
    {
        return "waves and tiles must be >= 1";
    }
    if(s.blockC() < s.ncols() || s.blockC() - s.ncols() >= 16)
    {
        return "block_c " + std::to_string(s.blockC())
               + " must be R*S*K = " + std::to_string(s.ncols()) + " rounded up to 16";
    }
    if(s.winH() * s.winW() > s.blockP())
    {
        return "input window larger than the GEMM pixels";
    }
    if(k.stages < 2)
    {
        return "stages must be >= 2";
    }
    if(k.wavesC * k.wavesP < 2)
    {
        return "needs at least two waves";
    }
    if(s.threadsPerBlock() > GFX1250_MAX_THREADS_PER_BLOCK)
    {
        return std::to_string(s.threadsPerBlock()) + " threads per block";
    }
    if(s.ldsBytes() > GFX1250_LDS_BYTES)
    {
        return "LDS " + std::to_string(s.ldsBytes()) + " B > " + std::to_string(GFX1250_LDS_BYTES);
    }
    const auto vgprs = 8 * k.tilesC * k.tilesP + 8 * (k.tilesC + k.tilesP) * (k.prefetch ? 2 : 1);
    if(vgprs > SMALL_WMMA_VGPR_BUDGET)
    {
        return "~" + std::to_string(vgprs) + " VGPRs of accumulators and fragments";
    }
    return schedulerReason(k.sched);
}

inline Refusal refuse(const SmallkWmmaKernel& kernel, const Problem& problem)
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
    const SmallkWmmaSpec s{kernel, p};
    if(auto why = smallkReason(s))
    {
        return why;
    }
    return smallTileEfficiencyReason(p, kernel.outH, kernel.outW);
}

inline Plan plan(const SmallkWmmaKernel& kernel, const Problem& problem)
{
    DirectProblem p{};
    directProblem(problem, p);
    const SmallkWmmaSpec s{kernel, p};
    return directPlan(kernel.baked.label, problem, s.grid(), s.threadsPerBlock());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
