// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdDirect.hpp"

/**
 * @file Gfx1250ConvFwdDepthwise.hpp
 * @brief Admission and launch of the gfx1250 scalar depthwise forward instances:
 *        `direct_depthwise`, `direct_depthwise_tiled`, `direct_depthwise_col` and
 *        `direct_depthwise_spatial` (cpg = kpg = 1, one lane per channel).
 *
 * A port of kernels/common/conv_direct_grouped.py: the specs DirectDepthwiseSpec,
 * DirectDepthwiseTiledSpec, DirectDepthwiseColSpec and DirectDepthwiseSpatialSpec as
 * direct_variants.make_direct_spec rebuilds them on the real problem (all four are
 * DIRECT_WAVE_PARAMETRIC: their wave is the identity's `wave_size`; the col spec takes its
 * dtype from the problem), the validators direct_variants.validate_direct_spec runs for
 * them (is_valid_depthwise_spec, is_valid_depthwise_tiled_spec with the spec's validate(),
 * DirectDepthwiseColSpec.validate() then is_valid_depthwise_col_spec,
 * is_valid_depthwise_spatial_spec) -- direct_spec_for_problem adds no rule after them --
 * and their branches of direct_launch_geometry.
 *
 * The validators read the gfx1250 ArchTarget (wave 32, 1024 threads per block, 256 VGPRs):
 * the engine serves gfx1250 only (its graph match), and its packaging adapter takes only
 * gfx1250 instances.
 *
 * Unchecked agreements (restated by hand, pinned only by the golden vectors):
 * - the spec defaults: depthwise block_w 8, block_waves 1, cpl 1; tiled block_w 8,
 *   block_h 4, block_waves 1, unroll_rows false, cpl 1; col block_w 1, block_waves 1,
 *   block_h 16, max_live_f32 none; spatial block_waves 1;
 * - gfx1250's arch_specs.json limits: wave 32, max_threads_per_block 1024, vgprs 256
 *   (the col live-f32 budget is vgprs * 3 / 8 = 96);
 * - I32_MAX, _DW_MAX_KH/_DW_MAX_KW 32, _DW_MAX_PRELOAD_TAPS 200, _DW_CPL_MAX_ACC 192,
 *   _DW_TILED_REG_BUDGET 256, _DW_TILED_UNROLL_MAX_FMAS 6144;
 * - channel_totals_reason, forward_padding_reason, dilation_reason, pad_w_reason,
 *   _depthwise_filter_reason, _depthwise_cpl_reason, the tiled spec's n_cols / live_regs /
 *   unrolled_fmas and the col spec's live_f32.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

namespace depthwise_detail
{

constexpr int64_t TARGET_VGPRS = 256;
constexpr int64_t MAX_FILTER = 32;
constexpr int64_t MAX_PRELOAD_TAPS = 200;
constexpr int64_t CPL_MAX_ACC = 192;
constexpr int64_t TILED_REG_BUDGET = 256;
constexpr int64_t TILED_UNROLL_MAX_FMAS = 6144;

/// The shared admission prefix: the direct view and the baked capability.
inline Refusal directPrefix(const DirectBaked& b, const Problem& problem, DirectProblem& p)
{
    if(auto why = directProblem(problem, p))
    {
        return why;
    }
    return directCapability(b, p);
}

inline Refusal channelTotalsReason(const DirectProblem& p)
{
    if(p.totalC() > I32_MAX || p.totalK() > I32_MAX)
    {
        return "groups * cpg and groups * kpg must fit in i32";
    }
    return std::nullopt;
}

inline Refusal dilationReason(const DirectProblem& p)
{
    if(p.dilated())
    {
        return "this direct conv kernel has no dilation";
    }
    return std::nullopt;
}

/// The row-streaming forward kernels index output rows by input rows: odd filter, "same"
/// padding PAD == (KH - 1) / 2.
inline Refusal forwardPaddingReason(const DirectProblem& p)
{
    if(auto why = dilationReason(p))
    {
        return why;
    }
    if(p.kh % 2 == 0 || p.kw % 2 == 0)
    {
        return "forward direct conv needs odd filter extents";
    }
    if(p.pad != floorDiv(p.kh - 1, 2))
    {
        return "forward direct conv needs 'same' padding PAD == (KH-1)/2";
    }
    return std::nullopt;
}

inline Refusal filterReason(const DirectProblem& p)
{
    if(p.kh < 1 || p.kh > MAX_FILTER)
    {
        return "KH must be in 1..32";
    }
    if(p.kw < 1 || p.kw > MAX_FILTER)
    {
        return "KW must be in 1..32";
    }
    return std::nullopt;
}

inline Refusal cplReason(int64_t cpl, int64_t groups)
{
    if(cpl != 1 && cpl != 2 && cpl != 4)
    {
        return "cpl must be 1, 2 or 4";
    }
    if(groups % cpl != 0)
    {
        return "channels are not a multiple of cpl";
    }
    return std::nullopt;
}

} // namespace depthwise_detail

// ---------------------------------------------------------------------------------------
// direct_depthwise (DirectDepthwiseSpec, row-streaming, weights preloaded)
// ---------------------------------------------------------------------------------------

struct DepthwiseKernel
{
    DirectBaked baked;
    int64_t blockW, blockWaves, cpl, waveSize;

    static DepthwiseKernel parse(const InstanceGeometry& g)
    {
        const auto wave = g.integer("wave_size");
        return {DirectBaked::parse(g),
                g.knob("block_w", 8),
                g.knob("block_waves", 1),
                g.knob("cpl", 1),
                wave};
    }

    int64_t threads() const
    {
        return blockWaves * waveSize;
    }
    int64_t blockCh() const
    {
        return blockWaves * waveSize * cpl;
    }
};

inline Refusal refuse(const DepthwiseKernel& k, const Problem& problem)
{
    using namespace depthwise_detail;
    DirectProblem p{};
    if(auto why = directPrefix(k.baked, problem, p))
    {
        return why;
    }
    if(auto why = channelTotalsReason(p))
    {
        return why;
    }
    if(auto why = forwardPaddingReason(p))
    {
        return why;
    }
    if(p.cpg != 1 || p.kpg != 1)
    {
        return "cpg and kpg must both be 1";
    }
    if(auto why = filterReason(p))
    {
        return why;
    }
    const auto taps = p.kh * p.kw;
    if(taps > MAX_PRELOAD_TAPS)
    {
        return "preloaded filter exceeds 200 taps";
    }
    if(auto why = cplReason(k.cpl, p.groups))
    {
        return why;
    }
    if(k.cpl > 1 && (k.cpl * taps > MAX_PRELOAD_TAPS || k.cpl * k.blockW * p.kh > CPL_MAX_ACC))
    {
        return "packed weights or accumulator rows spill";
    }
    return std::nullopt;
}

inline Plan plan(const DepthwiseKernel& k, const Problem& problem)
{
    DirectProblem p{};
    directProblem(problem, p);
    const Dim3 grid{ceilDiv(p.wo(), k.blockW), ceilDiv(p.groups, k.blockCh()), p.n};
    return directPlan(k.baked.label, problem, grid, k.threads());
}

// ---------------------------------------------------------------------------------------
// direct_depthwise_tiled (DirectDepthwiseTiledSpec, output-stationary, forward mode)
// ---------------------------------------------------------------------------------------

struct DepthwiseTiledKernel
{
    DirectBaked baked;
    int64_t blockW, blockH, blockWaves, cpl, waveSize;
    bool unrollRows;

    static DepthwiseTiledKernel parse(const InstanceGeometry& g)
    {
        const auto wave = g.integer("wave_size");
        // dgrad mode is the direct_depthwise_tiled_dgrad variant, not a forward instance.
        return {DirectBaked::parse(g),
                g.knob("block_w", 8),
                g.knob("block_h", 4),
                g.knob("block_waves", 1),
                g.knob("cpl", 1),
                wave,
                g.boolKnob("unroll_rows", false)};
    }

    int64_t threads() const
    {
        return blockWaves * waveSize;
    }
    int64_t blockCh() const
    {
        return blockWaves * waveSize * cpl;
    }
};

inline Refusal refuse(const DepthwiseTiledKernel& k, const Problem& problem)
{
    using namespace depthwise_detail;
    DirectProblem p{};
    if(auto why = directPrefix(k.baked, problem, p))
    {
        return why;
    }
    // DirectDepthwiseTiledSpec.validate() (forward mode).
    if(p.padW < -1)
    {
        return "pad_w must be -1 (PAD) or at least 0";
    }
    if(p.padColumns() != p.pad)
    {
        return "one padding for both axes";
    }
    if(p.cpg != 1 || p.kpg != 1)
    {
        return "requires cpg=kpg=1";
    }
    if(k.blockW < 1 || k.blockH < 1 || k.blockWaves < 1 || k.waveSize < 1)
    {
        return "block_w, block_h, block_waves and wave_size must be positive";
    }
    if(p.stride < 1 || p.pad < 0)
    {
        return "stride must be >= 1 and PAD >= 0";
    }
    const auto ho = p.ho();
    const auto wo = p.wo();
    if(ho < 1 || wo < 1)
    {
        return "filter does not fit the padded input";
    }
    // is_valid_depthwise_tiled_spec.
    if(auto why = dilationReason(p))
    {
        return why;
    }
    if(auto why = filterReason(p))
    {
        return why;
    }
    if(p.stride > p.kh)
    {
        return "stride > KH: some phases see no filter row";
    }
    if(k.waveSize != GFX1250_WAVE_SIZE)
    {
        return "wave_size differs from the gfx1250 wave";
    }
    if(k.threads() > GFX1250_MAX_THREADS_PER_BLOCK)
    {
        return "threads_per_block exceeds arch limit";
    }
    if(auto why = cplReason(k.cpl, p.groups))
    {
        return why;
    }
    const auto nCols = (k.blockW - 1) * p.stride + p.kw;
    const auto liveRegs = k.cpl * (k.blockH * k.blockW + k.blockH * nCols + 2 * p.kw);
    if(liveRegs > TILED_REG_BUDGET)
    {
        return "live registers per lane exceed the 256-register budget";
    }
    if(k.unrollRows && p.kh * k.blockH * k.blockW * p.kw > TILED_UNROLL_MAX_FMAS)
    {
        return "unrolling the filter-row loop emits too many FMAs";
    }
    return std::nullopt;
}

inline Plan plan(const DepthwiseTiledKernel& k, const Problem& problem)
{
    DirectProblem p{};
    directProblem(problem, p);
    const Dim3 grid{
        ceilDiv(p.wo(), k.blockW), ceilDiv(p.groups, k.blockCh()), p.n * ceilDiv(p.ho(), k.blockH)};
    return directPlan(k.baked.label, problem, grid, k.threads());
}

// ---------------------------------------------------------------------------------------
// direct_depthwise_col (DirectDepthwiseColSpec, filter-column loop outermost)
// ---------------------------------------------------------------------------------------

struct DepthwiseColKernel
{
    DirectBaked baked;
    int64_t blockW, blockWaves, blockH, waveSize;
    int64_t maxLiveF32; ///< -1: none (the arch budget alone).

    static DepthwiseColKernel parse(const InstanceGeometry& g)
    {
        const auto wave = g.integer("wave_size");
        // make_direct_spec passes the problem's dtype itself; a dtype knob is a TypeError.
        const auto maxLive = g.knob("max_live_f32", -1);
        if(g.hasKnob("max_live_f32") && maxLive < 0)
        {
            throw std::invalid_argument(g.label() + ": unported max_live_f32 knob");
        }
        return {DirectBaked::parse(g),
                g.knob("block_w", 1),
                g.knob("block_waves", 1),
                g.knob("block_h", 16),
                wave,
                maxLive};
    }

    int64_t threads() const
    {
        return blockWaves * waveSize;
    }
    int64_t blockCh() const
    {
        return blockWaves * waveSize;
    }
    /// resolve_max_live_f32: three eighths of the VGPR file, an explicit knob only tightening.
    int64_t liveF32Budget() const
    {
        const auto budget = depthwise_detail::TARGET_VGPRS * 3 / 8;
        return maxLiveF32 < 0 ? budget : std::min(maxLiveF32, budget);
    }
};

inline Refusal refuse(const DepthwiseColKernel& k, const Problem& problem)
{
    using namespace depthwise_detail;
    DirectProblem p{};
    if(auto why = directPrefix(k.baked, problem, p))
    {
        return why;
    }
    // DirectDepthwiseColSpec.validate().
    if(p.cpg != 1 || p.kpg != 1)
    {
        return "requires cpg=kpg=1";
    }
    if(auto why = dilationReason(p))
    {
        return why;
    }
    if(k.blockH <= 0)
    {
        return "block_h must be > 0";
    }
    // is_valid_depthwise_col_spec; its dtype check passes (fp16/bf16 from the problem), and
    // its repeated cpg/kpg and dilation checks are the ones above.
    if(auto why = channelTotalsReason(p))
    {
        return why;
    }
    if(p.stride < 1)
    {
        return "stride must be >= 1";
    }
    if(p.kh < 1 || p.kw < 1)
    {
        return "filter extents must be >= 1";
    }
    if(p.pad < 0)
    {
        return "PAD must be >= 0";
    }
    const auto ho = p.ho();
    const auto wo = p.wo();
    if(ho < 1 || wo < 1)
    {
        return "filter does not fit the padded input";
    }
    if(ho > p.h)
    {
        return "requires Ho <= H";
    }
    if(p.pad >= p.kh || p.pad >= p.kw)
    {
        return "PAD must be < min(KH, KW)";
    }
    if(k.blockW < 1)
    {
        return "block_w must be >= 1";
    }
    if(k.blockW > wo)
    {
        return "block_w > Wo";
    }
    if(k.blockWaves < 1)
    {
        return "block_waves must be >= 1";
    }
    if(k.waveSize != GFX1250_WAVE_SIZE)
    {
        return "wave_size differs from the gfx1250 wave";
    }
    if(k.threads() > GFX1250_MAX_THREADS_PER_BLOCK)
    {
        return "threads_per_block exceeds the arch limit";
    }
    if(k.blockH * k.blockW + p.kh > k.liveF32Budget())
    {
        return "live f32 per lane exceeds max_live_f32";
    }
    return std::nullopt;
}

inline Plan plan(const DepthwiseColKernel& k, const Problem& problem)
{
    DirectProblem p{};
    directProblem(problem, p);
    const Dim3 grid{
        ceilDiv(p.wo(), k.blockW), ceilDiv(p.groups, k.blockCh()), p.n * ceilDiv(p.ho(), k.blockH)};
    return directPlan(k.baked.label, problem, grid, k.threads());
}

// ---------------------------------------------------------------------------------------
// direct_depthwise_spatial (DirectDepthwiseSpatialSpec, groups <= wave, no channel tile)
// ---------------------------------------------------------------------------------------

struct DepthwiseSpatialKernel
{
    DirectBaked baked;
    int64_t blockWaves, waveSize;

    static DepthwiseSpatialKernel parse(const InstanceGeometry& g)
    {
        const auto wave = g.integer("wave_size");
        return {DirectBaked::parse(g), g.knob("block_waves", 1), wave};
    }

    int64_t threads() const
    {
        return blockWaves * waveSize;
    }
    /// Output columns per block: every wave covers wave_size / groups columns of all channels.
    int64_t blockW(int64_t groups) const
    {
        return blockWaves * (waveSize / groups);
    }
};

inline Refusal refuse(const DepthwiseSpatialKernel& k, const Problem& problem)
{
    using namespace depthwise_detail;
    DirectProblem p{};
    if(auto why = directPrefix(k.baked, problem, p))
    {
        return why;
    }
    if(auto why = channelTotalsReason(p))
    {
        return why;
    }
    if(auto why = forwardPaddingReason(p))
    {
        return why;
    }
    if(p.cpg != 1 || p.kpg != 1)
    {
        return "cpg and kpg must both be 1";
    }
    if(auto why = filterReason(p))
    {
        return why;
    }
    if(p.groups > k.waveSize)
    {
        return "groups > wave_size";
    }
    if(k.waveSize / p.groups == 0)
    {
        return "no W positions per wave";
    }
    return std::nullopt;
}

inline Plan plan(const DepthwiseSpatialKernel& k, const Problem& problem)
{
    DirectProblem p{};
    directProblem(problem, p);
    const Dim3 grid{ceilDiv(p.wo(), k.blockW(p.groups)), 1, p.n};
    return directPlan(k.baked.label, problem, grid, k.threads());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
