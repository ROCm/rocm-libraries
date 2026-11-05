// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdDirect.hpp"

/**
 * @file Gfx1250ConvFwdDirectWmma.hpp
 * @brief Admission and launch of the gfx1250 `direct_fwd_wmma` instances.
 *
 * A port of rocKE's dispatcher for `direct_fwd_wmma`: after the shared direct prefix
 * (Gfx1250ConvFwdDirect.hpp), kernels/conv/direct_variants.direct_spec_for_problem rebuilds
 * kernels/common/conv_direct_fwd_wmma.DirectFwdWmmaSpec on the real problem, runs
 * fwd_wmma_reason (with kernels/common/conv_fwd_wmma_gate.wmma_fwd_gate and
 * wmma_fwd_limits_reason) and refuses a tile efficiency (fwd_wmma_tile_efficiency) below
 * 0.5. The launch is DirectFwdWmmaSpec.launch_geometry(): grid() and threads_per_block.
 *
 * DirectFwdWmmaSpec's derived quantities (vp, halo, LDS footprint, grid) are FwdWmmaSpec's
 * members here, named as in Python. Every knob of the spec is ported, with the dataclass's
 * defaults when the instance does not spell one.
 *
 * Unchecked agreements (restated by hand, pinned only by the golden vectors):
 * - The spec defaults: waves_k 2, waves_p 4, tiles_k 4, tiles_p 4, tile_w 16, tile_n 1,
 *   ck 32, stages 2, wave_size 32, pack_groups 1, pp_group 1, every flag off, sched None.
 * - conv_fwd_wmma_gate: WMMA_K = 32, CH_PAD = 8, the halo extent
 *   `stride * (tile - 1) + (k - 1) * dil + 1`, the phase-split halo columns and lead columns,
 *   problem_dims_reason (N, H, W, groups, cpg, kpg, KH, KW >= 1), pad_w_reason
 *   (pad_w >= -1), the i32 tensor-byte and i32 block-id limits (I32_MAX = 2**31 - 1).
 * - gfx1250's arch facts (core/arch/data/arch_specs.json): wave32, TDM and the fp16 / bf16
 *   16x16x32 fp32 WMMA atom present (so those gate checks always pass), 1024 threads per
 *   block, 327680 B of LDS per workgroup.
 * - FWD_WMMA_VGPR_BUDGET = 512, POLY_ZERO_LANE_CH = 8, the 0xFFFF TDM descriptor limit.
 * - The scheduler strategies of rocke.core.codegen_policy.SchedulerStrategy (checked at
 *   parse: the strategy is a knob, so an unknown one would refuse every problem).
 * - The tile-efficiency floor 0.5 of direct_spec_for_problem.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

namespace direct_fwd_wmma
{

constexpr int64_t VGPR_BUDGET = 512; ///< FWD_WMMA_VGPR_BUDGET
constexpr int64_t POLY_ZERO_LANE_CH = 8;
constexpr double MIN_TILE_EFFICIENCY = 0.5;

/// wmma_fwd_halo_extent: input rows (columns) a tile of @p tile output rows (columns) reads.
inline int64_t haloExtent(int64_t stride, int64_t tile, int64_t k, int64_t dil)
{
    return stride * (tile - 1) + (k - 1) * dil + 1;
}

inline bool isSchedulerStrategy(const std::string& s)
{
    return s == "max-ilp" || s == "max-memory-clause" || s == "iterative-ilp"
           || s == "iterative-minreg" || s == "iterative-maxocc";
}

} // namespace direct_fwd_wmma

struct DirectFwdWmmaKernel
{
    DirectBaked baked;
    int64_t wavesK, wavesP, tilesK, tilesP, tileW, tileN, ck, stages, waveSize;
    int64_t packGroups, ppGroup;
    bool prefetch, splitBarrier, waveEpilogue, phaseSplit, rowRing, tapRing, pingpong;
    bool tapReuse, tapShift, stridedView, foldN, polyView;

    static DirectFwdWmmaKernel parse(const InstanceGeometry& g)
    {
        const auto on = [&g](const char* knob) { return g.boolKnob(knob, false); };
        if(g.hasKnob("sched") && !direct_fwd_wmma::isSchedulerStrategy(g.knobString("sched", "")))
        {
            throw std::invalid_argument(g.label() + ": unknown scheduler strategy "
                                        + g.knobString("sched", ""));
        }
        return {DirectBaked::parse(g),
                g.knob("waves_k", 2),
                g.knob("waves_p", 4),
                g.knob("tiles_k", 4),
                g.knob("tiles_p", 4),
                g.knob("tile_w", 16),
                g.knob("tile_n", 1),
                g.knob("ck", 32),
                g.knob("stages", 2),
                WMMA_SPEC_WAVE_SIZE,
                g.knob("pack_groups", 1),
                g.knob("pp_group", 1),
                on("prefetch"),
                on("split_barrier"),
                on("wave_epilogue"),
                on("phase_split"),
                on("row_ring"),
                on("tap_ring"),
                on("pingpong"),
                on("tap_reuse"),
                on("tap_shift"),
                on("strided_view"),
                on("fold_n"),
                on("poly_view")};
    }

    int64_t blockK() const
    {
        return wavesK * tilesK * 16;
    }
    int64_t blockP() const
    {
        return wavesP * tilesP * 16;
    }
    int64_t tileH() const ///< 0 when tile_w * tile_n is not positive (refused).
    {
        const auto d = tileW * tileN;
        return d > 0 ? blockP() / d : 0;
    }
    int64_t waves() const
    {
        return wavesK * wavesP;
    }
    int64_t threadsPerBlock() const
    {
        return waves() * waveSize;
    }
    int64_t xPitch() const
    {
        return ck + CH_PAD;
    }
    int64_t epiPitch() const
    {
        return blockK() + CH_PAD;
    }
};

/// DirectFwdWmmaSpec on one problem: the knobs, the real problem and the view `vp` the
/// tile loop runs (the problem itself, its polyphase view or its strided-GEMM view).
struct FwdWmmaSpec
{
    const DirectFwdWmmaKernel& k;
    DirectProblem problem;
    DirectProblem vp;

    FwdWmmaSpec(const DirectFwdWmmaKernel& kernel, const DirectProblem& p)
        : k(kernel)
        , problem(p)
        , vp(view(kernel, p))
    {
    }

    static DirectProblem view(const DirectFwdWmmaKernel& k, const DirectProblem& p)
    {
        auto v = p;
        if(k.polyView)
        {
            const auto m = p.kw / p.stride;
            v.h = 1;
            v.w = p.wo() + m - 1;
            v.cpg = p.stride * p.cpg;
            v.kh = 1;
            v.kw = m;
        }
        else if(k.stridedView)
        {
            v.n = k.foldN ? 1 : p.n;
            v.h = k.foldN ? p.n * p.ho() : p.ho();
            v.w = p.wo();
            v.cpg = p.kh * p.kw * p.cpg;
            v.kh = 1;
            v.kw = 1;
        }
        else
        {
            return v;
        }
        v.pad = 0;
        v.padW = -1;
        v.stride = 1;
        return v;
    }

    int64_t cpgEff() const
    {
        return k.packGroups * vp.cpg;
    }
    int64_t kpgEff() const
    {
        return k.packGroups * problem.kpg;
    }
    int64_t groupsEff() const
    {
        return problem.groups / k.packGroups;
    }
    int64_t ringTaps() const ///< Taps per weight slot.
    {
        if(k.tapRing)
        {
            return 1;
        }
        return k.rowRing ? vp.kw : vp.kh * vp.kw;
    }
    int64_t phasesW() const
    {
        return k.phaseSplit ? vp.stride : 1;
    }
    int64_t haloH() const
    {
        return direct_fwd_wmma::haloExtent(vp.stride, k.tileH(), vp.kh, vp.dilH);
    }
    int64_t haloW() const
    {
        return direct_fwd_wmma::haloExtent(vp.stride, k.tileW, vp.kw, vp.dilW);
    }
    int64_t haloCols() const ///< wmma_fwd_halo_cols: columns of one (phase) sub-halo.
    {
        return k.phaseSplit ? k.tileW + ((vp.kw - 1) * vp.dilW) / vp.stride : haloW();
    }
    int64_t leadCols() const ///< wmma_fwd_lead_cols
    {
        return k.phaseSplit ? ceilDiv(vp.padColumns(), vp.stride) : vp.padColumns();
    }
    int64_t phaseElems() const
    {
        return haloH() * haloCols() * k.xPitch() + leadCols() * k.xPitch();
    }
    int64_t imageElems() const
    {
        return k.phaseSplit ? phasesW() * phaseElems() : haloH() * haloW() * k.xPitch();
    }
    int64_t haloElems() const
    {
        return k.tileN * imageElems();
    }
    int64_t weiElems() const
    {
        return ringTaps() * k.blockK() * k.xPitch();
    }
    int64_t haloSlack() const
    {
        return k.phaseSplit ? 0 : vp.padColumns() * k.xPitch();
    }
    int64_t ldsBytes() const
    {
        const auto ring = k.rowRing || k.tapRing
                              ? 2 * (haloElems() + haloSlack()) + k.stages * weiElems()
                              : k.stages * (haloElems() + haloSlack() + weiElems());
        auto epi = k.blockP() * k.epiPitch();
        if(k.waveEpilogue)
        {
            epi = k.waves() * k.tilesP * 16 * (k.tilesK * 16 + CH_PAD);
        }
        return 2 * std::max(ring, epi);
    }
    int64_t gridX() const
    {
        const auto nKt = ceilDiv(kpgEff(), k.blockK());
        const auto nHt = ceilDiv(vp.ho(), k.tileH());
        const auto nWt = ceilDiv(vp.wo(), k.tileW);
        return ceilDiv(vp.n, k.tileN) * groupsEff() * nHt * nWt * nKt;
    }
    /// fwd_wmma_tile_efficiency: the share of the padded tile grid that is real work.
    double tileEfficiency() const
    {
        const auto work = vp.n * vp.ho() * vp.wo() * kpgEff();
        const auto padded = alignUp(vp.n, k.tileN) * alignUp(vp.ho(), k.tileH())
                            * alignUp(vp.wo(), k.tileW) * alignUp(kpgEff(), k.blockK());
        return static_cast<double>(work) / static_cast<double>(padded);
    }
};

/// wmma_fwd_gate's problem checks (problem_dims_reason, pad_w_reason); the arch checks
/// always pass on gfx1250 but the spec's own wave size.
inline Refusal fwdWmmaGate(const DirectFwdWmmaKernel& k, const DirectProblem& p)
{
    if(k.waveSize != GFX1250_WAVE_SIZE)
    {
        return "direct_fwd_wmma is a wave32 kernel";
    }
    const std::initializer_list<std::pair<const char*, int64_t>> dims = {{"N", p.n},
                                                                         {"H", p.h},
                                                                         {"W", p.w},
                                                                         {"groups", p.groups},
                                                                         {"cpg", p.cpg},
                                                                         {"kpg", p.kpg},
                                                                         {"KH", p.kh},
                                                                         {"KW", p.kw}};
    for(const auto& [name, value] : dims)
    {
        if(value < 1)
        {
            return std::string(name) + " must be >= 1";
        }
    }
    if(p.padW < -1)
    {
        return "pad_w must be -1 (PAD) or a column padding of at least 0";
    }
    return std::nullopt;
}

/// The strided_view / fold_n rules of fwd_wmma_reason, on the real problem.
inline Refusal stridedViewReason(const DirectFwdWmmaKernel& k, const DirectProblem& p)
{
    if(!k.stridedView)
    {
        return k.foldN ? Refusal("fold_n builds on strided_view") : std::nullopt;
    }
    if(p.pad != 0 || p.padColumns() != 0 || p.dilated() || p.kh != p.kw
       || (p.kh != 1 && p.kh != p.stride))
    {
        return "strided_view serves undilated 1x1 or KxK stride-K, no padding";
    }
    if(p.stride < 2)
    {
        return "strided_view serves stride > 1";
    }
    if(k.phaseSplit || k.packGroups > 1 || k.tapReuse)
    {
        return "strided_view is a 1x1 stride-1 loop (no phases, packs or reuse)";
    }
    if(p.kh > 1 && p.groups > 1)
    {
        return "a KxK strided_view serves ungrouped convs";
    }
    if(p.kh > 1 && k.ck != 0 && (p.kw * p.cpg) % k.ck != 0)
    {
        return "a patch run is not whole ck chunks";
    }
    if(k.foldN && (p.h != p.stride * p.ho() || k.tileN != 1))
    {
        return "fold_n needs H == stride * Ho and one image per tile";
    }
    return std::nullopt;
}

/// The poly_view rules of fwd_wmma_reason, on the real problem.
inline Refusal polyViewReason(const DirectFwdWmmaKernel& k, const DirectProblem& p)
{
    const auto s = p.stride;
    const auto padW = p.padColumns();
    if(k.stridedView || p.h != 1 || p.kh != 1 || p.pad != 0 || p.dilated())
    {
        return "poly_view serves 1-D (H = 1, 1 x R) undilated filters";
    }
    if(s < 2 || p.kw % s != 0 || padW < 0 || padW >= s || p.w % s != 0)
    {
        return "poly_view needs R = m * S, column padding below S and W % S == 0";
    }
    if(0 < 2 * padW && 2 * padW < s)
    {
        // The view's last pixel is then all input, not a trailing zero.
        return "poly_view needs column padding 0 or at least S / 2";
    }
    if(p.groups != 1)
    {
        return "poly_view serves ungrouped convs";
    }
    if(k.phaseSplit || k.packGroups > 1)
    {
        return "poly_view is a plain stride-1 1-D loop";
    }
    if(k.ck == 0 || p.cpg % k.ck != 0 || (padW * p.cpg) % k.ck != 0)
    {
        return "a chunk must lie in one input pixel";
    }
    if(padW != 0 && k.ck > direct_fwd_wmma::POLY_ZERO_LANE_CH * k.waveSize)
    {
        return "a padded poly_view zeroes at most 8 * wave_size channels of a chunk";
    }
    if(k.tileH() != 1)
    {
        return "poly_view tiles one row (tile_w * tile_n == block_p)";
    }
    return std::nullopt;
}

/// The shape rules of fwd_wmma_reason from the stride check on, on @p p: the real problem,
/// or the polyphase view with poly_view.
inline Refusal shapeReason(const FwdWmmaSpec& spec, const DirectProblem& p)
{
    const auto& k = spec.k;
    if(p.stride != 1 && p.stride != 2)
    {
        return "stride not in (1, 2)";
    }
    if(p.dilH < 1 || p.dilW < 1)
    {
        return "dilation < 1";
    }
    // Non-square filters: 1 x R stride-1 rows (dilated or not) and undilated rectangles.
    if(p.kh != p.kw)
    {
        if(k.packGroups > 1 || k.tapReuse || k.phaseSplit)
        {
            return "a non-square filter is served unpacked, without tap_reuse or phase split";
        }
        if(p.dilated() && (p.kh != 1 || p.stride != 1))
        {
            return "a dilated non-square filter is served only as 1 x R, stride 1";
        }
    }
    if(p.padColumns() != p.pad)
    {
        if(k.packGroups > 1)
        {
            return "per-axis padding is served unpacked";
        }
        if(p.dilated() && (p.kh != 1 || p.stride != 1))
        {
            return "per-axis padding with dilation is served only as 1 x R, stride 1";
        }
    }
    const auto maxDil = std::max(p.dilH, p.dilW);
    if(p.pad < 0 || p.pad > (p.kh - 1) * maxDil)
    {
        return "PAD outside 0..(K-1)*dilation";
    }
    if(p.padColumns() < 0 || p.padColumns() > (p.kw - 1) * maxDil)
    {
        return "PAD_W outside 0..(KW-1)*dilation";
    }
    if(p.ho() < 1 || p.wo() < 1)
    {
        return "empty output";
    }
    const auto g = k.packGroups;
    if(g < 1)
    {
        return "pack_groups < 1";
    }
    if(g > 1)
    {
        const auto packable = [](int64_t c) { return c == 2 || c == 4 || c == 8 || c == 16; };
        if(p.groups % g != 0)
        {
            return "groups not a multiple of pack_groups";
        }
        if(!packable(p.cpg) || !packable(p.kpg))
        {
            return "packed cpg / kpg not in (2, 4, 8, 16)";
        }
        if(spec.cpgEff() > k.ck)
        {
            return "packed cpg > ck (one X chunk)";
        }
        if(spec.kpgEff() > k.blockK())
        {
            return "packed kpg > block_k";
        }
    }
    if(spec.cpgEff() % 8 != 0 || spec.kpgEff() % 8 != 0)
    {
        return "cpg and kpg must be multiples of 8 (16-byte rows)";
    }
    return std::nullopt;
}

/// The tiling and schedule rules of fwd_wmma_reason (knobs only, but for tap_reuse's stride).
inline Refusal tilingReason(const DirectFwdWmmaKernel& k, const DirectProblem& p)
{
    if(k.ck % WMMA_K != 0 || k.ck <= 0)
    {
        return "ck must be a positive multiple of 32";
    }
    if(std::min({k.wavesK, k.wavesP, k.tilesK, k.tilesP}) < 1)
    {
        return "waves and tiles must be >= 1";
    }
    if(k.stages < 2)
    {
        return "stages must be >= 2";
    }
    const auto tw = k.tileW;
    if(tw < 1 || k.tileN < 1)
    {
        return "tile_w and tile_n must be >= 1";
    }
    if(k.blockP() % (tw * k.tileN) != 0)
    {
        return "tile_w x tile_n must divide block_p";
    }
    if(k.tileN > 1 && (k.tileH() * tw) % 16 != 0)
    {
        return "an image tile must hold whole 16-pixel tiles";
    }
    if(k.tileN > k.waves() - 1)
    {
        return "tile_n needs one more wave than images (one X transfer per image)";
    }
    if(k.pingpong)
    {
        const auto issuers = k.phaseSplit ? 2 * k.tileN + 1 : k.tileN + 1;
        if(k.rowRing || k.tapRing || k.splitBarrier || k.prefetch)
        {
            return "pingpong replaces the ring variants, split_barrier and prefetch";
        }
        if(k.waves() % 2 != 0 || issuers > k.waves() / 2)
        {
            return "pingpong needs an even wave count with every issuer in rank 0";
        }
        if(k.waveEpilogue)
        {
            return "pingpong uses the block epilogue";
        }
    }
    if(k.ppGroup < 1 || (k.ppGroup > 1 && !k.pingpong))
    {
        return "pp_group: >= 1, and > 1 only with pingpong";
    }
    if((k.rowRing || k.tapRing) && (k.packGroups > 1 || k.splitBarrier))
    {
        return "row_ring / tap_ring serve unpacked tiles without split_barrier";
    }
    if(k.rowRing && k.tapRing)
    {
        return "row_ring and tap_ring are exclusive";
    }
    if(k.tapReuse)
    {
        if(k.rowRing || k.tapRing)
        {
            return "tap_reuse needs all filter rows of a column in one weight slot";
        }
        if(k.pingpong)
        {
            return "tap_reuse is not combined with pingpong";
        }
        if(p.stride != 1 || k.tileN != 1 || k.phaseSplit)
        {
            return "tap_reuse serves stride 1, one image per block";
        }
        if(tw % 16 != 0 || (k.tilesP * 16) % tw != 0)
        {
            return "tap_reuse needs whole 16-pixel row pieces and whole rows per wave";
        }
    }
    if(k.tapShift)
    {
        if(!k.tapReuse)
        {
            return "tap_shift builds on tap_reuse";
        }
        if(p.dilW > 15)
        {
            return "tap_shift rotates within a 16-lane row";
        }
    }
    if(k.phaseSplit)
    {
        if(p.stride != 2)
        {
            return "phase_split serves stride 2 only";
        }
        if(k.tileN * 2 + 1 > k.waves())
        {
            return "phase_split needs 2 * tile_n + 1 waves (one transfer each)";
        }
    }
    if(k.waveEpilogue)
    {
        const auto wpx = k.tilesP * 16;
        if(wpx % tw != 0 || (k.tileH() * tw) % wpx != 0)
        {
            return "wave_epilogue needs whole rows of one image per wave";
        }
    }
    return std::nullopt;
}

/// The resource rules of fwd_wmma_reason: threads, TDM extents, LDS and VGPRs.
inline Refusal resourceReason(const FwdWmmaSpec& spec, const DirectProblem& p)
{
    const auto& k = spec.k;
    if(k.threadsPerBlock() > GFX1250_MAX_THREADS_PER_BLOCK)
    {
        return std::to_string(k.threadsPerBlock()) + " threads per block";
    }
    if(k.waves() < 2)
    {
        return "needs at least two waves (one TDM issuer per operand)";
    }
    if(std::max({k.xPitch(), k.epiPitch(), spec.haloW(), spec.haloH()}) > TDM_MAX_EXTENT)
    {
        return "tile too wide for a TDM descriptor";
    }
    if(spec.ldsBytes() > GFX1250_LDS_BYTES)
    {
        return "LDS " + std::to_string(spec.ldsBytes()) + " B > "
               + std::to_string(GFX1250_LDS_BYTES);
    }
    // Accumulators plus one A and one B fragment set per wave (two with prefetch, one per
    // grouped step with pingpong); tap_reuse keeps the rows it reuses.
    auto vgprs = 8 * k.tilesK * k.tilesP + 8 * (k.tilesK + k.tilesP) * (k.prefetch ? 2 : k.ppGroup);
    if(k.tapReuse)
    {
        vgprs += 8 * ((p.kh - 1) * p.dilH * (k.tileW / 16));
    }
    if(vgprs > direct_fwd_wmma::VGPR_BUDGET)
    {
        return "~" + std::to_string(vgprs) + " VGPRs of accumulators and fragments";
    }
    return std::nullopt;
}

/// wmma_fwd_limits_reason past the scheduler (checked at parse): i32 byte offsets into X, W
/// and Y of the real problem, and an i32 flat block id.
inline Refusal limitsReason(const FwdWmmaSpec& spec)
{
    const auto& p = spec.problem;
    const auto bytes = std::max({2 * p.n * p.h * p.w * p.totalC(),
                                 2 * p.totalK() * p.kh * p.kw * p.cpg,
                                 2 * p.n * p.ho() * p.wo() * p.totalK()});
    if(bytes > I32_MAX)
    {
        return "tensors of " + std::to_string(bytes) + " B exceed the i32 byte offsets";
    }
    if(spec.gridX() > I32_MAX)
    {
        return "workgroups exceed the i32 block id";
    }
    return std::nullopt;
}

inline Refusal refuse(const DirectFwdWmmaKernel& kernel, const Problem& problem)
{
    DirectProblem real{};
    if(auto why = directProblem(problem, real))
    {
        return why;
    }
    if(auto why = directCapability(kernel.baked, real))
    {
        return why;
    }
    const FwdWmmaSpec spec(kernel, real);
    if(auto why = fwdWmmaGate(kernel, real))
    {
        return why;
    }
    if(auto why = stridedViewReason(kernel, real))
    {
        return why;
    }
    if(kernel.polyView)
    {
        if(auto why = polyViewReason(kernel, real))
        {
            return why;
        }
    }
    // From here fwd_wmma_reason checks the view a poly_view kernel runs, else the real
    // problem (a strided_view kernel's too).
    const auto& p = kernel.polyView ? spec.vp : real;
    if(auto why = shapeReason(spec, p))
    {
        return why;
    }
    if(auto why = tilingReason(kernel, p))
    {
        return why;
    }
    if(auto why = resourceReason(spec, p))
    {
        return why;
    }
    if(auto why = limitsReason(spec))
    {
        return why;
    }
    if(spec.tileEfficiency() < direct_fwd_wmma::MIN_TILE_EFFICIENCY)
    {
        return "tile efficiency < 0.5 on this shape";
    }
    return std::nullopt;
}

inline Plan plan(const DirectFwdWmmaKernel& kernel, const Problem& problem)
{
    DirectProblem real{};
    if(auto why = directProblem(problem, real))
    {
        throw std::invalid_argument(kernel.baked.label + ": " + *why);
    }
    const FwdWmmaSpec spec(kernel, real);
    return directPlan(kernel.baked.label, problem, {spec.gridX(), 1, 1}, kernel.threadsPerBlock());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
