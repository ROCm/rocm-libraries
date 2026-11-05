// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdDirect.hpp"

/**
 * @file Gfx1250ConvFwdGstreamWmma.hpp
 * @brief Admission and launch of the gfx1250 `direct_fwd_gstream_wmma` instances, and the
 *        wave32 WMMA forward gate the other WMMA direct families include from here.
 *
 * Ports, from rocKE's library:
 * - kernels/common/conv_fwd_wmma_gate.py: wmma_fwd_gate, wmma_fwd_limits_reason,
 *   wmma_fwd_tensor_bytes_reason and the halo helpers (with conv_direct_grouped's
 *   problem_dims_reason, pad_w_reason, one_padding_reason, I32_MAX and
 *   rocke.core.codegen_policy.scheduler_strategy_reason);
 * - kernels/common/conv_direct_fwd_gstream_wmma.py: FwdGstreamWmmaSpec (its derived
 *   quantities, grid, threads_per_block), fwd_gstream_reason and
 *   fwd_gstream_tile_efficiency;
 * - kernels/conv/direct_variants.direct_spec_for_problem's rule after the validator: the
 *   tile-efficiency floor (< 0.5 refuses).
 *
 * The DirectConvProblem arithmetic is Python's: DirectProblem::ho() / wo() floor-divide,
 * and ceilDiv / alignUp are exact for any sign. A ZeroDivisionError of the spec is a
 * refusal (the dispatcher catches it).
 *
 * Unchecked agreements (restated by hand, not derived from the Python):
 * - gfx1250's ArchTarget: wave32, TDM, the 16x16x32 fp16/bf16 WMMA atom,
 *   limits.max_threads_per_block = 1024, lds_capacity_bytes = 327680;
 * - the scheduler strategies of codegen_policy.SchedulerStrategy;
 * - the spec dataclasses' knob defaults (FwdGstreamWmmaSpec here, the others in theirs);
 * - GSTREAM_VGPR_BUDGET = 448, CH_PAD = 8, WMMA_K = 32, the TDM extent cap 0xFFFF;
 * - these families take no wave size from the identity or the target
 *   (direct_spec_takes_target_wave is false for them) and have no wave_size knob: the spec
 *   keeps its default, 32 (WMMA_SPEC_WAVE_SIZE).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

// ---------------------------------------------------------------------------------------
// The shared WMMA forward gate (conv_fwd_wmma_gate.py).
// ---------------------------------------------------------------------------------------

/// codegen_policy.scheduler_strategy_reason: no strategy, or a known one.
inline Refusal schedulerReason(const std::optional<std::string>& sched)
{
    if(!sched)
    {
        return std::nullopt;
    }
    for(const char* known :
        {"max-ilp", "max-memory-clause", "iterative-ilp", "iterative-minreg", "iterative-maxocc"})
    {
        if(*sched == known)
        {
            return std::nullopt;
        }
    }
    return "unsupported scheduler_strategy '" + *sched + "'";
}

/// The `sched` knob: absent is no strategy.
inline std::optional<std::string> schedKnob(const InstanceGeometry& g)
{
    if(!g.hasKnob("sched"))
    {
        return std::nullopt;
    }
    return g.knobString("sched", "");
}

/// wmma_fwd_gate on gfx1250 for a spec of @p waveSize: a wave32 spec, every problem extent
/// at least 1 (problem_dims_reason), then the padding (pad_w_reason when @p perAxisPad,
/// else one_padding_reason). gfx1250 has TDM and both WMMA atoms; DataType is fp16/bf16.
inline Refusal wmmaFwdGate(const DirectProblem& p, int64_t waveSize, bool perAxisPad)
{
    if(waveSize != GFX1250_WAVE_SIZE)
    {
        return "a wave32 kernel";
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
            return std::string(name) + " must be >= 1 (got " + std::to_string(value) + ")";
        }
    }
    if(p.padW < -1)
    {
        return "pad_w must be -1 (PAD) or a column padding of at least 0";
    }
    if(!perAxisPad && p.padColumns() != p.pad)
    {
        return "one padding for both axes";
    }
    return std::nullopt;
}

/// wmma_fwd_tensor_bytes_reason: X, W and Y within i32 byte offsets.
inline Refusal wmmaFwdTensorBytesReason(const DirectProblem& p)
{
    const auto bytes = std::max({2 * p.n * p.h * p.w * p.totalC(),
                                 2 * p.totalK() * p.kh * p.kw * p.cpg,
                                 2 * p.n * p.ho() * p.wo() * p.totalK()});
    if(bytes > I32_MAX)
    {
        return "tensors of " + std::to_string(bytes) + " B exceed the i32 byte offsets";
    }
    return std::nullopt;
}

/// wmma_fwd_limits_reason: a known scheduler, i32 byte offsets, an i32 flat block id.
inline Refusal wmmaFwdLimitsReason(const DirectProblem& p,
                                   const std::optional<std::string>& sched,
                                   int64_t blocks)
{
    if(auto why = schedulerReason(sched))
    {
        return why;
    }
    if(auto why = wmmaFwdTensorBytesReason(p))
    {
        return why;
    }
    if(blocks > I32_MAX)
    {
        return std::to_string(blocks) + " workgroups exceed the i32 block id";
    }
    return std::nullopt;
}

/// wmma_fwd_halo_extent: input rows (columns) a tile of output rows (columns) reads.
inline int64_t wmmaFwdHaloExtent(int64_t stride, int64_t tile, int64_t k, int64_t dil)
{
    return stride * (tile - 1) + (k - 1) * dil + 1;
}

/// The direct_spec_for_problem floor: refuses when num / den < 0.5 (Python float division,
/// compared exactly on the integers), and a zero denominator (ZeroDivisionError).
inline Refusal tileEfficiencyReason(int64_t num, int64_t den)
{
    if(den == 0)
    {
        return "division by zero";
    }
    const bool below = den > 0 ? 2 * num < den : 2 * num > den;
    if(below)
    {
        std::array<char, 64> text{};
        std::snprintf(text.data(),
                      text.size(),
                      "tile efficiency %.2f < 0.5 on this shape",
                      static_cast<double>(num) / static_cast<double>(den));
        return std::string(text.data());
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------------------
// direct_fwd_gstream_wmma (FwdGstreamWmmaSpec).
// ---------------------------------------------------------------------------------------

constexpr int64_t GSTREAM_VGPR_BUDGET = 448;

struct GstreamWmmaKernel
{
    DirectBaked baked;
    int64_t wavesK, wavesP, tilesK, tilesP, tileW, walk, stages;
    bool phaseSplit, prefetch;
    int64_t pfDepth;
    std::optional<std::string> sched;
    int64_t waveSize;

    static GstreamWmmaKernel parse(const InstanceGeometry& g)
    {
        return {DirectBaked::parse(g),
                g.knob("waves_k", 1),
                g.knob("waves_p", 4),
                g.knob("tiles_k", 1),
                g.knob("tiles_p", 2),
                g.knob("tile_w", 16),
                g.knob("walk", 8),
                g.knob("stages", 2),
                g.boolKnob("phase_split", false),
                g.boolKnob("prefetch", true),
                g.knob("pf_depth", 1),
                schedKnob(g),
                WMMA_SPEC_WAVE_SIZE};
    }
};

/// FwdGstreamWmmaSpec's derived quantities on one problem.
struct GstreamWmmaSpec
{
    const GstreamWmmaKernel& k;
    const DirectProblem& p;

    bool span16() const
    {
        return p.cpg <= 16;
    }
    /// kpg rounded up to whole 16-row M tiles (8 / 16 channels per group: kpg itself).
    int64_t kpad() const
    {
        return span16() ? p.kpg : alignUp(p.kpg, 16);
    }
    int64_t blockK() const
    {
        return k.wavesK * k.tilesK * 16;
    }
    int64_t blockC() const
    {
        return floorDiv(std::max(blockK(), kpad()) * p.cpg, kpad());
    }
    bool shareB() const
    {
        return !span16() && kpad() % (k.tilesK * 16) == 0;
    }
    int64_t blockP() const
    {
        return k.wavesP * k.tilesP * 16;
    }
    int64_t tileH() const
    {
        return k.tileW > 0 ? floorDiv(blockP(), k.tileW) : 0;
    }
    int64_t haloH() const
    {
        return wmmaFwdHaloExtent(p.stride, tileH(), p.kh, p.dilH);
    }
    int64_t haloW() const
    {
        return wmmaFwdHaloExtent(p.stride, k.tileW, p.kw, p.dilW);
    }
    int64_t phasesW() const
    {
        return k.phaseSplit ? p.stride : 1;
    }
    /// wmma_fwd_halo_cols: a whole halo row, or one of the stride column phases.
    int64_t haloCols() const
    {
        return k.phaseSplit ? k.tileW + floorDiv((p.kw - 1) * p.dilW, p.stride) : haloW();
    }
    /// wmma_fwd_lead_cols, which the spec feeds the row padding PAD.
    int64_t leadCols() const
    {
        return k.phaseSplit ? ceilDiv(p.pad, p.stride) : p.pad;
    }
    int64_t xPitch() const
    {
        return blockC() + CH_PAD;
    }
    int64_t phaseElems() const
    {
        return haloH() * haloCols() * xPitch() + leadCols() * xPitch();
    }
    int64_t slotElems() const
    {
        if(k.phaseSplit)
        {
            return phasesW() * phaseElems();
        }
        return (haloH() * haloW() + p.pad) * xPitch();
    }
    int64_t ldsBytes() const
    {
        return 2 * k.stages * slotElems();
    }
    /// len(k_steps()): tap pairs (span16), else every tap's 32-channel chunks.
    int64_t kStepCount() const
    {
        const auto taps = p.kh * p.kw;
        return span16() ? ceilDiv(taps, 2) : taps * ceilDiv(p.cpg, WMMA_K);
    }
    int64_t threadsPerBlock() const
    {
        return k.wavesK * k.wavesP * k.waveSize;
    }
    int64_t nBands() const
    {
        return ceilDiv(p.ho(), tileH());
    }
    int64_t gridX() const
    {
        const auto kBlocks = floorDiv(p.groups * kpad(), blockK());
        const auto wTiles = ceilDiv(p.wo(), k.tileW);
        const auto walkers = ceilDiv(p.n * nBands(), k.walk);
        return kBlocks * wTiles * walkers;
    }
};

/// fwd_gstream_reason.
inline Refusal gstreamReason(const GstreamWmmaSpec& s)
{
    const auto& k = s.k;
    const auto& p = s.p;
    if(auto why = wmmaFwdGate(p, k.waveSize, false))
    {
        return why;
    }
    if(p.stride != 1 && p.stride != 2)
    {
        return "stride " + std::to_string(p.stride) + " not in (1, 2)";
    }
    if(p.dilated())
    {
        return "dilation is not served";
    }
    if(p.kh != p.kw)
    {
        return "filter is not square";
    }
    if(p.pad < 0 || p.pad > p.kh - 1)
    {
        return "PAD outside 0..K-1";
    }
    if(p.ho() < 1 || p.wo() < 1)
    {
        return "empty output";
    }
    if(p.cpg != p.kpg)
    {
        return "cpg != kpg";
    }
    if(p.cpg != 8 && p.cpg != 16 && (p.cpg < WMMA_K || p.cpg % 8 != 0))
    {
        return "cpg not 8, 16 or a multiple of 8 from 32";
    }
    if(p.cpg > 16 && p.cpg % WMMA_K != 0 && alignUp(p.cpg, WMMA_K) - p.cpg > CH_PAD)
    {
        return "cpg: its last k-step reads past the LDS row skew";
    }
    if(std::min({k.wavesK, k.wavesP, k.tilesK, k.tilesP}) < 1)
    {
        return "waves and tiles must be >= 1";
    }
    if(s.blockK() % s.kpad() != 0 && s.kpad() % s.blockK() != 0)
    {
        return "block_k is neither whole groups nor part of one";
    }
    if((p.groups * s.kpad()) % s.blockK() != 0)
    {
        return "K not a multiple of block_k " + std::to_string(s.blockK());
    }
    if(k.pfDepth < 1 || (k.pfDepth > 1 && !k.prefetch))
    {
        return "pf_depth: >= 1, and > 1 only with prefetch";
    }
    if(k.stages < 2 || k.walk < 1)
    {
        return "stages must be >= 2 and walk >= 1";
    }
    if(k.tileW < 1 || s.blockP() % k.tileW != 0)
    {
        return "tile_w must divide the pixels of a band";
    }
    if(k.phaseSplit)
    {
        if(p.stride != 2)
        {
            return "phase_split serves stride 2 only";
        }
        if(k.wavesK * k.wavesP < 2)
        {
            return "phase_split needs two waves (one transfer each)";
        }
    }
    if(s.threadsPerBlock() > GFX1250_MAX_THREADS_PER_BLOCK)
    {
        return std::to_string(s.threadsPerBlock()) + " threads per block";
    }
    if(std::max({s.xPitch(), s.haloW(), s.haloH()}) > TDM_MAX_EXTENT)
    {
        return "tile too wide for a TDM descriptor";
    }
    if(s.ldsBytes() > GFX1250_LDS_BYTES)
    {
        return "LDS " + std::to_string(s.ldsBytes()) + " B > " + std::to_string(GFX1250_LDS_BYTES);
    }
    const auto bSets = s.shareB() ? k.tilesP : k.tilesK * k.tilesP;
    const auto vgprs = 8 * k.tilesK * k.tilesP + 8 * k.tilesK * s.kStepCount()
                       + 8 * bSets * (k.prefetch ? 1 + k.pfDepth : 1);
    if(vgprs > GSTREAM_VGPR_BUDGET)
    {
        return "~" + std::to_string(vgprs) + " VGPRs of weights, accumulators and fragments";
    }
    return wmmaFwdLimitsReason(p, k.sched, s.gridX());
}

inline Refusal refuse(const GstreamWmmaKernel& kernel, const Problem& problem)
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
    const GstreamWmmaSpec s{kernel, p};
    if(auto why = gstreamReason(s))
    {
        return why;
    }
    // fwd_gstream_tile_efficiency: real pixels over the padded band grid.
    return tileEfficiencyReason(p.ho() * p.wo(),
                                s.nBands() * s.tileH() * alignUp(p.wo(), kernel.tileW));
}

inline Plan plan(const GstreamWmmaKernel& kernel, const Problem& problem)
{
    DirectProblem p{};
    directProblem(problem, p);
    const GstreamWmmaSpec s{kernel, p};
    return directPlan(kernel.baked.label, problem, {s.gridX(), 1, 1}, s.threadsPerBlock());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
