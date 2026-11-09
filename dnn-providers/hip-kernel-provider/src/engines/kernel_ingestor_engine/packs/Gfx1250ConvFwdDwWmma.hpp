// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdDirect.hpp"

/**
 * @file Gfx1250ConvFwdDwWmma.hpp
 * @brief Admission and launch of the gfx1250 `direct_dw_fwd_wmma` instances (Toeplitz WMMA
 *        depthwise forward, stride 1).
 *
 * A port of kernels/common/conv_direct_dw_fwd_wmma.py: DwFwdWmmaSpec (rebuilt on the real
 * problem by direct_variants.make_direct_spec with the instance's knobs; its wave is the
 * spec's own wave32, not the target's), dw_fwd_wmma_reason / is_valid_dw_fwd_wmma_spec
 * (the validator direct_variants.validate_direct_spec runs; direct_spec_for_problem adds no
 * rule after it for this family), and DwFwdWmmaSpec.grid() / threads_per_block (the
 * launch_geometry() direct_launch_geometry falls back to).
 *
 * Unchecked agreements (restated by hand, pinned only by the golden vectors):
 * - the spec defaults block_waves 4, block_h 8, w_fold 1, prefetch_rows 2;
 * - DW_FWD_WMMA_WAVE 32, DW_FWD_WMMA_MAX_WAVES 16, DW_FWD_WMMA_WIDE_WAVES 8,
 *   DW_FWD_WMMA_WIDE_FRAG_VGPRS 112, DW_FWD_WMMA_MAX_FRAG_VGPRS 224,
 *   DW_FWD_WMMA_MAX_UNROLL 1024;
 * - conv_direct_grouped's DW_DGRAD_MFMA_CH 8, DW_DGRAD_MFMA_W_FOLDS {1, 2, 4},
 *   DW_DGRAD_MFMA_MAX_PREFETCH_ROWS 4, DW_DGRAD_MFMA_LDS_BUDGET 64 KiB,
 *   DW_DGRAD_WIN_MAX_TENSOR_BYTES 2**30 - 1;
 * - the helpers _dw_mfma_passes, _dw_mfma_tile_w, _dw_mfma_in_passes, _dw_mfma_lds_bytes
 *   and the spec's frag_vgprs / unrolled_wmmas estimates;
 * - problem_dims_reason, dilation_reason and one_padding_reason (pad_w_reason included);
 * - DW_FWD_WMMA_ARCHES = (gfx1250,): the engine serves gfx1250 only (its graph match), and
 *   its packaging adapter takes only gfx1250 instances.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

struct DwFwdWmmaKernel
{
    DirectBaked baked;
    int64_t blockWaves, blockH, wFold, prefetchRows;

    static constexpr int64_t WAVE = 32;
    static constexpr int64_t CHANNELS_PER_WAVE = 8; // DW_DGRAD_MFMA_CH
    static constexpr int64_t MAX_WAVES = 16;
    static constexpr int64_t WIDE_WAVES = 8;
    static constexpr int64_t WIDE_FRAG_VGPRS = 112;
    static constexpr int64_t MAX_FRAG_VGPRS = 224;
    static constexpr int64_t MAX_UNROLL = 1024;
    static constexpr int64_t MAX_PREFETCH_ROWS = 4;
    static constexpr int64_t LDS_BUDGET = int64_t{64} * 1024;
    static constexpr int64_t MAX_TENSOR_BYTES = (int64_t{1} << 30) - 1;

    static DwFwdWmmaKernel parse(const InstanceGeometry& g)
    {
        return {DirectBaked::parse(g),
                g.knob("block_waves", 4),
                g.knob("block_h", 8),
                g.knob("w_fold", 1),
                g.knob("prefetch_rows", 2)};
    }

    int64_t threads() const
    {
        return blockWaves * WAVE;
    }
    int64_t blockCh() const
    {
        return blockWaves * CHANNELS_PER_WAVE;
    }
    int64_t tileW() const
    {
        return 32 / wFold;
    }
    /// 16x16x32 passes per filter row (_dw_mfma_passes).
    static int64_t passes(int64_t kw)
    {
        return (kw + 4) / 4;
    }
    /// 16-byte loads per thread and window row (_dw_mfma_in_passes).
    int64_t inPasses(int64_t kw) const
    {
        const auto window = tileW() + 4 * passes(kw) - 2;
        const auto perPx = blockCh() / CHANNELS_PER_WAVE;
        return ceilDiv(wFold * window * perPx, threads());
    }
    int64_t fragVgprs(const DirectProblem& p) const
    {
        const auto kwp = passes(p.kw);
        return 8 * (p.kh * kwp + p.kh + kwp) + 4 * prefetchRows * inPasses(p.kw);
    }
    int64_t ldsBytes(int64_t kw) const
    {
        const auto perPx = blockCh() / CHANNELS_PER_WAVE;
        const auto inPx = ceilDiv(inPasses(kw) * threads(), perPx);
        const auto outPx = wFold * tileW();
        return 2 * (inPx + outPx) * (blockCh() + CHANNELS_PER_WAVE) * 2;
    }
};

inline Refusal refuse(const DwFwdWmmaKernel& k, const Problem& problem)
{
    DirectProblem p{};
    if(auto why = directProblem(problem, p))
    {
        return why;
    }
    if(auto why = directCapability(k.baked, p))
    {
        return why;
    }
    // problem_dims_reason
    for(const int64_t extent : {p.n, p.h, p.w, p.groups, p.cpg, p.kpg, p.kh, p.kw})
    {
        if(extent < 1)
        {
            return "every problem extent must be >= 1";
        }
    }
    if(p.cpg != 1 || p.kpg != 1)
    {
        return "requires cpg=kpg=1";
    }
    if(p.stride != 1)
    {
        return "requires stride=1";
    }
    if(p.dilated())
    {
        return "no dilation";
    }
    if(p.padW < -1)
    {
        return "pad_w must be -1 (PAD) or at least 0";
    }
    if(p.padColumns() != p.pad)
    {
        return "one padding for both axes";
    }
    const auto ho = p.ho();
    const auto wo = p.wo();
    if(p.pad < 0 || p.pad >= std::max(p.kh, p.kw) || ho <= 0 || wo <= 0)
    {
        return "degenerate geometry";
    }
    if(k.blockH < 1)
    {
        return "block_h must be >= 1";
    }
    if(k.blockWaves < 1 || k.blockWaves > DwFwdWmmaKernel::MAX_WAVES)
    {
        return "block_waves must be 1..16";
    }
    if(k.wFold != 1 && k.wFold != 2 && k.wFold != 4)
    {
        return "w_fold must be one of 1, 2, 4";
    }
    if(k.prefetchRows < 1 || k.prefetchRows > DwFwdWmmaKernel::MAX_PREFETCH_ROWS)
    {
        return "prefetch_rows must be 1..4";
    }
    if(p.groups % DwFwdWmmaKernel::CHANNELS_PER_WAVE != 0)
    {
        return "groups % 8 != 0";
    }
    const auto frag = k.fragVgprs(p);
    if(k.blockWaves > DwFwdWmmaKernel::WIDE_WAVES && frag > DwFwdWmmaKernel::WIDE_FRAG_VGPRS)
    {
        return "wide blocks need fragments <= 112 VGPRs";
    }
    if(frag > DwFwdWmmaKernel::MAX_FRAG_VGPRS)
    {
        return "fragments exceed 224 VGPRs";
    }
    if(k.blockH * p.kh * DwFwdWmmaKernel::passes(p.kw) > DwFwdWmmaKernel::MAX_UNROLL)
    {
        return "unrolled body too large";
    }
    if(k.ldsBytes(p.kw) > DwFwdWmmaKernel::LDS_BUDGET)
    {
        return "LDS staging exceeds the 64 KiB budget";
    }
    // Every in-range byte offset plus a sentinel stays a valid unsigned 32-bit offset.
    const auto xBytes = p.n * p.h * p.w * p.groups * 2;
    const auto yBytes = p.n * ho * wo * p.groups * 2;
    if(std::max(xBytes, yBytes) > DwFwdWmmaKernel::MAX_TENSOR_BYTES)
    {
        return "X/Y exceed the sentinel addressing range";
    }
    // Grid Z = ceil(N / w_fold) * ceil(Ho / block_h) must not exceed the hardware cap.
    if(ceilDiv(p.n, k.wFold) * ceilDiv(ho, k.blockH) > MAX_GRID_DIM)
    {
        return "grid Z exceeds the hardware cap";
    }
    return std::nullopt;
}

inline Plan plan(const DwFwdWmmaKernel& k, const Problem& problem)
{
    DirectProblem p{};
    directProblem(problem, p);
    const Dim3 grid{ceilDiv(p.wo(), k.tileW()),
                    ceilDiv(p.groups, k.blockCh()),
                    ceilDiv(p.n, k.wFold) * ceilDiv(p.ho(), k.blockH)};
    return directPlan(k.baked.label, problem, grid, k.threads());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
