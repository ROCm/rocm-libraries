// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdConvArgs.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdInstance.hpp"

/**
 * @file Gfx1250ConvFwdDirect.hpp
 * @brief What every gfx1250 forward direct-conv family shares: the direct problem view,
 *        the baked-capability check, and the direct ABI.
 *
 * A port of dispatch/grouped_convolution.py's direct path ahead of the family's own
 * validator: `_direct_problem` (one stride; a column padding of its own rides `pad_w`),
 * kernels/conv/kernel_identity.identity_supports_problem for a direct identity (every
 * baked filter field, and a nonzero baked cpg/kpg, must match), and
 * kernels/conv/direct_variants.direct_spec_for_problem's checks before the spec is built
 * (only PER_AXIS_PAD_VARIANTS take a column padding of their own). The family headers then
 * port the spec rebuilt on the real problem: its validator, tile-efficiency floor, grid and
 * workgroup size. Every direct launch is one step with the direct ABI (ConvArgs).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

/// rocKE's DirectConvProblem: the direct-conv view of a forward convolution.
struct DirectProblem
{
    int64_t n, h, w, groups, cpg, kpg, kh, kw, pad, stride, dilH, dilW;
    int64_t padW; ///< -1: PAD (one padding for both axes).
    DataType dtype;

    int64_t padColumns() const ///< PAD_W
    {
        return padW < 0 ? pad : padW;
    }
    int64_t totalC() const
    {
        return groups * cpg;
    }
    int64_t totalK() const
    {
        return groups * kpg;
    }
    bool dilated() const
    {
        return dilH != 1 || dilW != 1;
    }
    int64_t ho() const
    {
        return floorDiv(h + 2 * pad - (kh - 1) * dilH - 1, stride) + 1;
    }
    int64_t wo() const
    {
        return floorDiv(w + 2 * padColumns() - (kw - 1) * dilW - 1, stride) + 1;
    }
};

/// The baked fields of a direct instance (KernelIdentity), read once.
struct DirectBaked : BakedShape
{
    std::string label;
    std::string algorithm;

    static DirectBaked parse(const InstanceGeometry& g)
    {
        return {BakedShape::parse(g), g.label(), g.algorithm()};
    }
};

/// The wave size of the WMMA direct specs (direct_fwd_wmma and the gstream, smallc, smallk
/// and tap-outer families). They are not DIRECT_WAVE_PARAMETRIC, so make_direct_spec gives
/// them neither the identity's nor the target's wave size, and no knob of theirs sets it:
/// the spec keeps its dataclass default, 32, whatever the identity's `wave_size` says.
constexpr int64_t WMMA_SPEC_WAVE_SIZE = 32;

/// Whether @p family takes a column padding of its own (direct_variants.PER_AXIS_PAD_VARIANTS).
inline bool takesColumnPadding(const std::string& family)
{
    return family == "direct_fwd_smallk_wmma" || family == "direct_fwd_wmma";
}

/// The direct view of @p p (`_direct_problem`), or the refusal of the shape outright.
inline Refusal directProblem(const Problem& p, DirectProblem& out)
{
    if(p.strideH != p.strideW)
    {
        return "direct conv takes one stride";
    }
    out = DirectProblem{p.n,
                        p.h,
                        p.w,
                        p.groups,
                        p.cpg(),
                        p.kpg(),
                        p.y,
                        p.x,
                        p.padH,
                        p.strideH,
                        p.dilationH,
                        p.dilationW,
                        p.padW == p.padH ? -1 : p.padW,
                        p.dtype};
    return std::nullopt;
}

/// identity_supports_problem for a direct identity, then direct_spec_for_problem's checks
/// ahead of the spec: every baked filter field matches, a nonzero baked cpg/kpg matches, and
/// a column padding of its own is taken only by the per-axis families.
inline Refusal directCapability(const DirectBaked& b, const DirectProblem& p)
{
    if(b.filterH != p.kh || b.filterW != p.kw)
    {
        return "baked filter differs";
    }
    if(b.strideH != p.stride || b.strideW != p.stride)
    {
        return "baked stride differs";
    }
    if(b.dilationH != p.dilH || b.dilationW != p.dilW)
    {
        return "baked dilation differs";
    }
    if(b.padH != p.pad || b.padW != p.padColumns())
    {
        return "baked padding differs";
    }
    if(b.cpg != 0 && b.cpg != p.cpg)
    {
        return "baked cpg differs";
    }
    if(b.kpg != 0 && b.kpg != p.kpg)
    {
        return "baked kpg differs";
    }
    if(p.padColumns() != p.pad && !takesColumnPadding(b.algorithm))
    {
        return "one padding for both axes";
    }
    return std::nullopt;
}

/// The one step of a direct launch: the direct ABI over the problem's geometry.
inline Plan directPlan(const std::string& label, const Problem& problem, Dim3 grid, int64_t threads)
{
    Step step;
    step.kernel = label;
    step.grid = grid;
    step.block = {threads, 1, 1};
    step.args = directFwdArgs(ConvGeometry::of(problem));
    return Plan{{std::move(step)}, {}};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
