/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Source of truth for the decision tree:
//   /home/jascampb/AutoResearch/MIOPEN_INTEGRATION.md
// Derived from gfx950 perf-DB analysis (val geomean_slowdown = 1.0171, below the
// ~1.05 measurement-noise floor; 0 invalid picks on 30k val problems).

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/gfx950_rules.hpp>

#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>

namespace miopen {
namespace ai {
namespace gfx950 {

namespace {

// Look up solver IDs once on first call and cache. Names must exactly match
// the MIOpen solver class names registered in solver.cpp.
struct SolverIds
{
    solver::Id fwd_nhwc;
    solver::Id bwd_nhwc;
    solver::Id wrw_nhwc;
    solver::Id group_fwd;
    solver::Id group_bwd;
    solver::Id group_wrw;
    solver::Id winograd_3x2;
    solver::Id winograd_2x3;

    SolverIds()
        : fwd_nhwc("ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC"),
          bwd_nhwc("ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC"),
          wrw_nhwc("ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC"),
          group_fwd("ConvHipImplicitGemmGroupFwdXdlops"),
          group_bwd("ConvHipImplicitGemmGroupBwdXdlops"),
          group_wrw("ConvHipImplicitGemmGroupWrwXdlops"),
          winograd_3x2("ConvBinWinogradRxSf3x2"),
          winograd_2x3("ConvBinWinogradRxSf2x3")
    {
    }
};

const SolverIds& Ids()
{
    static const SolverIds inst;
    return inst;
}

solver::Id PickWrW(const conv::ProblemDescription& p)
{
    if(p.GetGroupCount() != 1)
        return Ids().group_wrw;
    return Ids().wrw_nhwc;
}

solver::Id PickBwdData(const conv::ProblemDescription& p)
{
    const auto g  = p.GetGroupCount();
    const auto fy = static_cast<int>(p.GetWeightsHeight());
    const auto fx = static_cast<int>(p.GetWeightsWidth());
    const auto sh = p.GetKernelStrideH();
    const auto oc = static_cast<int>(p.GetOutChannels());
    const auto c  = static_cast<int>(p.GetInChannels());

    if(p.IsBfp16())
    {
        if(g != 1)
            return Ids().group_bwd;
        if(c <= 3) // low-channel (RGB/grayscale)
            return Ids().group_bwd;
        if((fy == 7 && fx == 7) || (fy == 11 && fx == 11) || (fy == 5 && fx == 5))
            return Ids().group_bwd;
        return Ids().bwd_nhwc;
    }

    if(p.IsFp32())
    {
        if(g != 1)
            return Ids().winograd_3x2;
        if(c <= 8)
        {
            if(fy == 11 && sh == 4) // Winograd not applicable here.
                return Ids().bwd_nhwc;
            return Ids().winograd_3x2;
        }
        if(fy == 7 && fx == 7 && sh >= 2)
            return Ids().winograd_3x2;
        if(fy == 4 && fx == 4)
            return Ids().winograd_3x2;
        if(fy == 5 && fx == 20 && sh == 2)
            return Ids().winograd_3x2;
        if(fy == 1 && fx == 1 && sh == 1 && oc <= 64)
            return Ids().winograd_3x2;
        return Ids().bwd_nhwc;
    }

    if(p.IsFp16())
    {
        if(g != 1)
            return Ids().winograd_2x3;
        if(c <= 8)
        {
            if(sh >= 2 &&
               ((fy == 3 && fx == 3) || (fy == 7 && fx == 7) || (fy == 5 && fx == 20)))
                return Ids().winograd_3x2;
            return Ids().group_bwd;
        }
        if(fy == 11)
            return Ids().group_bwd;
        if(fy == 7 && fx == 7 && sh >= 2)
            return Ids().winograd_3x2;
        if(fy == 7 && fx == 7)
            return Ids().group_bwd;
        if(fy == 5 && fx == 20 && sh == 1)
            return Ids().group_bwd;
        return Ids().bwd_nhwc;
    }

    return Ids().bwd_nhwc;
}

solver::Id PickFwdFp32(const conv::ProblemDescription& p)
{
    const auto g  = p.GetGroupCount();
    const auto fy = static_cast<int>(p.GetWeightsHeight());
    const auto fx = static_cast<int>(p.GetWeightsWidth());
    const auto sy = p.GetKernelStrideH();
    const auto sx = p.GetKernelStrideW();
    const auto c  = static_cast<int>(p.GetInChannels());

    if(g != 1)
    {
        if(g <= 84)
        {
            if(fy == 3 && fx == 3 && sy == 1 && sx == 1)
                return Ids().winograd_2x3;
            return Ids().winograd_3x2;
        }
        return Ids().group_fwd;
    }

    if(p.GetOutLayout() == "NHWC") // FWD_NHWC requires NCHW output.
        return Ids().winograd_3x2;
    if(c <= 8)
    {
        if(fy == 11 && sy == 4)
            return Ids().fwd_nhwc;
        return Ids().winograd_3x2;
    }
    if(fy == 7 && fx == 7)
        return Ids().winograd_3x2;
    if(fy == 5 && fx >= 10)
        return Ids().winograd_3x2;
    return Ids().fwd_nhwc;
}

solver::Id PickFwdFp16(const conv::ProblemDescription& p)
{
    const auto g  = p.GetGroupCount();
    const auto fy = static_cast<int>(p.GetWeightsHeight());
    const auto fx = static_cast<int>(p.GetWeightsWidth());
    const auto sh = p.GetKernelStrideH();
    const auto sw = p.GetKernelStrideW();
    const auto oc = static_cast<int>(p.GetOutChannels());

    if(p.GetInLayout() == "NHWC")
        return Ids().group_fwd;
    if(g == 1 && p.GetOutLayout() == "NHWC")
        return Ids().winograd_3x2;

    if(g == 1 && fy == 3 && fx == 3 && sh == 1 && sw == 1)
    {
        const auto c = static_cast<int>(p.GetInChannels());
        if(c <= 8)
            return Ids().winograd_3x2;
        if(c <= 64)
            return Ids().fwd_nhwc;
        return Ids().group_fwd;
    }

    if(g == 1 && fy == 7 && fx == 7 && sh == 2 && sw == 2)
        return Ids().winograd_3x2;
    if(g != 1 && fy == 3 && fx == 3 && sh == 2 && sw == 2 && (g == 32 || g == 64))
        return Ids().winograd_3x2;
    if(g == 1 && fy == 3 && fx == 3 && sh == 2 && sw == 2 && oc <= 128)
        return Ids().winograd_3x2;

    if(g != 1)
        return Ids().group_fwd;
    if(fx == 1 || fx == 7)
        return Ids().fwd_nhwc;
    if(oc == 64)
        return Ids().fwd_nhwc;
    return Ids().group_fwd;
}

solver::Id PickFwdBfp16(const conv::ProblemDescription& p)
{
    const auto g  = p.GetGroupCount();
    const auto fx = static_cast<int>(p.GetWeightsWidth());
    const auto oc = static_cast<int>(p.GetOutChannels());

    if(g != 1)
        return Ids().group_fwd;
    if(p.GetInLayout() == "NHWC")
        return Ids().group_fwd;
    if(fx == 1 || fx == 7)
        return Ids().fwd_nhwc;
    if(oc == 64)
        return Ids().fwd_nhwc;
    return Ids().group_fwd;
}

} // namespace

solver::Id PickSolver(const conv::ProblemDescription& problem)
{
    // 3D and transposed conv are not covered by the rules; defer to Tunanet.
    // TODO: per the integration brief, 3D could be a trivial direction-keyed
    // single-solver dispatch (DIR_3D[dir]); revisit if 3D shows wins.
    if(!problem.Is2d())
        return {};
    if(problem.GetConv().mode != miopenConvolution)
        return {};

    switch(problem.GetDirection())
    {
    case conv::Direction::BackwardWeights:
        return PickWrW(problem);
    case conv::Direction::BackwardData:
        return PickBwdData(problem);
    case conv::Direction::Forward:
        if(problem.IsFp16())
            return PickFwdFp16(problem);
        if(problem.IsBfp16())
            return PickFwdBfp16(problem);
        if(problem.IsFp32())
            return PickFwdFp32(problem);
        // int8 and other dtypes: defer to Tunanet.
        return {};
    }
    return {};
}

} // namespace gfx950
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
