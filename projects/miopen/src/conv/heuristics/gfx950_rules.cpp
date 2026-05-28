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

// Flat sequence of carve-out rules derived from gfx950 perf-DB analysis.
// Each rule's `iterNN` tag matches the iteration in the source decision tree;
// the order is significant — earlier rules win.

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

struct SolverIds
{
    solver::Id fwd_nhwc;
    solver::Id bwd_nhwc;
    solver::Id wrw_nhwc;
    solver::Id group_fwd;
    solver::Id group_bwd;
    solver::Id group_wrw;
    solver::Id group_3d_fwd;
    solver::Id group_3d_bwd;
    solver::Id group_3d_wrw;
    solver::Id winograd_3x2;
    solver::Id winograd_2x3;

    SolverIds()
        : fwd_nhwc("ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC"),
          bwd_nhwc("ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC"),
          wrw_nhwc("ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC"),
          group_fwd("ConvHipImplicitGemmGroupFwdXdlops"),
          group_bwd("ConvHipImplicitGemmGroupBwdXdlops"),
          group_wrw("ConvHipImplicitGemmGroupWrwXdlops"),
          group_3d_fwd("ConvHipImplicitGemm3DGroupFwdXdlops"),
          group_3d_bwd("ConvHipImplicitGemm3DGroupBwdXdlops"),
          group_3d_wrw("ConvHipImplicitGemm3DGroupWrwXdlops"),
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

enum class DType
{
    Fp32,
    Fp16,
    Bfp16,
    Other
};

DType GetDType(const conv::ProblemDescription& p)
{
    if(p.IsFp32())
        return DType::Fp32;
    if(p.IsFp16())
        return DType::Fp16;
    if(p.IsBfp16())
        return DType::Bfp16;
    return DType::Other;
}

int GetDirInt(const conv::ProblemDescription& p)
{
    switch(p.GetDirection())
    {
    case conv::Direction::Forward: return 1;
    case conv::Direction::BackwardData: return 2;
    case conv::Direction::BackwardWeights: return 4;
    }
    return 0;
}

solver::Id Pick3d(int direction)
{
    switch(direction)
    {
    case 1: return Ids().group_3d_fwd;
    case 2: return Ids().group_3d_bwd;
    case 4: return Ids().group_3d_wrw;
    default: return Ids().group_bwd;
    }
}

} // namespace

solver::Id PickSolver(const conv::ProblemDescription& problem)
{
    if(problem.GetConv().mode != miopenConvolution)
        return {};
    if(!problem.Is2d() && !problem.Is3d())
        return {};

    const int direction        = GetDirInt(problem);
    const DType dtype          = GetDType(problem);
    const int g                = static_cast<int>(problem.GetGroupCount());
    const int sp               = problem.Is3d() ? 3 : 2;
    const std::string& out_l   = problem.GetOutLayout();
    const int fy               = static_cast<int>(problem.GetWeightsHeight());
    const int fx               = static_cast<int>(problem.GetWeightsWidth());
    const int sh               = static_cast<int>(problem.GetKernelStrideH());
    const int sw               = static_cast<int>(problem.GetKernelStrideW());
    const int c                = static_cast<int>(problem.GetInChannels());
    const int oc               = static_cast<int>(problem.GetOutChannels());
    const int h                = static_cast<int>(problem.GetInHeight());
    const int w                = static_cast<int>(problem.GetInWidth());
    const int n                = static_cast<int>(problem.GetBatchSize());

    // NHWC out_layout: Group* rarely a candidate; route per (dir, dtype).
    if(g == 1 && sp == 2 && out_l == "NHWC" && (dtype == DType::Fp16 || dtype == DType::Fp32))
    {
        // iter17: dir=2 fp16 c<=4 -> W3.
        if(direction == 2 && dtype == DType::Fp16 && c <= 4)
            return Ids().winograd_3x2;
        if(direction == 1)
        {
            // iter106: fp16 fy=3 sh=1 c<=128 oc<=256 h<=4 -> GROUP_FWD.
            if(dtype == DType::Fp16 && fy == 3 && sh == 1 && c <= 128 && oc <= 256 && h <= 4)
                return Ids().group_fwd;
            return Ids().winograd_3x2;
        }
        if(direction == 2)
            return Ids().bwd_nhwc;
        if(direction == 4)
            return Ids().wrw_nhwc;
    }

    if(sp == 3)
        return Pick3d(direction);

    // iter81: dir=1 bf16 g=1 NCHW fy=1 sh=1 128<=c<=256 oc<=128 h<=8 -> FN.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       128 <= c && c <= 256 && oc <= 128 && h <= 8)
        return Ids().fwd_nhwc;

    // iter73: dir=1 fp16 g=1 c==3 fy>=7 sh=2 oc>=96 -> FN.
    if(direction == 1 && dtype == DType::Fp16 && g == 1 && c == 3 && fy >= 7 && sh == 2 && oc >= 96)
        return Ids().fwd_nhwc;

    // iter92: dir=1 fp32 g=1 NCHW fy=7 sh=2 c<=64 oc<=128 h in [64,256] n<=64 -> GF.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 7 && sh == 2 &&
       c <= 64 && oc <= 128 && 64 <= h && h <= 256 && n <= 64)
        return Ids().group_fwd;

    // iter113: dir=1 fp32 g=1 NCHW fy=7 sh=1 c<=32 oc<=128 -> FN.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 7 && sh == 1 &&
       c <= 32 && oc <= 128)
        return Ids().fwd_nhwc;

    // iter71: dir=1 fp32 g=1 c==3 fy>=7 sh=2 oc<=64 h>=128 w<=500 n<=32 -> GF.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && c == 3 && fy >= 7 && sh == 2 &&
       oc <= 64 && h >= 128 && w <= 500 && n <= 32)
        return Ids().group_fwd;

    // iter63: dir=4 fp32 g=1 c==3 fy>=7 sh=2 w<=500 n<=256 -> GW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && c == 3 && fy >= 7 && sh == 2 &&
       w <= 500 && n <= 256)
        return Ids().group_wrw;

    // iter70: dir=4 fp32 g=1 c==3 fy>=7 sh=2 w<=500 oc<=64 n<=624 -> GW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && c == 3 && fy >= 7 && sh == 2 &&
       w <= 500 && oc <= 64 && n <= 624)
        return Ids().group_wrw;

    // dir=4 fp32 g=1 c==3 fy>=7 sh=2 oc>=96 n>=256 -> WN (large-batch first-layer 7x7).
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && c == 3 && fy >= 7 && sh == 2 &&
       oc >= 96 && n >= 256)
        return Ids().wrw_nhwc;

    // iter94: dir=4 fp32 g=1 NCHW fy=1 sh=2 c[512,2048] oc[128,1024] h<=128 n<=64 -> GW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 2 &&
       512 <= c && c <= 2048 && 128 <= oc && oc <= 1024 && h <= 128 && n <= 64)
        return Ids().group_wrw;

    // iter120: dir=4 fp32 g=1 NCHW fy=5 sh=1 c=64 oc=192 n>=4096 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 5 && sh == 1 &&
       c == 64 && oc == 192 && n >= 4096)
        return Ids().wrw_nhwc;

    // iter121: dir=2 fp32 g=1 NCHW c=3 oc=64 fy=7 sh=2 h>=4000 w>=8000 -> BN.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 3 && oc == 64 &&
       fy == 7 && sh == 2 && h >= 4000 && w >= 8000)
        return Ids().bwd_nhwc;

    // iter119: dir=4 fp32 g=1 NCHW c=3 fy>=7 oc>=96 GW-MISS reroute -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 3 && fy >= 7 &&
       oc >= 96)
    {
        if((sh == 1 && n >= 128) || (sh == 2 && h >= 800 && n >= 32))
            return Ids().wrw_nhwc;
    }

    // iter118: dir=4 fp32 g=1 NCHW fy=1 sh=1 specific (c,oc) shape carve -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1)
    {
        if((c == 16 && oc == 96 && h >= 112 && n >= 512) ||
           (c == 64 && oc == 256 && h >= 56 && n >= 1024) ||
           (c == 256 && oc == 1024 && h >= 64 && n >= 400) ||
           (c == 512 && oc == 2048 && h >= 28 && n >= 512))
            return Ids().wrw_nhwc;
    }

    // iter117: dir=4 fp32 g=1 NCHW fy=3 sh=1 32<=c<=128 2c<=oc<=3c 53k<=h*n<=80k -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1)
    {
        if(32 <= c && c <= 128 && 2 * c <= oc && oc <= 3 * c && 53000 <= h * n && h * n <= 80000)
            return Ids().wrw_nhwc;
    }

    // iter116: dir=4 fp32 g=1 NCHW c==3 fy==3 oc<=64 large-spatial GW-MISS -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 3 && fy == 3 &&
       oc <= 64)
    {
        if((sh == 1 && h >= 224 && n >= 256) ||
           (sh == 2 && oc >= 8 && h >= 800 && n >= 128) ||
           (sh == 2 && oc >= 8 && h >= 1000 && n >= 64))
            return Ids().wrw_nhwc;
    }

    // iter115: dir=4 fp32 g=1 NCHW c==1 fy=5 sh=1 oc<=64 h>=128 w>=500 n>=256 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 1 && fy == 5 &&
       sh == 1 && oc <= 64 && h >= 128 && w >= 500 && n >= 256)
        return Ids().wrw_nhwc;

    // iter114: dir=4 fp32 g=1 NCHW fy=1 sh=1 c==512 oc<=16 h*n>=80000 -> GW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c == 512 && oc <= 16 && h * n >= 80000)
        return Ids().group_wrw;

    // iter112: dir=4 fp32 g=1 NCHW fy=1 sh=1 c<512 oc<=16 h>=16 -> GW (tiny-oc carve).
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c < 512 && oc <= 16 && h >= 16)
        return Ids().group_wrw;

    // iter95: dir=4 fp32 g=1 NCHW fy=1 sh=1 c<=2048 oc<=256 n>=8 h<=16 -> GW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 2048 && oc <= 256 && n >= 8 && h <= 16)
        return Ids().group_wrw;

    // dir=4 fp32 g=1: heavy-activation / large-channel carves -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1)
    {
        const long long nhw = static_cast<long long>(n) * h * w;
        // iter15: nhw>30M -> WN, except c==3 with sh!=2 or oc<=32.
        if(nhw > 30000000LL)
        {
            if(!(c == 3 && (sh != 2 || oc <= 32)))
                return Ids().wrw_nhwc;
        }
        // c*nhw >= 5e8 -> WN.
        if(static_cast<long long>(c) * nhw >= 500000000LL)
            return Ids().wrw_nhwc;
        // bytes_processed >= 5e9 -> WN. Approximated as
        // elem_size * (in_size + out_size + filter_size).
        const long long elem = static_cast<long long>(problem.GetInElementSize());
        const long long h_out = static_cast<long long>(problem.GetOutHeight());
        const long long w_out = static_cast<long long>(problem.GetOutWidth());
        const long long in_sz = static_cast<long long>(n) * c * h * w;
        const long long out_sz = static_cast<long long>(n) * oc * h_out * w_out;
        const long long fil_sz = static_cast<long long>(c) * oc * fy * fx;
        if(elem * (in_sz + out_sz + fil_sz) >= 5000000000LL)
            return Ids().wrw_nhwc;
        // 1x1 large c*oc>=2e6 -> WN.
        if(fy == 1 && static_cast<long long>(c) * oc >= 2000000LL)
            return Ids().wrw_nhwc;
        // iter19: fy>=3 c*oc>=1e6 -> WN.
        if(fy >= 3 && static_cast<long long>(c) * oc >= 1000000LL)
            return Ids().wrw_nhwc;
    }

    // dir=1 fp32 g>1 3x3 sh=2 NCHW c==oc -> W3 / GF carves.
    if(direction == 1 && dtype == DType::Fp32 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c == oc)
    {
        if(h >= 150 || (h >= 56 && n >= 128))
        {
            // iter39: n==1 prefers GF.
            if(n == 1)
                return Ids().group_fwd;
            // iter76: depthwise small-c n<=128 prefers GF.
            if(c == g && c <= 100 && n <= 128)
                return Ids().group_fwd;
            return Ids().winograd_3x2;
        }
        // iter69: large-c moderate-n h>=32 -> W3.
        if(c >= 512 && 32 <= n && n < 128 && h >= 32)
            return Ids().winograd_3x2;
    }

    // iter111: dir=2 fp32 g>1 NCHW fy=3 sh=2 c<=128 oc<=128 h<=64 -> W2.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 128 && oc <= 128 && h <= 64)
        return Ids().winograd_2x3;

    // iter68: dir=2 fp32 g>1 NCHW fy=3 sh=2 c==oc h<=16 c!=g n>1 -> GB.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c == oc && h <= 16 && c != g && n > 1)
        return Ids().group_bwd;

    // dir=2 fp32 g>1 3x3 sh=2 NCHW c==oc -> W3 / W2 carves (iter21/27/34/40).
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c == oc)
    {
        if(h >= 100 || n >= 128 || (h >= 50 && n >= 16))
        {
            if(c == g && h < 30)
                return Ids().winograd_2x3;
            if(n == 1)
                return Ids().winograd_2x3;
            return Ids().winograd_3x2;
        }
    }

    // iter89: dir=2 fp32 g=1 NCHW fy=1 sh=1 c<=256 oc<=32 h<=64 n<=4 -> BN.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 256 && oc <= 32 && h <= 64 && n <= 4)
        return Ids().bwd_nhwc;

    // iter90: dir=2 bf16 g=1 NCHW fy=1 sh=1 c<=128 oc<=64 h<=4 -> BN.
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 128 && oc <= 64 && h <= 4)
        return Ids().bwd_nhwc;

    // iter104: dir=1 bf16 g=1 NCHW fy=1 sh=1 c<=64 oc<=32 h<=128 -> GF.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 64 && oc <= 32 && h <= 128)
        return Ids().group_fwd;

    // iter102: dir=1 fp16 g=1 NCHW fy=1 sh=1 c<=32 oc<=512 n>=4 h<=128 -> FN.
    if(direction == 1 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 32 && oc <= 512 && n >= 4 && h <= 128)
        return Ids().fwd_nhwc;

    // iter101: dir=1 bf16 g=1 NCHW fy=1 sh=1 c<=32 oc<=512 n>=32 h<=128 -> FN.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 32 && oc <= 512 && n >= 32 && h <= 128)
        return Ids().fwd_nhwc;

    // iter91: dir=1 bf16 g=1 NCHW fy=1 sh=1 c<=64 oc<=32 h in [64,256] n<=64 -> FN.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 64 && oc <= 32 && 64 <= h && h <= 256 && n <= 64)
        return Ids().fwd_nhwc;

    // iter22: dir=2 fp32 g=1 NCHW 1x1 s=1 oc<=4 -> BN.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && oc <= 4)
        return Ids().bwd_nhwc;

    // iter32: dir=1 fp32 depthwise fy=5 c<=84 h>=42 NCHW -> W2.
    if(direction == 1 && dtype == DType::Fp32 && g > 1 && c == g && c == oc && fy == 5 &&
       c <= 84 && h >= 42 && out_l != "NHWC")
        return Ids().winograd_2x3;

    // iter108: dir=2 fp16 g>1 NCHW fy=3 sh=2 c[256,2048] h<=32 -> GB.
    if(direction == 2 && dtype == DType::Fp16 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       256 <= c && c <= 2048 && h <= 32)
        return Ids().group_bwd;

    // iter109: dir=1 fp16 g>1 NCHW fy=3 sh=2 c<=512 oc[128,2048] n>=128 32<=h<=128 -> W3.
    if(direction == 1 && dtype == DType::Fp16 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 512 && 128 <= oc && oc <= 2048 && n >= 128 && 32 <= h && h <= 128)
        return Ids().winograd_3x2;

    // iter107: dir=2 fp16 g>1 NCHW fy=3 sh=2 c<=1024 oc[128,2048] n>=128 h<=64 -> W3.
    if(direction == 2 && dtype == DType::Fp16 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 1024 && 128 <= oc && oc <= 2048 && n >= 128 && h <= 64)
        return Ids().winograd_3x2;

    // iter110: dir=2 fp16 g>1 NCHW fy=3 sh=1 c<=512 oc<=512 n>=128 16<=h<=64 -> W2.
    if(direction == 2 && dtype == DType::Fp16 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 512 && oc <= 512 && n >= 128 && 16 <= h && h <= 64)
        return Ids().winograd_2x3;

    // iter105: dir=2 fp32 g>1 NCHW fy=3 sh=1 c<=512 oc[128,2048] h<=16 -> GB.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 512 && 128 <= oc && oc <= 2048 && h <= 16)
        return Ids().group_bwd;

    // iter59: dir=2 bf16 g=1 NCHW oc<=4 fy=3 sh=2 -> BN.
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && oc <= 4 && fy == 3 &&
       sh == 2)
        return Ids().bwd_nhwc;

    // iter59b: dir=2 bf16 g=1 NCHW oc<=4 fy=1 sh=1 -> BN.
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && oc <= 4 && fy == 1 &&
       sh == 1)
        return Ids().bwd_nhwc;

    // iter97: dir=2 fp16 g=1 NCHW fy=3 sh=2 c<=128 oc<=64 h<=8 -> W3.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 128 && oc <= 64 && h <= 8)
        return Ids().winograd_3x2;

    // iter98: dir=2 fp32 g=1 NCHW fy=3 sh=2 c<=128 oc<=64 h<=16 -> W3.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 128 && oc <= 64 && h <= 16)
        return Ids().winograd_3x2;

    // iter99: dir=2 fp16 g=1 NCHW fy=3 sh=1 c<=128 oc<=64 h<=8 -> W3.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 128 && oc <= 64 && h <= 8)
        return Ids().winograd_3x2;

    // iter103: dir=2 fp32 g=1 NCHW fy=3 sh=1 c<=256 oc<=16 h<=14 -> W3 (before iter58).
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 256 && oc <= 16 && h <= 14)
        return Ids().winograd_3x2;

    // iter58: dir=2 g=1 NCHW fy=3 sh=1 oc<=4 fp{32,16}/bf16 -> BN.
    if(direction == 2 &&
       (dtype == DType::Fp32 || dtype == DType::Bfp16 || dtype == DType::Fp16) && g == 1 &&
       out_l != "NHWC" && oc <= 4 && fy == 3 && sh == 1)
        return Ids().bwd_nhwc;

    // iter57: dir=2 fp32 depthwise fy=3 sh=1 NCHW 14<=h<=30 n>=64 -> W2.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && c == g && c == oc && fy == 3 && sh == 1 &&
       14 <= h && h <= 30 && n >= 64 && out_l != "NHWC")
        return Ids().winograd_2x3;

    // iter43: dir=2 fp32 depthwise fy=7 c<=84 NCHW -> W3.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && c == g && c == oc && fy == 7 &&
       c <= 84 && out_l != "NHWC")
        return Ids().winograd_3x2;

    // iter42: dir=1 fp32 depthwise fy=7 c<=84 NCHW -> W3.
    if(direction == 1 && dtype == DType::Fp32 && g > 1 && c == g && c == oc && fy == 7 &&
       c <= 84 && out_l != "NHWC")
        return Ids().winograd_3x2;

    // iter41: dir=1 fp32 g=1 NCHW fy=3 c<=3 oc<=4 -> W3.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && c <= 3 &&
       oc <= 4)
        return Ids().winograd_3x2;

    // iter30: dir=1 fp32 g>1 c!=g c==oc fy=3 sh=1 NCHW h>=14 n>=64 -> W2.
    if(direction == 1 && dtype == DType::Fp32 && g > 1 && c != g && c == oc && fy == 3 && sh == 1 &&
       out_l != "NHWC" && h >= 14 && n >= 64)
        return Ids().winograd_2x3;

    // iter78: dir=4 bf16 g=1 NCHW fy=3 sh=1 c<=512 oc>=512 h<=32 -> GW.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 512 && oc >= 512 && h <= 32)
        return Ids().group_wrw;

    // iter88: dir=4 bf16 g=1 NCHW fy=3 sh=1 c<=64 oc<=32 h<=8 -> WN.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 64 && oc <= 32 && h <= 8)
        return Ids().wrw_nhwc;

    // iter85: dir=4 fp16 g=1 NCHW fy=3 sh=1 c<=128 oc<=64 h<=8 n>=8 -> WN (before iter77).
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 128 && oc <= 64 && h <= 8 && n >= 8)
        return Ids().wrw_nhwc;

    // iter77: dir=4 fp16 g=1 NCHW fy=3 sh=1 c<=512 oc<=512 h<=8 -> GW.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 512 && oc <= 512 && h <= 8)
        return Ids().group_wrw;

    // iter48: dir=4 fp16 g=1 NCHW fy=3 sh=1 c>=256 oc>=340 n>=64 h<=20 -> WN.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 256 && oc >= 340 && n >= 64 && h <= 20)
        return Ids().wrw_nhwc;

    // iter52: dir=4 fp16 g=1 NCHW fy=3 sh=2 c>=512 oc>=512 n>=600 h<100 -> WN.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c >= 512 && oc >= 512 && n >= 600 && h < 100)
        return Ids().wrw_nhwc;

    // iter51: dir=2 fp32 g=1 NCHW fy=1 sh=2 c>=64 oc>=64 n>=32 h>=800 -> BN.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 2 &&
       c >= 64 && oc >= 64 && n >= 32 && h >= 800)
        return Ids().bwd_nhwc;

    // iter54: dir=4 bf16 g=1 NCHW fy=3 sh=1 c>=256 oc>=128 n>=256 h>=80 -> WN.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 256 && oc >= 128 && n >= 256 && h >= 80)
        return Ids().wrw_nhwc;

    // iter60: dir=4 fp16 g=1 NCHW fy=1 sh=2 c>=64 oc>=64 n>=256 h>=100 -> WN.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 2 &&
       c >= 64 && oc >= 64 && n >= 256 && h >= 100)
        return Ids().wrw_nhwc;

    // iter62: dir=1 fp32 g=1 NCHW fy=7 sh=2 c<=64 h>=128 -> FN.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 7 && sh == 2 &&
       c <= 64 && h >= 128)
        return Ids().fwd_nhwc;

    // iter96: dir=4 fp32 g=1 NCHW fy=3 sh=1 c[256,512] oc<=512 n>=128 h<=8 -> GW (before iter64).
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       256 <= c && c <= 512 && oc <= 512 && n >= 128 && h <= 8)
        return Ids().group_wrw;

    // iter65: dir=4 fp32 g=1 NCHW fy=3 sh=1 c<=256 h<=4 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 256 && h <= 4)
        return Ids().wrw_nhwc;

    // iter64: dir=4 fp32 g=1 NCHW fy=3 sh=1 c>=256 oc>=256 h<=8 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 256 && oc >= 256 && h <= 8)
        return Ids().wrw_nhwc;

    // iter86: dir=4 fp32 g=1 NCHW fy=3 sh=1 c<=256 oc>=512 h[32,128] n<=4 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 256 && oc >= 512 && 32 <= h && h <= 128 && n <= 4)
        return Ids().wrw_nhwc;

    // iter87: dir=4 fp32 g=1 NCHW fy=3 sh=2 c<=512 oc>=512 h[16,64] n<=32 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 512 && oc >= 512 && 16 <= h && h <= 64 && n <= 32)
        return Ids().wrw_nhwc;

    // iter84: dir=4 fp32 g=1 NCHW fy=3 sh=1 c>=512 oc>=128 h<=64 n<=32 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 512 && oc >= 128 && h <= 64 && n <= 32)
        return Ids().wrw_nhwc;

    // iter83: dir=4 fp32 g=1 NCHW fy=5 sh=2 h<=32 n<=64 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 5 && sh == 2 &&
       h <= 32 && n <= 64)
        return Ids().wrw_nhwc;

    // iter93: dir=4 fp32 g=1 NCHW fy=1 sh=1 c[128,512] oc[256,2048] h<=128 n<=64 -> GW
    // (before iter82).
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       128 <= c && c <= 512 && 256 <= oc && oc <= 2048 && h <= 128 && n <= 64)
        return Ids().group_wrw;

    // iter82: dir=4 fp32 g=1 NCHW fy=1 sh=1 h<=16 n<=8 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       h <= 16 && n <= 8)
        return Ids().wrw_nhwc;

    // iter80: dir=4 bf16 g=1 NCHW fy=1 sh=1 c<=512 oc<=128 h<=8 -> WN.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 512 && oc <= 128 && h <= 8)
        return Ids().wrw_nhwc;

    // iter79: dir=4 fp32 g=1 NCHW fy=3 sh=2 c>=512 oc<=256 h<=64 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c >= 512 && oc <= 256 && h <= 64)
        return Ids().wrw_nhwc;

    // iter75: dir=4 fp32 g=1 NCHW fy=3 sh=2 c<=256 oc<=512 h<=16 -> GW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 256 && oc <= 512 && h <= 16)
        return Ids().group_wrw;

    // iter61: dir=4 fp32 g=1 NCHW fy=3 sh=2 c>=256 h<=16 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c >= 256 && h <= 16)
        return Ids().wrw_nhwc;

    // iter67: dir=4 fp16 g=1 NCHW fy=3 sh=2 c<=64 n>=32 h<=8 -> WN.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 64 && n >= 32 && h <= 8)
        return Ids().wrw_nhwc;

    // iter66: dir=4 fp32 g=1 NCHW fy=3 sh=2 c<=256 oc>=512 h<=32 -> WN.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 256 && oc >= 512 && h <= 32)
        return Ids().wrw_nhwc;

    // iter53: dir=4 bf16 g=1 NCHW fy=3 sh=1 c>=128 oc>=340 n>=64 38<=h<=50 -> WN.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 128 && oc >= 340 && n >= 64 && 38 <= h && h <= 50)
        return Ids().wrw_nhwc;

    // iter50: dir=4 bf16 g=1 NCHW fy=3 sh=1 c>=512 oc>=510 n>=100 h<=25 -> WN.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 512 && oc >= 510 && n >= 100 && h <= 25)
        return Ids().wrw_nhwc;

    // iter72: dir=4 fp16 g=1 NCHW fy=3 sh=1 c==256 oc==256 80<=h<=130 n<=128 -> GW (before iter49).
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c == 256 && oc == 256 && 80 <= h && h <= 130 && n <= 128)
        return Ids().group_wrw;

    // iter49: dir=4 fp16 g=1 NCHW fy=3 sh=1 c>=256 oc>=128 n>=128 h>=80 -> WN.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 256 && oc >= 128 && n >= 128 && h >= 80)
        return Ids().wrw_nhwc;

    // iter47: dir=4 bf16 g=1 NCHW fy=1 sh=2 c*oc>=5e5 n>=128 h>=50 -> WN.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 2 &&
       static_cast<long long>(c) * oc >= 500000LL && n >= 128 && h >= 50)
        return Ids().wrw_nhwc;

    // iter46: dir=4 fp16 g=1 NCHW fy=3 sh=2 c>=256 oc>=256 n>=256 h>=100 -> WN.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c >= 256 && oc >= 256 && n >= 256 && h >= 100)
        return Ids().wrw_nhwc;

    // iter45: dir=4 fp16 g=1 NCHW fy=1 sh=2 c*oc>=1e6 n>=256 -> WN.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 2 &&
       static_cast<long long>(c) * oc >= 1000000LL && n >= 256)
        return Ids().wrw_nhwc;

    // iter44: dir=2 fp16 depthwise fy=3 sh=2 NCHW h>=28 n>=4 c<=88 -> W2.
    if(direction == 2 && dtype == DType::Fp16 && g > 1 && c == g && c == oc && fy == 3 && sh == 2 &&
       out_l != "NHWC" && h >= 28 && n >= 4 && c <= 88)
        return Ids().winograd_2x3;

    // iter29: dir=2 fp16 g>1 c==oc fy=3 sh=2 NCHW h>=28 n>=32 -> W2.
    if(direction == 2 && dtype == DType::Fp16 && g > 1 && c == oc && fy == 3 && sh == 2 &&
       out_l != "NHWC" && h >= 28 && n >= 32)
        return Ids().winograd_2x3;

    // iter28: dir=2 fp32 g>1 c!=g c==oc fy=3 sh=1 NCHW h>=14 n>=16 -> W2.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && c != g && c == oc && fy == 3 && sh == 1 &&
       out_l != "NHWC" && h >= 14 && n >= 16)
        return Ids().winograd_2x3;

    // iter18: dir=2 fp32 depthwise fy=3 sh=1 h>=28 -> W2.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && c == g && fy == 3 && sh == 1 && h >= 28)
        return Ids().winograd_2x3;

    // dir=2 fp32 depthwise stride>=2 h>=16 -> W2 (iter31), with iter74 carve-out to GB.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && c == g && sh >= 2 && sw >= 2 && h >= 16)
    {
        if(fy == 3 && h < 30 && n <= 50)
            return Ids().group_bwd;
        return Ids().winograd_2x3;
    }

    // iter24/iter35/iter36: dir=1 bf16/fp16 g=1 NCHW c<=3 -> FN, with fy>=11 / fp16 fy=7 sh=2
    // carves to GF.
    if(direction == 1 && (dtype == DType::Bfp16 || dtype == DType::Fp16) && g == 1 &&
       out_l != "NHWC" && c <= 3)
    {
        if(fy >= 11)
            return Ids().group_fwd;
        if(dtype == DType::Fp16 && fy == 7 && sh == 2)
            return Ids().group_fwd;
        return Ids().fwd_nhwc;
    }

    // iter100: dir=1 bf16 g=1 NCHW fy=3 sh=1 c<=512 32<=oc<=512 h<=32 -> GF.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 512 && 32 <= oc && oc <= 512 && h <= 32)
        return Ids().group_fwd;

    // iter38: dir=1 bf16 g=1 NCHW c<=13 fy=3 sh=1 -> FN.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && c <= 13 && fy == 3 &&
       sh == 1)
        return Ids().fwd_nhwc;

    // dir=1 bf16 g=1: c*nhw>=1e9 -> FN with carve-outs (iter56 + c==oc 3x3 large-n).
    if(direction == 1 && dtype == DType::Bfp16 && g == 1)
    {
        const long long cnhw = static_cast<long long>(c) * n * h * w;
        if(cnhw >= 1000000000LL)
        {
            if(c == oc && fy == 3 && sh == 1 && n >= 32)
                return Ids().group_fwd;
            // iter56: 1x1 sh=1 -> GF.
            if(fy == 1 && sh == 1)
                return Ids().group_fwd;
            return Ids().fwd_nhwc;
        }
        // n=1 3x3 s=1 bf16 -> FN.
        if(n == 1 && fy == 3 && fx == 3 && sh == 1 && sw == 1)
            return Ids().fwd_nhwc;
    }

    // iter23: dir=1 fp32 g=1 NCHW c<=2 fy>=3 nhw>=1e6 -> FN.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c <= 2 && fy >= 3)
    {
        const long long nhw = static_cast<long long>(n) * h * w;
        if(nhw >= 1000000LL)
            return Ids().fwd_nhwc;
    }

    // iter25: dir=1 fp32 g=1 NCHW c==3 fy>=3 sh=1 nhw>=1e6 -> FN.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 3 && fy >= 3 &&
       sh == 1)
    {
        const long long nhw = static_cast<long long>(n) * h * w;
        if(nhw >= 1000000LL)
            return Ids().fwd_nhwc;
    }

    // iter20/33/37: dir=1 fp32 g=1 NCHW c==3 sh=2 fy>=3 -> FN with fy-dependent nhw threshold.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 3 && sh == 2 &&
       fy >= 3)
    {
        const long long nhw = static_cast<long long>(n) * h * w;
        const long long thr = (fy == 3) ? 100000LL : 3000000LL;
        if(nhw >= thr)
            return Ids().fwd_nhwc;
    }

    // dir=1 fp32 g=1 NCHW c<=4 stride>=2 nhw>=1e8 -> FN.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c <= 4 && sh >= 2)
    {
        const long long nhw = static_cast<long long>(n) * h * w;
        if(nhw >= 100000000LL)
            return Ids().fwd_nhwc;
    }

    // dir=1 fp32 n=1 g=1 fy>=3 c*h*w>=1e7 -> FN.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && n == 1 && fy >= 3)
    {
        const long long chw = static_cast<long long>(c) * h * w;
        if(chw >= 10000000LL)
            return Ids().fwd_nhwc;
    }

    // No rule matched: defer to TunaNet.
    return {};
}

} // namespace gfx950
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
