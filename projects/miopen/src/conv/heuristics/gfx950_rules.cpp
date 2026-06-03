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

// Flat sequence of carve-out rules derived from gfx950 perf-DB mining.
// Each rule's `iterNN` tag matches the iteration in the source decision tree;
// order is significant — earlier rules win. Final fall-through returns the
// per-direction Group2D default (mirrors the Python source).

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
    solver::Id winograd_rx_g1;
    solver::Id gemm_fwd_1x1_s1;
    solver::Id gemm_fwd_1x1_s2;
    solver::Id gemm_fwd_rest;
    solver::Id gemm_bwd_1x1_s1;
    solver::Id gemm_bwd_1x1_s2;
    solver::Id gemm_bwd_rest;
    solver::Id gemm_wrw_1x1_s1;
    solver::Id gemm_wrw_universal;
    solver::Id naive_fwd;
    solver::Id naive_bwd;
    solver::Id fft;

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
          winograd_2x3("ConvBinWinogradRxSf2x3"),
          winograd_rx_g1("ConvBinWinogradRxSf2x3g1"),
          gemm_fwd_1x1_s1("GemmFwd1x1_0_1"),
          gemm_fwd_1x1_s2("GemmFwd1x1_0_2"),
          gemm_fwd_rest("GemmFwdRest"),
          gemm_bwd_1x1_s1("GemmBwd1x1_stride1"),
          gemm_bwd_1x1_s2("GemmBwd1x1_stride2"),
          gemm_bwd_rest("GemmBwdRest"),
          gemm_wrw_1x1_s1("GemmWrw1x1_stride1"),
          gemm_wrw_universal("GemmWrwUniversal"),
          naive_fwd("ConvDirectNaiveConvFwd"),
          naive_bwd("ConvDirectNaiveConvBwd"),
          fft("fft")
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

solver::Id PickGroup2d(int direction)
{
    switch(direction)
    {
    case 1: return Ids().group_fwd;
    case 2: return Ids().group_bwd;
    case 4: return Ids().group_wrw;
    default: return Ids().group_bwd;
    }
}

solver::Id ApplyOverlay(solver::Id base,
                        int direction,
                        DType dtype,
                        const conv::ProblemDescription& problem);
bool ShouldAbstain(solver::Id chosen,
                   int direction,
                   DType dtype,
                   const conv::ProblemDescription& problem);

solver::Id PickBase(const conv::ProblemDescription& problem)
{
    if(problem.GetConv().mode != miopenConvolution)
        return {};
    if(!problem.Is2d() && !problem.Is3d())
        return {};

    const int direction      = GetDirInt(problem);
    const DType dtype        = GetDType(problem);
    const int g              = static_cast<int>(problem.GetGroupCount());
    const int sp             = problem.Is3d() ? 3 : 2;
    const std::string& out_l = problem.GetOutLayout();
    const int fy             = static_cast<int>(problem.GetWeightsHeight());
    const int fx             = static_cast<int>(problem.GetWeightsWidth());
    const int sh             = static_cast<int>(problem.GetKernelStrideH());
    const int sw             = static_cast<int>(problem.GetKernelStrideW());
    const int c              = static_cast<int>(problem.GetInChannels());
    const int oc             = static_cast<int>(problem.GetOutChannels());
    const int h              = static_cast<int>(problem.GetInHeight());
    const int w              = static_cast<int>(problem.GetInWidth());
    const int n              = static_cast<int>(problem.GetBatchSize());
    const long long hw       = static_cast<long long>(h) * w;
    const long long nhw      = static_cast<long long>(n) * h * w;

    // NHWC out_layout fp16/fp32: Group* rarely a candidate; route per (dir, dtype).
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

    // 3D: handle a couple of 1x1x1 carve-outs, then default to 3D group.
    if(sp == 3)
    {
        const int fz = static_cast<int>(problem.GetWeightsDepth());
        const int sd = static_cast<int>(problem.GetKernelStrideD());
        // iter146: dir=2 3D fp16/bf16 g=1 1x1x1 s=1 -> GEMM_BWD_1X1_S1.
        if(direction == 2 && (dtype == DType::Fp16 || dtype == DType::Bfp16) && g == 1 && fy == 1 &&
           fx == 1 && fz == 1 && sh == 1 && sw == 1 && sd == 1)
            return Ids().gemm_bwd_1x1_s1;
        // iter147: dir=1 3D fp16/bf16 g=1 1x1x1 s=1 -> GEMM_FWD_1X1_S1.
        if(direction == 1 && (dtype == DType::Fp16 || dtype == DType::Bfp16) && g == 1 && fy == 1 &&
           fx == 1 && fz == 1 && sh == 1 && sw == 1 && sd == 1)
            return Ids().gemm_fwd_1x1_s1;
        // iter277: fz>=2 fy=1 n=1 -> Gemm-by-direction.
        if(fz >= 2 && fy == 1 && n == 1)
        {
            switch(direction)
            {
            case 1: return Ids().gemm_fwd_rest;
            case 2: return Ids().gemm_bwd_rest;
            case 4: return Ids().gemm_wrw_universal;
            default: return Ids().group_bwd;
            }
        }
        // iter273: bf16 fz>=2 fy=1 -> 3DGroup (not Gemm).
        if(fz >= 2 && dtype == DType::Bfp16 && fy == 1)
            return Pick3d(direction);
        // iter268: real 3D (fz>=2) -> Gemm-by-direction.
        if(fz >= 2)
        {
            switch(direction)
            {
            case 1: return Ids().gemm_fwd_rest;
            case 2: return Ids().gemm_bwd_rest;
            case 4: return Ids().gemm_wrw_universal;
            default: return Ids().group_bwd;
            }
        }
        // iter269: degenerate-3D (fz=1) depth>=2 n=1 -> Gemm-by-direction; 1x1x1 -> Gemm1x1.
        const int depth_in = static_cast<int>(problem.GetInDepth());
        if(depth_in >= 2 && n == 1)
        {
            const bool is_1x1x1 = (fy == 1 && fx == 1 && fz == 1);
            if(is_1x1x1)
            {
                switch(direction)
                {
                case 1: return Ids().gemm_fwd_1x1_s1;
                case 2: return Ids().gemm_bwd_1x1_s1;
                case 4: return Ids().gemm_wrw_1x1_s1;
                default: return Ids().group_bwd;
                }
            }
            switch(direction)
            {
            case 1: return Ids().gemm_fwd_rest;
            case 2: return Ids().gemm_bwd_rest;
            case 4: return Ids().gemm_wrw_universal;
            default: return Ids().group_bwd;
            }
        }
        // iter274: dir=1 first-conv 3x3 stride-2 (depth=1 fz=1 g=1 c<=5 n<=128) -> NAIVE_FWD.
        if(direction == 1 && depth_in == 1 && fz == 1 && fy == 3 && fx == 3 && g == 1 && c <= 5 &&
           sh == 2 && sw == 2 && n <= 128)
            return Ids().naive_fwd;
        return Pick3d(direction);
    }

    // iter128: dir=4 fp32 3x3 s=2 c=3 oc in {24,32} h>=400 -> GEMM_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && fy == 3 && fx == 3 && sh == 2 &&
       sw == 2 && c == 3 && (oc == 24 || oc == 32) && h >= 400)
        return Ids().gemm_wrw_universal;

    // iter127: dir=4 fp32 3x3 s=1 c=3 oc=32 h>=200 -> GEMM_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && fy == 3 && fx == 3 && sh == 1 &&
       sw == 1 && c == 3 && oc == 32 && h >= 200)
        return Ids().gemm_wrw_universal;

    // iter246: dir=2 fp16/bf16 g=1 NCHW c=3 7x7 s=1 -> GEMM_BWD_REST.
    if(direction == 2 && (dtype == DType::Fp16 || dtype == DType::Bfp16) && g == 1 &&
       out_l != "NHWC" && c == 3 && fy == 7 && fx == 7 && sh == 1 && sw == 1)
        return Ids().gemm_bwd_rest;

    // iter250: dir=2 fp16/bf16 g=1 NCHW 1x1 s=2 c==2oc hw<=200 n>=256 -> GROUP_BWD.
    if(direction == 2 && (dtype == DType::Fp16 || dtype == DType::Bfp16) && g == 1 &&
       out_l != "NHWC" && fy == 1 && fx == 1 && sh == 2 && sw == 2 && c == 2 * oc && hw <= 200 &&
       n >= 256)
        return Ids().group_bwd;

    // iter248: dir=2 fp16/bf16 g=1 NCHW 1x1 s=2 oc==2c hw<=2000 n>=512 -> GROUP_BWD.
    if(direction == 2 && (dtype == DType::Fp16 || dtype == DType::Bfp16) && g == 1 &&
       out_l != "NHWC" && fy == 1 && fx == 1 && sh == 2 && sw == 2 && oc == 2 * c &&
       hw <= 2000 && n >= 512)
        return Ids().group_bwd;

    // iter171: dir=2 fp32 g=1 NCHW 1x1 s=2 c>=3*oc -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 && oc > 0 && c >= 3 * oc)
        return Ids().group_bwd;

    // iter172: dir=2 bf16 g=1 NCHW 1x1 s=2 oc>=4*c -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 && c > 0 && oc >= 4 * c)
        return Ids().group_bwd;

    // iter258: dir=2 fp16 g=1 NCHW 1x1 s=2 oc>=2c c<=128 hw>=700 n>=128 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 && oc >= 2 * c && c <= 128 && hw >= 700 && n >= 128)
        return Ids().group_bwd;

    // iter260: dir=2 bf16 mirror of iter258.
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 && oc >= 2 * c && c <= 128 && hw >= 700 && n >= 128)
        return Ids().group_bwd;

    // iter133/136: dir=2 g=1 NCHW 1x1 s=2 squeeze/expand by dtype -> GEMM_BWD_1X1_S2.
    if(direction == 2 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 && sh == 2 && sw == 2)
    {
        const bool sqz_or_exp =
            (dtype == DType::Fp32 && (c >= 2 * oc || oc >= 2 * c)) ||
            (dtype == DType::Fp16 && (oc >= 2 * c || c >= 2 * oc)) ||
            (dtype == DType::Bfp16 && (oc >= 2 * c || c >= 2 * oc));
        if(sqz_or_exp)
            return Ids().gemm_bwd_1x1_s2;
    }

    // iter202: dir=1 fp16/bf16 g=1 NCHW 1x1 s=1 n<=16 -> GROUP_FWD.
    if(direction == 1 && (dtype == DType::Fp16 || dtype == DType::Bfp16) && g == 1 &&
       out_l != "NHWC" && fy == 1 && fx == 1 && sh == 1 && sw == 1 && n <= 16)
        return Ids().group_fwd;

    // iter256: dir=1 fp32 g=1 NCHW 1x1 s=1 narrow-oc large-c large-spatial small-n -> GROUP_FWD.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && oc <= 16 && c >= 256 && hw >= 20000 && n <= 2)
        return Ids().group_fwd;

    // iter134/136/156-160/188/193/237: dir=1 g=1 NCHW 1x1 s=1 by dtype -> mostly GEMM_FWD_1X1_S1.
    if(direction == 1 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 && sh == 1 && sw == 1 &&
       (dtype == DType::Fp32 || dtype == DType::Fp16 || dtype == DType::Bfp16))
    {
        if(dtype == DType::Bfp16)
        {
            // iter237: 1D-shape h=1 w>=100 c>=512 oc>=c -> GEMM_FWD_1X1_S1.
            if(h == 1 && w >= 100 && c >= 512 && oc >= c)
                return Ids().gemm_fwd_1x1_s1;
            // iter193: large-batch override -> GEMM_FWD_1X1_S1.
            if(n >= 128 && hw <= 196)
                return Ids().gemm_fwd_1x1_s1;
            // iter156/160: bf16 carves -> GROUP_FWD.
            if((128 <= c && c <= 1024 && 16 <= hw && hw <= 256) ||
               (64 <= c && c < 128 && 16 <= hw && hw <= 256) ||
               (128 <= c && c <= 1024 && 256 < hw && hw <= 512) ||
               (64 <= c && c < 128 && 256 < hw && hw <= 1024))
                return Ids().group_fwd;
            // iter188: bf16 c>=256 hw<=49 -> GROUP_FWD.
            if(c >= 256 && hw <= 49)
                return Ids().group_fwd;
        }
        // iter261: fp16 c>=1024 oc<=192 hw<=49 n>=128 -> GEMM_FWD_1X1_S1 (carve back from iter157).
        if(dtype == DType::Fp16 && c >= 1024 && oc <= 192 && hw <= 49 && n >= 128)
            return Ids().gemm_fwd_1x1_s1;
        // iter157: fp16 128<=c<=2048 16<=hw<=128 -> GROUP_FWD.
        if(dtype == DType::Fp16 && 128 <= c && c <= 2048 && 16 <= hw && hw <= 128)
            return Ids().group_fwd;
        return Ids().gemm_fwd_1x1_s1;
    }

    // iter155: tiny-channel 3x3 NCHW g=1 -> Naive (fwd all dtypes, bwd bf16).
    if(sp == 2 && out_l != "NHWC" && g == 1 && fy == 3 && fx == 3 && c <= 8 && oc <= 8)
    {
        if(direction == 1 &&
           (dtype == DType::Fp32 || dtype == DType::Fp16 || dtype == DType::Bfp16))
            return Ids().naive_fwd;
        if(direction == 2 && dtype == DType::Bfp16)
            return Ids().naive_bwd;
    }

    // iter186: dir=1 fp32 depthwise 5x5/7x7 NCHW -> NAIVE_FWD.
    if(direction == 1 && dtype == DType::Fp32 && out_l != "NHWC" && g > 1 && g == c && g == oc &&
       fy == fx && (fy == 5 || fy == 7))
        return Ids().naive_fwd;

    // iter234: dir=2 fp32 depthwise 3x3 s=2 c<=32 n<=64 -> WINO_2X3.
    if(direction == 2 && dtype == DType::Fp32 && out_l != "NHWC" && g > 1 && g == c && g == oc &&
       fy == 3 && fx == 3 && sh == 2 && sw == 2 && c <= 32 && n <= 64)
        return Ids().winograd_2x3;

    // iter241: dir=2 fp32 depthwise 5x5 s=2 -> WINO_2X3.
    if(direction == 2 && dtype == DType::Fp32 && out_l != "NHWC" && g > 1 && g == c && g == oc &&
       fy == 5 && fx == 5 && sh == 2 && sw == 2)
        return Ids().winograd_2x3;

    // iter185: dir=2 fp32 depthwise 3x3 NCHW -> NAIVE_BWD.
    if(direction == 2 && dtype == DType::Fp32 && out_l != "NHWC" && g > 1 && g == c && g == oc &&
       fy == 3 && fx == 3)
        return Ids().naive_bwd;

    // iter184: dir=1 depthwise 3x3 NCHW all dtypes -> NAIVE_FWD.
    if(direction == 1 && out_l != "NHWC" && g > 1 && g == c && g == oc && fy == 3 && fx == 3 &&
       (dtype == DType::Fp32 || dtype == DType::Fp16 || dtype == DType::Bfp16))
        return Ids().naive_fwd;

    // iter195: dir=1 bf16 g=1 NCHW tiny-c narrow-w -> NAIVE_FWD.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && c <= 4 && w <= 4)
        return Ids().naive_fwd;

    // iter264: dir=2 fp32 g=1 NCHW 3x3 s=1 c>=768 oc<=192 hw<=300 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c >= 768 && oc <= 192 && hw <= 300)
        return Ids().winograd_rx_g1;

    // iter265: dir=2 fp16 g=1 NCHW 3x3 s=1 c>=2*oc oc<=192 hw<=300 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c >= 2 * oc && oc <= 192 && hw <= 300)
        return Ids().winograd_rx_g1;

    // iter249: dir=2 fp16 g=1 NCHW 3x3 s=1 oc=510 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && oc == 510)
        return Ids().winograd_rx_g1;

    // iter242: dir=2 fp16 g=1 NCHW 3x3 s=1 c=1 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c == 1)
        return Ids().winograd_rx_g1;

    // iter197: dir=2 fp32/fp16 g=1 NCHW 3x3 tiny-c -> NAIVE_BWD.
    if(direction == 2 && (dtype == DType::Fp32 || dtype == DType::Fp16) && g == 1 &&
       out_l != "NHWC" && fy == 3 && fx == 3 && c <= 2)
        return Ids().naive_bwd;

    // iter254: dir=2 fp16 depthwise 3x3 s=2 NCHW c=88 -> NAIVE_BWD.
    if(direction == 2 && dtype == DType::Fp16 && out_l != "NHWC" && g > 1 && g == c && g == oc &&
       fy == 3 && fx == 3 && sh == 2 && sw == 2 && c == 88)
        return Ids().naive_bwd;

    // iter194: dir=2 fp16 depthwise 3x3 s=1 NCHW -> NAIVE_BWD.
    if(direction == 2 && dtype == DType::Fp16 && out_l != "NHWC" && g > 1 && g == c && g == oc &&
       fy == 3 && fx == 3 && sh == 1 && sw == 1)
        return Ids().naive_bwd;

    // iter183: dir=2 bf16 depthwise 3x3 NCHW -> NAIVE_BWD.
    if(direction == 2 && dtype == DType::Bfp16 && out_l != "NHWC" && g > 1 && g == c && g == oc &&
       fy == 3 && fx == 3)
        return Ids().naive_bwd;

    // iter154: high-groups fwd/bwd NCHW (g>=96) -> Naive.
    if(out_l != "NHWC" && g >= 96 &&
       (dtype == DType::Fp32 || dtype == DType::Fp16 || dtype == DType::Bfp16))
    {
        if(direction == 1)
            return Ids().naive_fwd;
        if(direction == 2)
            return Ids().naive_bwd;
    }

    // iter153: dir=2 fp32 g=1 NCHW 7x7 s=2 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 7 && fx == 7 &&
       sh == 2 && sw == 2)
        return Ids().gemm_bwd_rest;

    // iter169: dir=2 fp32 g=1 NCHW 3x3 s=1 n=1 18<=h,w<=32 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && n == 1 && 18 <= h && h <= 32 && 18 <= w && w <= 32)
        return Ids().group_bwd;

    // iter239: dir=2 fp32 g=1 NCHW 3x3 s=1 c==oc 24<=h,w<=30 n<=64 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c == oc && 24 <= h && h <= 30 && 24 <= w && w <= 30 && n <= 64)
        return Ids().group_bwd;

    // iter180: dir=2 fp32 g=1 NCHW 3x3 s=1 c==oc>=256 400<=hw<800 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c == oc && c >= 256 && 400 <= hw && hw < 800)
        return Ids().winograd_rx_g1;

    // iter199: dir=2 fp16 g=1 NCHW 3x3 s=1 c>=128 n>=64 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c >= 128 && n >= 64)
        return Ids().group_bwd;

    // iter226: dir=2 fp16 g=1 NCHW 3x3 s=1 dense c==oc>=256 or oc>=2c spatial -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 &&
       (((c == oc) && c >= 256 && hw <= 2048) || ((oc >= 2 * c) && c >= 64 && hw <= 5400)))
        return Ids().group_bwd;

    // iter182: dir=2 fp16 g=1 NCHW 3x3 s=1 c<256 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c < 256)
        return Ids().winograd_rx_g1;

    // iter179: dir=2 fp16 g=1 NCHW 3x3 s=1 100<=hw<200 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && 100 <= hw && hw < 200)
        return Ids().winograd_rx_g1;

    // iter235: dir=2 fp32 g=1 NCHW 3x3 s=1 h<=38 oc>=c+128 c>=64 n>=32 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && h <= 38 && oc >= c + 128 && c >= 64 && n >= 32)
        return Ids().winograd_rx_g1;

    // iter233: dir=2 fp32 g=1 NCHW 3x3 s=1 oc>=512 c<oc hw<=900 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && oc >= 512 && c < oc && hw <= 900)
        return Ids().group_bwd;

    // iter178: dir=2 fp32 g=1 NCHW 3x3 s=1 100<=hw<200 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && 100 <= hw && hw < 200)
        return Ids().winograd_rx_g1;

    // iter176: dir=2 fp32 g=1 NCHW 3x3 s=1 c>=1024 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c >= 1024)
        return Ids().group_bwd;

    // iter232: dir=2 fp32 g=1 NCHW 3x3 s=1 oc>=c+128 hw<=1500 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && oc >= c + 128 && hw <= 1500)
        return Ids().group_bwd;

    // iter149/151/159: dir=2 fp32 g=1 NCHW 3x3 s=1 h,w>=18 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && h >= 18 && w >= 18)
        return Ids().winograd_rx_g1;

    // iter198: dir=2 fp32 g=1 NCHW 3x3 s=2 n==1 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 2 && sw == 2 && n == 1)
        return Ids().gemm_bwd_rest;

    // iter192: dir=2 fp32 g=1 NCHW 11x11 s=1 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 11 &&
       fx == 11 && sh == 1 && sw == 1)
        return Ids().gemm_bwd_rest;

    // iter189: dir=2 fp32 g=1 NCHW 3x3 s=1 4<=hw<18 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && 4 <= hw && hw < 18)
        return Ids().winograd_rx_g1;

    // iter166: dir=2 fp32 g=1 NCHW 5x5 s=2 h,w>=64 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 5 && fx == 5 &&
       sh == 2 && sw == 2 && h >= 64 && w >= 64)
        return Ids().winograd_rx_g1;

    // iter268: REMOVED iter167 (dir=1 bf16 3x3 s=1 n=1 huge-spatial -> NAIVE_FWD).
    // 5/29 data refresh: GroupFwd is now best on 33/38 of this cohort; fall through.

    // iter238: dir=2 fp16 g=1 NCHW 3x3 s=2 c==oc<=64 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 2 && sw == 2 && c == oc && c <= 64)
        return Ids().winograd_rx_g1;

    // iter162: dir=2 fp16 g=1 NCHW 3x3 s=1 h,w>=32 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && h >= 32 && w >= 32)
        return Ids().winograd_rx_g1;

    // iter268: REMOVED iter163 + iter170 (dir=1 fp32 3x3 s=1 large-spatial n<=1 -> NAIVE_FWD).
    // 5/29 refresh: GroupFwd now best on 25/44 (iter163) and 30/49 (iter170);
    // Naive sumlog +116/+139 vs Group +5/+8. Fall through to default GROUP_FWD.

    // iter255: dir=1 fp32 g=1 NCHW 1x1 s=2 c>=2048 oc<=c/2 hw<=64 -> GROUP_FWD.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 && c >= 2048 && oc * 2 <= c && hw <= 64)
        return Ids().group_fwd;

    // iter210/213/217/251/266: dir=1 fp32 g=1 NCHW 1x1 s=2 -> GEMM_FWD_1X1_S2.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 &&
       (n >= 4 || (c >= 1024 && oc >= c) || (n <= 2 && hw >= 100000) ||
        (c == 256 && oc == 512 && n == 2 && hw >= 40000)))
        return Ids().gemm_fwd_1x1_s2;

    // iter257: dir=4 fp16 g=1 NCHW 3x3 s=1 oc>=256 oc%32!=0 -> WRW_NHWC (odd-channel reroute).
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && oc >= 256 && (oc % 32) != 0)
        return Ids().wrw_nhwc;

    // iter247: dir=4 fp32 g=1 NCHW 3x3 s=1 oc=720 n<=2 hw<=9000 -> GEMM_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && oc == 720 && n <= 2 && hw <= 9000)
        return Ids().gemm_wrw_universal;

    // iter208: dir=4 fp32 g=1 NCHW 1x1 s=2 c>=512 n<=4 oc>256 hw<44377 -> GEMM_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 && c >= 512 && n <= 4 && oc > 256 && hw < 44377)
        return Ids().gemm_wrw_universal;

    // iter219: dir=2 fp32 g=1 NCHW 7x7 s=1 c<=16 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 7 && fx == 7 &&
       sh == 1 && sw == 1 && c <= 16)
        return Ids().gemm_bwd_rest;

    // iter220: dir=1 fp32 g=1 NCHW 3x3 s=1 c>=256 oc>=256 n>=256 hw<=1024 -> GROUP_FWD.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c >= 256 && oc >= 256 && n >= 256 && hw <= 1024)
        return Ids().group_fwd;

    // iter221: dir=1 bf16 g=1 NCHW 3x3 s=1 n=1 c==oc>=512 -> GEMM_FWD_REST.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && n == 1 && c >= 512 && c == oc)
        return Ids().gemm_fwd_rest;

    // iter222: dir=4 bf16 g=1 NCHW 3x3 s=1 c>=512 oc>=2048 hw<=2304 -> GEMM_WRW.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c >= 512 && oc >= 2048 && hw <= 2304)
        return Ids().gemm_wrw_universal;

    // iter223: dir=2 fp32 g=1 NCHW 3x3 s=1 c==oc>=128 hw<=676 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c == oc && c >= 128 && hw <= 676)
        return Ids().group_bwd;

    // iter227: dir=2 fp32 g=1 NCHW 5x5 s=1 c<=3 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 5 && fx == 5 &&
       sh == 1 && sw == 1 && c <= 3)
        return Ids().gemm_bwd_rest;

    // iter243: dir=1 fp32 g=1 NCHW 5x5 s=1 hw<=900 n<=512 -> fft.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 5 && fx == 5 &&
       sh == 1 && sw == 1 && hw <= 900 && n <= 512)
        return Ids().fft;

    // iter224/231: dir=2 fp32 g=1 NCHW 5x5 s=1 c>=32 hw<=900 n<512 -> fft.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 5 && fx == 5 &&
       sh == 1 && sw == 1 && c >= 32 && hw <= 900 && n < 512)
        return Ids().fft;

    // iter244: dir=1 fp32 g=1 NCHW 3x3 s=2 n<=2 c==oc 256<=c<=512 -> GEMM_FWD_REST.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 2 && sw == 2 && n <= 2 && c == oc && 256 <= c && c <= 512)
        return Ids().gemm_fwd_rest;

    // iter225: dir=1 fp32 g=1 NCHW 3x3 s=1 n=1 c>=256 oc>=256 hw>=900000 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && n == 1 && c >= 256 && oc >= 256 && hw >= 900000)
        return Ids().fwd_nhwc;

    // iter228: dir=2 fp16 g=1 NCHW s=1 fy*fx>=64 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && sh == 1 && sw == 1 &&
       fy * fx >= 64)
        return Ids().gemm_bwd_rest;

    // iter230: dir=2 fp16 g=1 NCHW 5x5 s=1 c<=3 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 5 && fx == 5 &&
       sh == 1 && sw == 1 && c <= 3)
        return Ids().gemm_bwd_rest;

    // iter229: dir=2 bf16 g=1 NCHW s=1 fy*fx>=64 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && sh == 1 && sw == 1 &&
       fy * fx >= 64)
        return Ids().gemm_bwd_rest;

    // iter211/212: dir=2 fp16 g=1 NCHW 1x1 s=1 c<=512 (n<=2 oc<=c) or (n<=8 oc<c) -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && c <= 512 &&
       ((n <= 2 && oc <= c) || (n <= 8 && oc < c)))
        return Ids().winograd_rx_g1;

    // iter205: dir=2 fp16/bf16 g=1 NCHW 1x1 s=1 n>=128 -> GEMM_BWD_1X1_S1.
    if(direction == 2 && (dtype == DType::Fp16 || dtype == DType::Bfp16) && g == 1 &&
       out_l != "NHWC" && fy == 1 && fx == 1 && sh == 1 && sw == 1 && n >= 128)
        return Ids().gemm_bwd_1x1_s1;

    // iter204/209/215: dir=2 fp32 g=1 NCHW 3x3 s=1 c<=64 or (c<=256 n<=4 oc<=c) -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && (c <= 64 || (c <= 256 && n <= 4 && oc <= c)))
        return Ids().winograd_rx_g1;

    // iter245: dir=4 fp32 g=1 NCHW 1x1 s=1 c*oc>=1e6 hw>=400 -> GEMM_WRW_1X1_S1.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && static_cast<long long>(c) * oc >= 1000000LL && hw >= 400)
        return Ids().gemm_wrw_1x1_s1;

    // iter203/206: dir=4 fp32 g=1 NCHW 1x1 s=1 n<=4 49<hw<=2048 -> GEMM_WRW_1X1_S1.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && n <= 4 && 49 < hw && hw <= 2048)
        return Ids().gemm_wrw_1x1_s1;

    // iter200: dir=4 fp32 g=1 NCHW 3x3 s=2 c>=256 n<=2 -> GEMM_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 2 && sw == 2 && c >= 256 && n <= 2)
        return Ids().gemm_wrw_universal;

    // iter175: dir=4 fp32 g=1 NCHW fy=3 fx=1 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 1)
        return Ids().group_wrw;

    // iter168: dir=1 fp32 g=1 NCHW 3x3 s=1 h,w<=32 n=1 -> GEMM_FWD_REST.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && h <= 32 && w <= 32 && n == 1)
        return Ids().gemm_fwd_rest;

    // iter177: dir=1 fp32 g=1 NCHW 3x3 s=1 c>=1280 -> GROUP_FWD.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && c >= 1280)
        return Ids().group_fwd;

    // iter150/152/164: dir=1 fp32 g=1 NCHW 3x3 s=1 h,w>=6 -> WINO_RX_G1.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && h >= 6 && w >= 6)
        return Ids().winograd_rx_g1;

    // iter187: dir=2 fp16 g=1 NCHW 1x1 s=1 oc<=64 c<=32 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && oc <= 64 && c <= 32)
        return Ids().winograd_rx_g1;

    // iter262: dir=2 bf16 g=1 NCHW 1x1 s=1 n<=16 c+oc>=2048 hw>=200 -> GEMM_BWD_1X1_S1.
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && n <= 16 && (c + oc) >= 2048 && hw >= 200)
        return Ids().gemm_bwd_1x1_s1;

    // iter263: dir=2 fp16 mirror of iter262 with hw>=256.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && n <= 16 && (c + oc) >= 2048 && hw >= 256)
        return Ids().gemm_bwd_1x1_s1;

    // iter259: dir=2 bf16 g=1 NCHW 1x1 s=1 oc<=8 -> GEMM_BWD_1X1_S1 (carve from iter201).
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && oc <= 8)
        return Ids().gemm_bwd_1x1_s1;

    // iter201: dir=2 fp16/bf16 g=1 NCHW 1x1 s=1 n<=16 -> GROUP_BWD.
    if(direction == 2 && (dtype == DType::Fp16 || dtype == DType::Bfp16) && g == 1 &&
       out_l != "NHWC" && fy == 1 && fx == 1 && sh == 1 && sw == 1 && n <= 16)
        return Ids().group_bwd;

    // iter253: dir=2 fp32 g=1 NCHW 1x1 s=1 hw<=16 n>=256 c+oc>=512 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && hw <= 16 && n >= 256 && c + oc >= 512)
        return Ids().group_bwd;

    // iter252: dir=2 fp32 g=1 NCHW 1x1 s=1 hw==1 c+oc>=4096 (FC-like) -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && hw == 1 && c + oc >= 4096)
        return Ids().group_bwd;

    // iter236: dir=2 fp32 g=1 NCHW 1x1 s=1 hw<=64 c*oc>=5e5 n>=48 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && hw <= 64 && static_cast<long long>(c) * oc >= 500000LL && n >= 48)
        return Ids().group_bwd;

    // iter158/161/190: dir=2 g=1 NCHW 1x1 s=1 bf16/fp16 carves -> GROUP_BWD; else GEMM_BWD_1X1_S1.
    if(direction == 2 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 && sh == 1 && sw == 1 &&
       (dtype == DType::Fp32 || dtype == DType::Fp16 || dtype == DType::Bfp16))
    {
        if((dtype == DType::Bfp16 || dtype == DType::Fp16) && 128 <= c && c <= 2048 &&
           16 <= hw && hw <= 128)
            return Ids().group_bwd;
        if(dtype == DType::Bfp16 &&
           ((64 <= c && c < 128 && 16 <= hw && hw <= 128) ||
            (128 <= c && c <= 2048 && 4 <= hw && hw < 16)))
            return Ids().group_bwd;
        if(dtype == DType::Fp16 && 128 <= c && c <= 2048 && 4 <= hw && hw < 16)
            return Ids().group_bwd;
        return Ids().gemm_bwd_1x1_s1;
    }

    // iter141: dir=2 fp32 g=1 asymmetric 5x20 or 20x5 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 &&
       ((fy == 5 && fx == 20) || (fy == 20 && fx == 5)))
        return Ids().gemm_bwd_rest;

    // iter181: dir=1 fp32 g=1 NCHW 1x1 s=2 c<256 -> GEMM_FWD_1X1_S2.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 && c < 256)
        return Ids().gemm_fwd_1x1_s2;

    // iter140: dir=1 fp32 g=1 NCHW 1x1 s=2 oc>=3c -> GEMM_FWD_1X1_S2.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 && oc >= 3 * c)
        return Ids().gemm_fwd_1x1_s2;

    // iter139: dir=2 fp32 g=1 c=3 3x3 s=2 h>=400 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && c == 3 && fy == 3 && fx == 3 &&
       sh == 2 && sw == 2 && h >= 400)
        return Ids().gemm_bwd_rest;

    // iter145: dir=1 fp32 g=1 c=3 3x3 s in {1,2} -> WINO_RX_G1.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && c == 3 && fy == 3 && fx == 3 &&
       (sh == 1 || sh == 2) && sw == sh)
        return Ids().winograd_rx_g1;

    // iter138: dir=2 fp32 g=1 3x3 s=1 c>=64 h,w>=80 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && fy == 3 && fx == 3 && sh == 1 &&
       sw == 1 && c >= 64 && h >= 80 && w >= 80)
        return Ids().winograd_rx_g1;

    // iter137: dir=4 fp32 depthwise 3x3 s=2 (h>=100 or (h>=50 c>=64)) -> GEMM_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g > 1 && c == oc && c == g && fy == 3 &&
       fx == 3 && sh == 2 && sw == 2 && (h >= 100 || (h >= 50 && c >= 64)))
        return Ids().gemm_wrw_universal;

    // iter129: dir=2 fp16/bf16 g=1 c=3 7x7 s=2 h>500 -> GEMM_BWD.
    if(direction == 2 && (dtype == DType::Fp16 || dtype == DType::Bfp16) && g == 1 && c == 3 &&
       fy == 7 && fx == 7 && sh == 2 && sw == 2 && h > 500)
        return Ids().gemm_bwd_rest;

    // iter124: dir=2 fp32 g=1 c=3 7x7 s=2 h>300 -> GEMM_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && c == 3 && fy == 7 && fx == 7 &&
       sh == 2 && sw == 2 && h > 300)
        return Ids().gemm_bwd_rest;

    // iter123/125: dir=4 fp32 g>=8 3x3 s=2 200<=h<=240 large-n -> GEMM_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g >= 8 && fy == 3 && fx == 3 && sh == 2 &&
       sw == 2 && 200 <= h && h <= 240 &&
       (n >= 64 || (n >= 32 && (c + oc >= 1024 || n >= 48))))
        return Ids().gemm_wrw_universal;

    // iter126: dir=1 fp32 g=1 3x3 s=1 n=1 hw>=2e6 -> GEMM_FWD_REST.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && fy == 3 && fx == 3 && sh == 1 &&
       sw == 1 && n == 1 && hw >= 2000000LL)
        return Ids().gemm_fwd_rest;

    // iter122: dir=1 fp32 g=1 NCHW 3x3 s=1 n=1 1500<hw<=30000 -> WINO_RX_G1.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && n == 1 && 1500 < hw && hw <= 30000)
        return Ids().winograd_rx_g1;

    // iter81: dir=1 bf16 g=1 NCHW fy=1 sh=1 128<=c<=256 oc<=128 h<=8 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       128 <= c && c <= 256 && oc <= 128 && h <= 8)
        return Ids().fwd_nhwc;

    // iter73: dir=1 fp16 g=1 c=3 fy>=7 sh=2 oc>=96 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Fp16 && g == 1 && c == 3 && fy >= 7 && sh == 2 &&
       oc >= 96)
        return Ids().fwd_nhwc;

    // iter92: dir=1 fp32 g=1 NCHW fy=7 sh=2 c<=64 oc<=128 64<=h<=256 n<=64 -> GROUP_FWD.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 7 && sh == 2 &&
       c <= 64 && oc <= 128 && 64 <= h && h <= 256 && n <= 64)
        return Ids().group_fwd;

    // iter113: dir=1 fp32 g=1 NCHW fy=7 sh=1 c<=32 oc<=128 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 7 && sh == 1 &&
       c <= 32 && oc <= 128)
        return Ids().fwd_nhwc;

    // iter71: dir=1 fp32 g=1 c=3 fy>=7 sh=2 oc<=64 h>=128 w<=500 n<=32 -> GROUP_FWD.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && c == 3 && fy >= 7 && sh == 2 &&
       oc <= 64 && h >= 128 && w <= 500 && n <= 32)
        return Ids().group_fwd;

    // iter63: dir=4 fp32 g=1 c=3 fy>=7 sh=2 w<=500 n<=256 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && c == 3 && fy >= 7 && sh == 2 &&
       w <= 500 && n <= 256)
        return Ids().group_wrw;

    // iter70: dir=4 fp32 g=1 c=3 fy>=7 sh=2 w<=500 oc<=64 n<=624 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && c == 3 && fy >= 7 && sh == 2 &&
       w <= 500 && oc <= 64 && n <= 624)
        return Ids().group_wrw;

    // dir=4 fp32 g=1 c=3 fy>=7 sh=2 oc>=96 n>=256 -> WRW_NHWC (large-batch first-layer 7x7).
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && c == 3 && fy >= 7 && sh == 2 &&
       oc >= 96 && n >= 256)
        return Ids().wrw_nhwc;

    // iter94: dir=4 fp32 g=1 NCHW fy=1 sh=2 512<=c<=2048 128<=oc<=1024 h<=128 n<=64 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 2 &&
       512 <= c && c <= 2048 && 128 <= oc && oc <= 1024 && h <= 128 && n <= 64)
        return Ids().group_wrw;

    // iter120: dir=4 fp32 g=1 NCHW fy=5 sh=1 c=64 oc=192 n>=4096 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 5 && sh == 1 &&
       c == 64 && oc == 192 && n >= 4096)
        return Ids().wrw_nhwc;

    // iter121: dir=2 fp32 g=1 NCHW c=3 oc=64 fy=7 sh=2 h>=4000 w>=8000 -> BWD_NHWC.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 3 && oc == 64 &&
       fy == 7 && sh == 2 && h >= 4000 && w >= 8000)
        return Ids().bwd_nhwc;

    // iter119: dir=4 fp32 g=1 NCHW c=3 fy>=7 oc>=96 GW-MISS reroute -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 3 && fy >= 7 &&
       oc >= 96)
    {
        if((sh == 1 && n >= 128) || (sh == 2 && h >= 800 && n >= 32))
            return Ids().wrw_nhwc;
    }

    // iter118: dir=4 fp32 g=1 NCHW fy=1 sh=1 specific (c,oc) shape carves -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1)
    {
        if((c == 16 && oc == 96 && h >= 112 && n >= 512) ||
           (c == 64 && oc == 256 && h >= 56 && n >= 1024) ||
           (c == 256 && oc == 1024 && h >= 64 && n >= 400) ||
           (c == 512 && oc == 2048 && h >= 28 && n >= 512))
            return Ids().wrw_nhwc;
    }

    // iter117: dir=4 fp32 g=1 NCHW fy=3 sh=1 32<=c<=128 2c<=oc<=3c 53k<=h*n<=80k -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1)
    {
        const long long hn = static_cast<long long>(h) * n;
        if(32 <= c && c <= 128 && 2 * c <= oc && oc <= 3 * c && 53000LL <= hn && hn <= 80000LL)
            return Ids().wrw_nhwc;
    }

    // iter116: dir=4 fp32 g=1 NCHW c=3 fy=3 oc<=64 large-spatial GW-MISS -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 3 && fy == 3 &&
       oc <= 64)
    {
        if((sh == 1 && h >= 224 && n >= 256) ||
           (sh == 2 && oc >= 8 && h >= 800 && n >= 128) ||
           (sh == 2 && oc >= 8 && h >= 1000 && n >= 64))
            return Ids().wrw_nhwc;
    }

    // iter115: dir=4 fp32 g=1 NCHW c=1 fy=5 sh=1 oc<=64 h>=128 w>=500 n>=256 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 1 && fy == 5 &&
       sh == 1 && oc <= 64 && h >= 128 && w >= 500 && n >= 256)
        return Ids().wrw_nhwc;

    // iter114: dir=4 fp32 g=1 NCHW fy=1 sh=1 c=512 oc<=16 h*n>=80000 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c == 512 && oc <= 16 && static_cast<long long>(h) * n >= 80000LL)
        return Ids().group_wrw;

    // iter112: dir=4 fp32 g=1 NCHW fy=1 sh=1 c<512 oc<=16 h>=16 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c < 512 && oc <= 16 && h >= 16)
        return Ids().group_wrw;

    // iter95: dir=4 fp32 g=1 NCHW fy=1 sh=1 c<=2048 oc<=256 n>=8 h<=16 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 2048 && oc <= 256 && n >= 8 && h <= 16)
        return Ids().group_wrw;

    // dir=4 fp32 g=1: heavy-activation / large-channel carves -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1)
    {
        // iter15: nhw>30M -> WN, except c=3 with sh!=2 or oc<=32.
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
        const long long elem   = static_cast<long long>(problem.GetInElementSize());
        const long long h_out  = static_cast<long long>(problem.GetOutHeight());
        const long long w_out  = static_cast<long long>(problem.GetOutWidth());
        const long long in_sz  = static_cast<long long>(n) * c * h * w;
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
            if(n == 1)
                return Ids().group_fwd;
            if(c == g && c <= 100 && n <= 128)
                return Ids().group_fwd;
            return Ids().winograd_3x2;
        }
        if(c >= 512 && 32 <= n && n < 128 && h >= 32)
            return Ids().winograd_3x2;
    }

    // iter111: dir=2 fp32 g>1 NCHW fy=3 sh=2 c<=128 oc<=128 h<=64 -> WINO_2X3.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 128 && oc <= 128 && h <= 64)
        return Ids().winograd_2x3;

    // iter68: dir=2 fp32 g>1 NCHW fy=3 sh=2 c==oc h<=16 c!=g n>1 -> GROUP_BWD.
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

    // iter89: dir=2 fp32 g=1 NCHW fy=1 sh=1 c<=256 oc<=32 h<=64 n<=4 -> BWD_NHWC.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 256 && oc <= 32 && h <= 64 && n <= 4)
        return Ids().bwd_nhwc;

    // iter90: dir=2 bf16 g=1 NCHW fy=1 sh=1 c<=128 oc<=64 h<=4 -> BWD_NHWC.
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 128 && oc <= 64 && h <= 4)
        return Ids().bwd_nhwc;

    // iter104: dir=1 bf16 g=1 NCHW fy=1 sh=1 c<=64 oc<=32 h<=128 -> GROUP_FWD.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 64 && oc <= 32 && h <= 128)
        return Ids().group_fwd;

    // iter102: dir=1 fp16 g=1 NCHW fy=1 sh=1 c<=32 oc<=512 n>=4 h<=128 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 32 && oc <= 512 && n >= 4 && h <= 128)
        return Ids().fwd_nhwc;

    // iter101: dir=1 bf16 g=1 NCHW fy=1 sh=1 c<=32 oc<=512 n>=32 h<=128 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 32 && oc <= 512 && n >= 32 && h <= 128)
        return Ids().fwd_nhwc;

    // iter91: dir=1 bf16 g=1 NCHW fy=1 sh=1 c<=64 oc<=32 64<=h<=256 n<=64 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 64 && oc <= 32 && 64 <= h && h <= 256 && n <= 64)
        return Ids().fwd_nhwc;

    // iter22: dir=2 fp32 g=1 NCHW 1x1 s=1 oc<=4 -> BWD_NHWC.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 1 && sw == 1 && oc <= 4)
        return Ids().bwd_nhwc;

    // iter32: dir=1 fp32 depthwise fy=5 c<=84 h>=42 NCHW -> WINO_2X3.
    if(direction == 1 && dtype == DType::Fp32 && g > 1 && c == g && c == oc && fy == 5 &&
       c <= 84 && h >= 42 && out_l != "NHWC")
        return Ids().winograd_2x3;

    // iter108: dir=2 fp16 g>1 NCHW fy=3 sh=2 256<=c<=2048 h<=32 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp16 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       256 <= c && c <= 2048 && h <= 32)
        return Ids().group_bwd;

    // iter109: dir=1 fp16 g>1 NCHW fy=3 sh=2 c<=512 128<=oc<=2048 n>=128 32<=h<=128 -> WINO_3X2.
    if(direction == 1 && dtype == DType::Fp16 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 512 && 128 <= oc && oc <= 2048 && n >= 128 && 32 <= h && h <= 128)
        return Ids().winograd_3x2;

    // iter107: dir=2 fp16 g>1 NCHW fy=3 sh=2 c<=1024 128<=oc<=2048 n>=128 h<=64 -> WINO_3X2.
    if(direction == 2 && dtype == DType::Fp16 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 1024 && 128 <= oc && oc <= 2048 && n >= 128 && h <= 64)
        return Ids().winograd_3x2;

    // iter110: dir=2 fp16 g>1 NCHW fy=3 sh=1 c<=512 oc<=512 n>=128 16<=h<=64 -> WINO_2X3.
    if(direction == 2 && dtype == DType::Fp16 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 512 && oc <= 512 && n >= 128 && 16 <= h && h <= 64)
        return Ids().winograd_2x3;

    // iter105: dir=2 fp32 g>1 NCHW fy=3 sh=1 c<=512 128<=oc<=2048 h<=16 -> GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 512 && 128 <= oc && oc <= 2048 && h <= 16)
        return Ids().group_bwd;

    // iter59: dir=2 bf16 g=1 NCHW oc<=4 fy=3 sh=2 -> BWD_NHWC.
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && oc <= 4 && fy == 3 &&
       sh == 2)
        return Ids().bwd_nhwc;

    // iter59b: dir=2 bf16 g=1 NCHW oc<=4 fy=1 sh=1 -> BWD_NHWC.
    if(direction == 2 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && oc <= 4 && fy == 1 &&
       sh == 1)
        return Ids().bwd_nhwc;

    // iter97: dir=2 fp16 g=1 NCHW fy=3 sh=2 c<=128 oc<=64 h<=8 -> WINO_3X2.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 128 && oc <= 64 && h <= 8)
        return Ids().winograd_3x2;

    // iter98: dir=2 fp32 g=1 NCHW fy=3 sh=2 c<=128 oc<=64 h<=16 -> WINO_3X2.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 128 && oc <= 64 && h <= 16)
        return Ids().winograd_3x2;

    // iter99: dir=2 fp16 g=1 NCHW fy=3 sh=1 c<=128 oc<=64 h<=8 -> WINO_3X2.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 128 && oc <= 64 && h <= 8)
        return Ids().winograd_3x2;

    // iter103: dir=2 fp32 g=1 NCHW fy=3 sh=1 c<=256 oc<=16 h<=14 -> WINO_3X2 (before iter58).
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 256 && oc <= 16 && h <= 14)
        return Ids().winograd_3x2;

    // iter58: dir=2 g=1 NCHW fy=3 sh=1 oc<=4 fp32/bf16/fp16 -> BWD_NHWC.
    if(direction == 2 &&
       (dtype == DType::Fp32 || dtype == DType::Bfp16 || dtype == DType::Fp16) && g == 1 &&
       out_l != "NHWC" && oc <= 4 && fy == 3 && sh == 1)
        return Ids().bwd_nhwc;

    // iter57: dir=2 fp32 depthwise fy=3 sh=1 NCHW 14<=h<=30 n>=64 -> WINO_2X3.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && c == g && c == oc && fy == 3 &&
       sh == 1 && 14 <= h && h <= 30 && n >= 64 && out_l != "NHWC")
        return Ids().winograd_2x3;

    // iter43: dir=2 fp32 depthwise fy=7 c<=84 NCHW -> WINO_3X2.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && c == g && c == oc && fy == 7 &&
       c <= 84 && out_l != "NHWC")
        return Ids().winograd_3x2;

    // iter42: dir=1 fp32 depthwise fy=7 c<=84 NCHW -> WINO_3X2.
    if(direction == 1 && dtype == DType::Fp32 && g > 1 && c == g && c == oc && fy == 7 &&
       c <= 84 && out_l != "NHWC")
        return Ids().winograd_3x2;

    // iter41: dir=1 fp32 g=1 NCHW fy=3 c<=3 oc<=4 -> WINO_3X2.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && c <= 3 &&
       oc <= 4)
        return Ids().winograd_3x2;

    // iter30: dir=1 fp32 g>1 c!=g c==oc fy=3 sh=1 NCHW h>=14 n>=64 -> WINO_2X3.
    if(direction == 1 && dtype == DType::Fp32 && g > 1 && c != g && c == oc && fy == 3 &&
       sh == 1 && out_l != "NHWC" && h >= 14 && n >= 64)
        return Ids().winograd_2x3;

    // iter78: dir=4 bf16 g=1 NCHW fy=3 sh=1 c<=512 oc>=512 h<=32 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 512 && oc >= 512 && h <= 32)
        return Ids().group_wrw;

    // iter88: dir=4 bf16 g=1 NCHW fy=3 sh=1 c<=64 oc<=32 h<=8 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 64 && oc <= 32 && h <= 8)
        return Ids().wrw_nhwc;

    // iter85: dir=4 fp16 g=1 NCHW fy=3 sh=1 c<=128 oc<=64 h<=8 n>=8 -> WRW_NHWC (before iter77).
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 128 && oc <= 64 && h <= 8 && n >= 8)
        return Ids().wrw_nhwc;

    // iter77: dir=4 fp16 g=1 NCHW fy=3 sh=1 c<=512 oc<=512 h<=8 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 512 && oc <= 512 && h <= 8)
        return Ids().group_wrw;

    // iter48: dir=4 fp16 g=1 NCHW fy=3 sh=1 c>=256 oc>=340 n>=64 h<=20 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 256 && oc >= 340 && n >= 64 && h <= 20)
        return Ids().wrw_nhwc;

    // iter52: dir=4 fp16 g=1 NCHW fy=3 sh=2 c>=512 oc>=512 n>=600 h<100 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c >= 512 && oc >= 512 && n >= 600 && h < 100)
        return Ids().wrw_nhwc;

    // iter51: dir=2 fp32 g=1 NCHW fy=1 sh=2 c>=64 oc>=64 n>=32 h>=800 -> BWD_NHWC.
    if(direction == 2 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 2 &&
       c >= 64 && oc >= 64 && n >= 32 && h >= 800)
        return Ids().bwd_nhwc;

    // iter54: dir=4 bf16 g=1 NCHW fy=3 sh=1 c>=256 oc>=128 n>=256 h>=80 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 256 && oc >= 128 && n >= 256 && h >= 80)
        return Ids().wrw_nhwc;

    // iter60: dir=4 fp16 g=1 NCHW fy=1 sh=2 c>=64 oc>=64 n>=256 h>=100 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 2 &&
       c >= 64 && oc >= 64 && n >= 256 && h >= 100)
        return Ids().wrw_nhwc;

    // iter62: dir=1 fp32 g=1 NCHW fy=7 sh=2 c<=64 h>=128 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 7 && sh == 2 &&
       c <= 64 && h >= 128)
        return Ids().fwd_nhwc;

    // iter96: dir=4 fp32 g=1 NCHW fy=3 sh=1 256<=c<=512 oc<=512 n>=128 h<=8 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       256 <= c && c <= 512 && oc <= 512 && n >= 128 && h <= 8)
        return Ids().group_wrw;

    // iter65: dir=4 fp32 g=1 NCHW fy=3 sh=1 c<=256 h<=4 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 256 && h <= 4)
        return Ids().wrw_nhwc;

    // iter64: dir=4 fp32 g=1 NCHW fy=3 sh=1 c>=256 oc>=256 h<=8 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 256 && oc >= 256 && h <= 8)
        return Ids().wrw_nhwc;

    // iter86: dir=4 fp32 g=1 NCHW fy=3 sh=1 c<=256 oc>=512 32<=h<=128 n<=4 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 256 && oc >= 512 && 32 <= h && h <= 128 && n <= 4)
        return Ids().wrw_nhwc;

    // iter87: dir=4 fp32 g=1 NCHW fy=3 sh=2 c<=512 oc>=512 16<=h<=64 n<=32 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 512 && oc >= 512 && 16 <= h && h <= 64 && n <= 32)
        return Ids().wrw_nhwc;

    // iter84: dir=4 fp32 g=1 NCHW fy=3 sh=1 c>=512 oc>=128 h<=64 n<=32 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 512 && oc >= 128 && h <= 64 && n <= 32)
        return Ids().wrw_nhwc;

    // iter83: dir=4 fp32 g=1 NCHW fy=5 sh=2 h<=32 n<=64 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 5 && sh == 2 &&
       h <= 32 && n <= 64)
        return Ids().wrw_nhwc;

    // iter93: dir=4 fp32 g=1 NCHW fy=1 sh=1 128<=c<=512 256<=oc<=2048 h<=128 n<=64 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       128 <= c && c <= 512 && 256 <= oc && oc <= 2048 && h <= 128 && n <= 64)
        return Ids().group_wrw;

    // iter82: dir=4 fp32 g=1 NCHW fy=1 sh=1 h<=16 n<=8 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       h <= 16 && n <= 8)
        return Ids().wrw_nhwc;

    // iter80: dir=4 bf16 g=1 NCHW fy=1 sh=1 c<=512 oc<=128 h<=8 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 1 &&
       c <= 512 && oc <= 128 && h <= 8)
        return Ids().wrw_nhwc;

    // iter79: dir=4 fp32 g=1 NCHW fy=3 sh=2 c>=512 oc<=256 h<=64 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c >= 512 && oc <= 256 && h <= 64)
        return Ids().wrw_nhwc;

    // iter75: dir=4 fp32 g=1 NCHW fy=3 sh=2 c<=256 oc<=512 h<=16 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 256 && oc <= 512 && h <= 16)
        return Ids().group_wrw;

    // iter61: dir=4 fp32 g=1 NCHW fy=3 sh=2 c>=256 h<=16 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c >= 256 && h <= 16)
        return Ids().wrw_nhwc;

    // iter67: dir=4 fp16 g=1 NCHW fy=3 sh=2 c<=64 n>=32 h<=8 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 64 && n >= 32 && h <= 8)
        return Ids().wrw_nhwc;

    // iter66: dir=4 fp32 g=1 NCHW fy=3 sh=2 c<=256 oc>=512 h<=32 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c <= 256 && oc >= 512 && h <= 32)
        return Ids().wrw_nhwc;

    // iter53: dir=4 bf16 g=1 NCHW fy=3 sh=1 c>=128 oc>=340 n>=64 38<=h<=50 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 128 && oc >= 340 && n >= 64 && 38 <= h && h <= 50)
        return Ids().wrw_nhwc;

    // iter50: dir=4 bf16 g=1 NCHW fy=3 sh=1 c>=512 oc>=510 n>=100 h<=25 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 512 && oc >= 510 && n >= 100 && h <= 25)
        return Ids().wrw_nhwc;

    // iter72: dir=4 fp16 g=1 NCHW fy=3 sh=1 c=256 oc=256 80<=h<=130 n<=128 -> GROUP_WRW.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c == 256 && oc == 256 && 80 <= h && h <= 130 && n <= 128)
        return Ids().group_wrw;

    // iter49: dir=4 fp16 g=1 NCHW fy=3 sh=1 c>=256 oc>=128 n>=128 h>=80 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c >= 256 && oc >= 128 && n >= 128 && h >= 80)
        return Ids().wrw_nhwc;

    // iter47: dir=4 bf16 g=1 NCHW fy=1 sh=2 c*oc>=5e5 n>=128 h>=50 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 2 &&
       static_cast<long long>(c) * oc >= 500000LL && n >= 128 && h >= 50)
        return Ids().wrw_nhwc;

    // iter46: dir=4 fp16 g=1 NCHW fy=3 sh=2 c>=256 oc>=256 n>=256 h>=100 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 2 &&
       c >= 256 && oc >= 256 && n >= 256 && h >= 100)
        return Ids().wrw_nhwc;

    // iter45: dir=4 fp16 g=1 NCHW fy=1 sh=2 c*oc>=1e6 n>=256 -> WRW_NHWC.
    if(direction == 4 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 1 && sh == 2 &&
       static_cast<long long>(c) * oc >= 1000000LL && n >= 256)
        return Ids().wrw_nhwc;

    // iter44: dir=2 fp16 depthwise fy=3 sh=2 NCHW h>=28 n>=4 c<=88 -> WINO_2X3.
    if(direction == 2 && dtype == DType::Fp16 && g > 1 && c == g && c == oc && fy == 3 &&
       sh == 2 && out_l != "NHWC" && h >= 28 && n >= 4 && c <= 88)
        return Ids().winograd_2x3;

    // iter29: dir=2 fp16 g>1 c==oc fy=3 sh=2 NCHW h>=28 n>=32 -> WINO_2X3.
    if(direction == 2 && dtype == DType::Fp16 && g > 1 && c == oc && fy == 3 && sh == 2 &&
       out_l != "NHWC" && h >= 28 && n >= 32)
        return Ids().winograd_2x3;

    // iter28: dir=2 fp32 g>1 c!=g c==oc fy=3 sh=1 NCHW h>=14 n>=16 -> WINO_2X3.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && c != g && c == oc && fy == 3 &&
       sh == 1 && out_l != "NHWC" && h >= 14 && n >= 16)
        return Ids().winograd_2x3;

    // iter18: dir=2 fp32 depthwise fy=3 sh=1 h>=28 -> WINO_2X3.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && c == g && fy == 3 && sh == 1 && h >= 28)
        return Ids().winograd_2x3;

    // dir=2 fp32 depthwise stride>=2 h>=16 -> WINO_2X3 (iter31), with iter74 carve to GROUP_BWD.
    if(direction == 2 && dtype == DType::Fp32 && g > 1 && c == g && sh >= 2 && sw >= 2 && h >= 16)
    {
        if(fy == 3 && h < 30 && n <= 50)
            return Ids().group_bwd;
        return Ids().winograd_2x3;
    }

    // iter24/35/36: dir=1 bf16/fp16 g=1 NCHW c<=3 -> FWD_NHWC, with fy>=11 / fp16 fy=7 sh=2 -> GF.
    if(direction == 1 && (dtype == DType::Bfp16 || dtype == DType::Fp16) && g == 1 &&
       out_l != "NHWC" && c <= 3)
    {
        if(fy >= 11)
            return Ids().group_fwd;
        if(dtype == DType::Fp16 && fy == 7 && sh == 2)
            return Ids().group_fwd;
        return Ids().fwd_nhwc;
    }

    // iter100: dir=1 bf16 g=1 NCHW fy=3 sh=1 c<=512 32<=oc<=512 h<=32 -> GROUP_FWD.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && sh == 1 &&
       c <= 512 && 32 <= oc && oc <= 512 && h <= 32)
        return Ids().group_fwd;

    // iter38: dir=1 bf16 g=1 NCHW c<=13 fy=3 sh=1 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && c <= 13 && fy == 3 &&
       sh == 1)
        return Ids().fwd_nhwc;

    // dir=1 bf16 g=1: c*nhw>=1e9 -> FWD_NHWC with carves (iter56 + c==oc 3x3 large-n -> GF).
    if(direction == 1 && dtype == DType::Bfp16 && g == 1)
    {
        const long long cnhw = static_cast<long long>(c) * n * h * w;
        if(cnhw >= 1000000000LL)
        {
            if(c == oc && fy == 3 && sh == 1 && n >= 32)
                return Ids().group_fwd;
            if(fy == 1 && sh == 1)
                return Ids().group_fwd;
            return Ids().fwd_nhwc;
        }
        // n=1 3x3 s=1 bf16 -> FWD_NHWC.
        if(n == 1 && fy == 3 && fx == 3 && sh == 1 && sw == 1)
            return Ids().fwd_nhwc;
    }

    // iter23: dir=1 fp32 g=1 NCHW c<=2 fy>=3 nhw>=1e6 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c <= 2 && fy >= 3 &&
       nhw >= 1000000LL)
        return Ids().fwd_nhwc;

    // iter25: dir=1 fp32 g=1 NCHW c=3 fy>=3 sh=1 nhw>=1e6 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 3 && fy >= 3 &&
       sh == 1 && nhw >= 1000000LL)
        return Ids().fwd_nhwc;

    // iter20/33/37: dir=1 fp32 g=1 NCHW c=3 sh=2 fy>=3 -> FWD_NHWC with fy-dependent threshold.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c == 3 && sh == 2 &&
       fy >= 3)
    {
        const long long thr = (fy == 3) ? 100000LL : 3000000LL;
        if(nhw >= thr)
            return Ids().fwd_nhwc;
    }

    // dir=1 fp32 g=1 NCHW c<=4 stride>=2 nhw>=1e8 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && c <= 4 && sh >= 2 &&
       nhw >= 100000000LL)
        return Ids().fwd_nhwc;

    // dir=1 fp32 n=1 g=1 fy>=3 c*h*w>=1e7 -> FWD_NHWC.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && n == 1 && fy >= 3 &&
       static_cast<long long>(c) * h * w >= 10000000LL)
        return Ids().fwd_nhwc;

    return PickGroup2d(direction);
}

// Helper: does the C++ runtime's in_layout look like the parquet's "NaN"?
// In production, ComputeInLayout always populates a non-empty layout string,
// so this is effectively always false. Overlay/abstain entries gated on NaN
// in_layout are dead at runtime; we still gate them via this helper to match
// the Python contract exactly.
bool LayoutNaN(const conv::ProblemDescription& p) { return p.GetInLayout().empty(); }

bool LayoutNCHW(const conv::ProblemDescription& p) { return p.GetInLayout() == "NCHW"; }

bool LayoutNHWC(const conv::ProblemDescription& p) { return p.GetInLayout() == "NHWC"; }

solver::Id ApplyOverlay(solver::Id base,
                        int direction,
                        DType dtype,
                        const conv::ProblemDescription& problem)
{
    const auto& IDS    = Ids();
    const bool is2d    = problem.Is2d();
    const bool is3d    = problem.Is3d();
    const int g        = static_cast<int>(problem.GetGroupCount());
    const int fy       = static_cast<int>(problem.GetWeightsHeight());
    const int fx       = static_cast<int>(problem.GetWeightsWidth());
    const int sh       = static_cast<int>(problem.GetKernelStrideH());
    const int sw       = static_cast<int>(problem.GetKernelStrideW());
    const int dilh     = static_cast<int>(problem.GetDilationH());
    const int dilw     = static_cast<int>(problem.GetDilationW());
    const int c        = static_cast<int>(problem.GetInChannels());
    const int oc       = static_cast<int>(problem.GetOutChannels());
    const int h        = static_cast<int>(problem.GetInHeight());
    const int w        = static_cast<int>(problem.GetInWidth());
    const int n        = static_cast<int>(problem.GetBatchSize());
    const long long hw = static_cast<long long>(h) * w;
    const bool nan_l   = LayoutNaN(problem);
    const bool nchw_l  = LayoutNCHW(problem);
    const bool nhwc_l  = LayoutNHWC(problem);

    // 1st overlay: dir2 fp16 GROUP_BWD -> WINO_RX_G1 (small-oc + narrow-width + small-batch).
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Fp16 &&
       oc <= 160 && w <= 14 && n <= 8 && g == 1)
        return IDS.winograd_rx_g1;
    // iter276: dir=4 fp32 GROUP_WRW g>1 3x3 s=1 n<=2 -> GEMM_WRW (must precede next overlay).
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Fp32 &&
       g > 1 && is2d && fy == 3 && fx == 3 && sh == 1 && sw == 1 && n <= 2)
        return IDS.gemm_wrw_universal;
    // dir4 fp32 GROUP_WRW -> WINO_RX_G1: 3x3 s=1 tiny-batch hw<=600.
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Fp32 &&
       fy == 3 && fx == 3 && sh == 1 && sw == 1 && n <= 2 && hw <= 600)
        return IDS.winograd_rx_g1;
    // iter288: dir=4 fp32 GROUP_WRW n=1 hw<=4096 fy>=3 -> GEMM_WRW.
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Fp32 &&
       n == 1 && hw <= 4096 && fy >= 3)
        return IDS.gemm_wrw_universal;
    // dir1 fp32 GEMM_FWD_1X1_S1 -> GROUP_FWD: oc<=64 n<=1.
    if(base == IDS.gemm_fwd_1x1_s1 && direction == 1 && dtype == DType::Fp32 &&
       oc <= 64 && n <= 1)
        return IDS.group_fwd;
    // dir2 fp32 GROUP_BWD -> WINO_RX_G1: 3x3 s=1 g=1 oc<=96.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Fp32 &&
       g == 1 && sh == 1 && sw == 1 && fy == 3 && fx == 3 && oc <= 96)
        return IDS.winograd_rx_g1;
    // dir4 bf16 GROUP_WRW -> GEMM_WRW_1X1_S1: 1x1 s=1 n==2 hw<=256.
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Bfp16 &&
       fy == 1 && fx == 1 && sh == 1 && sw == 1 && n == 2 && hw <= 256)
        return IDS.gemm_wrw_1x1_s1;
    // iter268: dir1 fp32/bf16 huge-spatial 3x3 s=1 n=1 g=1 out_layout!=NHWC -> GROUP_FWD.
    if(direction == 1 && (dtype == DType::Fp32 || dtype == DType::Bfp16) &&
       (base == IDS.gemm_fwd_rest || base == IDS.fwd_nhwc || base == IDS.winograd_rx_g1) &&
       g == 1 && is2d && problem.GetOutLayout() != "NHWC" && fy == 3 && fx == 3 && sh == 1 &&
       sw == 1 && n == 1 && hw >= 2500000LL)
        return IDS.group_fwd;
    // iter268b: dir1 fp16/bf16 GROUP_FWD 3x3 s=1 n<=2 g=1 NaN-layout hw>=786432 -> GemmFwdRest.
    if(base == IDS.group_fwd && direction == 1 &&
       (dtype == DType::Fp16 || dtype == DType::Bfp16) && g == 1 && is2d && nan_l &&
       fy == 3 && fx == 3 && sh == 1 && sw == 1 && n <= 2 && hw >= 786432LL)
        return IDS.gemm_fwd_rest;
    // iter269b: dir1 all-dtypes FWD_NHWC|GROUP_FWD fy in (5,7) fx in (5,7) n=1 g=1 NaN c>3 -> GemmFwdRest.
    if(direction == 1 && (base == IDS.fwd_nhwc || base == IDS.group_fwd) && g == 1 && is2d &&
       nan_l && (fy == 5 || fy == 7) && (fx == 5 || fx == 7) && n == 1 && c > 3)
        return IDS.gemm_fwd_rest;
    // iter270 fp32 g any: dir=2 fp32 GROUP_BWD n=1 NaN fy>1 -> GemmBwdRest.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Fp32 &&
       n == 1 && is2d && nan_l && fy > 1)
        return IDS.gemm_bwd_rest;
    // iter270 fp16/bf16 narrowed to g>1: dir=2 GROUP_BWD n=1 NaN fy>1 g>1 -> GemmBwdRest.
    if(base == IDS.group_bwd && direction == 2 &&
       (dtype == DType::Fp16 || dtype == DType::Bfp16) &&
       n == 1 && is2d && nan_l && fy > 1 && g > 1)
        return IDS.gemm_bwd_rest;
    // iter295 fp32: dir=2 GROUP_BWD n=1 NaN 1x1 g>1 -> GEMM_BWD_1X1_S1.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Fp32 &&
       n == 1 && is2d && nan_l && fy == 1 && fx == 1 && g > 1)
        return IDS.gemm_bwd_1x1_s1;
    // iter295b fp16/bf16: same predicate plus oc not in {128,192}.
    if(base == IDS.group_bwd && direction == 2 &&
       (dtype == DType::Fp16 || dtype == DType::Bfp16) &&
       n == 1 && is2d && nan_l && fy == 1 && fx == 1 && oc != 128 && oc != 192 && g > 1)
        return IDS.gemm_bwd_1x1_s1;
    // iter272: dir=1 fp32 WINO_RX_G1 g=1 3x3 s=1 h=w=28 c==oc>=256 -> GROUP_FWD.
    if(base == IDS.winograd_rx_g1 && direction == 1 && dtype == DType::Fp32 &&
       g == 1 && is2d && fy == 3 && fx == 3 && sh == 1 && h == 28 && w == 28 &&
       c == oc && c >= 256)
        return IDS.group_fwd;
    // iter271b non-1x1: dir=1 GROUP_FWD n=1 NaN g in {2,4,8,16} not 1x1 -> GemmFwdRest.
    if(base == IDS.group_fwd && direction == 1 &&
       (dtype == DType::Fp32 || dtype == DType::Fp16 || dtype == DType::Bfp16) &&
       n == 1 && is2d && nan_l && (g == 2 || g == 4 || g == 8 || g == 16) &&
       !(fy == 1 && fx == 1))
        return IDS.gemm_fwd_rest;
    // iter271b 1x1: dir=1 GROUP_FWD n=1 NaN g in {2,4,8,16} 1x1 -> GEMM_FWD_1X1_S1.
    if(base == IDS.group_fwd && direction == 1 &&
       (dtype == DType::Fp32 || dtype == DType::Fp16 || dtype == DType::Bfp16) &&
       n == 1 && is2d && nan_l && (g == 2 || g == 4 || g == 8 || g == 16) &&
       fy == 1 && fx == 1)
        return IDS.gemm_fwd_1x1_s1;
    // iter271 non-1x1: dir=4 GROUP_WRW n=1 NaN g in {2,4,8,16} not 1x1 -> GemmWrwUniversal.
    if(base == IDS.group_wrw && direction == 4 &&
       (dtype == DType::Fp32 || dtype == DType::Fp16 || dtype == DType::Bfp16) &&
       n == 1 && is2d && nan_l && (g == 2 || g == 4 || g == 8 || g == 16) &&
       !(fy == 1 && fx == 1))
        return IDS.gemm_wrw_universal;
    // iter271 1x1: dir=4 GROUP_WRW n=1 NaN g in {2,4,8,16} 1x1 -> GEMM_WRW_1X1_S1.
    if(base == IDS.group_wrw && direction == 4 &&
       (dtype == DType::Fp32 || dtype == DType::Fp16 || dtype == DType::Bfp16) &&
       n == 1 && is2d && nan_l && (g == 2 || g == 4 || g == 8 || g == 16) &&
       fy == 1 && fx == 1)
        return IDS.gemm_wrw_1x1_s1;
    // iter278: dir=4 bf16 GROUP_WRW n=1 -> GEMM_WRW (g in {32,64} OR (sh=1 fy>=3 hw<=50000)).
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Bfp16 && n == 1 && is2d &&
       ((g == 32 || g == 64) || (sh == 1 && fy >= 3 && hw <= 50000LL)))
        return IDS.gemm_wrw_universal;
    // iter279: dir=4 fp16 GROUP_WRW n=1 -> GEMM_WRW (g in {32,64} OR (sh=1 fy>=5 hw<=50000)).
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Fp16 && n == 1 && is2d &&
       ((g == 32 || g == 64) || (sh == 1 && fy >= 5 && hw <= 50000LL)))
        return IDS.gemm_wrw_universal;
    // iter280: dir=1 bf16 GROUP_FWD n=1 -> GemmFwdRest (g in {32,64} OR (sh=1 fy>=5)).
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Bfp16 && n == 1 && is2d &&
       ((g == 32 || g == 64) || (sh == 1 && fy >= 5)))
        return IDS.gemm_fwd_rest;
    // iter281: dir=1 fp16 GROUP_FWD n=1 NaN -> GemmFwdRest.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Fp16 && n == 1 && is2d && nan_l &&
       ((g == 32 || g == 64) ||
        (sh == 1 && fy >= 3) ||
        (sh == 2 && fy == 3 && hw > 50000LL)))
        return IDS.gemm_fwd_rest;
    // iter282: dir=1 fp32 GROUP_FWD n=1 -> GemmFwdRest (g in {32,64} OR (sh=1 fy>=5)).
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Fp32 && n == 1 && is2d &&
       ((g == 32 || g == 64) || (sh == 1 && fy >= 5)))
        return IDS.gemm_fwd_rest;
    // iter290: dir=1 fp16 GROUP_FWD g=1 NCHW 1x1 s=1 h=w=1 -> ASM_FWD_NHWC.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Fp16 && g == 1 && is2d &&
       nchw_l && fy == 1 && fx == 1 && sh == 1 && sw == 1 && h == 1 && w == 1)
        return IDS.fwd_nhwc;
    // iter298: dir=4 bf16 GROUP_WRW g=1 NCHW 1x1 s=1 h=w=1 -> ASM_WRW_NHWC.
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Bfp16 && g == 1 && is2d &&
       nchw_l && fy == 1 && fx == 1 && sh == 1 && sw == 1 && h == 1 && w == 1)
        return IDS.wrw_nhwc;
    // iter299: dir=1 bf16 GROUP_FWD g=1 NCHW 1x1 s=1 h=w=1 -> ASM_FWD_NHWC.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Bfp16 && g == 1 && is2d &&
       nchw_l && fy == 1 && fx == 1 && sh == 1 && sw == 1 && h == 1 && w == 1)
        return IDS.fwd_nhwc;
    // iter301: dir=2 bf16 GROUP_BWD g=1 NCHW 1x1 s=1 h=w=1 -> ASM_BWD_NHWC.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Bfp16 && g == 1 && is2d &&
       nchw_l && fy == 1 && fx == 1 && sh == 1 && sw == 1 && h == 1 && w == 1)
        return IDS.bwd_nhwc;
    // iter302: dir=1 fp16 NHWC WINO_3X2 -> GROUP_FWD.
    if(base == IDS.winograd_3x2 && direction == 1 && dtype == DType::Fp16 && is2d && nhwc_l)
        return IDS.group_fwd;
    // iter292: dir=1 fp16 GROUP_FWD g=1 NCHW c<=3 -> ASM_FWD_NHWC.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Fp16 && g == 1 && is2d &&
       nchw_l && c <= 3)
        return IDS.fwd_nhwc;
    // iter297: dir=1 bf16 mirror of iter292.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Bfp16 && g == 1 && is2d &&
       nchw_l && c <= 3)
        return IDS.fwd_nhwc;
    // iter291: dir=1 fp16 GROUP_FWD g=1 NCHW fy==fx>=2 sh=2 sw=2 -> GemmFwdRest.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Fp16 && g == 1 && is2d &&
       nchw_l && fy == fx && fy >= 2 && sh == 2 && sw == 2)
        return IDS.gemm_fwd_rest;
    // iter289: dir=1 fp16 GROUP_FWD g=1 NCHW 1x1 s=1 -> GEMM_FWD_1X1_S1.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Fp16 && g == 1 && is2d &&
       nchw_l && fy == 1 && fx == 1 && sh == 1 && sw == 1)
        return IDS.gemm_fwd_1x1_s1;
    // iter325: dir=1 fp16 GROUP_FWD g=1 NCHW 3x3 s=1 c>=512 -> ASM_FWD_NHWC (precedes iter303).
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Fp16 && is2d && nchw_l && g == 1 &&
       fy == 3 && fx == 3 && sh == 1 && c >= 512)
        return IDS.fwd_nhwc;
    // iter303 (narrowed by iter325): dir=1 fp16 GROUP_FWD g=1 NCHW 3x3 s=1 c<512 hw<16384 -> WINO_RX_G1.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Fp16 && g == 1 && is2d &&
       nchw_l && fy == 3 && fx == 3 && sh == 1 && sw == 1 && c < 512 && hw < 16384)
        return IDS.winograd_rx_g1;
    // iter326: dir=1 fp16 WINO_3X2 NaN g=1 h=w=1 -> GROUP_FWD.
    if(base == IDS.winograd_3x2 && direction == 1 && dtype == DType::Fp16 && is2d && nan_l &&
       g == 1 && h == 1 && w == 1)
        return IDS.group_fwd;
    // iter324: dir=2 fp16/bf16 GROUP_BWD n=1 NaN g=1 fy>=3 h=w=28 -> GemmBwdRest.
    if(base == IDS.group_bwd && direction == 2 &&
       (dtype == DType::Fp16 || dtype == DType::Bfp16) &&
       is2d && nan_l && n == 1 && g == 1 && fy >= 3 && h == 28 && w == 28)
        return IDS.gemm_bwd_rest;
    // iter323: dir=1 fp32 GROUP_FWD n=1 NaN g=1 h==w in {56,112} -> GemmFwdRest.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Fp32 && is2d && nan_l &&
       n == 1 && g == 1 && h == w && (h == 56 || h == 112))
        return IDS.gemm_fwd_rest;
    // iter322: dir=1 bf16 GROUP_FWD n=1 NaN g=1 3x3 sh=2 -> GemmFwdRest.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Bfp16 && is2d && nan_l &&
       n == 1 && g == 1 && fy == 3 && fx == 3 && sh == 2)
        return IDS.gemm_fwd_rest;
    // iter321: dir=1 fp16 GROUP_FWD n=1 NCHW g=1 1x1 -> ASM_FWD_NHWC.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Fp16 && is2d && nchw_l &&
       n == 1 && g == 1 && fy == 1 && fx == 1)
        return IDS.fwd_nhwc;
    // iter320: dir=2 bf16 GROUP_BWD n=1 NaN g=1 c>=64 h>=112 -> ASM_BWD_NHWC.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Bfp16 && is2d && nan_l &&
       n == 1 && g == 1 && c >= 64 && h >= 112)
        return IDS.bwd_nhwc;
    // iter319: dir=2 fp16 GROUP_BWD n=1 NaN g=1 c>=64 h>=112 -> ASM_BWD_NHWC.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Fp16 && is2d && nan_l &&
       n == 1 && g == 1 && c >= 64 && h >= 112)
        return IDS.bwd_nhwc;
    // iter318: dir=1 fp32/bf16 GROUP_FWD n=1 NaN g=1 h=w=28 -> GemmFwdRest.
    if(base == IDS.group_fwd && direction == 1 &&
       (dtype == DType::Fp32 || dtype == DType::Bfp16) &&
       is2d && nan_l && n == 1 && g == 1 && h == 28 && w == 28)
        return IDS.gemm_fwd_rest;
    // iter317: dir=2 fp32 GROUP_BWD n=1 NaN g=1 h>=56 -> ASM_BWD_NHWC.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Fp32 && is2d && nan_l &&
       n == 1 && g == 1 && h >= 56)
        return IDS.bwd_nhwc;
    // iter316: dir=4 fp32 GROUP_WRW n=1 NaN g=1 h=w=112 -> ASM_WRW_NHWC.
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Fp32 && is2d && nan_l &&
       n == 1 && g == 1 && h == 112 && w == 112)
        return IDS.wrw_nhwc;
    // iter315/327: dir=1 fp16 ASM_FWD_NHWC NCHW c<=3 g=1 fy>=3 sh>=2 n<=2 -> GemmFwdRest.
    if(base == IDS.fwd_nhwc && direction == 1 && dtype == DType::Fp16 && is2d && nchw_l &&
       n <= 2 && c <= 3 && g == 1 && fy >= 3 && sh >= 2)
        return IDS.gemm_fwd_rest;
    // iter314: dir=1 bf16 GROUP_FWD 1x1 s=1 n=1 NaN g=1 -> GEMM_FWD_1X1_S1.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Bfp16 && is2d && nan_l &&
       n == 1 && g == 1 && fy == 1 && fx == 1 && sh == 1 && sw == 1)
        return IDS.gemm_fwd_1x1_s1;
    // iter311: dir=2 fp16/fp32 ASM_BWD_NHWC h=w=1 -> GROUP_BWD.
    if(base == IDS.bwd_nhwc && direction == 2 &&
       (dtype == DType::Fp16 || dtype == DType::Fp32) && is2d && h == 1 && w == 1)
        return IDS.group_bwd;
    // iter310: dir=1 bf16 GEMM_FWD_1X1_S1 NCHW g=1 c<=3 h=w=1 -> NAIVE_FWD.
    if(base == IDS.gemm_fwd_1x1_s1 && direction == 1 && dtype == DType::Bfp16 && is2d && nchw_l &&
       g == 1 && c <= 3 && h == 1 && w == 1)
        return IDS.naive_fwd;
    // iter309: dir=2 fp32 GROUP_BWD 3x3 sh=2 g=1 NaN c<=3 256<=hw<1024 -> WINO_RX_G1.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Fp32 && is2d && nan_l &&
       g == 1 && c <= 3 && fy == 3 && fx == 3 && sh == 2 && hw >= 256 && hw < 1024)
        return IDS.winograd_rx_g1;
    // iter308: dir=4 fp32 ASM_WRW_NHWC 3x3 sh=1 n<=4 g=1 h<=8 NaN -> WINO_RX_G1.
    if(base == IDS.wrw_nhwc && direction == 4 && dtype == DType::Fp32 && is2d && nan_l && g == 1 &&
       fy == 3 && fx == 3 && sh == 1 && n <= 4 && h <= 8)
        return IDS.winograd_rx_g1;
    // iter307: dir=2 fp32 NAIVE_BWD 3x3 s=1 g=1 c==1 NaN -> WINO_RX_G1.
    if(base == IDS.naive_bwd && direction == 2 && dtype == DType::Fp32 && is2d && nan_l && c == 1 &&
       fy == 3 && fx == 3 && sh == 1 && sw == 1 && g == 1)
        return IDS.winograd_rx_g1;
    // iter306: WINO_RX_G1 base + dilation>1 -> reroute to Group by direction.
    if(base == IDS.winograd_rx_g1 &&
       (dtype == DType::Fp16 || dtype == DType::Fp32 || dtype == DType::Bfp16) &&
       (dilh > 1 || dilw > 1))
    {
        if(direction == 1) return IDS.group_fwd;
        if(direction == 2) return IDS.group_bwd;
        if(direction == 4) return IDS.group_wrw;
    }
    // iter332: dir=2 fp32 GROUP_BWD 3x3 sh=1 g=1 NaN h=w=8 n in {32,256} -> WINO_RX_G1.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Fp32 && is2d && nan_l && g == 1 &&
       fy == 3 && sh == 1 && h == 8 && w == 8 && (n == 32 || n == 256))
        return IDS.winograd_rx_g1;
    // dir=1 bf16 GROUP_FWD 1x1 sh=1 NaN g=1 h=w in {14,28} n in {16,64} -> GEMM_FWD_1X1_S1.
    if(base == IDS.group_fwd && direction == 1 && dtype == DType::Bfp16 && is2d && nan_l && g == 1 &&
       fy == 1 && fx == 1 && sh == 1 && (h == 14 || h == 28) && h == w && (n == 16 || n == 64))
        return IDS.gemm_fwd_1x1_s1;
    // dir=2 bf16 GEMM_BWD_1X1_S1 NCHW g=1 1x1 sh=1 h=w=14 n=64 -> GROUP_BWD.
    if(base == IDS.gemm_bwd_1x1_s1 && direction == 2 && dtype == DType::Bfp16 && is2d && nchw_l &&
       g == 1 && fy == 1 && fx == 1 && sh == 1 && h == 14 && w == 14 && n == 64)
        return IDS.group_bwd;
    // dir=4 fp32 ASM_WRW_NHWC NaN g=1 1x1 sh=1 h=w=1 n=2 -> NAIVE_WRW. (No NAIVE_WRW Id; use Wino_G1 instead? skip — keep semantic by emitting wrw via gemm fall-back.)
    // The Python returns "ConvDirectNaiveConvWrw" which is NaiveWrw; we have no
    // SolverIds member for it. Add inline.
    {
        static const solver::Id NAIVE_WRW("ConvDirectNaiveConvWrw");
        if(base == IDS.wrw_nhwc && direction == 4 && dtype == DType::Fp32 && is2d && nan_l && g == 1 &&
           fy == 1 && fx == 1 && sh == 1 && h == 1 && w == 1 && n == 2)
            return NAIVE_WRW;
    }
    // dir=1 bf16 GEMM_FWD_1X1_S1 NCHW g=1 1x1 sh=1 h=w=56 n=64 -> GROUP_FWD.
    if(base == IDS.gemm_fwd_1x1_s1 && direction == 1 && dtype == DType::Bfp16 && is2d && nchw_l &&
       g == 1 && fy == 1 && fx == 1 && sh == 1 && h == 56 && w == 56 && n == 64)
        return IDS.group_fwd;
    // iter331: dir=2 fp16 GEMM_BWD_1X1_S1 NaN g=1 1x1 sh=1 h=w=7 n=128 -> GROUP_BWD.
    if(base == IDS.gemm_bwd_1x1_s1 && direction == 2 && dtype == DType::Fp16 && is2d && nan_l &&
       g == 1 && fy == 1 && fx == 1 && sh == 1 && h == 7 && w == 7 && n == 128)
        return IDS.group_bwd;
    // dir=4 fp16 GROUP_WRW NaN g=1 1x1 sh=1 h=w=7 n<=2 -> WINO_RX_G1.
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Fp16 && is2d && nan_l && g == 1 &&
       fy == 1 && fx == 1 && sh == 1 && h == 7 && w == 7 && n <= 2)
        return IDS.winograd_rx_g1;
    // dir=4 fp32 ASM_WRW_NHWC NaN g=1 1x1 sh=1 h=w=7 n<=4 -> WINO_RX_G1.
    if(base == IDS.wrw_nhwc && direction == 4 && dtype == DType::Fp32 && is2d && nan_l && g == 1 &&
       fy == 1 && fx == 1 && sh == 1 && h == 7 && w == 7 && n <= 4)
        return IDS.winograd_rx_g1;
    // dir=4 fp16 GROUP_WRW NaN g=1 1x1 sh=1 h=w=28 n=2 -> GEMM_WRW_1X1_S1.
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Fp16 && is2d && nan_l && g == 1 &&
       fy == 1 && fx == 1 && sh == 1 && h == 28 && w == 28 && n == 2)
        return IDS.gemm_wrw_1x1_s1;
    // dir=4 fp32 GROUP_WRW NaN g=1 1x1 sh=1 h=w=56 n=1 -> GEMM_WRW_1X1_S1.
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Fp32 && is2d && nan_l && g == 1 &&
       fy == 1 && fx == 1 && sh == 1 && h == 56 && w == 56 && n == 1)
        return IDS.gemm_wrw_1x1_s1;
    // dir=2 bf16 GROUP_BWD NaN g=1 1x1 sh=1 h=w=28 n=16 -> GEMM_BWD_1X1_S1.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Bfp16 && is2d && nan_l && g == 1 &&
       fy == 1 && fx == 1 && sh == 1 && h == 28 && w == 28 && n == 16)
        return IDS.gemm_bwd_1x1_s1;
    // iter330: bf16 ASM_FWD_NHWC scalar 1x1 NCHW -> GEMM_FWD_1X1_S1.
    if(base == IDS.fwd_nhwc && direction == 1 && dtype == DType::Bfp16 && is2d && nchw_l && g == 1 &&
       fy == 1 && fx == 1 && sh == 1 && h == 1 && w == 1)
        return IDS.gemm_fwd_1x1_s1;
    // bf16 ASM_BWD_NHWC scalar 1x1 NCHW -> GEMM_BWD_1X1_S1.
    if(base == IDS.bwd_nhwc && direction == 2 && dtype == DType::Bfp16 && is2d && nchw_l && g == 1 &&
       fy == 1 && fx == 1 && sh == 1 && h == 1 && w == 1)
        return IDS.gemm_bwd_1x1_s1;
    // fp32 GROUP_BWD 3x3 sh=1 g=1 NaN h=w in {15,27} -> WINO_RX_G1.
    if(base == IDS.group_bwd && direction == 2 && dtype == DType::Fp32 && is2d && nan_l && g == 1 &&
       fy == 3 && sh == 1 && ((h == 15 && w == 15) || (h == 27 && w == 27)))
        return IDS.winograd_rx_g1;
    // fp16 GROUP_WRW 3x3 sh=1 g=1 NaN h=w=5 -> WINO_RX_G1.
    if(base == IDS.group_wrw && direction == 4 && dtype == DType::Fp16 && is2d && nan_l && g == 1 &&
       fy == 3 && sh == 1 && h == 5 && w == 5)
        return IDS.winograd_rx_g1;
    // bf16 ASM_FWD_NHWC h=24 w=16 NCHW -> NAIVE_FWD.
    if(base == IDS.fwd_nhwc && direction == 1 && dtype == DType::Bfp16 && is2d && nchw_l && g == 1 &&
       fy == 1 && sh == 1 && h == 24 && w == 16)
        return IDS.naive_fwd;
    // iter329: dir=2 bf16 GEMM_BWD_1X1_S1 NCHW h=24 w=16 fy=1 sh=1 g=1 -> GROUP_BWD.
    if(base == IDS.gemm_bwd_1x1_s1 && direction == 2 && dtype == DType::Bfp16 && is2d && nchw_l &&
       g == 1 && fy == 1 && fx == 1 && sh == 1 && h == 24 && w == 16)
        return IDS.group_bwd;
    // iter328: dir=1 fp32 3D-fake first-conv c=3 h=32 fy=3 fz=1 depth=1 NaN g=1 -> NAIVE_FWD.
    if(base == IDS.group_3d_fwd && direction == 1 && dtype == DType::Fp32 && is3d && nan_l &&
       g == 1 && c == 3 && fy == 3 && h == 32)
    {
        const int fz_ = static_cast<int>(problem.GetWeightsDepth());
        const int depth_ = static_cast<int>(problem.GetInDepth());
        if(fz_ == 1 && depth_ == 1)
            return IDS.naive_fwd;
    }
    // iter304: dir=2 fp32 NAIVE_BWD depthwise 3x3 sh=2 n<=4 NaN -> GemmBwdRest.
    if(base == IDS.naive_bwd && direction == 2 && dtype == DType::Fp32 && is2d && nan_l && n <= 4 &&
       fy == 3 && fx == 3 && sh == 2 && sw == 2 && g >= 64 && c == oc)
        return IDS.gemm_bwd_rest;

    return base;
}

bool ShouldAbstain(solver::Id chosen,
                   int direction,
                   DType dtype,
                   const conv::ProblemDescription& problem)
{
    const auto& IDS    = Ids();
    const int g        = static_cast<int>(problem.GetGroupCount());
    const int fy       = static_cast<int>(problem.GetWeightsHeight());
    const int fx       = static_cast<int>(problem.GetWeightsWidth());
    const int sh       = static_cast<int>(problem.GetKernelStrideH());
    const int c        = static_cast<int>(problem.GetInChannels());
    const int oc       = static_cast<int>(problem.GetOutChannels());
    const int h        = static_cast<int>(problem.GetInHeight());
    const int w        = static_cast<int>(problem.GetInWidth());
    const int n        = static_cast<int>(problem.GetBatchSize());
    const bool nan_l   = LayoutNaN(problem);
    const bool nchw_l  = LayoutNCHW(problem);
    (void)g; (void)fx; (void)c; (void)oc; (void)n; (void)nan_l;

    static const solver::Id NAIVE_WRW("ConvDirectNaiveConvWrw");
    (void)NAIVE_WRW;

    // Whole-cohort abstentions (top1 < TunaNet baseline). iter337-339.
    auto whole = [&](int dr, DType dt, const solver::Id& ch) {
        return direction == dr && dtype == dt && chosen == ch;
    };
    if(whole(2, DType::Fp16, IDS.winograd_3x2)) return true;
    if(whole(2, DType::Bfp16, IDS.bwd_nhwc))    return true;
    if(whole(2, DType::Fp16, IDS.winograd_2x3)) return true;
    if(whole(1, DType::Fp32, IDS.fft))          return true;
    if(whole(4, DType::Bfp16, IDS.wrw_nhwc))    return true;
    if(whole(2, DType::Fp32, IDS.winograd_2x3)) return true;
    if(whole(2, DType::Fp32, IDS.winograd_3x2)) return true;
    if(whole(4, DType::Fp16, IDS.gemm_wrw_universal)) return true;
    if(whole(1, DType::Bfp16, IDS.fwd_nhwc))    return true;
    if(whole(1, DType::Fp16, IDS.fwd_nhwc))     return true;
    if(whole(1, DType::Fp32, IDS.fwd_nhwc))     return true;
    if(whole(4, DType::Bfp16, IDS.gemm_wrw_universal)) return true;
    if(whole(2, DType::Fp32, IDS.fft))          return true;
    // iter337
    if(whole(1, DType::Fp16, IDS.gemm_fwd_rest))  return true;
    if(whole(1, DType::Bfp16, IDS.gemm_fwd_rest)) return true;
    if(whole(1, DType::Fp32, IDS.gemm_fwd_rest))  return true;
    if(whole(4, DType::Fp32, IDS.wrw_nhwc))       return true;
    if(whole(4, DType::Fp32, IDS.gemm_wrw_1x1_s1)) return true;
    // iter338
    if(whole(2, DType::Fp32, IDS.group_bwd))      return true;
    if(whole(4, DType::Fp32, IDS.winograd_rx_g1)) return true;
    if(whole(1, DType::Fp32, IDS.group_fwd))      return true;
    if(whole(2, DType::Fp16, IDS.gemm_bwd_1x1_s2))return true;
    if(whole(2, DType::Fp16, IDS.winograd_rx_g1)) return true;
    if(whole(4, DType::Fp16, IDS.winograd_rx_g1)) return true;
    if(whole(2, DType::Fp32, IDS.gemm_bwd_rest))  return true;
    if(whole(2, DType::Fp16, IDS.gemm_bwd_rest))  return true;
    if(whole(2, DType::Bfp16, IDS.gemm_bwd_1x1_s2))return true;
    if(whole(2, DType::Bfp16, IDS.gemm_bwd_rest)) return true;
    if(whole(1, DType::Fp32, IDS.gemm_fwd_1x1_s2)) return true;
    if(whole(4, DType::Fp16, IDS.wrw_nhwc))       return true;
    if(whole(1, DType::Bfp16, IDS.gemm_fwd_1x1_s1)) return true;
    if(whole(1, DType::Fp32, IDS.winograd_rx_g1)) return true;
    // iter339
    if(whole(4, DType::Fp32, IDS.gemm_wrw_universal)) return true;
    if(whole(2, DType::Fp16, IDS.gemm_bwd_1x1_s1)) return true;
    if(whole(2, DType::Fp16, IDS.group_bwd))      return true;
    if(whole(1, DType::Fp32, IDS.gemm_fwd_1x1_s1)) return true;
    if(whole(2, DType::Bfp16, IDS.gemm_bwd_1x1_s1)) return true;
    if(whole(1, DType::Fp16, IDS.gemm_fwd_1x1_s1)) return true;
    if(whole(4, DType::Fp32, IDS.group_3d_wrw))   return true;
    if(whole(1, DType::Bfp16, IDS.group_3d_fwd))  return true;
    if(whole(4, DType::Bfp16, IDS.gemm_wrw_1x1_s1)) return true;
    if(whole(4, DType::Fp16, IDS.gemm_wrw_1x1_s1)) return true;
    if(whole(2, DType::Fp32, IDS.gemm_bwd_1x1_s1)) return true;
    if(whole(2, DType::Fp32, IDS.winograd_rx_g1)) return true;

    // ----------------------------------------------------------------------
    // Sub-cohort abstentions: tight (h,w,fy,sh) bucket carves.
    // Only NCHW/NHWC variants reach C++ runtime (in_layout is never NaN here).
    auto sub = [&](int dr, DType dt, const solver::Id& ch) {
        return direction == dr && dtype == dt && chosen == ch;
    };

    // bf16 GEMM_FWD_1X1_S1 NCHW h=24 w=16 fy=1 sh=1
    if(sub(1, DType::Bfp16, IDS.gemm_fwd_1x1_s1) && nchw_l && h == 24 && w == 16 && fy == 1 &&
       sh == 1)
        return true;
    // bf16 GEMM_FWD_1X1_S1 NCHW h=w=16 fy=1 sh=1
    if(sub(1, DType::Bfp16, IDS.gemm_fwd_1x1_s1) && nchw_l && h == 16 && w == 16 && fy == 1 &&
       sh == 1)
        return true;
    // bf16 GROUP_BWD NCHW h=w=4 fy=1 sh=1
    if(sub(2, DType::Bfp16, IDS.group_bwd) && nchw_l && h == 4 && w == 4 && fy == 1 && sh == 1)
        return true;
    // bf16 GROUP_BWD NCHW h=192 w=128 fy=1 sh=2
    if(sub(2, DType::Bfp16, IDS.group_bwd) && nchw_l && h == 192 && w == 128 && fy == 1 && sh == 2)
        return true;
    // bf16 GROUP_FWD NCHW h=192 w=128 fy=1 sh=2
    if(sub(1, DType::Bfp16, IDS.group_fwd) && nchw_l && h == 192 && w == 128 && fy == 1 && sh == 2)
        return true;
    // bf16 GROUP_FWD NCHW h=w=192 fy=1 sh=1: not in source — skip.
    // bf16 GROUP_FWD NCHW h=48 w=32 fy=1 sh=1 (iter344)
    if(sub(1, DType::Bfp16, IDS.group_fwd) && nchw_l && h == 48 && w == 32 && fy == 1 && sh == 1)
        return true;
    // bf16 GROUP_FWD NCHW h=w in {14,28,56} fy=1 sh=1 (iter342)
    if(sub(1, DType::Bfp16, IDS.group_fwd) && nchw_l && fy == 1 && sh == 1 && h == w &&
       (h == 14 || h == 28 || h == 56))
        return true;
    // bf16 GROUP_FWD NCHW h=225 w=225 fy=1 sh=1, h=48 w=32 fy=1 sh=2, h=8 w=32 fy=1 sh=1 (iter343)
    if(sub(1, DType::Bfp16, IDS.group_fwd) && nchw_l && fy == 1 &&
       ((h == 225 && w == 225 && sh == 1) || (h == 48 && w == 32 && sh == 2) ||
        (h == 8 && w == 32 && sh == 1)))
        return true;
    // bf16 GROUP_BWD NCHW h=48 w=32 fy=3 sh=1 (iter345)
    if(sub(2, DType::Bfp16, IDS.group_bwd) && nchw_l && h == 48 && w == 32 && fy == 3 && sh == 1)
        return true;
    // bf16 GROUP_WRW NCHW h=1 w=30 fy=1 sh=1 (iter345)
    if(sub(4, DType::Bfp16, IDS.group_wrw) && nchw_l && h == 1 && w == 30 && fy == 1 && sh == 1)
        return true;
    // bf16 GROUP_WRW NCHW h=48 w=32 fy=1 sh=1 (iter342)
    if(sub(4, DType::Bfp16, IDS.group_wrw) && nchw_l && fy == 1 && sh == 1 && h == 48 && w == 32)
        return true;
    // bf16 GROUP_WRW NCHW h=8 w=32 fy=1 sh=1 (rules.py:3630)
    if(sub(4, DType::Bfp16, IDS.group_wrw) && nchw_l && fy == 1 && sh == 1 && h == 8 && w == 32)
        return true;
    // bf16 GROUP_WRW NCHW h=100 w=1 fy=4 sh=4 (rules.py:3874)
    if(sub(4, DType::Bfp16, IDS.group_wrw) && nchw_l && fy == 4 && sh == 4 && h == 100 && w == 1)
        return true;
    // bf16 GROUP_FWD NCHW h=96 w=64 fy=1 sh=2 (rules.py:3726)
    if(sub(1, DType::Bfp16, IDS.group_fwd) && nchw_l && fy == 1 && sh == 2 && h == 96 && w == 64)
        return true;
    // bf16 GROUP_FWD NCHW h=w=19 fy=1 sh=1 (rules.py:3732)
    if(sub(1, DType::Bfp16, IDS.group_fwd) && nchw_l && fy == 1 && sh == 1 && h == 19 && w == 19)
        return true;
    // bf16 GROUP_FWD NCHW h=192 w=128 fy=1 sh=1 (rules.py:3849)
    if(sub(1, DType::Bfp16, IDS.group_fwd) && nchw_l && fy == 1 && sh == 1 && h == 192 && w == 128)
        return true;
    // bf16 GROUP_BWD common h=w in {7,8,14,28,56} fy=1 fx=1 sh=1 (iter342 any layout)
    if(sub(2, DType::Bfp16, IDS.group_bwd) && fy == 1 && fx == 1 && sh == 1 && h == w &&
       (h == 7 || h == 8 || h == 14 || h == 28 || h == 56))
        return true;
    // bf16 GROUP_BWD NCHW h=48 w=32 fy=1 sh=1 (iter342)
    if(sub(2, DType::Bfp16, IDS.group_bwd) && nchw_l && fy == 1 && sh == 1 && h == 48 && w == 32)
        return true;
    // fp16 GROUP_WRW common h=w in {7,14,17,28,56} fy=1 fx=1 sh=1 (iter342)
    if(sub(4, DType::Fp16, IDS.group_wrw) && fy == 1 && fx == 1 && sh == 1 && h == w &&
       (h == 7 || h == 14 || h == 17 || h == 28 || h == 56))
        return true;
    // bf16 GROUP_WRW common h=w in {7,28} fy=1 fx=1 sh=1 (iter342)
    if(sub(4, DType::Bfp16, IDS.group_wrw) && fy == 1 && fx == 1 && sh == 1 && h == w &&
       (h == 7 || h == 28))
        return true;
    // iter345: fp32 GROUP_WRW NaN h=w in {8,17,55,57,64} fy=1 sh=1  (NaN-only — skip at runtime)
    // iter345: bf16 GROUP_BWD NaN h=w=17 fy=1 sh=1 — NaN-only — skip.

    return false;
}

} // namespace

solver::Id PickSolver(const conv::ProblemDescription& problem)
{
    if(problem.GetConv().mode != miopenConvolution)
        return {};
    if(!problem.Is2d() && !problem.Is3d())
        return {};

    const int direction = GetDirInt(problem);
    const DType dtype   = GetDType(problem);
    solver::Id base     = PickBase(problem);
    if(!base.IsValid())
        return {};
    solver::Id chosen = ApplyOverlay(base, direction, dtype, problem);
    if(ShouldAbstain(chosen, direction, dtype, problem))
        return {};
    return chosen;
}

} // namespace gfx950
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
