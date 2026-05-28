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

} // namespace

solver::Id PickSolver(const conv::ProblemDescription& problem)
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

    // iter167: dir=1 bf16 g=1 NCHW 3x3 s=1 n=1 huge-spatial -> NAIVE_FWD.
    if(direction == 1 && dtype == DType::Bfp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && n == 1 && ((h >= 2048 && w >= 2048) || hw >= 4194304LL))
        return Ids().naive_fwd;

    // iter238: dir=2 fp16 g=1 NCHW 3x3 s=2 c==oc<=64 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 2 && sw == 2 && c == oc && c <= 64)
        return Ids().winograd_rx_g1;

    // iter162: dir=2 fp16 g=1 NCHW 3x3 s=1 h,w>=32 -> WINO_RX_G1.
    if(direction == 2 && dtype == DType::Fp16 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && h >= 32 && w >= 32)
        return Ids().winograd_rx_g1;

    // iter163: dir=1 fp32 g=1 NCHW 3x3 s=1 h>=2048 n<=1 -> NAIVE_FWD.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && h >= 2048 && n <= 1)
        return Ids().naive_fwd;

    // iter170: dir=1 fp32 g=1 NCHW 3x3 s=1 n=1 1280<=h<2048 w>=1280 (or huge hw) -> NAIVE_FWD.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 3 && fx == 3 &&
       sh == 1 && sw == 1 && n == 1 &&
       ((1280 <= h && h < 2048 && w >= 1280) || hw >= 4194304LL))
        return Ids().naive_fwd;

    // iter255: dir=1 fp32 g=1 NCHW 1x1 s=2 c>=2048 oc<=c/2 hw<=64 -> GROUP_FWD.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 && c >= 2048 && oc * 2 <= c && hw <= 64)
        return Ids().group_fwd;

    // iter210/213/217/251: dir=1 fp32 g=1 NCHW 1x1 s=2 -> GEMM_FWD_1X1_S2.
    if(direction == 1 && dtype == DType::Fp32 && g == 1 && out_l != "NHWC" && fy == 1 && fx == 1 &&
       sh == 2 && sw == 2 &&
       (n >= 4 || (c >= 1024 && oc >= c) || (n <= 2 && hw >= 100000)))
        return Ids().gemm_fwd_1x1_s2;

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

} // namespace gfx950
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
