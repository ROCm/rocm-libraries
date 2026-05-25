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
//   /home/jascampb/AutoResearchGfx942/rules.py
// Faithful port: each branch carries the perf-DB delta from the Python comment.

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/gfx942_rules.hpp>

#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>

namespace miopen {
namespace ai {
namespace gfx942 {

namespace {

struct SolverIds
{
    solver::Id fwd_nhwc;
    solver::Id group_fwd;
    solver::Id bwd_nhwc;
    solver::Id group_bwd;
    solver::Id wrw_nhwc;
    solver::Id group_wrw;
    solver::Id winograd_3x2;
    solver::Id winograd_2x3;
    solver::Id winograd_2x3_g1;
    solver::Id gemm_fwd_1x1;
    solver::Id gemm_fwd_1x1_s2;
    solver::Id gemm_fwd_rest;
    solver::Id gemm_bwd_1x1;
    solver::Id gemm_bwd_1x1_s2;
    solver::Id gemm_bwd_rest;
    solver::Id gemm_wrw_1x1;
    solver::Id gemm_wrw_universal;
    solver::Id naive_fwd;
    solver::Id naive_bwd;
    solver::Id dir3d_fwd;
    solver::Id dir3d_bwd;
    solver::Id dir3d_wrw;
    solver::Id transposed_wino;

    SolverIds()
        : fwd_nhwc("ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC"),
          group_fwd("ConvHipImplicitGemmGroupFwdXdlops"),
          bwd_nhwc("ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC"),
          group_bwd("ConvHipImplicitGemmGroupBwdXdlops"),
          wrw_nhwc("ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC"),
          group_wrw("ConvHipImplicitGemmGroupWrwXdlops"),
          winograd_3x2("ConvBinWinogradRxSf3x2"),
          winograd_2x3("ConvBinWinogradRxSf2x3"),
          winograd_2x3_g1("ConvBinWinogradRxSf2x3g1"),
          gemm_fwd_1x1("GemmFwd1x1_0_1"),
          gemm_fwd_1x1_s2("GemmFwd1x1_0_2"),
          gemm_fwd_rest("GemmFwdRest"),
          gemm_bwd_1x1("GemmBwd1x1_stride1"),
          gemm_bwd_1x1_s2("GemmBwd1x1_stride2"),
          gemm_bwd_rest("GemmBwdRest"),
          gemm_wrw_1x1("GemmWrw1x1_stride1"),
          gemm_wrw_universal("GemmWrwUniversal"),
          naive_fwd("ConvDirectNaiveConvFwd"),
          naive_bwd("ConvDirectNaiveConvBwd"),
          dir3d_fwd("ConvHipImplicitGemm3DGroupFwdXdlops"),
          dir3d_bwd("ConvHipImplicitGemm3DGroupBwdXdlops"),
          dir3d_wrw("ConvHipImplicitGemm3DGroupWrwXdlops"),
          transposed_wino("TransposedConvBinWinogradRxSf2x3g1")
    {
    }
};

const SolverIds& Ids()
{
    static const SolverIds inst;
    return inst;
}

// direction=4
solver::Id PickWrW(const conv::ProblemDescription& p)
{
    const auto g  = static_cast<int>(p.GetGroupCount());
    const auto fy = static_cast<int>(p.GetWeightsHeight());
    const auto fx = static_cast<int>(p.GetWeightsWidth());
    const auto sh = p.GetKernelStrideH();
    const auto sw = p.GetKernelStrideW();
    const auto c  = static_cast<int>(p.GetInChannels());
    const auto n  = static_cast<int>(p.GetBatchSize());
    const auto h  = static_cast<int>(p.GetInHeight());
    const auto w  = static_cast<int>(p.GetInWidth());
    const auto hw = h * w;

    if(g != 1)
    {
        if(p.IsFp32() && fy == 3 && fx == 3)
        {
            if(g == c)
            {
                // dw 3x3 s2 → GemmWrwUniversal; s1 → Winograd (large-spatial flips back).
                if(sh == 2)
                    return Ids().gemm_wrw_universal;
                if(sh == 1)
                {
                    if(hw > 784)
                        return Ids().gemm_wrw_universal;
                    return Ids().winograd_2x3;
                }
            }
            else
            {
                // non-dw 3x3 s2: batch<=2 or (batch>=48 + hw>=3481) → GemmWrwUniversal.
                if(sh == 2 && (n <= 2 || (n >= 48 && hw >= 3481)))
                    return Ids().gemm_wrw_universal;
            }
        }
        return Ids().group_wrw;
    }

    if(fy == 1 && fx == 1 && sh == 1 && sw == 1)
    {
        // GEMM_WRW_1X1 beats NHWC ASM for small-batch + sufficient channels.
        if(p.IsFp32() && n == 1 && 16 < hw && hw <= 4096)
            return Ids().gemm_wrw_1x1;
        if(hw >= 49 &&
           ((p.IsFp32() && n == 2 && c >= 1024) ||
            (p.IsBfp16() && n <= 2 && c >= 128) ||
            (p.IsFp16() && n <= 2 && c >= 256)))
            return Ids().gemm_wrw_1x1;
    }
    // gfx942 fp32 wrw 3x3 s1 g=1 small-batch (n<=4) low-spatial (h*w<=196):
    // WINOGRAD_2X3 beats NHWC ASM (train -43, val -9 lsum).
    if(p.IsFp32() && fy == 3 && fx == 3 && sh == 1 && sw == 1 && n <= 4 && hw <= 196)
        return Ids().winograd_2x3;
    return Ids().wrw_nhwc;
}

// direction=2
solver::Id PickBwdData(const conv::ProblemDescription& p)
{
    const auto g  = static_cast<int>(p.GetGroupCount());
    const auto fy = static_cast<int>(p.GetWeightsHeight());
    const auto fx = static_cast<int>(p.GetWeightsWidth());
    const auto sh = p.GetKernelStrideH();
    const auto oc = static_cast<int>(p.GetOutChannels());
    const auto c  = static_cast<int>(p.GetInChannels());
    const auto n  = static_cast<int>(p.GetBatchSize());
    const auto h  = static_cast<int>(p.GetInHeight());
    const auto w  = static_cast<int>(p.GetInWidth());
    const auto hw = h * w;

    if(p.IsBfp16())
    {
        // NHWC-output: Winograd / Gemm bwd solvers are invalid here.
        if(p.GetOutLayout() == "NHWC")
            return Ids().bwd_nhwc;
        if(g != 1)
        {
            // gfx942 bf16 bwd depthwise (g == c) 1x3/3x3 — NAIVE wins 100% (354 train).
            if(g == c && ((fy == 3 && fx == 3) || (fy == 1 && fx == 3)))
                return Ids().naive_bwd;
            return Ids().group_bwd;
        }
        // Very-low-channel inputs (RGB/grayscale): GROUP_BWD dominates,
        // except 5x5 stride-2 where GemmBwdRest is the broad winner.
        if(c <= 3)
        {
            if(fy == 3 && fx == 3)
                return Ids().bwd_nhwc;
            if(fy == 5 && fx == 5)
                return Ids().gemm_bwd_rest;
            if(fy == 1 && fx == 1 && sh == 1)
                return Ids().gemm_bwd_1x1;
            if(fy == 5 && fx == 20 && sh == 1)
                return Ids().gemm_bwd_rest;
            // 7x7 s>=2 c<=3: GEMM_BWD_REST broadly; mid-spatial+batch flips to GROUP_BWD.
            if(fy == 7 && fx == 7 && sh >= 2)
            {
                if(4096 < hw && hw <= 250000 && n >= 9)
                    return Ids().group_bwd;
                return Ids().gemm_bwd_rest;
            }
            return Ids().group_bwd;
        }
        if(fy == 7 && fx == 7)
            return Ids().group_bwd;
        if(fy == 11 && fx == 11)
            return Ids().group_bwd;
        // gfx942 bf16 bwd g=1 5x5 stride-2 c>=4: NHWC ASM beats GROUP_BWD
        // (train -27, val -14 lsum).
        if(fy == 5 && fx == 5 && sh >= 2)
            return Ids().bwd_nhwc;
        // gfx942: bf16 bwd 1x1 s1 — GEMM_BWD_1X1 wins broadly. Very-thin spatial
        // (h<=2) flips to NHWC. Invalid for NHWC input layout.
        if(fy == 1 && fx == 1 && sh == 1)
        {
            if(p.GetInLayout() == "NHWC")
                return Ids().bwd_nhwc;
            if(h <= 2)
                return Ids().bwd_nhwc;
            // small-spatial (hw<=64) + large-batch (n>128): ASM-NHWC beats GemmBwd.
            if(hw <= 64 && n > 128)
                return Ids().bwd_nhwc;
            // c==oc high-c (c>=1024): ASM-NHWC beats GemmBwd.
            if(c >= 1024 && c == oc)
                return Ids().bwd_nhwc;
            return Ids().gemm_bwd_1x1;
        }
        return Ids().bwd_nhwc;
    }

    if(p.IsFp32())
    {
        // NHWC-output: Winograd and Gemm bwd solvers are invalid here.
        if(p.GetOutLayout() == "NHWC")
            return Ids().bwd_nhwc;
        // h<=2 fp32 bwd 1x1 s1 → BWD_NHWC.
        if(g == 1 && fy == 1 && fx == 1 && sh == 1 && h <= 2)
            return Ids().bwd_nhwc;
        if(g != 1)
        {
            // 5x5 s=1 — NAIVE valid & wins; 5x5 s>=2 — NAIVE invalid, W2X3 wins.
            if(fy == 5 && fx == 5)
            {
                if(sh == 1)
                    return Ids().naive_bwd;
                return Ids().winograd_2x3;
            }
            if(fy == 1 && fx == 3)
                return Ids().naive_bwd;
            // depthwise 3x3 — NAIVE wins ~95%. Carve: s=2 c<=24 hw<=3300 → W2X3.
            if(g == c && fy == 3 && fx == 3)
            {
                if(sh == 2 && c <= 24 && hw <= 3300)
                    return Ids().winograd_2x3;
                // s2 n==1 c>=64 hw>=784: NAIVE unmeasured; W2X3 wins.
                if(sh == 2 && n == 1 && c >= 64 && hw >= 784)
                    return Ids().winograd_2x3;
                return Ids().naive_bwd;
            }
            if(fy == 3 && fx == 3 && sh == 1)
                return Ids().winograd_2x3;
            return Ids().winograd_3x2;
        }
        // Low-channel fp32 bwd: per-filter routing.
        if(c <= 8)
        {
            if(fy == 11 && sh == 4)
                return Ids().bwd_nhwc;
            if(fy == 3 && fx == 3 && sh == 1)
                return Ids().winograd_2x3_g1;
            if(fy == 5 && fx == 5)
                return Ids().winograd_2x3_g1;
            if(sh == 1 && ((fy == 7 && fx == 7) || (fy == 5 && fx == 20)))
                return Ids().gemm_bwd_rest;
            if(fy == 7 && fx == 7 && sh == 2)
            {
                // Mid-spatial+batch+moderate-h: W3X2; otherwise GemmBwdRest.
                if(hw > 4096 && n >= 9 && h <= 305)
                    return Ids().winograd_3x2;
                return Ids().gemm_bwd_rest;
            }
            return Ids().winograd_3x2;
        }
        // 1x1 s1 — GEMM_BWD_1X1 wins broadly (h<=2 routed above).
        if(fy == 1 && fx == 1 && sh == 1)
        {
            if(c >= 1024 && c == oc)
                return Ids().bwd_nhwc;
            return Ids().gemm_bwd_1x1;
        }
        // 1x1 s2 g=1 mid-channel (64<c<=256): GEMM_BWD_1X1_S2 beats NHWC, except
        // small-spatial (hw<=4096) where NHWC wins.
        if(fy == 1 && fx == 1 && sh == 2 && 64 < c && c <= 256)
        {
            if(hw <= 4096)
                return Ids().bwd_nhwc;
            return Ids().gemm_bwd_1x1_s2;
        }
        // 3x3 s1 g=1 — Winograd 2x3-g1 dominates up through c<=512, except
        // dilated / asymmetric-padded shapes (Winograd rejects).
        if(fy == 3 && fx == 3 && sh == 1)
        {
            if(c <= 512)
            {
                const auto dy = p.GetDilationH();
                const auto dx = p.GetDilationW();
                const auto ph = p.GetPadH();
                const auto pw = p.GetPadW();
                if(dy != 1 || dx != 1 || ph != pw || ph > 2)
                    return Ids().bwd_nhwc;
                // tiny-spatial (h*w<=49): NHWC beats W2X3g1.
                if(hw <= 49)
                    return Ids().bwd_nhwc;
                // wide-out (oc>512): GemmBwdRest beats NHWC.
                if(oc > 512)
                    return Ids().gemm_bwd_rest;
                // mid-spatial (hw<=192) + mid-out (oc>=128) + small-batch (n<=16): NHWC.
                if(hw <= 192 && oc >= 128 && n <= 16)
                    return Ids().bwd_nhwc;
                return Ids().winograd_2x3_g1;
            }
            return Ids().bwd_nhwc;
        }
        // 3x3 s2 g=1 — low-C Winograd 3x2.
        if(fy == 3 && fx == 3 && sh == 2)
        {
            if(c <= 64)
                return Ids().winograd_3x2;
            return Ids().bwd_nhwc;
        }
        if(fy == 7 && fx == 7 && sh >= 2)
            return Ids().winograd_3x2;
        if(fy == 4 && fx == 4)
            return Ids().winograd_3x2;
        if(fy == 5 && fx == 20 && sh == 2)
            return Ids().winograd_3x2;
        return Ids().bwd_nhwc;
    }

    if(p.IsFp16())
    {
        // NHWC-output: Winograd / WinoRage / Gemm bwd solvers are invalid here.
        if(p.GetOutLayout() == "NHWC")
            return Ids().bwd_nhwc;
        // NHWC-layout inputs: WinoRage/Winograd are invalid; route to GROUP.
        if(p.GetInLayout() == "NHWC")
            return Ids().group_bwd;
        if(g != 1)
        {
            // fp16 bwd grouped — per-filter routing.
            if(c <= 8 && sh >= 2 &&
               ((fy == 3 && fx == 3) || (fy == 7 && fx == 7) || (fy == 5 && fx == 20)))
                return Ids().winograd_3x2;
            if(c <= 8)
                return Ids().group_bwd;
            if(fy == 1 && fx == 3 && sh == 1)
                return Ids().naive_bwd;
            // depthwise 3x3 / 5x5 — NAIVE wins ~100%.
            if(g == c && fy == 3 && fx == 3)
            {
                // s=2 c<=32: NAIVE often unmeasured; W2X3 wins.
                if(sh == 2 && c <= 32)
                    return Ids().winograd_2x3;
                return Ids().naive_bwd;
            }
            if(g == c && fy == 5 && fx == 5)
            {
                // NAIVE invalid for s>=2 here; W2X3 wins.
                if(sh == 1)
                    return Ids().naive_bwd;
                return Ids().winograd_2x3;
            }
            if(fy == 3 && fx == 3 && sh == 2)
                return Ids().winograd_3x2;
            if(fy == 3 && fx == 3 && sh == 1)
                return Ids().winograd_2x3;
            if(fy == 5 && fx == 5 && sh == 2)
                return Ids().winograd_2x3;
            // grouped 1x1 s1: WINOGRAD_3X2 beats GROUP_BWD.
            if(fy == 1 && fx == 1 && sh == 1)
                return Ids().winograd_3x2;
            return Ids().group_bwd;
        }
        // Low-channel fp16 bwd.
        if(c <= 8)
        {
            if(sh >= 2 &&
               ((fy == 3 && fx == 3) || (fy == 7 && fx == 7) || (fy == 5 && fx == 20)))
            {
                // low-c 7x7 s2 h>305: GemmBwdRest beats Wino3x2.
                if(fy == 7 && fx == 7 && h > 305)
                    return Ids().gemm_bwd_rest;
                return Ids().winograd_3x2;
            }
            if(sh == 1)
            {
                if(fy == 3 && fx == 3)
                    return Ids().winograd_2x3_g1;
                if((fy == 7 && fx == 7) || (fy == 5 && fx == 20))
                    return Ids().gemm_bwd_rest;
                if(fy == 1 && fx == 1)
                    return Ids().bwd_nhwc;
                if(fy == 5 && fx == 5)
                    return Ids().winograd_2x3_g1;
            }
            // low-C 5x5 stride-2 — Winograd 2x3-g1 wins.
            if(sh == 2 && fy == 5 && fx == 5)
                return Ids().winograd_2x3_g1;
            return Ids().group_bwd;
        }
        if(fy == 11)
            return Ids().group_bwd;
        // 1x1 s1 — GEMM_BWD_1X1 wins broadly except very-thin spatial (h<=2).
        if(fy == 1 && fx == 1 && sh == 1)
        {
            if(h <= 2)
                return Ids().bwd_nhwc;
            // c==oc high-c (c>=1024): ASM-NHWC beats GemmBwd.
            if(c >= 1024 && c == oc)
                return Ids().bwd_nhwc;
            return Ids().gemm_bwd_1x1;
        }
        // 3x3 s1 (WinoRage disabled): ASM NHWC wins broadly.
        if(fy == 3 && fx == 3 && sh == 1)
        {
            // low-c (c<=64): W2X3g1 beats ASM.
            if(c <= 64)
                return Ids().winograd_2x3_g1;
            return Ids().bwd_nhwc;
        }
        // 3x3 s2 g=1 — low-C Winograd 3x2.
        if(fy == 3 && fx == 3 && sh == 2)
        {
            if(c <= 64)
                return Ids().winograd_3x2;
            return Ids().bwd_nhwc;
        }
        if(fy == 5 && fx == 5 && sh == 1)
            return Ids().bwd_nhwc;
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

// direction=1, bf16. (Python `_fwd_bf16_fp16` — only invoked for bf16; fp16 has
// its own dedicated path.)
solver::Id PickFwdBfp16(const conv::ProblemDescription& p)
{
    const auto g  = static_cast<int>(p.GetGroupCount());
    const auto c  = static_cast<int>(p.GetInChannels());
    const auto fy = static_cast<int>(p.GetWeightsHeight());
    const auto fx = static_cast<int>(p.GetWeightsWidth());

    if(g != 1)
    {
        // bf16 depthwise (g == c) 3x3 / 1x3 — NAIVE wins 100%.
        if(g == c && ((fy == 3 && fx == 3) || (fy == 1 && fx == 3)))
            return Ids().naive_fwd;
        return Ids().group_fwd;
    }
    if(p.GetInLayout() == "NHWC")
        return Ids().group_fwd;

    const auto sh = p.GetKernelStrideH();
    const auto sw = p.GetKernelStrideW();
    const auto oc = static_cast<int>(p.GetOutChannels());
    const auto h  = static_cast<int>(p.GetInHeight());
    const auto w  = static_cast<int>(p.GetInWidth());
    const auto n  = static_cast<int>(p.GetBatchSize());
    const auto hw = h * w;

    // bf16 1x1 s1 g=1: spatial/batch-tiered routing.
    if(fy == 1 && fx == 1 && sh == 1)
    {
        // large-spatial (h*w>4096) + mid-batch (n>8): GEMM slow, NHWC wins.
        if(hw > 4096 && n > 8)
            return Ids().fwd_nhwc;
        // mid-spatial (h*w>=512) + n>16: NHWC wins.
        if(hw >= 512 && n > 16)
            return Ids().fwd_nhwc;
        if(hw >= 8 && n <= 32)
            return Ids().gemm_fwd_1x1;
        // small-spatial (hw in [8,128]) mid-batch (32<n<=64): GEMM beats ASM.
        if(8 <= hw && hw <= 128 && 32 < n && n <= 64)
            return Ids().gemm_fwd_1x1;
        // high-c (c>=1024) tiny-spatial (hw<=49) huge-batch (n>=64): GROUP_FWD.
        if(c >= 1024 && hw <= 49 && n >= 64)
            return Ids().group_fwd;
        return Ids().fwd_nhwc;
    }
    // g=1 non-1x1: NHWC ASM beats GROUP on most shapes, except 5x5 where GROUP wins.
    if(fy == 5 && fx == 5)
        return Ids().group_fwd;
    // 4x1 s4: NAIVE wins 100%.
    if(fy == 4 && fx == 1 && sh == 4 && sw == 1)
        return Ids().naive_fwd;
    // 3x3 tiny problem (c<=8 oc<=8): NAIVE beats ASM-NHWC.
    if(fy == 3 && fx == 3 && c <= 8 && oc <= 8)
        return Ids().naive_fwd;
    // 3x3 s1 mid-channel (128<=c<=192): GROUP_FWD beats NHWC.
    if(fy == 3 && fx == 3 && sh == 1 && sw == 1 && 128 <= c && c <= 192)
        return Ids().group_fwd;
    // 3x3 s1 high-c (512<c<=2048) mid-batch (4<n<=64): GROUP_FWD beats NHWC.
    if(fy == 3 && fx == 3 && sh == 1 && sw == 1 && 512 < c && c <= 2048 && 4 < n && n <= 64)
        return Ids().group_fwd;
    // 3x3 s2 mid-channel small-spatial: hw<=196 AND 32<c<=128 → GROUP.
    if(fy == 3 && fx == 3 && sh == 2 && sw == 2 && 32 < c && c <= 128 && hw <= 196)
        return Ids().group_fwd;
    // 3x3 s2 64<c<=128 large-spatial (hw>=1024) small-batch (n<=4): GROUP.
    if(fy == 3 && fx == 3 && sh == 2 && sw == 2 && 64 < c && c <= 128 && hw >= 1024 && n <= 4)
        return Ids().group_fwd;
    return Ids().fwd_nhwc;
}

// direction=1, fp16
solver::Id PickFwdFp16(const conv::ProblemDescription& p)
{
    const auto g  = static_cast<int>(p.GetGroupCount());
    const auto c  = static_cast<int>(p.GetInChannels());
    const auto oc = static_cast<int>(p.GetOutChannels());
    const auto fy = static_cast<int>(p.GetWeightsHeight());
    const auto fx = static_cast<int>(p.GetWeightsWidth());
    const auto sh = p.GetKernelStrideH();
    const auto sw = p.GetKernelStrideW();
    const auto h  = static_cast<int>(p.GetInHeight());
    const auto w  = static_cast<int>(p.GetInWidth());
    const auto n  = static_cast<int>(p.GetBatchSize());
    const auto hw = h * w;

    if(p.GetInLayout() == "NHWC")
        return Ids().group_fwd;
    // NHWC-output fp16 g=1: FWD_NHWC invalid in 95% of cases; Winograd always valid.
    // Exception: 3x3 s1 NHWC-out has measured FWD_NHWC that crushes W3x2.
    if(g == 1 && p.GetOutLayout() == "NHWC")
    {
        if(fy == 3 && fx == 3 && sh == 1 && sw == 1)
            return Ids().fwd_nhwc;
        return Ids().winograd_3x2;
    }
    if(g == 1 && fy == 3 && fx == 3 && sh == 1 && sw == 1)
    {
        // 3x3 s1 g=1 tiny problem (c<=8 oc<=8): NAIVE beats Winograd.
        if(c <= 8 && oc <= 8)
            return Ids().naive_fwd;
        // huge-batch (n>=256) c>=128: FWD_NHWC beats W2X3g1.
        if(n >= 256 && c >= 128)
            return Ids().fwd_nhwc;
        // mid-c (65<=c<=192): W2X3g1 beats ASM.
        if(c <= 192)
        {
            // Sub-carve: c in (64,128] mid-spatial (hw>=196) large-batch (n>=16): FWD_NHWC.
            if(64 < c && c <= 128 && hw >= 196 && n >= 16)
                return Ids().fwd_nhwc;
            return Ids().winograd_2x3_g1;
        }
        return Ids().fwd_nhwc;
    }
    // 2x2 s1 g=1 (WinoRage disabled): W3x2 wins.
    if(g == 1 && fy == 2 && fx == 2 && sh == 1 && sw == 1)
        return Ids().winograd_3x2;
    if(g == 1 && fy == 1 && fx == 1 && sh == 1 && sw == 1)
    {
        // very-low-c (c<=16, oc<=32): W3x2 beats GemmFwd.
        if(c <= 16 && oc <= 32)
            return Ids().winograd_3x2;
        if(c <= 32)
            return Ids().gemm_fwd_1x1;
        // large-spatial (hw>=4096) large-batch (n>=64): GemmFwd1x1 beats FWD_NHWC.
        if(hw >= 4096 && n >= 64)
            return Ids().gemm_fwd_1x1;
        if(h <= 2 || n > 64)
            return Ids().fwd_nhwc;
        // n>=64 hw>=64: FWD_NHWC beats GemmFwd.
        if(n >= 64 && hw >= 64)
            return Ids().fwd_nhwc;
        return Ids().gemm_fwd_1x1;
    }
    if(g == 1 && fy == 1 && fx == 1 && sh == 2 && sw == 2)
    {
        // 1x1 s2 g=1: large spatial (h>=64) -> Gemm_0_2; thinner -> NHWC.
        if(h >= 64)
            return Ids().gemm_fwd_1x1_s2;
        return Ids().fwd_nhwc;
    }
    if(g == 1 && fy == 7 && fx == 7 && sh == 2 && sw == 2)
        return Ids().winograd_3x2;
    if(g != 1 && fy == 3 && fx == 3 && sh == 2 && sw == 2 && (g == 32 || g == 64))
    {
        // grouped 3x3 s2 g in (32,64) dilated: Winograd rejects dilation → GroupFwd.
        const auto dy = p.GetDilationH();
        const auto dx = p.GetDilationW();
        if(dy != 1 || dx != 1)
            return Ids().group_fwd;
        return Ids().winograd_3x2;
    }
    // grouped 3x3 s2 non-dw g in (2,6): GROUP_FWD beats W3x2.
    if((g == 2 || g == 6) && fy == 3 && fx == 3 && sh == 2 && sw == 2 && g != c &&
       p.GetInLayout() != "NHWC")
        return Ids().group_fwd;
    // Depthwise (g == c, g > 1): per-kernel h*w threshold routing.
    if(g != 1 && g == c && p.GetInLayout() != "NHWC")
    {
        if(fy == 1 && fx == 1 && hw < 262144)
            return Ids().naive_fwd;
        // dw 3x3 s1 c>256: NAIVE wins broadly.
        if(fy == 3 && fx == 3 && sh == 1 && sw == 1 && c > 256)
            return Ids().naive_fwd;
        // dw 3x3 s2 c>=256 large-spatial (hw>=32768): NAIVE beats GroupFwd.
        if(fy == 3 && fx == 3 && sh == 2 && sw == 2 && c >= 256 && hw >= 32768)
            return Ids().naive_fwd;
        // dw 3x3 s2 c<256 hw>=32768 n>=2: NAIVE beats GroupFwd.
        if(fy == 3 && fx == 3 && sh == 2 && sw == 2 && c < 256 && hw >= 32768 && n >= 2)
            return Ids().naive_fwd;
        if(fy == 3 && fx == 3 && hw < 32768)
        {
            // dw 3x3 sh=1 low-c: W2X3 beats NAIVE.
            if(sh == 1 && sw == 1 && c <= 64)
            {
                // dw 3x3 s1 c<=64 small-spatial (hw<=1024): NAIVE beats W2X3.
                if(hw <= 1024)
                    return Ids().naive_fwd;
                return Ids().winograd_2x3;
            }
            // dw 3x3 s1 n=1 mid-c (c<=160): Win2x3 beats Naive (NAIVE often unmeasured).
            if(sh == 1 && sw == 1 && n == 1 && c <= 160)
                return Ids().winograd_2x3;
            // dw 3x3 s2 n=1 hw>=8192 c<=180: W3x2 beats Naive.
            if(sh == 2 && sw == 2 && n == 1 && hw >= 8192 && c <= 180)
                return Ids().winograd_3x2;
            return Ids().naive_fwd;
        }
        // dw 3x3 s1 c<=64 large-spatial (hw>=262144): W2X3 beats GroupFwd.
        if(fy == 3 && fx == 3 && sh == 1 && sw == 1 && c <= 64 && hw >= 262144)
            return Ids().winograd_2x3;
        // dw 5x5 s2 16<c<=64: WIN2X3 dominates.
        if(fy == 5 && fx == 5 && sh == 2 && sw == 2 && 16 < c && c <= 64)
            return Ids().winograd_2x3;
        // dw 5x5 s2 n=1 64<c<=128: WIN2X3.
        if(fy == 5 && fx == 5 && sh == 2 && sw == 2 && n == 1 && 64 < c && c <= 128)
            return Ids().winograd_2x3;
        if(fy == 5 && fx == 5 && hw < 8192)
        {
            // dw 5x5 s1 n=1 mid-c (c<=128): GroupFwd beats Naive (NAIVE often unmeasured).
            if(sh == 1 && sw == 1 && n == 1 && c <= 128)
                return Ids().group_fwd;
            // dw 5x5 s1 n>=2 c<=64 hw>=3136: W2X3 beats Naive.
            if(sh == 1 && sw == 1 && n >= 2 && c <= 64 && hw >= 3136)
                return Ids().winograd_2x3;
            return Ids().naive_fwd;
        }
        if((fy == 1 && fx == 5) || (fy == 5 && fx == 1))
            return Ids().naive_fwd;
        // dw 7x7 s1 hw<=4096 c>=350 n>=2: NAIVE wins.
        if(fy == 7 && fx == 7 && sh == 1 && sw == 1 && hw <= 4096 && c >= 350 && n >= 2)
            return Ids().naive_fwd;
        if(fy == 7 && fx == 7 && hw < 1024)
            return Ids().naive_fwd;
        if(fy == 9 && fx == 9 && hw < 1024)
            return Ids().naive_fwd;
    }
    if(g != 1 && fy == 5 && fx == 5)
    {
        // grouped (non-dw) 5x5: NAIVE wins most when valid.
        if(p.GetInLayout() == "NHWC" || hw >= 8192)
            return Ids().group_fwd;
        return Ids().naive_fwd;
    }
    if(g != 1 && fy == 1 && fx == 3)
        return Ids().naive_fwd;
    // grouped 3x3 huge c/g (>=512): GemmFwdRest crushes everything.
    if(g != 1 && g != c && fy == 3 && fx == 3 && p.GetInLayout() != "NHWC" && g > 0 &&
       (c / g) >= 512)
        return Ids().gemm_fwd_rest;
    // grouped (non-dw) 3x3: Winograd 2x3 (s=1) / 3x2 (s=2) beat GROUP.
    if(g != 1 && g != c && fy == 3 && fx == 3 && p.GetInLayout() != "NHWC")
    {
        const auto dy = p.GetDilationH();
        const auto dx = p.GetDilationW();
        if(dy == 1 && dx == 1)
        {
            if(sh == 1 && sw == 1)
                return Ids().winograd_2x3;
            if(sh == 2 && sw == 2)
            {
                // grouped 3x3 s2 c>=256 c/g>=32: W3x2 unmeasured for ~9/80; GroupFwd.
                if(c >= 256 && g > 0 && (c / g) >= 32)
                    return Ids().group_fwd;
                return Ids().winograd_3x2;
            }
        }
    }
    if(g == 1 && fy == 3 && fx == 3 && sh == 2 && sw == 2)
    {
        // low-C → W3x2; mid-C → GROUP; high-C → NHWC ASM. Dilated → NHWC.
        const auto dy = p.GetDilationH();
        const auto dx = p.GetDilationW();
        if(dy != 1 || dx != 1)
            return Ids().fwd_nhwc;
        // 3x3 s2 tiny problem (c<=8 oc<=8): NAIVE beats W3x2.
        if(c <= 8 && oc <= 8)
            return Ids().naive_fwd;
        if(c <= 32)
            return Ids().winograd_3x2;
        if(c <= 128)
            return Ids().group_fwd;
        return Ids().fwd_nhwc;
    }
    // grouped (non-dw) 1x1 s1 g=4: GEMM beats other choices.
    if(g != 1 && g != c && g == 4 && fy == 1 && fx == 1 && sh == 1 && sw == 1)
        return Ids().gemm_fwd_1x1;
    if(g != 1)
        return Ids().group_fwd;
    // g=1 fy=1 fx=3: NHWC ASM wins.
    if(fy == 1 && fx == 3)
        return Ids().fwd_nhwc;
    if(fx == 1 || fx == 7)
        return Ids().fwd_nhwc;
    if(oc == 64)
        return Ids().fwd_nhwc;
    // g=1 fy=fx=stride large (>=16x16 s>=10): NHWC ASM beats GROUP_FWD (downsample pattern).
    if(fy >= 16 && fx >= 16 && sh >= 10)
        return Ids().fwd_nhwc;
    return Ids().group_fwd;
}

// direction=1, fp32
solver::Id PickFwdFp32(const conv::ProblemDescription& p)
{
    const auto g  = static_cast<int>(p.GetGroupCount());
    const auto fy = static_cast<int>(p.GetWeightsHeight());
    const auto fx = static_cast<int>(p.GetWeightsWidth());
    const auto sy = p.GetKernelStrideH();
    const auto sx = p.GetKernelStrideW();
    const auto c  = static_cast<int>(p.GetInChannels());
    const auto oc = static_cast<int>(p.GetOutChannels());
    const auto h  = static_cast<int>(p.GetInHeight());
    const auto w  = static_cast<int>(p.GetInWidth());
    const auto n  = static_cast<int>(p.GetBatchSize());
    const auto hw = h * w;

    if(g != 1)
    {
        // grouped 5x5 — NAIVE wins ~95%.
        if(fy == 5 && fx == 5)
            return Ids().naive_fwd;
        // depthwise 3x3 (g == c) — NAIVE wins ~98%. Carve: s=2 h>=300 → GroupFwd.
        if(fy == 3 && fx == 3 && g == c)
        {
            if(sy == 2 && h >= 300)
                return Ids().group_fwd;
            return Ids().naive_fwd;
        }
        if(g <= 84)
        {
            if(fy == 3 && fx == 3 && sy == 1 && sx == 1)
                return Ids().winograd_2x3;
            return Ids().winograd_3x2;
        }
        return Ids().group_fwd;
    }
    // NHWC-out: 1x1 s1 / 3x3 s1 → FWD_NHWC; others → Winograd 3x2.
    if(p.GetOutLayout() == "NHWC")
    {
        if(sy == 1 && sx == 1 && ((fy == 1 && fx == 1) || (fy == 3 && fx == 3)))
            return Ids().fwd_nhwc;
        return Ids().winograd_3x2;
    }
    // g=1 3x3 tiny problem (c<=8 oc<=8): NAIVE beats Winograd.
    if(c <= 8 && oc <= 8 && fy == 3 && fx == 3)
        return Ids().naive_fwd;
    // Low-channel inputs (RGB-style): Winograd_3x2 wins. Exception 11x11 stride-4.
    if(c <= 8)
    {
        if(fy == 11 && sy == 4)
            return Ids().fwd_nhwc;
        return Ids().winograd_3x2;
    }
    // 1x1 s1 — GEMM wins broadly; thin-spatial (h<=2) → NHWC.
    if(fy == 1 && fx == 1 && sy == 1 && sx == 1)
    {
        if(h <= 2)
            return Ids().fwd_nhwc;
        return Ids().gemm_fwd_1x1;
    }
    if(fy == 1 && fx == 1 && sy == 2 && sx == 2)
    {
        // 1x1 s2 g=1: GEMM_FWD_1X1_S2 wins for h>=17; NHWC for thin.
        if(h >= 17)
            return Ids().gemm_fwd_1x1_s2;
        return Ids().fwd_nhwc;
    }
    // 3x3 s1 g=1 — Winograd 2x3-g1 dominates up through mid-C.
    if(fy == 3 && fx == 3 && sy == 1 && sx == 1)
    {
        const auto dy = p.GetDilationH();
        const auto dx = p.GetDilationW();
        if(dy != 1 || dx != 1)
            return Ids().fwd_nhwc;
        if(c <= 512)
        {
            // low-c (c<=64): W2X3g1 beats NHWC even on tiny-spatial.
            if(c <= 64)
                return Ids().winograd_2x3_g1;
            // tiny-spatial (h*w<=49): NHWC beats Winograd.
            if(hw <= 49)
                return Ids().fwd_nhwc;
            if(c > 256)
            {
                // 256<c<=512 with mid-spatial (784<hw<=4096): W2X3_G1 wins.
                if(c <= 512 && 784 < hw && hw <= 4096)
                    return Ids().winograd_2x3_g1;
                return Ids().fwd_nhwc;
            }
            // mid-spatial (hw<=192) + mid-out (oc>=128) + small-batch (n<=16): NHWC.
            if(hw <= 192 && oc >= 128 && n <= 16)
                return Ids().fwd_nhwc;
            return Ids().winograd_2x3_g1;
        }
        return Ids().fwd_nhwc;
    }
    // 3x3 s2 g=1 — Winograd 3x2 at low C.
    if(fy == 3 && fx == 3 && sy == 2 && sx == 2)
    {
        if(c <= 64)
            return Ids().winograd_3x2;
        return Ids().fwd_nhwc;
    }
    // 7x7 s=1 g=1 — NHWC wins for c>8 (c<=8 routed above).
    if(fy == 7 && fx == 7 && sy == 1)
        return Ids().fwd_nhwc;
    if(fy == 7 && fx == 7)
        return Ids().winograd_3x2;
    if(fy == 5 && fx >= 10)
        return Ids().winograd_3x2;
    return Ids().fwd_nhwc;
}

// 3D problems (spatial_dim == 3)
solver::Id PickSolver3d(const conv::ProblemDescription& p)
{
    const auto dir = p.GetDirection();
    const auto fy  = static_cast<int>(p.GetWeightsHeight());
    const auto fx  = static_cast<int>(p.GetWeightsWidth());
    const auto fz  = static_cast<int>(p.GetWeightsDepth());
    const auto sh  = p.GetKernelStrideH();
    const auto sw  = p.GetKernelStrideW();
    const auto sd  = p.GetKernelStrideD();
    const auto c   = static_cast<int>(p.GetInChannels());
    const auto oc  = static_cast<int>(p.GetOutChannels());
    const auto h   = static_cast<int>(p.GetInHeight());
    const auto w   = static_cast<int>(p.GetInWidth());
    const auto n   = static_cast<int>(p.GetBatchSize());

    // 3D bwd 1x1x1 s1 → GemmBwd1x1_stride1 across bf16/fp16/fp32.
    if(dir == conv::Direction::BackwardData && fy == 1 && fx == 1 && fz == 1 && sh == 1 &&
       sw == 1 && sd == 1)
        return Ids().gemm_bwd_1x1;
    // 3D fp32 bwd 2x2x2 s222 / 3x3x3 sXYZ: GemmBwdRest beats 3DGroupBwd.
    if(dir == conv::Direction::BackwardData && p.IsFp32())
    {
        if((fy == 2 && fx == 2 && fz == 2) || (fy == 3 && fx == 3 && fz == 3))
        {
            // 3x3x3 narrow carve: c<256 prefers 3DGroupBwd.
            if(fy == 3 && fx == 3 && fz == 3 && c < 256)
                return Ids().dir3d_bwd;
            return Ids().gemm_bwd_rest;
        }
    }
    // 3D fp32 fwd 1x1x1 s1: GemmFwd1x1_0_1 beats 3DGroupFwd.
    if(dir == conv::Direction::Forward && p.IsFp32() && fy == 1 && fx == 1 && fz == 1 &&
       sh == 1 && sw == 1 && sd == 1)
        return Ids().gemm_fwd_1x1;
    // 3D fwd 3x3x1 s=2,2,1 low-c (c<=3): NaiveFwd beats 3DGroupFwd.
    if(dir == conv::Direction::Forward && c <= 3 && fy == 3 && fx == 3 && fz == 1 && sh == 2 &&
       sw == 2 && sd == 1)
    {
        // NAIVE often unmeasured at hw=1024 + (oc=64 or n>=400): route to 3DGroupFwd.
        if(h * w == 1024 && (oc == 64 || n >= 400))
            return Ids().dir3d_fwd;
        return Ids().naive_fwd;
    }
    switch(dir)
    {
    case conv::Direction::Forward: return Ids().dir3d_fwd;
    case conv::Direction::BackwardData: return Ids().dir3d_bwd;
    case conv::Direction::BackwardWeights: return Ids().dir3d_wrw;
    }
    return Ids().wrw_nhwc; // unreachable
}

} // namespace

solver::Id PickSolver(const conv::ProblemDescription& problem)
{
    // gfx942 transposed conv: TransposedConvBin* solvers are the measured winners.
    if(problem.GetConv().mode == miopenTranspose)
        return Ids().transposed_wino;
    if(problem.GetConv().mode != miopenConvolution)
        return {};

    if(problem.Is3d())
        return PickSolver3d(problem);

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
        // int8 fwd → GROUP_FWD per Python BUCKET.
        if(problem.IsInt8())
            return Ids().group_fwd;
        return {};
    }
    return {};
}

} // namespace gfx942
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
