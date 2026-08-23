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

// Pure-CPU unit tests for the naive-conv work-size threshold
// (ConvDirectNaiveConvExceedsWorkLimit). Naive stays *applicable* at any size
// (it is the universal fallback); this threshold is consumed at the selection
// layer (EvaluateInvokers, src/conv/solver_finders.cpp) to keep the un-tiled
// naive kernel from being *benchmarked* on shapes so large its single launch
// would trip the OS GPU watchdog (TDR) when a non-naive alternative also
// applies. See src/solver/conv/conv_direct_naive_conv.cpp for the metric.
//
// These tests exercise the metric directly on ProblemDescriptions. They never
// construct a Handle and never launch a kernel, so they are safe on machines
// where the huge shapes would otherwise TDR.

#include <cstdint>

#include <gtest/gtest.h>

#include <miopen/conv/problem_description.hpp>
#include <miopen/solver/conv_direct_naive_conv.hpp>

#include "unit_TensorDescriptor.hpp"
#include "unit_conv_ConvolutionDescriptor.hpp"
#include "lib_env_var.hpp"
#include "gtest_common.hpp"

namespace {

using miopen::unit_tests::ConvolutionDescriptorParams;
using miopen::unit_tests::TensorDescriptorParams;
using Direction = miopen::conv::Direction;

// Mirror of the env override read by the gate (declared UINT64 in the library).
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK)

// Build a 2D conv ProblemDescription with square spatial dims, odd square
// filter, unit stride/dilation and 'same' padding (so out spatial == in
// spatial). Layout/dtype are irrelevant to the work metric, so we use the
// simplest (NCHW, fp32); lens are always in NCHW logical order.
miopen::conv::ProblemDescription Make2dProblem(
    std::size_t n, std::size_t c, std::size_t hw, std::size_t k, std::size_t fyx, Direction dir)
{
    const int pad = static_cast<int>(fyx / 2);
    TensorDescriptorParams in{miopenFloat, {n, c, hw, hw}};
    TensorDescriptorParams wei{miopenFloat, {k, c, fyx, fyx}};
    TensorDescriptorParams out{miopenFloat, {n, k, hw, hw}};
    ConvolutionDescriptorParams conv{{pad, pad}, {1, 1}, {1, 1}};
    return miopen::conv::ProblemDescription{in.GetTensorDescriptor(),
                                            wei.GetTensorDescriptor(),
                                            out.GetTensorDescriptor(),
                                            conv.GetConvolutionDescriptor(),
                                            dir};
}

// 3D variant (NCDHW), same conventions.
miopen::conv::ProblemDescription Make3dProblem(
    std::size_t n, std::size_t c, std::size_t dhw, std::size_t k, std::size_t fzyx, Direction dir)
{
    const int pad = static_cast<int>(fzyx / 2);
    TensorDescriptorParams in{miopenFloat, {n, c, dhw, dhw, dhw}};
    TensorDescriptorParams wei{miopenFloat, {k, c, fzyx, fzyx, fzyx}};
    TensorDescriptorParams out{miopenFloat, {n, k, dhw, dhw, dhw}};
    ConvolutionDescriptorParams conv{{pad, pad, pad}, {1, 1, 1}, {1, 1, 1}};
    return miopen::conv::ProblemDescription{in.GetTensorDescriptor(),
                                            wei.GetTensorDescriptor(),
                                            out.GetTensorDescriptor(),
                                            conv.GetConvolutionDescriptor(),
                                            dir};
}

// Depthwise 2D conv (group_count == in_channels == out_channels), so
// C_per_group == 1. Same spatial conventions as Make2dProblem; weights are
// {k, C_per_group=1, fyx, fyx}. Because the work metric folds in the group count
// (below), every real depthwise conv lands far under the limit, so the selection
// gate never work-defers it -- Naive keeps competing on merit with no special
// case. These tests lock that safety invariant in.
miopen::conv::ProblemDescription
MakeDepthwise2dProblem(std::size_t n, std::size_t c, std::size_t hw, std::size_t fyx, Direction dir)
{
    const int pad   = static_cast<int>(fyx / 2);
    const int group = static_cast<int>(c); // depthwise: group == c == k
    TensorDescriptorParams in{miopenFloat, {n, c, hw, hw}};
    TensorDescriptorParams wei{miopenFloat, {c, 1, fyx, fyx}}; // {k, C_per_group, fyx, fyx}
    TensorDescriptorParams out{miopenFloat, {n, c, hw, hw}};
    ConvolutionDescriptorParams conv{{pad, pad}, {1, 1}, {1, 1}, group};
    return miopen::conv::ProblemDescription{in.GetTensorDescriptor(),
                                            wei.GetTensorDescriptor(),
                                            out.GetTensorDescriptor(),
                                            conv.GetConvolutionDescriptor(),
                                            dir};
}

bool ExceedsWorkLimit(const miopen::conv::ProblemDescription& p)
{
    return miopen::solver::conv::ConvDirectNaiveConvExceedsWorkLimit(p);
}

} // namespace

// The confirmed-TDR SDXL VAE decode conv (c128 k128 768^2 3x3, N=1) is ~87 GMAC,
// far above the ~16 GMAC default limit -> over threshold (naive skipped in the
// mini-bench when a non-naive alternative also applies).
TEST(CPU_ConvNaiveWorkGate_NONE, HugeShapeExceedsLimit)
{
    EXPECT_TRUE(ExceedsWorkLimit(Make2dProblem(1, 128, 768, 128, 3, Direction::Forward)));
}

// A resnet50-class 3x3 (~0.1 GMAC) is orders of magnitude below the limit -> under
// threshold (naive runs normally in the mini-bench).
TEST(CPU_ConvNaiveWorkGate_NONE, SmallShapeWithinLimit)
{
    EXPECT_FALSE(ExceedsWorkLimit(Make2dProblem(1, 64, 56, 64, 3, Direction::Forward)));
}

// A large 3D conv (~98 GMAC) is over threshold too -- the metric folds in output
// depth and the 3D filter volume.
TEST(CPU_ConvNaiveWorkGate_NONE, HugeShape3dExceedsLimit)
{
    EXPECT_TRUE(ExceedsWorkLimit(Make3dProblem(1, 64, 96, 64, 3, Direction::Forward)));
}

// The MAC total is identical for fwd/bwd/wrw, so one constant governs all three
// directions: the huge shape is over threshold and the small shape is not,
// regardless of direction.
TEST(CPU_ConvNaiveWorkGate_NONE, DirectionInvariant)
{
    for(const auto dir : {Direction::Forward, Direction::BackwardData, Direction::BackwardWeights})
    {
        EXPECT_TRUE(ExceedsWorkLimit(Make2dProblem(1, 128, 768, 128, 3, dir)));
        EXPECT_FALSE(ExceedsWorkLimit(Make2dProblem(1, 64, 56, 64, 3, dir)));
    }
}

// Depthwise safety invariant for the selection gate: because the work metric folds
// in the group count (C_per_group == 1 for depthwise), the same c/k/hw/filter that
// is far *over* the limit as a dense conv is ~C times smaller as depthwise and lands
// well *under* it. This is why Naive is never work-deferred for depthwise and keeps
// competing (TDR-safe). A large-spatial depthwise (c=k=g=1024, 64^2, 3x3 ~= 38 MMAC)
// stays under threshold; the matching dense conv (~38 GMAC) is over.
TEST(CPU_ConvNaiveWorkGate_NONE, DepthwiseFarUnderLimit)
{
    EXPECT_FALSE(ExceedsWorkLimit(MakeDepthwise2dProblem(1, 1024, 64, 3, Direction::Forward)));
}

TEST(CPU_ConvNaiveWorkGate_NONE, DenseCounterpartOverLimit)
{
    // Identical c/k/hw/filter as the depthwise above but dense (group == 1): ~C
    // times more work, pushing it over the limit -- confirms the metric is
    // group-aware and the depthwise case is not merely small by coincidence.
    EXPECT_TRUE(ExceedsWorkLimit(Make2dProblem(1, 1024, 64, 1024, 3, Direction::Forward)));
}

// The env override raises the limit; a huge shape that is normally over threshold
// falls under it when the cap is set above its work count (how a user with a raised
// TdrDelay opts naive back into the mini-bench on big shapes).
TEST(CPU_ConvNaiveWorkGate_NONE, EnvOverrideRaisesLimit)
{
    const auto huge = Make2dProblem(1, 128, 768, 128, 3, Direction::Forward);
    ASSERT_TRUE(ExceedsWorkLimit(huge)); // over threshold with the default limit
    ScopedEnvironment<std::uint64_t> raise(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK,
                                           std::uint64_t{1} << 60);
    EXPECT_FALSE(ExceedsWorkLimit(huge));
}

// The env override also lowers the limit: a small shape that is normally under
// threshold goes over it when the cap is dropped below its work count. Confirms the
// override is honored in both directions rather than only disabling the check.
TEST(CPU_ConvNaiveWorkGate_NONE, EnvOverrideLowersLimit)
{
    const auto small = Make2dProblem(1, 64, 56, 64, 3, Direction::Forward);
    ASSERT_FALSE(ExceedsWorkLimit(small)); // under threshold with the default limit
    ScopedEnvironment<std::uint64_t> lower(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK,
                                           std::uint64_t{1000});
    EXPECT_TRUE(ExceedsWorkLimit(small));
}
