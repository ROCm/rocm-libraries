// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Solver-selection tests for the small-batch 1x1 NHWC backward GEMM exclusion on gfx942.
// These cover which solvers the gate turns off (across the wrw and bwd-data GEMM families),
// not the per-solver numerics, so they live apart from the per-solver functionality tests.
// Mocked devices keep the result deterministic on any host GPU.

#include "unit_conv_solver.hpp"

namespace {

// 1x1 NHWC fp32 backward problem. hw=1 selects the point-output path (GemmWrwUniversal /
// GemmBwdRest); stride=2 is the downsample case that must stay applicable.
auto MakeProblem(miopen::ExecutionContext& ctx,
                 miopen::conv::Direction dir,
                 std::size_t n,
                 int stride,
                 std::size_t hw)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    auto problem   = TestCase{{n, 8, hw, hw},
                              {8, 8, 1, 1},
                              {0, 0},
                              {stride, stride},
                              {1, 1},
                            miopenFloat,
                            miopenTensorNHWC}
                       .GetProblemDescription(dir);
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);
    return problem;
}

} // namespace

TEST(CPU_UnitTestConvSolverSmallBatchGemmExclusion_NONE, Wrw)
{
    auto gfx942   = MockHandle{DevDescription{"gfx942", 304, 64}, true};
    auto other    = MockHandle{DevDescription{"gfx90a", 104, 64}, true};
    auto ctx942   = miopen::ExecutionContext{&gfx942};
    auto ctxother = miopen::ExecutionContext{&other};

    const auto dir          = miopen::conv::Direction::BackwardWeights;
    const auto s1_small_942 = MakeProblem(ctx942, dir, 2, 1, 8);   // 1x1 stride-1, within 1..3
    const auto s1_small_oth = MakeProblem(ctxother, dir, 2, 1, 8); // same shape, other arch
    const auto s1_large_942 = MakeProblem(ctx942, dir, 4, 1, 8);   // above the exclusion
    const auto s2_small_942 = MakeProblem(ctx942, dir, 2, 2, 8);   // 1x1 stride-2: NOT excluded
    const auto po_small_942 = MakeProblem(ctx942, dir, 2, 1, 1);   // point-output: GemmWrwUniversal

    const auto stride1   = miopen::solver::conv::GemmWrw1x1_stride1{};
    const auto universal = miopen::solver::conv::GemmWrwUniversal{};

    // Excluded on gfx942; both GemmWrw1x1_stride1 and GemmWrwUniversal honour the gate.
    EXPECT_FALSE(stride1.IsApplicable(ctx942, s1_small_942));
    EXPECT_FALSE(universal.IsApplicable(ctx942, po_small_942));
    // Not excluded: other arch, larger batch, and the 1x1 stride-2 downsample.
    EXPECT_TRUE(stride1.IsApplicable(ctxother, s1_small_oth));
    EXPECT_TRUE(stride1.IsApplicable(ctx942, s1_large_942));
    EXPECT_TRUE(universal.IsApplicable(ctx942, s2_small_942));
}

TEST(CPU_UnitTestConvSolverSmallBatchGemmExclusion_NONE, BwdData)
{
    auto gfx942   = MockHandle{DevDescription{"gfx942", 304, 64}, true};
    auto other    = MockHandle{DevDescription{"gfx90a", 104, 64}, true};
    auto ctx942   = miopen::ExecutionContext{&gfx942};
    auto ctxother = miopen::ExecutionContext{&other};

    const auto dir          = miopen::conv::Direction::BackwardData;
    const auto s1_small_942 = MakeProblem(ctx942, dir, 2, 1, 8);   // 1x1 stride-1, within 1..3
    const auto s1_small_oth = MakeProblem(ctxother, dir, 2, 1, 8); // same shape, other arch
    const auto s1_large_942 = MakeProblem(ctx942, dir, 4, 1, 8);   // above the exclusion
    const auto s2_small_942 = MakeProblem(ctx942, dir, 2, 2, 8);   // 1x1 stride-2: NOT excluded
    const auto po_small_942 = MakeProblem(ctx942, dir, 2, 1, 1);   // point-output: GemmBwdRest

    const auto stride1 = miopen::solver::conv::GemmBwd1x1_stride1{};
    const auto rest    = miopen::solver::conv::GemmBwdRest{};

    // Excluded on gfx942; both GemmBwd1x1_stride1 and GemmBwdRest honour the gate.
    EXPECT_FALSE(stride1.IsApplicable(ctx942, s1_small_942));
    EXPECT_FALSE(rest.IsApplicable(ctx942, po_small_942));
    // Not excluded: other arch, larger batch, and the 1x1 stride-2 downsample (claimed by
    // GemmBwdRest) — the gate's stride-1 restriction leaves the downsample untouched.
    EXPECT_TRUE(stride1.IsApplicable(ctxother, s1_small_oth));
    EXPECT_TRUE(stride1.IsApplicable(ctx942, s1_large_942));
    EXPECT_TRUE(rest.IsApplicable(ctx942, s2_small_942));
}
