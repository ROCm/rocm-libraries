// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

namespace {

// Backward data: x is dx (the long side), w is (Ci of dy, Co of dx, 1, K). Channel counts that are
// not multiples of the channel tile, batch 2, padding and lengths whose last dx positions get no
// tap cover the edge paths.
auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // Audio decoder upsampling (k = 2 x stride) at stride 2, 4 and 8.
        TestCase{{2, 64, 1, 602}, {64, 64, 1, 4}, {0, 1}, {1, 2}, {1, 1}, 1, datatype},
        TestCase{{1, 128, 1, 1501}, {128, 128, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, datatype},
        TestCase{{1, 70, 1, 1203}, {40, 70, 1, 16}, {0, 4}, {1, 8}, {1, 1}, 1, datatype},
        // Channels not multiples of the tiles; padding and an odd dx length.
        TestCase{{1, 37, 1, 801}, {50, 37, 1, 12}, {0, 5}, {1, 2}, {1, 1}, 1, datatype},
        // Stride 5 with k = 9 (taps per phase differ between phases).
        TestCase{{1, 96, 1, 647}, {32, 96, 1, 9}, {0, 2}, {1, 5}, {1, 1}, 1, datatype},
        // Stride wider than the filter: some phases get no tap.
        TestCase{{1, 24, 1, 500}, {16, 24, 1, 3}, {0, 0}, {1, 5}, {1, 1}, 1, datatype},
        // Stride 1: the input gradient of a conv1d.
        TestCase{{2, 8, 1, 1000}, {16, 8, 1, 3}, {0, 1}, {1, 1}, {1, 1}, 1, datatype},
        // The filter width bound.
        TestCase{{1, 16, 1, 800}, {24, 16, 1, 64}, {0, 0}, {1, 8}, {1, 1}, 1, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(
            Gpu::gfx103X | Gpu::gfx110X | Gpu::gfx115X | Gpu::gfx117X | Gpu::gfx120X);
        return p;
    }();
    return params;
}

bool IsApplicableFor(miopen::ExecutionContext& ctx,
                     std::vector<size_t>&& x,
                     std::vector<size_t>&& w,
                     std::vector<int>&& pad,
                     std::vector<int>&& stride,
                     std::vector<int>&& dilation,
                     int groups,
                     miopenDataType_t type)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    auto problem   = TestCase{std::move(x),
                            std::move(w),
                            std::move(pad),
                            std::move(stride),
                            std::move(dilation),
                            groups,
                            type}
                       .GetProblemDescription(miopen::conv::Direction::BackwardData);
    problem.SetupFloats(ctx);
    return miopen::solver::conv::ConvHipImplicitGemmFmaBwdData1dNoWorkspace{}.IsApplicable(ctx,
                                                                                           problem);
}

} // namespace

using GPU_UnitTestConvSolverHipImplicitGemmFmaBwdData1dNoWorkspace_FP16 =
    GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverHipImplicitGemmFmaBwdData1dNoWorkspace_BFP16 =
    GPU_UnitTestConvSolverBwd_BFP16;

using CPU_UnitTestConvSolverHipImplicitGemmFmaBwdData1dNoWorkspaceDevApplicabilityBwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmFmaBwdData1dNoWorkspace_FP16,
       ConvHipImplicitGemmFmaBwdData1dNoWorkspace)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmFmaBwdData1dNoWorkspace{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmFmaBwdData1dNoWorkspace_BFP16,
       ConvHipImplicitGemmFmaBwdData1dNoWorkspace)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmFmaBwdData1dNoWorkspace{});
};

TEST_P(CPU_UnitTestConvSolverHipImplicitGemmFmaBwdData1dNoWorkspaceDevApplicabilityBwd_NONE,
       ConvHipImplicitGemmFmaBwdData1dNoWorkspace)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmFmaBwdData1dNoWorkspace{});
};

// Both sides of each bound, on mocked devices so the result does not depend on the host GPU.
TEST(CPU_UnitTestConvSolverHipImplicitGemmFmaBwdData1dNoWorkspaceBounds_NONE,
     ConvHipImplicitGemmFmaBwdData1dNoWorkspace)
{
    auto gfx1151 = MockHandle{DevDescription{"gfx1151", 20, 32}, true};
    auto gfx1201 = MockHandle{DevDescription{"gfx1201", 32, 32}, true};
    auto gfx1030 = MockHandle{DevDescription{"gfx1030", 40, 32}, true};
    auto gfx1010 = MockHandle{DevDescription{"gfx1010", 20, 32}, true};
    auto gfx942  = MockHandle{DevDescription{"gfx942", 304, 64}, true};
    auto ctx     = miopen::ExecutionContext{&gfx1151};
    auto ctx1201 = miopen::ExecutionContext{&gfx1201};
    auto ctx1030 = miopen::ExecutionContext{&gfx1030};
    auto ctx1010 = miopen::ExecutionContext{&gfx1010};
    auto ctx942  = miopen::ExecutionContext{&gfx942};

    const auto f16 = miopenHalf;

    // RDNA only; fp16 and bf16 (gfx101X / gfx103X included), not fp32, which needs no workspace in
    // ConvHipImplicitGemmFmaBwdData1d already.
    EXPECT_FALSE(IsApplicableFor(
        ctx, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, miopenFloat));
    EXPECT_TRUE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, f16));
    EXPECT_TRUE(
        IsApplicableFor(ctx1201, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, f16));
    EXPECT_TRUE(
        IsApplicableFor(ctx1030, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, f16));
    EXPECT_TRUE(
        IsApplicableFor(ctx1010, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, f16));
    EXPECT_FALSE(
        IsApplicableFor(ctx942, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, f16));
    EXPECT_TRUE(IsApplicableFor(
        ctx1030, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, miopenHalf));
    EXPECT_TRUE(IsApplicableFor(
        ctx1010, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, miopenBFloat16));
    EXPECT_FALSE(IsApplicableFor(
        ctx942, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, miopenHalf));

    // Filter width <= 64, stride <= 8.
    EXPECT_TRUE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 64}, {0, 0}, {1, 8}, {1, 1}, 1, f16));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 65}, {0, 0}, {1, 8}, {1, 1}, 1, f16));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 18}, {0, 0}, {1, 9}, {1, 1}, 1, f16));

    // groups == 1, dilation 1, 1D only.
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 32, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 2, f16));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 2}, 1, f16));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 8, 512}, {64, 64, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, f16));
}

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmFmaBwdData1dNoWorkspace_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmFmaBwdData1dNoWorkspace_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverHipImplicitGemmFmaBwdData1dNoWorkspaceDevApplicabilityBwd_NONE,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(GetConvTestCases(miopenHalf)[0])));
