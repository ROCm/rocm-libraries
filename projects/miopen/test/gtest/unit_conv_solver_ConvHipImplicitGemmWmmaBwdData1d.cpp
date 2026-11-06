// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

namespace {

// Backward data: x is dx (the long side), w is (Ci of dy, Co of dx, 1, K). Channel counts that are
// not multiples of the 16 / 64 tiles, batch 2, padding and lengths whose last dx positions get no
// tap cover the edge paths.
auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // Audio decoder upsampling (k = 2 x stride): stride 2, 4, 8-tap at stride 4.
        TestCase{{2, 64, 1, 602}, {64, 64, 1, 4}, {0, 1}, {1, 2}, {1, 1}, 1, datatype},
        TestCase{{1, 128, 1, 1501}, {128, 128, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, datatype},
        // Channels not multiples of the tiles; padding and an odd dx length.
        TestCase{{1, 37, 1, 801}, {50, 37, 1, 12}, {0, 5}, {1, 2}, {1, 1}, 1, datatype},
        // Stride 5 with k = 9 (taps per phase differ between phases).
        TestCase{{1, 96, 1, 647}, {32, 96, 1, 9}, {0, 2}, {1, 5}, {1, 1}, 1, datatype},
        // Stride 8 at k = 16 (32-q tile).
        TestCase{{1, 70, 1, 1203}, {40, 70, 1, 16}, {0, 4}, {1, 8}, {1, 1}, 1, datatype},
        // Stride 1: the input gradient of a conv1d.
        TestCase{{2, 8, 1, 1000}, {16, 8, 1, 3}, {0, 1}, {1, 1}, {1, 1}, 1, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx110X | Gpu::gfx115X |
                                                              Gpu::gfx120X);
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
    return miopen::solver::conv::ConvHipImplicitGemmWmmaBwdData1d{}.IsApplicable(ctx, problem);
}

} // namespace

using GPU_UnitTestConvSolverHipImplicitGemmWmmaBwdData1d_FP16  = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverHipImplicitGemmWmmaBwdData1d_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;

using CPU_UnitTestConvSolverHipImplicitGemmWmmaBwdData1dDevApplicabilityBwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmWmmaBwdData1d_FP16, ConvHipImplicitGemmWmmaBwdData1d)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmWmmaBwdData1d{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmWmmaBwdData1d_BFP16, ConvHipImplicitGemmWmmaBwdData1d)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmWmmaBwdData1d{});
};

TEST_P(CPU_UnitTestConvSolverHipImplicitGemmWmmaBwdData1dDevApplicabilityBwd_NONE,
       ConvHipImplicitGemmWmmaBwdData1d)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmWmmaBwdData1d{});
};

// Both sides of each bound, on mocked devices so the result does not depend on the host GPU.
TEST(CPU_UnitTestConvSolverHipImplicitGemmWmmaBwdData1dBounds_NONE,
     ConvHipImplicitGemmWmmaBwdData1d)
{
    auto gfx1151 = MockHandle{DevDescription{"gfx1151", 20, 32}, true};
    auto gfx1201 = MockHandle{DevDescription{"gfx1201", 32, 32}, true};
    auto gfx1030 = MockHandle{DevDescription{"gfx1030", 40, 32}, true};
    auto ctx     = miopen::ExecutionContext{&gfx1151};
    auto ctx1201 = miopen::ExecutionContext{&gfx1201};
    auto ctx1030 = miopen::ExecutionContext{&gfx1030};

    const auto bf16 = miopenBFloat16;
    EXPECT_TRUE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, bf16));
    EXPECT_TRUE(IsApplicableFor(
        ctx, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, miopenHalf));

    // fp16 / bf16 only; gfx110X / gfx115X and gfx120X (both WMMA layouts), not gfx103X.
    EXPECT_FALSE(IsApplicableFor(
        ctx, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, miopenFloat));
    EXPECT_TRUE(IsApplicableFor(
        ctx1201, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, bf16));
    EXPECT_FALSE(IsApplicableFor(
        ctx1030, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 1, bf16));

    // LDS bound (stride x taps per phase <= 20): stride 7 at k = 14 and stride 8 at k = 16 (with a
    // 32-q tile) fit; stride 8 at k = 24 and stride 4 at k = 24 (24 > 20) do not.
    EXPECT_TRUE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 14}, {0, 0}, {1, 7}, {1, 1}, 1, bf16));
    EXPECT_TRUE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 16}, {0, 4}, {1, 8}, {1, 1}, 1, bf16));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 24}, {0, 8}, {1, 8}, {1, 1}, 1, bf16));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 24}, {0, 0}, {1, 4}, {1, 1}, 1, bf16));

    // groups == 1, dilation 1, 1D only.
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 32, 1, 8}, {0, 2}, {1, 4}, {1, 1}, 2, bf16));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 8}, {0, 2}, {1, 4}, {1, 2}, 1, bf16));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 8, 512}, {64, 64, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, bf16));
}

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmWmmaBwdData1d_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmWmmaBwdData1d_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipImplicitGemmWmmaBwdData1dDevApplicabilityBwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));
