// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // Atrous 3x3 at stride 1: dilation 2 and 4 (MobileNetV2 DeepLab), and 6 and
        // 12 (separable ASPP), with partial row and column tiles and a batch of 2.
        TestCase{{1, 64, 33, 33}, {64, 1, 3, 3}, {2, 2}, {1, 1}, {2, 2}, 64, datatype},
        TestCase{{1, 96, 41, 37}, {96, 1, 3, 3}, {4, 4}, {1, 1}, {4, 4}, 96, datatype},
        TestCase{{1, 64, 33, 33}, {64, 1, 3, 3}, {6, 6}, {1, 1}, {6, 6}, 64, datatype},
        TestCase{{2, 64, 33, 31}, {64, 1, 3, 3}, {12, 12}, {1, 1}, {12, 12}, 64, datatype},
        // Larger filters: 5x5 at dilation 2 (MobileNetV3 DeepLab), 7x7 at dilation 3.
        TestCase{{1, 64, 33, 33}, {64, 1, 5, 5}, {4, 4}, {1, 1}, {2, 2}, 64, datatype},
        TestCase{{1, 64, 40, 40}, {64, 1, 7, 7}, {9, 9}, {1, 1}, {3, 3}, 64, datatype},
        // Dilation along one axis only, and unpadded.
        TestCase{{1, 64, 30, 40}, {64, 1, 3, 3}, {5, 1}, {1, 1}, {5, 1}, 64, datatype},
        TestCase{{1, 64, 30, 40}, {64, 1, 3, 3}, {1, 7}, {1, 1}, {1, 7}, 64, datatype},
        TestCase{{1, 64, 30, 30}, {64, 1, 3, 3}, {0, 0}, {1, 1}, {3, 3}, 64, datatype},
        // Strides: a whole number of dilation steps (2 and 2), and not (2 and 3, 3 and
        // 2), where the outputs of a lane are 3 and 2 apart.
        TestCase{{1, 64, 32, 32}, {64, 1, 3, 3}, {2, 2}, {2, 2}, {2, 2}, 64, datatype},
        TestCase{{1, 64, 31, 31}, {64, 1, 3, 3}, {3, 3}, {2, 2}, {3, 3}, 64, datatype},
        TestCase{{1, 64, 37, 41}, {64, 1, 3, 3}, {2, 2}, {3, 3}, {2, 2}, 64, datatype},
        // 1xK on H = 1, the depthwise conv1d layout (adjacent output columns):
        // Conv-TasNet's k = 3 at dilation 8 and 128, and k = 5 at stride 2.
        TestCase{{1, 64, 1, 999}, {64, 1, 1, 3}, {0, 8}, {1, 1}, {1, 8}, 64, datatype},
        TestCase{{1, 64, 1, 1500}, {64, 1, 1, 3}, {0, 128}, {1, 1}, {1, 128}, 64, datatype},
        TestCase{{2, 64, 1, 777}, {64, 1, 1, 5}, {0, 6}, {1, 2}, {1, 3}, 64, datatype},
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

// Depthwise problem with 64 channels, for the applicability bounds test.
bool IsApplicableFor(miopen::ExecutionContext& ctx,
                     miopenDataType_t datatype,
                     std::vector<size_t>&& x,
                     std::vector<size_t>&& w,
                     std::vector<int>&& pad,
                     std::vector<int>&& stride,
                     std::vector<int>&& dilation)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    auto problem   = TestCase{std::move(x),
                            std::move(w),
                            std::move(pad),
                            std::move(stride),
                            std::move(dilation),
                            64,
                            datatype}
                       .GetProblemDescription(miopen::conv::Direction::Forward);
    problem.SetupFloats(ctx);
    return miopen::solver::conv::ConvDirectDepthwiseFwd2DDilated{}.IsApplicable(ctx, problem);
}

} // namespace

using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilated_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilated_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilated_FP32  = GPU_UnitTestConvSolverFwd_FP32;

using CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilatedDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilated_FP16, ConvDirectDepthwiseFwd2DDilated)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2DDilated{});
}

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilated_BFP16, ConvDirectDepthwiseFwd2DDilated)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2DDilated{});
}

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilated_FP32, ConvDirectDepthwiseFwd2DDilated)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2DDilated{});
}

TEST_P(CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilatedDevApplicabilityFwd_NONE,
       ConvDirectDepthwiseFwd2DDilated)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2DDilated{});
}

// Both sides of each bound, on a mocked gfx1151 so the result does not depend on
// the host GPU.
TEST(CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilatedBounds_NONE,
     ConvDirectDepthwiseFwd2DDilated)
{
    auto gfx1151   = MockHandle{DevDescription{"gfx1151", 20, 32}, true};
    auto ctx       = miopen::ExecutionContext{&gfx1151};
    const auto f32 = miopenFloat;
    const auto f16 = miopenHalf;

    // Dilated only: undilated filters, and dilation along an axis a 1xK filter does
    // not span, are left to ConvDirectDepthwiseFwd2D.
    EXPECT_FALSE(IsApplicableFor(ctx, f32, {1, 64, 48, 48}, {64, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}));
    EXPECT_FALSE(IsApplicableFor(ctx, f32, {1, 64, 1, 512}, {64, 1, 1, 3}, {0, 1}, {1, 1}, {4, 1}));
    EXPECT_TRUE(IsApplicableFor(ctx, f32, {1, 64, 1, 512}, {64, 1, 1, 3}, {0, 4}, {1, 1}, {1, 4}));

    // Any dilation at stride 1: 3x3 needs 21 window slots in fp32, 27 in fp16.
    EXPECT_TRUE(
        IsApplicableFor(ctx, f32, {1, 64, 65, 65}, {64, 1, 3, 3}, {18, 18}, {1, 1}, {18, 18}));
    EXPECT_TRUE(
        IsApplicableFor(ctx, f16, {1, 64, 65, 65}, {64, 1, 3, 3}, {18, 18}, {1, 1}, {18, 18}));

    // Filter extent up to 7.
    EXPECT_TRUE(IsApplicableFor(ctx, f32, {1, 64, 48, 48}, {64, 1, 7, 7}, {6, 6}, {1, 1}, {2, 2}));
    EXPECT_FALSE(IsApplicableFor(ctx, f32, {1, 64, 48, 48}, {64, 1, 9, 9}, {8, 8}, {1, 1}, {2, 2}));

    // Window bound: at a horizontal stride of 3 and dilation 2 the outputs of a lane
    // are 3 window columns apart; 7x7 then needs 119 slots in fp32 (2 outputs per
    // lane) and 161 in fp16 (4 outputs per lane).
    EXPECT_TRUE(IsApplicableFor(ctx, f32, {1, 64, 48, 48}, {64, 1, 7, 7}, {6, 6}, {1, 3}, {2, 2}));
    EXPECT_FALSE(IsApplicableFor(ctx, f16, {1, 64, 48, 48}, {64, 1, 7, 7}, {6, 6}, {1, 3}, {2, 2}));
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilated_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilated_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilated_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDilatedDevApplicabilityFwd_NONE,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(GetConvTestCases(miopenHalf)[0])));
