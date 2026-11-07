// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // Shape reported in #12334 and minimum supported channel count.
        TestCase{{2, 32, 16, 16}, {32, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, datatype},

        // YOLO26x depthwise shapes.
        TestCase{{1, 384, 80, 80}, {384, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 384, datatype},
        TestCase{{1, 768, 40, 40}, {768, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 768, datatype},
        TestCase{{1, 384, 20, 20}, {384, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 384, datatype},
        // Strided, unpadded, larger filter and odd-sized variants. The 33x17
        // case yields out_h = 17, a prime, so the row block cannot divide the
        // output height and the last row block runs short.
        TestCase{{2, 128, 33, 17}, {128, 1, 3, 3}, {1, 1}, {2, 2}, {1, 1}, 128, datatype},
        TestCase{{1, 96, 41, 41}, {96, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, 96, datatype},
        TestCase{{1, 64, 56, 56}, {64, 1, 5, 5}, {2, 2}, {1, 1}, {1, 1}, 64, datatype},
        // Dilated. Stride 1 is the atrous case, where no input row survives to
        // the next output row and the window is reloaded per row; stride 2 is
        // the case where rows are reused across the slide.
        TestCase{{1, 64, 28, 28}, {64, 1, 3, 3}, {2, 2}, {1, 1}, {2, 2}, 64, datatype},
        TestCase{{1, 64, 28, 28}, {64, 1, 3, 3}, {2, 2}, {2, 2}, {2, 2}, 64, datatype},
        // Asymmetric filter, padding, stride and dilation, so that swapping a
        // height constant for a width one cannot pass unnoticed.
        TestCase{{1, 64, 32, 32}, {64, 1, 5, 3}, {2, 1}, {1, 1}, {1, 1}, 64, datatype},
        TestCase{{1, 64, 32, 32}, {64, 1, 3, 3}, {1, 1}, {2, 1}, {1, 1}, 64, datatype},
        TestCase{{1, 64, 32, 32}, {64, 1, 3, 3}, {2, 1}, {1, 1}, {2, 1}, 64, datatype},
        // Largest register window the general bound admits: 7x7 needs 105 of the
        // 128 float slots kMaxWindowRegs allows.
        TestCase{{1, 64, 24, 24}, {64, 1, 7, 7}, {3, 3}, {1, 1}, {1, 1}, 64, datatype},
        // Undilated filters past kMaxWindowRegs: 9x9 at stride 1, and an
        // asymmetric 9x8 at the horizontal stride bound of 4.
        TestCase{{1, 64, 24, 24}, {64, 1, 9, 9}, {4, 4}, {1, 1}, {1, 1}, 64, datatype},
        TestCase{{1, 64, 29, 45}, {64, 1, 9, 8}, {4, 3}, {2, 4}, {1, 1}, 64, datatype},
        // 1xK filters on H = 1, the depthwise conv1d layout: K = 31 at dilation 1,
        // and a dilated width of 31 from K = 16 at dilation 2 and stride 2.
        TestCase{{1, 64, 1, 1500}, {64, 1, 1, 31}, {0, 15}, {1, 1}, {1, 1}, 64, datatype},
        TestCase{{2, 64, 1, 777}, {64, 1, 1, 16}, {0, 15}, {1, 2}, {1, 2}, 64, datatype},
        // A single output pixel, over an input narrower than one lane's column
        // span, so every row load takes the bounds-checked path.
        TestCase{{1, 64, 3, 3}, {64, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, 64, datatype},
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

// fp32 depthwise problem with 64 channels, for the applicability bounds test.
bool IsApplicableFor(miopen::ExecutionContext& ctx,
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
                            miopenFloat}
                       .GetProblemDescription(miopen::conv::Direction::Forward);
    problem.SetupFloats(ctx);
    return miopen::solver::conv::ConvDirectDepthwiseFwd2D{}.IsApplicable(ctx, problem);
}

} // namespace

using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP32  = GPU_UnitTestConvSolverFwd_FP32;

using CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP16, ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_BFP16, ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP32, ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

TEST_P(CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDevApplicabilityFwd_NONE,
       ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

// Both sides of each filter bound, on a mocked gfx1151 so the result does not depend
// on the host GPU. Most rejected shapes are ones the kernel would spill on.
TEST(CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DFilterBounds_NONE, ConvDirectDepthwiseFwd2D)
{
    auto gfx1151 = MockHandle{DevDescription{"gfx1151", 20, 32}, true};
    auto ctx     = miopen::ExecutionContext{&gfx1151};

    // General bound: extent 7 within kMaxWindowRegs. 7x7 at stride 2 and dilation 2
    // needs 244 window slots.
    EXPECT_TRUE(IsApplicableFor(ctx, {1, 64, 48, 48}, {64, 1, 7, 7}, {3, 3}, {1, 1}, {1, 1}));
    EXPECT_FALSE(IsApplicableFor(ctx, {1, 64, 48, 48}, {64, 1, 7, 7}, {6, 6}, {2, 2}, {2, 2}));

    // Undilated up to 9x9 at a horizontal stride of at most 4.
    EXPECT_TRUE(IsApplicableFor(ctx, {1, 64, 48, 48}, {64, 1, 9, 9}, {4, 4}, {1, 1}, {1, 1}));
    EXPECT_TRUE(IsApplicableFor(ctx, {1, 64, 48, 48}, {64, 1, 9, 9}, {4, 4}, {1, 4}, {1, 1}));
    EXPECT_FALSE(IsApplicableFor(ctx, {1, 64, 48, 48}, {64, 1, 9, 9}, {4, 4}, {1, 5}, {1, 1}));
    EXPECT_FALSE(IsApplicableFor(ctx, {1, 64, 48, 48}, {64, 1, 9, 9}, {8, 8}, {1, 1}, {2, 2}));
    EXPECT_FALSE(IsApplicableFor(ctx, {1, 64, 48, 48}, {64, 1, 11, 11}, {5, 5}, {1, 1}, {1, 1}));

    // 1xK up to a dilated width of 31.
    EXPECT_TRUE(IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 1, 1, 31}, {0, 15}, {1, 1}, {1, 1}));
    EXPECT_TRUE(IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 1, 1, 16}, {0, 15}, {1, 1}, {1, 2}));
    EXPECT_TRUE(IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 1, 1, 11}, {0, 15}, {1, 1}, {1, 3}));
    EXPECT_FALSE(IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 1, 1, 33}, {0, 16}, {1, 1}, {1, 1}));
    EXPECT_FALSE(IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 1, 1, 17}, {0, 16}, {1, 1}, {1, 2}));
    EXPECT_FALSE(IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 1, 1, 31}, {0, 45}, {1, 1}, {1, 3}));
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));
