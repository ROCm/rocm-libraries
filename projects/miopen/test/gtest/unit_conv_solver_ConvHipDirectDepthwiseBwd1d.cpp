// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

namespace {

// Backward data: x is dx (the long side), so dy is derived from x, the filter, padding and
// stride. Lengths are chosen so the last dx positions get no tap (the output_padding case of a
// ConvTranspose1d) and lane groups do not divide the output evenly.
auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // BigVGAN anti-aliased upsampling (kaiser filter): k = 12, stride 2, batch 2.
        TestCase{{2, 64, 1, 1034}, {64, 1, 1, 12}, {0, 0}, {1, 2}, {1, 1}, 64, datatype},
        // Same with padding and an odd dx length, so the last dx position gets no tap.
        TestCase{{1, 48, 1, 1001}, {48, 1, 1, 12}, {0, 5}, {1, 2}, {1, 1}, 48, datatype},
        // Hann resampler filters: k = 29 at stride 2, k = 43 at stride 3, few channels.
        TestCase{{1, 2, 1, 2077}, {2, 1, 1, 29}, {0, 0}, {1, 2}, {1, 1}, 2, datatype},
        TestCase{{1, 32, 1, 3001}, {32, 1, 1, 43}, {0, 0}, {1, 3}, {1, 1}, 32, datatype},
        // Stride 1: the input gradient of a depthwise conv1d (k = 7, same padding).
        TestCase{{2, 128, 1, 777}, {128, 1, 1, 7}, {0, 3}, {1, 1}, {1, 1}, 128, datatype},
        // Filter width and stride at the bounds.
        TestCase{{1, 32, 1, 4100}, {32, 1, 1, 64}, {0, 28}, {1, 8}, {1, 1}, 32, datatype},
        // Stride wider than the filter: dx positions between taps stay zero.
        TestCase{{1, 32, 1, 401}, {32, 1, 1, 3}, {0, 1}, {1, 4}, {1, 1}, 32, datatype},
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

// fp32 depthwise conv1d problem for the applicability test.
bool IsApplicableFor(miopen::ExecutionContext& ctx,
                     std::vector<size_t>&& x,
                     std::vector<size_t>&& w,
                     std::vector<int>&& pad,
                     std::vector<int>&& stride,
                     std::vector<int>&& dilation,
                     int groups)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    auto problem   = TestCase{std::move(x),
                            std::move(w),
                            std::move(pad),
                            std::move(stride),
                            std::move(dilation),
                            groups,
                            miopenFloat}
                       .GetProblemDescription(miopen::conv::Direction::BackwardData);
    problem.SetupFloats(ctx);
    return miopen::solver::conv::ConvHipDirectDepthwiseBwd1d{}.IsApplicable(ctx, problem);
}

} // namespace

using GPU_UnitTestConvSolverHipDirectDepthwiseBwd1d_FP32  = GPU_UnitTestConvSolverBwd_FP32;
using GPU_UnitTestConvSolverHipDirectDepthwiseBwd1d_FP16  = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverHipDirectDepthwiseBwd1d_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;

using CPU_UnitTestConvSolverHipDirectDepthwiseBwd1dDevApplicabilityBwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipDirectDepthwiseBwd1d_FP32, ConvHipDirectDepthwiseBwd1d)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectDepthwiseBwd1d{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectDepthwiseBwd1d_FP16, ConvHipDirectDepthwiseBwd1d)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectDepthwiseBwd1d{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectDepthwiseBwd1d_BFP16, ConvHipDirectDepthwiseBwd1d)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectDepthwiseBwd1d{});
};

TEST_P(CPU_UnitTestConvSolverHipDirectDepthwiseBwd1dDevApplicabilityBwd_NONE,
       ConvHipDirectDepthwiseBwd1d)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectDepthwiseBwd1d{});
};

// Both sides of each bound, on mocked devices so the result does not depend on the host GPU.
TEST(CPU_UnitTestConvSolverHipDirectDepthwiseBwd1dBounds_NONE, ConvHipDirectDepthwiseBwd1d)
{
    auto gfx1151 = MockHandle{DevDescription{"gfx1151", 20, 32}, true};
    auto gfx1030 = MockHandle{DevDescription{"gfx1030", 40, 32}, true};
    auto gfx942  = MockHandle{DevDescription{"gfx942", 304, 64}, true};
    auto ctx     = miopen::ExecutionContext{&gfx1151};
    auto ctx1030 = miopen::ExecutionContext{&gfx1030};
    auto ctx942  = miopen::ExecutionContext{&gfx942};

    // Filter width up to 64 and stride up to 8.
    EXPECT_TRUE(IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 1, 1, 64}, {0, 0}, {1, 8}, {1, 1}, 64));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 1, 1, 65}, {0, 0}, {1, 8}, {1, 1}, 64));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 1, 1, 12}, {0, 0}, {1, 9}, {1, 1}, 64));

    // Dilation 1 only.
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 1, 1, 12}, {0, 0}, {1, 2}, {1, 2}, 64));

    // 1D only: H == 1 and a 1xK filter.
    EXPECT_FALSE(IsApplicableFor(ctx, {1, 64, 8, 512}, {64, 1, 1, 12}, {0, 0}, {1, 2}, {1, 1}, 64));
    EXPECT_FALSE(IsApplicableFor(ctx, {1, 64, 8, 512}, {64, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 64));

    // Depthwise with one channel per group only.
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {64, 64, 1, 12}, {0, 0}, {1, 2}, {1, 1}, 1));
    EXPECT_FALSE(
        IsApplicableFor(ctx, {1, 64, 1, 4096}, {128, 1, 1, 12}, {0, 0}, {1, 2}, {1, 1}, 64));

    // RDNA only, including gfx103X.
    EXPECT_TRUE(
        IsApplicableFor(ctx1030, {1, 64, 1, 4096}, {64, 1, 1, 12}, {0, 0}, {1, 2}, {1, 1}, 64));
    EXPECT_FALSE(
        IsApplicableFor(ctx942, {1, 64, 1, 4096}, {64, 1, 1, 12}, {0, 0}, {1, 2}, {1, 1}, 64));
}

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectDepthwiseBwd1d_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectDepthwiseBwd1d_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectDepthwiseBwd1d_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipDirectDepthwiseBwd1dDevApplicabilityBwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));
