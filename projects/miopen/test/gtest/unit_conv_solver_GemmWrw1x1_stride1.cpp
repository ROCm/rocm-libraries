/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "unit_conv_solver.hpp"

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{1, 8, 8, 8}, {8, 8, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

// Channel-last pointwise shapes, which take the single unbatched GEMM path.
auto GetConvTestCasesNhwc(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNHWC, {4, 16, 14, 14}},
                 {datatype, miopenTensorNHWC, {32, 16, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}}},
        TestCase{{datatype, miopenTensorNDHWC, {4, 16, 4, 4, 4}},
                 {datatype, miopenTensorNDHWC, {32, 16, 1, 1, 1}},
                 datatype, {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}}},
        // A 1x1 input is also point-output, so GemmWrwUniversal claims it as well.
        TestCase{{datatype, miopenTensorNHWC, {4, 16, 1, 1}},
                 {datatype, miopenTensorNHWC, {32, 16, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}}},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        p.SetTolerance(Gpu::gfx90A, miopenHalf, 2.0f);
        return p;
    }();
    return params;
}

const auto& GetTestParamsNhwc()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        p.SetTolerance(Gpu::gfx90A, miopenHalf, 2.0f);
        // On the 14x14 fp32 case, which reduces over k = 784, gfx90a and gfx1151 measure
        // 1.17 eps against the 1.0 eps default, while gfx950 measures 0.55 to 0.86 over 20
        // seeds.
        p.SetTolerance(Gpu::All, miopenFloat, 2.0f);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP16  = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;
using GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP32  = GPU_UnitTestConvSolverWrw_FP32;
using CPU_UnitTestConvSolverGemmWrw1x1Stride1DevApplicabilityWrw_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP16, GemmWrw1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmWrw1x1_stride1{});
};

TEST_P(GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_BFP16, GemmWrw1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmWrw1x1_stride1{});
};

TEST_P(GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP32, GemmWrw1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmWrw1x1_stride1{});
};

TEST_P(CPU_UnitTestConvSolverGemmWrw1x1Stride1DevApplicabilityWrw_NONE, GemmWrw1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmWrw1x1_stride1{});
};

TEST(CPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_NONE, NHWCUsesStride1Solver)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    // Batch 4 stays above the gfx942 small-batch exclusion (see Gfx942SmallBatchNhwcExcluded).
    const auto test_case =
        TestCase{{4, 8, 8, 8}, {8, 8, 1, 1}, {0, 0}, {1, 1}, {1, 1}, miopenFloat, miopenTensorNHWC};
    const auto problem = test_case.GetProblemDescription(miopen::conv::Direction::BackwardWeights);
    auto context       = miopen::ExecutionContext{&get_handle()};
    problem.SetupFloats(context);
    problem.SetupComputeType(context);

    EXPECT_TRUE(miopen::solver::conv::GemmWrw1x1_stride1{}.IsApplicable(context, problem));
    EXPECT_FALSE(miopen::solver::conv::GemmWrwUniversal{}.IsApplicable(context, problem));
}

// Small-batch 2D NHWC wgrad excludes the GEMM family on gfx942. Uses mocked devices so the
// result is deterministic on any host GPU.
TEST(CPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_NONE, Gfx942SmallBatchNhwcExcluded)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto gfx942   = MockHandle{DevDescription{"gfx942", 304, 64}, true};
    auto other    = MockHandle{DevDescription{"gfx90a", 104, 64}, true};
    auto ctx942   = miopen::ExecutionContext{&gfx942};
    auto ctxother = miopen::ExecutionContext{&other};

    // hw=8 -> 1x1-filter path (GemmWrw1x1_stride1); hw=1 -> point-output path (GemmWrwUniversal).
    const auto make = [](miopen::ExecutionContext& ctx, std::size_t n, int stride, std::size_t hw) {
        auto p = TestCase{{n, 8, hw, hw},
                          {8, 8, 1, 1},
                          {0, 0},
                          {stride, stride},
                          {1, 1},
                          miopenFloat,
                          miopenTensorNHWC}
                     .GetProblemDescription(miopen::conv::Direction::BackwardWeights);
        p.SetupFloats(ctx);
        p.SetupComputeType(ctx);
        return p;
    };
    const auto s1_small_942 = make(ctx942, 2, 1, 8);   // 1x1 stride-1, within the 1..3 exclusion
    const auto s1_small_oth = make(ctxother, 2, 1, 8); // same shape, other arch
    const auto s1_large_942 = make(ctx942, 4, 1, 8);   // above the exclusion
    const auto s2_small_942 = make(ctx942, 2, 2, 8);   // 1x1 stride-2 downsample: NOT excluded
    const auto po_small_942 = make(ctx942, 2, 1, 1);   // point-output: handled by GemmWrwUniversal

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

TEST(CPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_NONE, NDHWCUsesStride1Solver)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    const auto test_case = TestCase{{1, 8, 4, 5, 6},
                                    {8, 8, 1, 1, 1},
                                    {0, 0, 0},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    miopenFloat,
                                    miopenTensorNDHWC};
    const auto problem = test_case.GetProblemDescription(miopen::conv::Direction::BackwardWeights);
    auto context       = miopen::ExecutionContext{&get_handle()};
    problem.SetupFloats(context);
    problem.SetupComputeType(context);

    EXPECT_TRUE(miopen::solver::conv::GemmWrw1x1_stride1{}.IsApplicable(context, problem));
    EXPECT_FALSE(miopen::solver::conv::GemmWrwUniversal{}.IsApplicable(context, problem));
}

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Channel-last smoke tests
INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParamsNhwc()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesNhwc(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_BFP16,
                         testing::Combine(testing::Values(GetTestParamsNhwc()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesNhwc(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP32,
                         testing::Combine(testing::Values(GetTestParamsNhwc()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesNhwc(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverGemmWrw1x1Stride1DevApplicabilityWrw_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         CPU_UnitTestConvSolverGemmWrw1x1Stride1DevApplicabilityWrw_NONE,
                         testing::Combine(testing::Values(GetTestParamsNhwc()),
                                          testing::Values(GetConvTestCasesNhwc(miopenFloat)[0])));
