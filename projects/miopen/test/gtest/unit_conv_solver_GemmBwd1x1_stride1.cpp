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
        // A 1x1 input is also point-output, so GemmBwdRest claims it as well.
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

} // namespace

using GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_FP16  = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_FP32  = GPU_UnitTestConvSolverBwd_FP32;

using CPU_UnitTestConvSolverGemmBwd1x1_Stride1DevApplicabilityBwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_FP16, GemmBwd1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmBwd1x1_stride1{});
};

TEST_P(GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_BFP16, GemmBwd1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmBwd1x1_stride1{});
};

TEST_P(GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_FP32, GemmBwd1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmBwd1x1_stride1{});
};

TEST_P(CPU_UnitTestConvSolverGemmBwd1x1_Stride1DevApplicabilityBwd_NONE, GemmBwd1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmBwd1x1_stride1{});
};

// Small-batch 2D NHWC backward-data excludes the GEMM family on gfx942. Uses mocked devices so
// the result is deterministic on any host GPU.
TEST(CPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_NONE, Gfx942SmallBatchNhwcExcluded)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto gfx942   = MockHandle{DevDescription{"gfx942", 304, 64}, true};
    auto other    = MockHandle{DevDescription{"gfx90a", 104, 64}, true};
    auto ctx942   = miopen::ExecutionContext{&gfx942};
    auto ctxother = miopen::ExecutionContext{&other};

    const auto make = [](miopen::ExecutionContext& ctx, std::size_t n, int stride, std::size_t hw) {
        auto p = TestCase{{n, 8, hw, hw},
                          {8, 8, 1, 1},
                          {0, 0},
                          {stride, stride},
                          {1, 1},
                          miopenFloat,
                          miopenTensorNHWC}
                     .GetProblemDescription(miopen::conv::Direction::BackwardData);
        p.SetupFloats(ctx);
        p.SetupComputeType(ctx);
        return p;
    };
    const auto s1_small_942 = make(ctx942, 2, 1, 8);   // 1x1 stride-1, within the 1..3 exclusion
    const auto s1_small_oth = make(ctxother, 2, 1, 8); // same shape, other arch
    const auto s1_large_942 = make(ctx942, 4, 1, 8);   // above the exclusion
    const auto s2_small_942 = make(ctx942, 2, 2, 8);   // 1x1 stride-2 downsample: NOT excluded
    const auto po_small_942 = make(ctx942, 2, 1, 1);   // point-output: handled by GemmBwdRest

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

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Channel-last smoke tests
INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesNhwc(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesNhwc(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         GPU_UnitTestConvSolverGemmBwd1x1_Stride1Bwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesNhwc(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverGemmBwd1x1_Stride1DevApplicabilityBwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         CPU_UnitTestConvSolverGemmBwd1x1_Stride1DevApplicabilityBwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCasesNhwc(miopenFloat)[0])));
