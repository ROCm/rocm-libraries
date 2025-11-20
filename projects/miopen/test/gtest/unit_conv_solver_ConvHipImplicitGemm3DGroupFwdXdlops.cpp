/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

auto GetConvSmokeTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                 {datatype, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                 datatype, {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}}},
        // clang-format on
    };
}
// for tf32
auto GetConvSmokeTestCases(const std::string& compute_type_str)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    if(compute_type_str != "TF32")
    {
        throw std::runtime_error("Unsupported compute_type in smoke tests");
    }

    return std::vector{
        // clang-format off
        TestCase{{miopenFloat, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                 {miopenFloat, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                 miopenFloat, {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1, false, true}},
        // clang-format on
    };
}

auto GetConvFullTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                 {datatype, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                 datatype, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}}, // non-zero padding
        TestCase{{datatype, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                 {datatype, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                 datatype, {{0, 0, 0}, {2, 2, 2}, {1, 1, 1}}}, // stride > 1
        TestCase{{datatype, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                 {datatype, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                 datatype, {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}}, // dilation > 1
        TestCase{{datatype, miopenTensorNDHWC, {1, 64, 12, 24, 48}},
                 {datatype, miopenTensorNDHWC, {384, 64, 1, 1, 1}},
                 datatype, {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}}}, // some different NCHW and k parameters
        // clang-format on
    };
}
// for tf32
auto GetConvFullTestCases(const std::string& compute_type_str)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    // using datatype = miopenDataType_t::miopenFloat;
    if(compute_type_str != "TF32")
    {
        throw std::runtime_error("Unsupported compute_type in smoke tests");
    }

    return std::vector{
        // clang-format off
        TestCase{{miopenFloat, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                 {miopenFloat, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                 miopenFloat, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, false, true}}, // non-zero padding
        TestCase{{miopenFloat, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                 {miopenFloat, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                 miopenFloat, {{0, 0, 0}, {2, 2, 2}, {1, 1, 1}, 1, false, true}}, // stride > 1
        TestCase{{miopenFloat, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                 {miopenFloat, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                 miopenFloat, {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, 1, false, true}}, // dilation > 1
        TestCase{{miopenFloat, miopenTensorNDHWC, {1, 64, 12, 24, 48}},
                 {miopenFloat, miopenTensorNDHWC, {384, 64, 1, 1, 1}},
                 miopenFloat, {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1, false, true}}, // some different NCHW and k parameters
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
// If MIOpen is built without CK these tests will fail, skip them to avoid failing
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        Gpu supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
#else
        Gpu supportedDevices = Gpu::None;
#endif
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
        p.Tunable(5);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_FP32  = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_TF32  = GPU_UnitTestConvSolverFwd_TF32;
using CPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlopsDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_FP16,
       ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_BFP16,
       ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_FP32,
       ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};
TEST_P(GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_TF32,
       ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};
TEST_P(CPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlopsDevApplicability_NONE,
       ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_TF32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvSmokeTestCases("TF32"))));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlops_TF32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases("TF32"))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipImplicitGemm3DGroupFwdXdlopsDevApplicability_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvSmokeTestCases(miopenHalf)[0])));
