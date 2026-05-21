// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "unit_conv_solver_group_xdlops.hpp"

namespace {

// numeric part of test case
using TestCase     = miopen::unit_tests::GroupXdlopsNumericData;
using TestDataType = miopen::unit_tests::TestDataType;

template <TestDataType type>
std::vector<TestCase> GetConvSmokeTestCases()
{
    const bool tf32_compute = type == TestDataType::TF32;
    return {
        // clang-format off
        TestCase{{64, 32, 28, 28, 28}, {32, 32, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, false, tf32_compute}
        // clang-format on
    };
}

template <TestDataType type>
std::vector<TestCase> GetConvFullTestCases()
{
    const bool tf32_compute = type == TestDataType::TF32;
    return {
        // clang-format off
        // Group Count 1
        TestCase{{6, 448, 3, 118, 182}, {896, 448, 1, 1, 1}, {0, 0, 0}, {1, 2, 2}, {1, 1, 1}, 1, false, tf32_compute},
        TestCase{{128, 3, 2, 14, 14}, {320, 3, 2, 14, 14}, {0, 0, 0}, {2, 14, 14}, {1, 1, 1}, 1, false, tf32_compute},

        // Group Count > 1  (2, 3, 5, 16)
        TestCase{{128, 32, 28, 28, 28}, {32, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 2, false, tf32_compute},
        TestCase{{48, 48, 28, 28, 28}, {48, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 3, false, tf32_compute},
        TestCase{{120, 60, 28, 28, 28}, {60, 12, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 5, false, tf32_compute},
        TestCase{{64, 32, 28, 28, 28}, {32, 2, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, 16, false, tf32_compute},
        // clang-format on  
    };
}

auto GetDevApplicabilityConvCase()
{
    // For device applicability checks
    return GetConvTestForGroupXdlops<miopenHalf>(miopenTensorNDHWC,
                                                 std::move(GetConvSmokeTestCases<TestDataType::FP16>()[0]));
}

// Deterministic test case (for CPU deterministic applicability test)
auto GetDeterministicConvCase()
{
    TestCase test_case = {
        // clang-format off
                TestCase{{64, 32, 28, 28, 28}, {32, 32, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, true}
        // clang-format on
    };

    return GetConvTestForGroupXdlops<miopenHalf>(miopenTensorNDHWC, std::move(test_case));
}

template <TestDataType type>
miopen::unit_tests::UnitTestConvSolverParams GetTestParams()
{
// CK dynamic-library tests are HIP-only; runtime plugin availability is checked by the harness.
#if MIOPEN_BACKEND_HIP
    Gpu supportedDevices;
    if constexpr(type == TestDataType::FP32)
    {
        supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
    }
    else if constexpr(type == TestDataType::TF32)
    {
        supportedDevices = Gpu::gfx94X | Gpu::gfx950;
    }
    else
    {
        supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950 | Gpu::gfx110X |
                           Gpu::gfx115X | Gpu::gfx120X;
    }
#else
    Gpu supportedDevices = Gpu::None;
#endif
    miopen::unit_tests::UnitTestConvSolverParams p(supportedDevices);
    p.ExcludeDevice("gfx1103");
    p.Tunable(5);
    p.UsesCKDynamicLib();

    // Increased tolerance factor to 2 because of the following errors observed :
    // Expected: (error) < (threshold), actual: 1.4733528696833642e-07 vs 1.1920928955078125e-07
    p.SetTolerance(supportedDevices, miopenFloat, 2.0f);
    return p;
}

} // namespace

// For I8 datatype we get "Empty code object path", so it requires additional
// investigation/debugging

using GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward, miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward,
                                                      miopenFloat>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_TF32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward,
                                                      miopenFloat>;

using CPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlopsDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;
using CPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlopsDeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_FP16, ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_BFP16,
       ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_FP32, ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_TF32, ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlopsDevApplicability_FP16,
       ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlopsDeterministicApplicability_NONE,
       ConvHipImplicitGemm3DGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::TF32>())));

// Full tests
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::TF32>())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                                          testing::Values(GetDevApplicabilityConvCase())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlopsDeterministicApplicability_NONE,
    testing::Combine(testing::Values(Gpu::None), testing::Values(GetDeterministicConvCase())));
