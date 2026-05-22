// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "unit_conv_solver_group_xdlops.hpp"

namespace {

using TestCase     = miopen::unit_tests::GroupXdlopsNumericData;
using TestDataType = miopen::unit_tests::TestDataType;

// Non-deterministic test cases (for GPU smoke tests)
template <TestDataType type>
std::vector<TestCase> GetConvSmokeTestCases()
{
    const bool tf32_compute = type == TestDataType::TF32;

    return {
        // clang-format off
        TestCase{{1, 32, 8, 8}, {32, 32, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, false, tf32_compute}
        // clang-format on
    };
}

template <TestDataType type>
std::vector<TestCase> GetConvFullTestCases()
{
    const bool tf32_compute = type == TestDataType::TF32;

    return {
        // clang-format off
        TestCase{{1, 32, 8, 8}, {32, 32, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false, tf32_compute}, // non-zero padding
        TestCase{{1, 64, 24, 48}, {96, 64, 1, 1}, {0, 0}, {2, 2}, {1, 1}, 1, false, tf32_compute}, // stride > 1
        TestCase{{1, 32, 8, 8}, {32, 32, 3, 3}, {0, 0}, {1, 1}, {3, 3}, 1, false, tf32_compute}, // dilation > 1
        TestCase{{1, 64, 24, 48}, {96, 64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, false, tf32_compute},
        // Group count = 2 and 4
        TestCase{{1, 32, 8, 8}, {32, 16, 3, 3}, {0, 0}, {1, 1}, {3, 3}, 2, false, tf32_compute}, // dilation > 1
        TestCase{{1, 64, 24, 48}, {96, 16, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 4, false, tf32_compute},
        // clang-format on
    };
}

auto GetDevApplicabilityConvCase()
{
    // For device applicability checks
    return GetConvTestForGroupXdlops<miopenHalf>(
        miopenTensorNHWC, std::move(GetConvSmokeTestCases<TestDataType::FP16>()[0]));
}

// Deterministic test case (for CPU deterministic applicability test)
auto GetDeterministicConvCase()
{
    TestCase test_case = {
        // clang-format off
        TestCase{{1, 32, 8, 8}, {32, 32, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, true},
        // clang-format on
    };

    return GetConvTestForGroupXdlops<miopenHalf>(miopenTensorNHWC, std::move(test_case));
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
    return p;
}

} // namespace

using GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData,
                                                      miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData,
                                                      miopenFloat>;
using GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_TF32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData,
                                                      miopenFloat>;
using CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;
using CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP16, ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_BFP16, ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP32, ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_TF32, ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDevApplicability_FP16,
       ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDeterministicApplicability_NONE,
       ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::TF32>())));

// Full tests
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::TF32>())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                                          testing::Values(GetDevApplicabilityConvCase())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDeterministicApplicability_NONE,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(GetDeterministicConvCase())));
