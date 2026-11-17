// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>

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
        TestCase {{1, 4, 8, 28, 28}, {4, 4, 3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1, false, tf32_compute}
        // clang-format on
    };
}

template <TestDataType type>
std::vector<TestCase> GetConvFullTestCases()
{
    const bool tf32_compute = type == TestDataType::TF32;
    return {
        // clang-format off
        // Group Count = 1
        TestCase {{1, 1, 8, 8, 8}, {1, 1, 2, 2, 2}, {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, 1, false, tf32_compute},
        TestCase {{6, 448, 3, 118, 182}, {896, 448, 1, 1, 1}, {0, 0, 0}, {1, 2, 2}, {1, 1, 1}, 1, false, tf32_compute},

        // Group Count > 1 (2, 3, 4)
        TestCase {{128, 2, 28, 28, 28}, {2, 1, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, 2, false, tf32_compute},
        TestCase {{128, 2, 28, 28, 28}, {2, 1, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, 2, false, tf32_compute},
        TestCase {{256, 9, 2, 14, 14}, {27, 3, 2, 14, 14}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 3, false, tf32_compute},
        TestCase {{128, 4, 28, 28, 28}, {8, 1, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, 4, false, tf32_compute}
        // clang-format on
    };
}

auto GetDevApplicabilityConvCase()
{
    // For device applicability checks
    return GetConvTestForGroupXdlops<miopenHalf>(
        miopenTensorNDHWC, std::move(GetConvSmokeTestCases<TestDataType::FP16>()[0]));
}

// Deterministic test case (for CPU deterministic applicability test)
auto GetDeterministicConvCase()
{
    TestCase test_case = {
        // clang-format off
        TestCase {{1, 4, 8, 28, 28}, {4, 4, 3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1, true}
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
    else if constexpr(type == TestDataType::TF32 || type == TestDataType::BF16)
    {
        supportedDevices = Gpu::gfx94X | Gpu::gfx950;
    }
    else
    {
        supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950 | Gpu::gfx110X |
                           Gpu::gfx115X | Gpu::gfx120X | Gpu::gfx125X;
    }
#else
    Gpu supportedDevices = Gpu::None;
#endif
    miopen::unit_tests::UnitTestConvSolverParams p(supportedDevices);
    p.ExcludeDevice("gfx1103");
    p.Tunable(5);
    p.UsesCKDynamicLib();

    // The five-config tuning sample can select different FP32 accumulation orders.
    // Use the general FP32 bound here; the long-reduction shape is calibrated separately.
    if constexpr(type == TestDataType::FP32)
    {
        p.SetTolerance(supportedDevices, miopenFloat, 50.0f);
    }
    else
    {
        p.SetTolerance(supportedDevices, miopenFloat, 30.0f);
    }

    // Tolerance bump for FP16 on gfx110X and gfx115X due to observed precision differences.
    p.SetTolerance(Gpu::gfx110X | Gpu::gfx115X, miopenHalf, 4.0f);

    return p;
}

void ApplyFp32ShapeTolerance(miopen::unit_tests::UnitTestConvSolverParams& params,
                             const TestCase& test_case)
{
    constexpr std::array<std::size_t, 5> input_lengths{128, 2, 28, 28, 28};
    constexpr std::array<std::size_t, 5> weight_lengths{2, 1, 3, 3, 3};
    constexpr std::array<int, 3> padding{2, 2, 2};
    constexpr std::array<int, 3> unit{1, 1, 1};
    const auto matches = [](const auto& actual, const auto& expected) {
        return std::equal(actual.begin(), actual.end(), expected.begin(), expected.end());
    };

    if(test_case.group_count == 2 && !test_case.deterministic && !test_case.tf32_compute &&
       matches(test_case.x, input_lengths) && matches(test_case.w, weight_lengths) &&
       matches(test_case.pad, padding) && matches(test_case.stride, unit) &&
       matches(test_case.dilation, unit))
    {
        // Each weight reduces 128 * 28^3 valid FP32 products. Low-split-K
        // CK configurations reach 1.25e-5 normalized RMS error against the FP64 reference.
        // 128 * FP32 epsilon (~1.53e-5) budgets the measured accumulation roundoff for
        // this shape in both layouts, without widening any other case's validation.
        params.SetTolerance(params.supported_devs, miopenFloat, 128.0f);
    }
}

} // namespace

// For I8 datatype we get "Empty code object path", so it requires additional
// investigation/debugging

using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenFloat>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_TF32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenFloat>;

using CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;
using CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP16, ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_BFP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP32, ConvHipImplicitGemm3DGroupWrwXdlops)
{
    auto [params, layout, numeric_data] = GetParam();
    ApplyFp32ShapeTolerance(params, numeric_data);
    const auto conv_config =
        miopen::unit_tests::GetConvTestForGroupXdlops<miopenFloat>(layout, std::move(numeric_data));
    this->RunTestImpl(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{},
                      params,
                      miopen::conv::Direction::BackwardWeights,
                      conv_config,
                      miopenConvolutionAlgoImplicitGEMM);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_TF32, ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDevApplicability_FP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDeterministicApplicability_NONE,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::TF32>())));

// Full tests
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::TF32>())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                                          testing::Values(GetDevApplicabilityConvCase())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDeterministicApplicability_NONE,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(GetDeterministicConvCase())));
