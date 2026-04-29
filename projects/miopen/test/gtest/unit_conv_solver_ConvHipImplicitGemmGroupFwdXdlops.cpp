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
        TestCase{{1, 64, 8, 8}, {96, 64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, false, tf32_compute},
        // clang-format on
    };
}

template <TestDataType type>
std::vector<TestCase> GetConvFullTestCases()
{
    const bool tf32_compute = type == TestDataType::TF32;

    return {
        // clang-format off
        TestCase{{1, 32, 8, 8}, {48, 32, 1, 1}, {1, 1}, {1, 1}, {1, 1}, 1, false, tf32_compute}, // non-zero padding
        TestCase{{1, 32, 8, 8}, {48, 32, 1, 1}, {0, 0}, {2, 2}, {1, 1}, 1, false, tf32_compute}, // stride > 1
        TestCase{{1, 32, 8, 8}, {48, 32, 1, 1}, {0, 0}, {1, 1}, {2, 2}, 1, false, tf32_compute}, // dilation > 1
        TestCase{{1, 32, 24, 48}, {192, 32, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, false, tf32_compute},
        // Group count = 2 and 4
        TestCase{{1, 32, 8, 8}, {48, 16, 1, 1}, {0, 0}, {1, 1}, {2, 2}, 2, false, tf32_compute}, // dilation > 1
        TestCase{{1, 32, 24, 48}, {96, 8, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 4, false, tf32_compute},
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
        TestCase{{1, 64, 8, 8}, {96, 64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, true}
        // clang-format on
    };

    return GetConvTestForGroupXdlops<miopenHalf>(miopenTensorNHWC, std::move(test_case));
}

template <TestDataType type>
miopen::unit_tests::UnitTestConvSolverParams GetTestParams()
{
// If MIOpen is built without CK these tests will fail, skip them to avoid failing
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
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

using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_I8 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward, miopenInt8>;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward, miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward,
                                                      miopenFloat>;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_TF32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward,
                                                      miopenFloat>;
using CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;
using CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_I8, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP16, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_BFP16, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP32, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_TF32, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDevApplicability_FP16,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDeterministicApplicability_NONE,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_I8,
    testing::Combine(testing::Values(GetTestParams<TestDataType::I8>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::I8>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::TF32>())));

// Full tests
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_I8,
    testing::Combine(testing::Values(GetTestParams<TestDataType::I8>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::I8>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::TF32>())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                                          testing::Values(GetDevApplicabilityConvCase())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDeterministicApplicability_NONE,
    testing::Combine(testing::Values(Gpu::None), testing::Values(GetDeterministicConvCase())));
