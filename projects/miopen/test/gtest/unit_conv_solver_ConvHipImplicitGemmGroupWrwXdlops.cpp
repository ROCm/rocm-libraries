// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "unit_conv_solver_group_xdlops.hpp"
#include <gtest/group_conv.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

#include <algorithm>
#include <string>

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
        TestCase{{1, 64, 8, 8}, {96, 64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, false, tf32_compute}
        // clang-format on
    };
}

template <TestDataType type>
std::vector<TestCase> GetConvFullTestCases()
{
    const bool tf32_compute = type == TestDataType::TF32;

    return {
        // clang-format off
        TestCase{{1, 64, 8, 8}, {96, 64, 1, 1}, {1, 1}, {1, 1}, {1, 1}, 1, false, tf32_compute}, // non-zero padding
        TestCase{{1, 64, 8, 8}, {96, 64, 1, 1}, {0, 0}, {2, 2}, {1, 1}, 1, false, tf32_compute}, // stride > 1

        // Group count = 2 and 4
        TestCase{{1, 64, 8, 8}, {96, 32, 1, 1}, {0, 0}, {1, 1}, {2, 2}, 2, false, tf32_compute}, // dilation > 1
        TestCase{{1, 64, 8, 8}, {96, 16, 1, 1}, {0, 0}, {2, 2}, {1, 1}, 4, false, tf32_compute}, // stride > 1
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
    return p;
}

} // namespace

// Solver itself supports I8 in isApplicable, but CK returns 0 compatible kernels

using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenHalf>;

using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenBFloat16>;

using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenFloat>;

using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_TF32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenFloat>;

using CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;
using CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_TF32, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDevApplicability_FP16,
       ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDeterministicApplicability_NONE,
       ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST(GPU_GroupWrwRankedSelection_FP16, ValidatesSplitOneAndPreservesFallback)
{
    miopen::Handle handle;
    miopen::ExecutionContext ctx(&handle);
    const auto& arch = handle.GetDeviceName();
    std::string type;
    if(arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0)
    {
        type = "DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmMultipleD_Wmma_"
               "CShuffleV3<MNKPadding, CRR> BlkSize: 256, BlkTile: 128x32x128, WaveTile: "
               "16x16, WaveMap: 1x2, VmemReadVec: 1x1, BlkGemmPipelineScheduler: Intrawave, "
               "BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>";
    }
    else if(arch.rfind("gfx9", 0) == 0)
    {
        type = "DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<"
               "MNKPadding, CRR> BlkSize: 128, BlkTile: 16x32x64, WaveTile: 16x16, WaveMap: "
               "1x1, VmemReadVec: 1x4, BlkGemmPipelineScheduler: Intrawave, "
               "BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>";
    }
    else
        GTEST_SKIP() << "No grouped WRW ranking for this architecture";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(arch);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "Grouped-convolution CK plugin is unavailable";

    // These ranked kernels have non-unit tuning hints. Defaults still validate split one.

    group_conv::GroupConvTestConfig<2u> conv{1, 1, 64, 96, {8, 8}, {1, 1}, {0, 0}, {1, 1}, {1, 1}};
    const auto x_desc = miopen::TensorDescriptor(miopenHalf, miopenTensorNHWC, conv.GetInput());
    const auto w_desc = miopen::TensorDescriptor(miopenHalf, miopenTensorNHWC, conv.GetWeights());
    auto conv_desc    = conv.GetConv();
    const auto y_desc = conv_desc.GetForwardOutputTensor(x_desc, w_desc, miopenHalf);
    const auto make_problem = [&] {
        return miopen::conv::ProblemDescription(
            y_desc, w_desc, x_desc, conv_desc, miopen::conv::Direction::BackwardWeights);
    };
    auto problem = make_problem();

    const auto valid_kernels = loader.FillValidKernels(
        miopen::solver::CKSolverType::GrpConvWrw, problem, miopenHalf, false);
    if(std::find(valid_kernels.begin(), valid_kernels.end(), type) == valid_kernels.end())
        GTEST_SKIP() << "Ranked CK instance is not present for this problem";

    using Config = miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupWrwXdlops;
    Config config;
    config.valid_kernels  = {type};
    config.split_k        = 7;
    config.kernel_id      = type + "+7";
    const auto default_id = type + "+1";
    const bool supported  = loader.IsArgsSupported(
        miopen::solver::CKSolverType::GrpConvWrw, problem, default_id, miopenHalf, false);
    config.DefaultKernelFromList(ctx, problem);
    EXPECT_EQ(config.split_k, supported ? 1 : 7);
    EXPECT_EQ(config.kernel_id, supported ? default_id : type + "+7");

    // Deterministic problems also validate split one without adopting a ranked tuning hint.
    conv_desc.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);
    problem = make_problem();
    config.split_k   = 1;
    config.kernel_id = default_id;
    config.DefaultKernelFromList(ctx, problem);
    EXPECT_EQ(config.split_k, 1);
    EXPECT_EQ(config.kernel_id, type + "+1");

    // A stale DB/type name is not an alias for any current ranked kernel.
    Config invalid;
    invalid.valid_kernels = {"obsolete_group_wrw_kernel"};
    invalid.kernel_id     = "obsolete_group_wrw_kernel+1";
    invalid.DefaultKernelFromList(ctx, problem);
    EXPECT_EQ(invalid.index, 0);
    EXPECT_EQ(invalid.split_k, 1);
    EXPECT_EQ(invalid.kernel_id, "obsolete_group_wrw_kernel+1");
    EXPECT_FALSE(loader.IsArgsSupported(
        miopen::solver::CKSolverType::GrpConvWrw, problem, invalid.kernel_id, miopenHalf, false));
}

// Smoke tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::TF32>())));

// Full tests

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP32>())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::TF32>())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                                          testing::Values(GetDevApplicabilityConvCase())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDeterministicApplicability_NONE,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(GetDeterministicConvCase())));
