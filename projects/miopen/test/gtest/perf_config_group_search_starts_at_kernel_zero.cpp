// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Regression tests for the valid_kernels[0] ("first kernel") tuning skip in the 2D grouped CK
// xdlops solvers (Fwd / Bwd / Wrw). Before the fix, the empty-valid_kernels branch of
// SetNextValue() returned without seeding kernel_id, so GenericSearch's ComputedIterator skipped
// the still-unset first config and enumeration began at valid_kernels[1] -- valid_kernels[0] was
// never tuned even though it is applicable (IsValid() == true). After the fix the first
// enumerated config is valid_kernels[0], with split_k = 1 for the split-k Bwd/Wrw solvers. The
// 3D Fwd/Bwd equivalents live in their own perf_config_HipImplicitGemm3DGroup*Xdlops.cpp files.
//
// These exercise the real CK lazy-init path inside SetNextValue(), so they require a ROCm device
// and the installed CK grouped-conv library; both are checked and the test GTEST_SKIPs otherwise.

#include <gtest/gtest.h>
#include <gtest/group_conv.hpp>

#include <miopen/tensor.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

#include <string>

namespace {

using Problem   = miopen::conv::ProblemDescription;
using Direction = miopen::conv::Direction;

using FwdConfig = miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupFwdXdlops;
using BwdConfig = miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupBwdXdlops;
using WrwConfig = miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupWrwXdlops;

// Builds a 2D grouped-conv ProblemDescription for the requested direction. input == x,
// output == y; the backward directions pass tensors in (output, weights, input) order, matching
// the RunSolver fixture in group_conv.hpp.
Problem MakeSearchProblem(group_conv::GroupConvTestConfig<2u> conv,
                          miopenDataType_t data_type,
                          miopenTensorLayout_t layout,
                          Direction direction)
{
    const auto x_desc    = miopen::TensorDescriptor(data_type, layout, conv.GetInput());
    const auto w_desc    = miopen::TensorDescriptor(data_type, layout, conv.GetWeights());
    const auto conv_desc = conv.GetConv();
    const auto y_desc    = conv_desc.GetForwardOutputTensor(x_desc, w_desc, data_type);

    if(direction == Direction::Forward)
        return Problem(x_desc, w_desc, y_desc, conv_desc, direction);
    return Problem(y_desc, w_desc, x_desc, conv_desc, direction);
}

} // namespace

TEST(GPU_PerfConfig_HipImplicitGemmGroupFwdXdlops_FP16, SearchStartsAtKernelZero)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.empty())
        GTEST_SKIP() << "No ROCm-capable device is detected";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem =
        MakeSearchProblem({1, 256, 192, 192, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
                          miopenHalf,
                          miopenTensorNHWC,
                          Direction::Forward);

    FwdConfig cfg(false);
    ASSERT_TRUE(cfg.SetNextValue(problem));
    ASSERT_FALSE(cfg.valid_kernels.empty());
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels.front());
    EXPECT_TRUE(cfg.IsValid(problem));
}

TEST(GPU_PerfConfig_HipImplicitGemmGroupBwdXdlops_FP16, SearchStartsAtKernelZero)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.empty())
        GTEST_SKIP() << "No ROCm-capable device is detected";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem =
        MakeSearchProblem({8, 256, 192, 192, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
                          miopenHalf,
                          miopenTensorNHWC,
                          Direction::BackwardData);

    BwdConfig cfg(false);
    ASSERT_TRUE(cfg.SetNextValue(problem));
    ASSERT_FALSE(cfg.valid_kernels.empty());
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels.front() + "+1");
    EXPECT_TRUE(cfg.IsValid(problem));
}

TEST(GPU_PerfConfig_HipImplicitGemmGroupWrwXdlops_FP32, SearchStartsAtKernelZero)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.empty())
        GTEST_SKIP() << "No ROCm-capable device is detected";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem =
        MakeSearchProblem({8, 256, 192, 192, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
                          miopenFloat,
                          miopenTensorNHWC,
                          Direction::BackwardWeights);

    WrwConfig cfg(false);
    ASSERT_TRUE(cfg.SetNextValue(problem));
    ASSERT_FALSE(cfg.valid_kernels.empty());
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels.front() + "+1");
    EXPECT_TRUE(cfg.IsValid(problem));
}
