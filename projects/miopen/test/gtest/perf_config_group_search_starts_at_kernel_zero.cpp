// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Regression tests for the valid_kernels[0] ("first kernel") tuning skip in the grouped CK xdlops
// solvers. Before the fix, the empty-valid_kernels branch of SetNextValue() returned without
// seeding kernel_id, so GenericSearch's ComputedIterator skipped the still-unset first config and
// enumeration began at valid_kernels[1] -- valid_kernels[0] was never tuned even though it is
// applicable (IsValid() == true). After the fix the first enumerated config is valid_kernels[0],
// with split_k = 1 for the split-k Bwd/Wrw solvers.
//
// Two layers of coverage:
//   * GPU_* : exercise the real CK lazy-init path end-to-end for the 2D Fwd/Bwd/Wrw solvers, so
//     they require a ROCm device and the installed CK grouped-conv library (GTEST_SKIP otherwise).
//     They use a canonical group=1, C=K=64, 3x3 shape that CK should always enumerate kernels for;
//     if SetNextValue() returns no kernels the ASSERT fails on purpose -- that signals a real CK /
//     applicability regression, not a bad test shape. The 3D Fwd/Bwd end-to-end equivalents live in
//     their own perf_config_HipImplicitGemm3DGroup*Xdlops.cpp files.
//   * CPU_* : host-only checks of SeedFirstValidKernel() -- the populate-branch seed extracted for
//     testability -- for all five fixed solvers. valid_kernels is injected directly, so no device
//     or CK library is needed and these run in the CPU-only test stage.

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

using FwdConfig   = miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupFwdXdlops;
using BwdConfig   = miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupBwdXdlops;
using WrwConfig   = miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupWrwXdlops;
using Fwd3DConfig = miopen::solver::conv::PerformanceConfigHipImplicitGemm3DGroupFwdXdlops;
using Bwd3DConfig = miopen::solver::conv::PerformanceConfigHipImplicitGemm3DGroupBwdXdlops;

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
        MakeSearchProblem({1, 64, 64, 64, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
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
        MakeSearchProblem({1, 64, 64, 64, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
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
        MakeSearchProblem({1, 64, 64, 64, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
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

// --- Host-only (no device / CK library): SeedFirstValidKernel() must make valid_kernels[0] the
// first enumerated config. valid_kernels is injected directly; index/split_k are pre-clobbered to
// prove the seed resets them. Non-split solvers seed kernel_id = valid_kernels[0]; split-k solvers
// seed split_k = 1 and kernel_id = valid_kernels[0] + "+1".

TEST(CPU_PerfConfigGroupFwdSeed_NONE, SeedsKernelZero)
{
    FwdConfig cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 7;
    cfg.SeedFirstValidKernel();
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.kernel_id, "A");
}

TEST(CPU_PerfConfigGroupBwdSeed_NONE, SeedsKernelZero)
{
    BwdConfig cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 7;
    cfg.split_k       = 64;
    cfg.SeedFirstValidKernel();
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, "A+1");
}

TEST(CPU_PerfConfigGroupWrwSeed_NONE, SeedsKernelZero)
{
    WrwConfig cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 7;
    cfg.split_k       = 64;
    cfg.SeedFirstValidKernel();
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, "A+1");
}

TEST(CPU_PerfConfig3DGroupFwdSeed_NONE, SeedsKernelZero)
{
    Fwd3DConfig cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 7;
    cfg.SeedFirstValidKernel();
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.kernel_id, "A");
}

TEST(CPU_PerfConfig3DGroupBwdSeed_NONE, SeedsKernelZero)
{
    Bwd3DConfig cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 7;
    cfg.SeedFirstValidKernel();
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.kernel_id, "A");
}
