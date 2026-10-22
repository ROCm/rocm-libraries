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
//     they require a ROCm device (GTEST_SKIP otherwise) and a canonical group=1, C=K=64, 3x3 shape.
//     Each test skips where the solver is inapplicable (CK ships no instances for the dtype/arch,
//     e.g. FP32 grouped bwd-weight on gfx11/gfx12) and otherwise asserts the search starts at
//     valid_kernels[0]. The 3D Fwd/Bwd equivalents live in
//     perf_config_HipImplicitGemm3DGroup*Xdlops.cpp.
//   * CPU_* : host-only advance walks for the non-split solvers (2D Fwd, 3D Fwd/Bwd). With
//     valid_kernels injected directly, SetNextValue() bypasses the CK populate branch and only the
//     advance path runs -- no device or CK library is needed and these run in the CPU-only stage.

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

// 3D variant (distinct name, not an overload: a 2-element nested brace can also aggregate-init the
// 3-field 3D size structs, which would make overload resolution ambiguous). Only used by the
// host-only advance tests, whose SetNextValue() path ignores the problem; it just needs to be a
// well-formed ProblemDescription for the requested direction.
Problem MakeSearchProblem3D(group_conv::GroupConvTestConfig<3u> conv,
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

// Skip when the solver is inapplicable on this device (CK ships no instances for the dtype/arch);
// where it is applicable the search must enumerate, so callers still assert SetNextValue().
template <class Solver>
bool SolverApplicable(const Problem& problem)
{
    auto& handle = get_handle();
    auto ctx     = miopen::ExecutionContext{};
    ctx.SetStream(&handle);
    return Solver{}.IsApplicable(ctx, problem);
}

} // namespace

TEST(GPU_PerfConfig_HipImplicitGemmGroupFwdXdlops_FP16, SearchStartsAtKernelZero)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.empty())
        GTEST_SKIP() << "No ROCm-capable device is detected";

    const auto problem =
        MakeSearchProblem({1, 64, 64, 64, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
                          miopenHalf,
                          miopenTensorNHWC,
                          Direction::Forward);

    if(!SolverApplicable<miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops>(problem))
        GTEST_SKIP() << "solver not applicable for " << device_name;

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

    const auto problem =
        MakeSearchProblem({1, 64, 64, 64, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
                          miopenHalf,
                          miopenTensorNHWC,
                          Direction::BackwardData);

    if(!SolverApplicable<miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops>(problem))
        GTEST_SKIP() << "solver not applicable for " << device_name;

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

    const auto problem =
        MakeSearchProblem({1, 64, 64, 64, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
                          miopenFloat,
                          miopenTensorNHWC,
                          Direction::BackwardWeights);

    if(!SolverApplicable<miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops>(problem))
        GTEST_SKIP() << "solver not applicable for " << device_name;

    WrwConfig cfg(false);
    ASSERT_TRUE(cfg.SetNextValue(problem));
    ASSERT_FALSE(cfg.valid_kernels.empty());
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels.front() + "+1");
    EXPECT_TRUE(cfg.IsValid(problem));
}

// --- Host-only advance walk for the non-split solvers (2D Fwd, 3D Fwd, 3D Bwd). With valid_kernels
// pre-populated, SetNextValue() bypasses the CK populate branch and runs only the advance path: it
// must visit index 0..N-1 in order and then return false without stepping out of bounds. (The
// split-k Bwd/Wrw advance walk is covered in perf_config_group_deterministic_set_next_value.cpp.)

TEST(CPU_PerfConfigGroupFwdAdvance_NONE, WalksEveryKernelThenExhausts)
{
    const auto problem =
        MakeSearchProblem({1, 64, 64, 64, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
                          miopenHalf,
                          miopenTensorNHWC,
                          Direction::Forward);

    FwdConfig cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 0;
    cfg.kernel_id     = cfg.valid_kernels.front();
    ASSERT_EQ(cfg.index, 0);
    ASSERT_EQ(cfg.kernel_id, "A");

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 1);
    EXPECT_EQ(cfg.kernel_id, "B");

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 2);
    EXPECT_EQ(cfg.kernel_id, "C");

    EXPECT_FALSE(cfg.SetNextValue(problem)); // exhausted, last kernel included, no overrun
    EXPECT_EQ(cfg.index, 2);
}

TEST(CPU_PerfConfigGroupFwdAdvance_NONE, SingleKernelExhaustsWithoutReemit)
{
    const auto problem =
        MakeSearchProblem({1, 64, 64, 64, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
                          miopenHalf,
                          miopenTensorNHWC,
                          Direction::Forward);

    // One valid kernel: the seed (index 0) is the whole search, so SetNextValue must report
    // exhaustion immediately and must not re-emit or step past index 0.
    FwdConfig cfg(false);
    cfg.valid_kernels = {"A"};
    cfg.index         = 0;
    cfg.kernel_id     = cfg.valid_kernels.front();

    EXPECT_FALSE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.kernel_id, "A");
}

TEST(CPU_PerfConfig3DGroupFwdAdvance_NONE, WalksEveryKernelThenExhausts)
{
    const auto problem = MakeSearchProblem3D(
        {1, 64, 64, 64, {4, 28, 28}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        miopenHalf,
        miopenTensorNDHWC,
        Direction::Forward);

    Fwd3DConfig cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 0;
    cfg.kernel_id     = cfg.valid_kernels.front();
    ASSERT_EQ(cfg.index, 0);

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 1);
    EXPECT_EQ(cfg.kernel_id, "B");

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 2);
    EXPECT_EQ(cfg.kernel_id, "C");

    EXPECT_FALSE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 2);
}

TEST(CPU_PerfConfig3DGroupBwdAdvance_NONE, WalksEveryKernelThenExhausts)
{
    const auto problem = MakeSearchProblem3D(
        {1, 64, 64, 64, {4, 28, 28}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        miopenHalf,
        miopenTensorNDHWC,
        Direction::BackwardData);

    Bwd3DConfig cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 0;
    cfg.kernel_id     = cfg.valid_kernels.front();
    ASSERT_EQ(cfg.index, 0);

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 1);
    EXPECT_EQ(cfg.kernel_id, "B");

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 2);
    EXPECT_EQ(cfg.kernel_id, "C");

    EXPECT_FALSE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 2);
}
