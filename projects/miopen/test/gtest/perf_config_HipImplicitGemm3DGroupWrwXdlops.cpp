// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Tests for the two SetNextValue() enumeration bugs in the 3D grouped backward-weights (Wrw) CK
// xdlops solver:
//   1. the empty-valid_kernels populate branch fell through instead of emitting the seeded first
//      config, dropping (valid_kernels[0], split_k 1) from the search; and
//   2. deterministic mode used an inverted `!NextLinear` exit, so GetAllConfigs returned empty and
//      only the GenericSearch fallback config was ever tuned -- deterministic 3D-Wrw was untuned.
//
// GPU_* exercises the real CK populate path end-to-end, in both non-deterministic and
// deterministic mode, and needs a ROCm device plus the installed CK grouped-conv library
// (GTEST_SKIP otherwise). It uses a canonical group=1, C=K=64 shape that CK should always
// enumerate kernels for; a missing kernel fails the ASSERT on purpose, signalling a real
// CK/applicability regression rather than a bad test shape. CPU_* checks the deterministic
// and non-deterministic advance on the host by injecting valid_kernels directly -- no device
// needed (same technique as perf_config_group_deterministic_set_next_value.cpp).

#include <gtest/gtest.h>
#include <gtest/group_conv.hpp>

#include <miopen/tensor.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

#include <set>
#include <string>

namespace {

using Problem   = miopen::conv::ProblemDescription;
using Direction = miopen::conv::Direction;
using Config    = miopen::solver::conv::PerformanceConfigHipImplicitGemm3DGroupWrwXdlops;

// Builds a 3D grouped Wrw ProblemDescription. input == x, output == y; Wrw passes tensors in
// (output, weights, input) order, matching the RunSolver fixture in group_conv.hpp. The advance
// tests only read the deterministic attribute off the problem, not its geometry.
Problem MakeWrwProblem(group_conv::GroupConvTestConfig<3u> conv,
                       miopenDataType_t data_type,
                       miopenTensorLayout_t layout,
                       bool deterministic)
{
    const auto x_desc = miopen::TensorDescriptor(data_type, layout, conv.GetInput());
    const auto w_desc = miopen::TensorDescriptor(data_type, layout, conv.GetWeights());
    auto conv_desc    = conv.GetConv();
    if(deterministic)
        conv_desc.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);
    const auto y_desc = conv_desc.GetForwardOutputTensor(x_desc, w_desc, data_type);
    return Problem(y_desc, w_desc, x_desc, conv_desc, Direction::BackwardWeights);
}

group_conv::GroupConvTestConfig<3u> CanonicalShape()
{
    return {1, 64, 64, 64, {4, 28, 28}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
}

} // namespace

TEST(GPU_PerfConfig_HipImplicitGemm3DGroupWrwXdlops_FP32, SearchStartsAtKernelZero)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.empty())
        GTEST_SKIP() << "No ROCm-capable device is detected";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem =
        MakeWrwProblem(CanonicalShape(), miopenFloat, miopenTensorNDHWC, /*deterministic=*/false);

    Config cfg(false);
    ASSERT_TRUE(cfg.SetNextValue(problem));
    ASSERT_FALSE(cfg.valid_kernels.empty());
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels.front() + "+1");
    EXPECT_TRUE(cfg.IsValid(problem));
}

// Same, in deterministic mode -- the configuration that SILOTIGER-1243 reported, where the
// inverted exit made GetAllConfigs return empty. The seed must still be emitted, and the first
// advance must step to the next kernel while holding split_k at 1.
TEST(GPU_PerfConfig_HipImplicitGemm3DGroupWrwXdlops_FP32, DeterministicSearchStartsAtKernelZero)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.empty())
        GTEST_SKIP() << "No ROCm-capable device is detected";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem =
        MakeWrwProblem(CanonicalShape(), miopenFloat, miopenTensorNDHWC, /*deterministic=*/true);

    Config cfg(false);
    ASSERT_TRUE(cfg.SetNextValue(problem));
    ASSERT_FALSE(cfg.valid_kernels.empty());
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels.front() + "+1");
    EXPECT_TRUE(cfg.IsValid(problem));

    if(cfg.valid_kernels.size() < 2)
        GTEST_SKIP() << "Only one applicable CK instance; nothing left to advance to";

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 1);
    EXPECT_EQ(cfg.split_k, 1) << "Deterministic mode must hold split_k at 1";
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels[1] + "+1");
}

// Deterministic walk: with three valid kernels, deterministic SetNextValue() must visit index 1
// then index 2 before exhausting. The inverted-NextLinear bug exhausted after the first call.
TEST(CPU_PerfConfig3DGroupWrwDeterministic_NONE, WalksEveryKernelThenExhausts)
{
    const auto problem =
        MakeWrwProblem(CanonicalShape(), miopenFloat, miopenTensorNDHWC, /*deterministic=*/true);

    Config cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 0;
    cfg.split_k       = 1;

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 1);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, "B+1");

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 2);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, "C+1");

    EXPECT_FALSE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 2) << "Exhaustion must leave the cursor on the last kernel, not wrap it";
}

// Hang regression: deterministic mode with exactly one valid kernel must terminate immediately.
// The inverted bug fell through and could spin on a single-kernel index space.
TEST(CPU_PerfConfig3DGroupWrwDeterministic_NONE, SingleKernelTerminates)
{
    const auto problem =
        MakeWrwProblem(CanonicalShape(), miopenFloat, miopenTensorNDHWC, /*deterministic=*/true);

    // One valid kernel: the seed is the whole search, so SetNextValue must report exhaustion
    // immediately and leave the cursor untouched -- no re-emit, no step past index 0.
    Config cfg(false);
    cfg.valid_kernels = {"A"};
    cfg.index         = 0;
    cfg.split_k       = 1;
    cfg.kernel_id     = cfg.valid_kernels.front() + "+1";

    int iterations = 0;
    while(cfg.SetNextValue(problem))
    {
        ++iterations;
        ASSERT_LT(iterations, 10) << "SetNextValue looped instead of terminating";
    }
    EXPECT_EQ(iterations, 0);
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, "A+1");
}

// Non-deterministic smoke: the (untouched) non-deterministic branch must still terminate and
// explore more than one (index, split_k) combination.
TEST(CPU_PerfConfig3DGroupWrwNonDeterministic_NONE, VisitsMultipleValues)
{
    const auto problem =
        MakeWrwProblem(CanonicalShape(), miopenFloat, miopenTensorNDHWC, /*deterministic=*/false);

    Config cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 0;
    cfg.split_k       = 1;
    cfg.kernel_id     = cfg.valid_kernels[cfg.index] + "+" + std::to_string(cfg.split_k);

    std::set<int> visited_indices{cfg.index};
    std::set<int> visited_split_ks{cfg.split_k};
    int iterations = 0;
    while(cfg.SetNextValue(problem))
    {
        visited_indices.insert(cfg.index);
        visited_split_ks.insert(cfg.split_k);
        ++iterations;
        ASSERT_LT(iterations, 500) << "Non-deterministic SetNextValue did not terminate";
    }
    EXPECT_GT(visited_indices.size(), 1u);
    EXPECT_GT(visited_split_ks.size(), 1u);
}
