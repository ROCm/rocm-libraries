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
// CK/applicability regression rather than a bad test shape.
//
// The host-only SetNextValue() advance checks for this config (single-kernel termination,
// deterministic walk, non-deterministic coverage) live in
// perf_config_group_deterministic_set_next_value.cpp, where they share the templated helpers
// with the 2D grouped Bwd/Wrw configs.

#include <gtest/gtest.h>
#include <gtest/group_conv.hpp>

#include <miopen/tensor.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

#include "get_handle.hpp"

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

// The 3D grouped Wrw xdlops solver only enumerates kernels on xdlops-capable (CDNA) architectures.
// On other devices (e.g. gfx110X/RDNA) the CK library still loads but produces no valid kernels, so
// SetNextValue() legitimately has nothing to seed. The GPU tests use this to skip there rather than
// failing the kernel-enumeration asserts; on CDNA the solver is applicable and the asserts run.
bool SolverApplicableHere(const Problem& problem, std::string& reason)
{
    auto& handle = get_handle();
    auto ctx     = miopen::ExecutionContext{};
    ctx.SetStream(&handle);
    const miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops solver{};
    if(solver.IsApplicable(ctx, problem))
        return true;
    reason = solver.SolverDbId() + " is not applicable on this device (requires an xdlops/CDNA "
                                   "architecture)";
    return false;
}

} // namespace

// Parameterized only so that INSTANTIATE_TEST_SUITE_P can put these tests under the
// "Standard/" prefix used by test_categories.yaml; the int param itself is unused.
struct GPU_PerfConfig_HipImplicitGemm3DGroupWrwXdlops_FP32 : public ::testing::TestWithParam<int>
{
};

TEST_P(GPU_PerfConfig_HipImplicitGemm3DGroupWrwXdlops_FP32, SearchStartsAtKernelZero)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.empty())
        GTEST_SKIP() << "No ROCm-capable device is detected";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem =
        MakeWrwProblem(CanonicalShape(), miopenFloat, miopenTensorNDHWC, /*deterministic=*/false);

    std::string skip_reason;
    if(!SolverApplicableHere(problem, skip_reason))
        GTEST_SKIP() << skip_reason;

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
TEST_P(GPU_PerfConfig_HipImplicitGemm3DGroupWrwXdlops_FP32, DeterministicSearchStartsAtKernelZero)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.empty())
        GTEST_SKIP() << "No ROCm-capable device is detected";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem =
        MakeWrwProblem(CanonicalShape(), miopenFloat, miopenTensorNDHWC, /*deterministic=*/true);

    std::string skip_reason;
    if(!SolverApplicableHere(problem, skip_reason))
        GTEST_SKIP() << skip_reason;

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

INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_PerfConfig_HipImplicitGemm3DGroupWrwXdlops_FP32,
                         ::testing::Values(0));
