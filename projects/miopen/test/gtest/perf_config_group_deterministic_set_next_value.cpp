// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Regression tests for a deterministic-mode search hang in the grouped Bwd/Wrw CK xdlops
// solvers: miopenFindConvolutionBackwardDataAlgorithm() (and the analogous BackwardWeights
// find) hung on MI300X when TF32 + deterministic + exhaustive find coincided with an empty
// perf-db cache. Root cause was an inverted return-value polarity bug in the `is_deterministic`
// branch of SetNextValue(): NextLinear() returns true on wraparound (exhausted) / false on
// normal advance, but the old code treated a normal advance (`!NextLinear(...)` true) as
// "exhausted" and a wraparound (`NextLinear(...)` true) as "advanced" - exactly backwards. With
// exactly one valid kernel this spun forever (hang); with more than one it exhausted the search
// after the very first call (silent under-tuning).
//
// These tests run entirely on the host: they bypass the CkImplLibLoader lazy-init path inside
// SetNextValue() by directly assigning a non-empty `valid_kernels` list before calling it, so no
// GPU or CK shared library is required.

#include <gtest/gtest.h>
#include <gtest/group_conv.hpp>

#include <miopen/tensor.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>

#include <set>
#include <string>

namespace {

using Problem   = miopen::conv::ProblemDescription;
using BwdConfig = miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupBwdXdlops;
using WrwConfig = miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupWrwXdlops;

// Builds a minimal grouped-conv ProblemDescription whose only purpose is to carry the
// "deterministic" convolution attribute through to SetNextValue(); the tensor shapes are
// otherwise arbitrary since these tests never touch the CK library lookup.
Problem MakeProblem(miopen::conv::Direction direction, bool deterministic)
{
    group_conv::GroupConvTestConfig<2u> conv{1, 1, 4, 4, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}};

    const auto x_desc = miopen::TensorDescriptor(miopenFloat, miopenTensorNHWC, conv.GetInput());
    const auto w_desc = miopen::TensorDescriptor(miopenFloat, miopenTensorNHWC, conv.GetWeights());
    auto conv_desc    = conv.GetConv();
    if(deterministic)
        conv_desc.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);
    const auto y_desc = conv_desc.GetForwardOutputTensor(x_desc, w_desc, miopenFloat);

    return Problem(y_desc, w_desc, x_desc, conv_desc, direction);
}

// Hang regression: with exactly one valid kernel, deterministic SetNextValue() must return
// false immediately. The original bug spun on a single-kernel index space forever.
template <typename Config>
void CheckHangRegressionSingleKernel(const Problem& problem)
{
    Config cfg(false);
    cfg.valid_kernels = {"kernel_A"};
    cfg.index         = 0;

    int iterations = 0;
    while(cfg.SetNextValue(problem))
    {
        ++iterations;
        ASSERT_LT(iterations, 10) << "SetNextValue looped instead of terminating (hang regression)";
    }

    EXPECT_EQ(iterations, 0);
    EXPECT_EQ(cfg.index, 0);
}

// Under-tuning regression: with three valid kernels, deterministic SetNextValue() must visit
// index 1 then index 2 before exhausting. The original bug exhausted on the very first call.
template <typename Config>
void CheckUnderTuningThreeKernels(const Problem& problem)
{
    Config cfg(false);
    cfg.valid_kernels = {"A", "B", "C"};
    cfg.index         = 0;

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 1);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels[cfg.index] + "+1");

    ASSERT_TRUE(cfg.SetNextValue(problem));
    EXPECT_EQ(cfg.index, 2);
    EXPECT_EQ(cfg.split_k, 1);
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels[cfg.index] + "+1");

    EXPECT_FALSE(cfg.SetNextValue(problem));
}

// Non-deterministic smoke guard: the non-deterministic branch (untouched by this fix) must
// still terminate and explore more than one (index, split_k) combination. Deliberately loose -
// Bwd uses NextTwoPower<1,128> while Wrw uses NextCKSplitkValue<1,128>, so exact sequences
// differ and are not asserted here.
template <typename Config>
void CheckNonDeterministicVisitsMultipleValues(const Problem& problem)
{
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

} // namespace

TEST(CPU_PerfConfigGroupBwdDeterministic_NONE, HangRegressionSingleKernel)
{
    const auto problem = MakeProblem(miopen::conv::Direction::BackwardData, /*deterministic=*/true);
    CheckHangRegressionSingleKernel<BwdConfig>(problem);
}

TEST(CPU_PerfConfigGroupBwdDeterministic_NONE, UnderTuningThreeKernels)
{
    const auto problem = MakeProblem(miopen::conv::Direction::BackwardData, /*deterministic=*/true);
    CheckUnderTuningThreeKernels<BwdConfig>(problem);
}

TEST(CPU_PerfConfigGroupWrwDeterministic_NONE, HangRegressionSingleKernel)
{
    const auto problem =
        MakeProblem(miopen::conv::Direction::BackwardWeights, /*deterministic=*/true);
    CheckHangRegressionSingleKernel<WrwConfig>(problem);
}

TEST(CPU_PerfConfigGroupWrwDeterministic_NONE, UnderTuningThreeKernels)
{
    const auto problem =
        MakeProblem(miopen::conv::Direction::BackwardWeights, /*deterministic=*/true);
    CheckUnderTuningThreeKernels<WrwConfig>(problem);
}

TEST(CPU_PerfConfigGroupBwdNonDeterministic_NONE, VisitsMultipleIndicesAndSplitKs)
{
    const auto problem =
        MakeProblem(miopen::conv::Direction::BackwardData, /*deterministic=*/false);
    CheckNonDeterministicVisitsMultipleValues<BwdConfig>(problem);
}

TEST(CPU_PerfConfigGroupWrwNonDeterministic_NONE, VisitsMultipleIndicesAndSplitKs)
{
    const auto problem =
        MakeProblem(miopen::conv::Direction::BackwardWeights, /*deterministic=*/false);
    CheckNonDeterministicVisitsMultipleValues<WrwConfig>(problem);
}
