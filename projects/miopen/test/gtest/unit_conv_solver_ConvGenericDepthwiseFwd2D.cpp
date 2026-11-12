// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

#include <cstring>
#include <string>
#include <vector>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/stringutils.hpp>

#include "conv_tensor_gen.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"

namespace {
// Shapes without a matching CK instance, served by the general-shape HIP kernel on gfx950.
auto GetConvGenericTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // 3x3 on a 24x24 plane (no 24-wide instance), odd channel count: cache-read variant
        TestCase{{datatype, miopenTensorNCHW, {3, 13, 24, 24}},
                 {datatype, miopenTensorNCHW, {13, 1, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}, 13}},
        // 5x5 on a 19x19 plane, odd output width (scalar stores): LDS variant
        TestCase{{datatype, miopenTensorNCHW, {2, 5, 19, 19}},
                 {datatype, miopenTensorNCHW, {5, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 5}},
        // stride 2 without padding, batch 1
        TestCase{{datatype, miopenTensorNCHW, {1, 8, 28, 28}},
                 {datatype, miopenTensorNCHW, {8, 1, 3, 3}},
                 datatype, {{0, 0}, {2, 2}, {1, 1}, 8}},
        // rectangular filter, mixed stride, dilation and padding
        TestCase{{datatype, miopenTensorNCHW, {2, 5, 29, 38}},
                 {datatype, miopenTensorNCHW, {5, 1, 5, 3}},
                 datatype, {{2, 0}, {2, 1}, {2, 1}, 5}},
        // 4096 small planes: several planes per workgroup, last group partial
        TestCase{{datatype, miopenTensorNCHW, {64, 64, 9, 9}},
                 {datatype, miopenTensorNCHW, {64, 1, 7, 7}},
                 datatype, {{3, 3}, {1, 1}, {1, 1}, 64}},
        // 700-wide rows: two column tiles with a shared halo
        TestCase{{datatype, miopenTensorNCHW, {1, 2, 6, 700}},
                 {datatype, miopenTensorNCHW, {2, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 2}},
        // 113x113 at dilation 8 with a 15x15 filter: the LDS tile would need 13,333 floats,
        // past the 8,192-float budget, so only the direct variant launches this shape
        TestCase{{datatype, miopenTensorNCHW, {1, 1, 113, 113}},
                 {datatype, miopenTensorNCHW, {1, 1, 15, 15}},
                 datatype, {{0, 0}, {1, 1}, {8, 8}, 1}},
        // clang-format on
    };
}

// Shapes at the edge of the admitted 32-bit index range. No memory is ever allocated for
// them: they only exercise the solver's applicability and launch configuration, where the
// ceilings and the row balancing have to stay inside int64.
auto GetConvGenericIndexLimitTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // one row of INT32_MAX columns
        TestCase{{datatype, miopenTensorNCHW, {1, 1, 1, 2147483647}},
                 {datatype, miopenTensorNCHW, {1, 1, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}, 1}},
        // INT32_MAX-512 rows of one column
        TestCase{{datatype, miopenTensorNCHW, {1, 1, 2147483135, 1}},
                 {datatype, miopenTensorNCHW, {1, 1, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}, 1}},
        // clang-format on
    };
}
auto GetGenericTestParams()
{
    // The general-shape kernel is plain HIP (no CK library) and is enabled on gfx950 only.
#if MIOPEN_BACKEND_HIP
    Gpu supportedDevices = Gpu::gfx950;
#else
    Gpu supportedDevices = Gpu::None;
#endif
    auto params = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
    params.Tunable(5);
    return params;
}

// Where both variants of the general kernel launch a shape they accumulate every output in
// the same order, so they must agree bit for bit. Some shapes fit only one variant, for
// instance when the LDS tile is too large; those run the one variant that is offered. The
// solver test above checks the tuned variant against the reference.
template <typename T>
void CheckGenericVariantsAgree(const miopen::unit_tests::ConvTestCase& conv_config)
{
    auto&& handle = get_handle();
    if(!miopen::StartsWith(handle.GetDeviceName(), "gfx950"))
        GTEST_SKIP() << "general depthwise kernel is enabled on gfx950 only";

    auto input           = tensor<T>{conv_config.GetXTensorDescriptor()};
    auto weights         = tensor<T>{conv_config.GetWTensorDescriptor()};
    const auto conv_desc = conv_config.GetConv();
    const auto out_desc =
        conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
    input.generate(GenConvData<T, T>{weights.desc.GetLengths()});
    weights.generate(GenConvData<T, T>{weights.desc.GetLengths()});
    const auto in_dev  = handle.Write(input.data);
    const auto wei_dev = handle.Write(weights.data);

    const auto problem = miopen::conv::ProblemDescription(
        input.desc, weights.desc, out_desc, conv_desc, miopen::conv::Direction::Forward);
    auto ctx = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);

    const auto solver = miopen::solver::conv::ConvGenericDepthwiseFwd2D{};
    ASSERT_TRUE(solver.IsApplicable(ctx, problem));

    int launched = 0;
    std::vector<std::vector<T>> results;
    for(const int use_lds : {1, 0})
    {
        const std::string id = use_lds == 1 ? "lds" : "direct";
        const auto config =
            miopen::solver::conv::PerformanceConfigConvGenericDepthwiseFwd2D(use_lds);
        if(!solver.IsValidPerformanceConfig(ctx, problem, config))
            continue;
        ++launched;
        const auto sol = solver.GetSolution(ctx, problem, config);
        ASSERT_TRUE(sol.Succeeded()) << id;

        std::vector<T> out(out_desc.GetElementSpace(), T{});
        auto out_dev       = handle.Write(out);
        const auto tensors = miopen::ConvFwdTensors{
            input.desc, in_dev.get(), weights.desc, wei_dev.get(), out_desc, out_dev.get()};
        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        invoker(handle, miopen::conv::DataInvokeParams{tensors, nullptr, 0, false});
        handle.Finish();
        results.push_back(handle.Read<T>(out_dev, out.size()));
    }
    ASSERT_GT(launched, 0);
    if(results.size() == 2)
    {
        ASSERT_EQ(results[0].size(), results[1].size());
        EXPECT_EQ(0,
                  std::memcmp(results[0].data(), results[1].data(), results[0].size() * sizeof(T)));
    }
}

// The index-limit shapes are never allocated. The solver has to apply, every variant it
// offers has to build, and the grid it asks for has to cover the outputs without passing the
// 2^31 work-item limit. Before the int64 rewrite the configuration arithmetic wrapped here.
void CheckGenericLaunchGrid(const miopen::unit_tests::ConvTestCase& conv_config)
{
    auto&& handle = get_handle();
    if(!miopen::StartsWith(handle.GetDeviceName(), "gfx950"))
        GTEST_SKIP() << "general depthwise kernel is enabled on gfx950 only";

    const auto in_desc   = conv_config.GetXTensorDescriptor();
    const auto wei_desc  = conv_config.GetWTensorDescriptor();
    const auto conv_desc = conv_config.GetConv();
    const auto out_desc =
        conv_desc.GetForwardOutputTensor(in_desc, wei_desc, conv_config.GetYDataType());

    const auto problem = miopen::conv::ProblemDescription(
        in_desc, wei_desc, out_desc, conv_desc, miopen::conv::Direction::Forward);
    auto ctx = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);

    const auto solver = miopen::solver::conv::ConvGenericDepthwiseFwd2D{};
    ASSERT_TRUE(solver.IsApplicable(ctx, problem));

    // One thread writes at most eight consecutive outputs.
    constexpr std::size_t max_threads    = 2147483647;
    constexpr std::size_t max_per_thread = 8;
    constexpr std::size_t block_size     = 256;
    const auto outputs                   = out_desc.GetElementSize();

    int launched = 0;
    for(const int use_lds : {1, 0})
    {
        const std::string id = use_lds == 1 ? "lds" : "direct";
        const auto config =
            miopen::solver::conv::PerformanceConfigConvGenericDepthwiseFwd2D(use_lds);
        if(!solver.IsValidPerformanceConfig(ctx, problem, config))
            continue;
        ++launched;
        const auto sol = solver.GetSolution(ctx, problem, config);
        ASSERT_TRUE(sol.Succeeded()) << id;
        ASSERT_EQ(std::size_t{1}, sol.construction_params.size()) << id;
        const auto& kernel = sol.construction_params.front();
        EXPECT_EQ(block_size, kernel.l_wk[0]) << id;
        EXPECT_EQ(std::size_t{0}, kernel.g_wk[0] % block_size) << id;
        EXPECT_LE(kernel.g_wk[0], max_threads) << id;
        EXPECT_GE(kernel.g_wk[0] * max_per_thread, outputs) << id;
    }
    ASSERT_GT(launched, 0);
}

} // namespace

// General-shape kernel (gfx950): shapes no CK instance covers
using GPU_UnitTestConvSolverConvGenericDepthwiseFwd2D_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverConvGenericDepthwiseFwd2D_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using CPU_UnitTestConvSolverConvGenericDepthwiseFwd2DDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverConvGenericDepthwiseFwd2D_FP16, ConvGenericDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvGenericDepthwiseFwd2D{});
    CheckGenericVariantsAgree<half_float::half>(std::get<2>(GetParam()));
};

TEST_P(GPU_UnitTestConvSolverConvGenericDepthwiseFwd2D_BFP16, ConvGenericDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvGenericDepthwiseFwd2D{});
    CheckGenericVariantsAgree<bfloat16>(std::get<2>(GetParam()));
};

TEST_P(CPU_UnitTestConvSolverConvGenericDepthwiseFwd2DDevApplicability_FP16,
       ConvGenericDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvGenericDepthwiseFwd2D{});
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvGenericDepthwiseFwd2D_FP16,
                         testing::Combine(testing::Values(GetGenericTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::Values(GetConvGenericTestCases(miopenHalf)[0],
                                                          GetConvGenericTestCases(miopenHalf)[1])));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverConvGenericDepthwiseFwd2D_FP16,
                         testing::Combine(testing::Values(GetGenericTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvGenericTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvGenericDepthwiseFwd2D_BFP16,
    testing::Combine(testing::Values(GetGenericTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::Values(GetConvGenericTestCases(miopenBFloat16)[1])));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverConvGenericDepthwiseFwd2D_BFP16,
    testing::Combine(testing::Values(GetGenericTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGenericTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverConvGenericDepthwiseFwd2DDevApplicability_FP16,
                         testing::Combine(testing::Values(GetGenericTestParams()),
                                          testing::Values(GetConvGenericTestCases(miopenHalf)[0])));

// General-shape kernel (gfx950): shapes at the edge of the admitted 32-bit index range
using GPU_UnitTestConvSolverConvGenericDepthwiseFwd2DLimit_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using CPU_UnitTestConvSolverConvGenericDepthwiseFwd2DLimitDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

// RunTest is deliberately not called: these shapes are far too large to allocate.
TEST_P(GPU_UnitTestConvSolverConvGenericDepthwiseFwd2DLimit_FP16, ConvGenericDepthwiseFwd2D)
{
    CheckGenericLaunchGrid(std::get<2>(GetParam()));
};

TEST_P(CPU_UnitTestConvSolverConvGenericDepthwiseFwd2DLimitDevApplicability_FP16,
       ConvGenericDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvGenericDepthwiseFwd2D{});
};

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvGenericDepthwiseFwd2DLimit_FP16,
    testing::Combine(testing::Values(GetGenericTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGenericIndexLimitTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverConvGenericDepthwiseFwd2DLimitDevApplicability_FP16,
    testing::Combine(testing::Values(GetGenericTestParams()),
                     testing::ValuesIn(GetConvGenericIndexLimitTestCases(miopenHalf))));
