// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Numerical 2D Fwd test for ConvHipImplicitGemmGroupFwdXdlops on a shape with
// element-strides exceeding INT_MAX. This complements
// conv_api_solution_count_2d_large_stride.cpp (which only verifies CompileSolution
// success): RunTestImpl actually launches the kernel and compares against a CPU
// reference, catching int32 wraparound that can occur inside the CK kernel even
// after MIOpen's host-side widening.
//
// Shape: x = (1, 96, 4736, 4736), w = (16, 96, 1, 1), group=1, pad=0, stride=1.
//   element count of x = 96 * 4736 * 4736 = 2.153 B (just above INT_MAX = 2.147 B).
//   FP16 footprint of x ~= 4.3 GB; FP32 ~= 8.6 GB. The full test allocates several
//   such tensors (X, W, Y on device plus host-side reference), so heavyweight
//   instances are gated at runtime by an explicit memory estimate.
//
// Heavyweight: requires a 64 GB-class GPU (FP32 needs even more) and is excluded
// from the standard (per-PR) test category via test_categories.yaml. Each variant
// dynamically skips when the device cannot fit the estimated working set.

#include <algorithm>
#include <cstddef>
#include <utility>

#include "get_handle.hpp"
#include "unit_conv_solver_group_xdlops.hpp"

namespace {

using TestCase     = miopen::unit_tests::GroupXdlopsNumericData;
using TestDataType = miopen::unit_tests::TestDataType;

template <TestDataType type>
std::vector<TestCase> GetLargeStrideFwdTestCases()
{
    return {
        // clang-format off
        TestCase{{1, 96, 4736, 4736}, {16, 96, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, false, false},
        // clang-format on
    };
}

template <TestDataType type>
miopen::unit_tests::UnitTestConvSolverParams GetTestParams()
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    // Restricted to gfx90A, gfx94X, and gfx950: covered by CI and manually
    // qualified for the large-stride kernel-launch path on this shape.
    Gpu supportedDevices = Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
#else
    Gpu supportedDevices = Gpu::None;
#endif
    miopen::unit_tests::UnitTestConvSolverParams p(supportedDevices);
    p.Tunable(5);
    p.UsesCKDynamicLib();
    return p;
}

// Conservative working-set estimate for the configured Fwd test. Sums the workspace
// (queried from the solver), the X/W/Y device tensors, and 4× the largest tensor
// for the host-side input/weights/output/reference allocations. Adds headroom for
// runtime/library reservations, allocator fragmentation, and (on consumer cards)
// the display compositor — using max(+1 GiB, +10%) to cover both the absolute and
// the proportional components.
struct MemoryEstimate
{
    std::size_t required;
    std::size_t available;
};

template <miopenDataType_t datatype>
MemoryEstimate EstimateRequiredMemoryFwd(TestCase tc,
                                         miopenTensorLayout_t layout,
                                         const miopen::solver::conv::ConvSolverInterface& solver)
{
    auto conv_case = miopen::unit_tests::GetConvTestForGroupXdlops<datatype>(layout, std::move(tc));
    const auto x_desc = conv_case.GetXTensorDescriptor();
    const auto w_desc = conv_case.GetWTensorDescriptor();
    const auto y_desc =
        conv_case.GetConv().GetForwardOutputTensor(x_desc, w_desc, conv_case.GetYDataType());

    auto&& handle      = get_handle();
    const auto problem = miopen::conv::ProblemDescription(
        x_desc, w_desc, y_desc, conv_case.GetConv(), miopen::conv::Direction::Forward);
    auto ctx = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);

    const std::size_t ws_size =
        solver.MayNeedWorkspace() ? solver.GetWorkspaceSize(ctx, problem) : 0;
    const std::size_t x_bytes = x_desc.GetNumBytes();
    const std::size_t y_bytes = y_desc.GetNumBytes();
    const std::size_t w_size  = w_desc.GetNumBytes();
    const std::size_t h_bytes = std::max(x_bytes, y_bytes);

    const std::size_t raw_mem      = ws_size + x_bytes + y_bytes + w_size + 4 * h_bytes;
    const std::size_t headroom     = std::max<std::size_t>(1ULL << 30, raw_mem / 10);
    const std::size_t required_mem = raw_mem + headroom;
    const std::size_t device_mem   = handle.GetGlobalMemorySize();

    return {required_mem, device_mem};
}

} // namespace

#define SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(datatype, solver_expr)                           \
    do                                                                                      \
    {                                                                                       \
        miopen::unit_tests::UnitTestConvSolverParams _params;                               \
        miopenTensorLayout_t _layout;                                                       \
        TestCase _tc;                                                                       \
        std::tie(_params, _layout, _tc) = this->GetParam();                                 \
        const auto _mem = EstimateRequiredMemoryFwd<datatype>(_tc, _layout, (solver_expr)); \
        if(_mem.available < _mem.required)                                                  \
        {                                                                                   \
            GTEST_SKIP() << "Insufficient device memory: need " << _mem.required            \
                         << " bytes, device has " << _mem.available;                        \
        }                                                                                   \
    } while(0)

using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward, miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward,
                                                      miopenFloat>;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP16,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(miopenHalf, solver);
    this->RunTest(solver);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_BFP16,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(miopenBFloat16, solver);
    this->RunTest(solver);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP32,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(miopenFloat, solver);
    this->RunTest(solver);
};

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetLargeStrideFwdTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetLargeStrideFwdTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                     testing::ValuesIn(GetLargeStrideFwdTestCases<TestDataType::FP32>())));
