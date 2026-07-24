// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Numerical 2D Fwd test for ConvHipImplicitGemmGroupFwdXdlops on a shape with
// element-strides exceeding INT_MAX (ROCM-23997). This complements
// conv_api_solution_count_2d_large_stride.cpp (which only verifies
// CompileSolution success): RunTest actually launches the kernel and compares
// against a reference, catching int32 wraparound that can occur inside the CK
// kernel even after MIOpen's host-side widening.
//
// Shape: x = (1, 96, 4736, 4736), w = (16, 96, 1, 1), group=1, pad=0, stride=1.
//   element count of x = 96 * 4736 * 4736 ~= 2.153 B (just above INT_MAX ~= 2.147 B).
//   FP16 footprint of x ~= 4.3 GB; FP32 ~= 8.6 GB. Heavyweight instances are
//   gated at runtime by an explicit memory estimate (SKIP_IF_INSUFFICIENT_DEVICE_MEMORY).
//
// Heavyweight: excluded from the default gtest filter via a negative filter in
// CMakeLists.txt (see add_gtest_negative_filter for "*LargeStride*").

#include <algorithm>
#include <cstddef>

#include "unit_conv_solver.hpp"
#include "get_handle.hpp"

namespace {

using miopen::unit_tests::ConvolutionDescriptorParams;
using miopen::unit_tests::ConvTestCase;
using miopen::unit_tests::TensorDescriptorParams;

std::vector<ConvTestCase> GetLargeStrideFwdTestCases(miopenDataType_t dt)
{
    std::vector<ConvTestCase> cases;
    for(auto layout : {miopenTensorNHWC, miopenTensorNCHW})
    {
        cases.push_back(ConvTestCase{TensorDescriptorParams{dt, layout, {1, 96, 4736, 4736}},
                                     TensorDescriptorParams{dt, layout, {16, 96, 1, 1}},
                                     dt,
                                     ConvolutionDescriptorParams{{0, 0}, {1, 1}, {1, 1}, 1}});
    }
    return cases;
}

const miopen::unit_tests::UnitTestConvSolverParams& GetTestParams()
{
    static const auto params = [] {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        // Restricted to gfx90A, gfx94X, and gfx950: covered by CI and manually
        // qualified for the large-stride kernel-launch path on this shape.
        Gpu supportedDevices = Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
#else
        Gpu supportedDevices = Gpu::None;
#endif
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
        p.Tunable(5);
        return p;
    }();
    return params;
}

// Conservative working-set estimate. Sums the workspace (queried from the
// solver), the X/W/Y device tensors, and 4x the largest tensor for the
// host-side input/weights/output/reference allocations. Adds headroom for
// runtime/library reservations, allocator fragmentation, and (on consumer
// cards) the display compositor -- using max(+1 GiB, +10%) to cover both the
// absolute and the proportional components.
struct MemoryEstimate
{
    std::size_t required;
    std::size_t available;
};

MemoryEstimate EstimateRequiredMemoryFwd(const ConvTestCase& conv_config,
                                         const miopen::solver::conv::ConvSolverInterface& solver)
{
    auto&& handle = get_handle();

    const auto x_desc    = conv_config.GetXTensorDescriptor();
    const auto w_desc    = conv_config.GetWTensorDescriptor();
    const auto conv_desc = conv_config.GetConv();
    const auto y_desc = conv_desc.GetForwardOutputTensor(x_desc, w_desc, conv_config.GetYDataType());

    const auto problem = miopen::conv::ProblemDescription(
        x_desc, w_desc, y_desc, conv_desc, miopen::conv::Direction::Forward);
    auto ctx = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);

    const std::size_t ws_size =
        solver.MayNeedWorkspace() ? solver.GetWorkspaceSize(ctx, problem) : 0;
    const std::size_t x_bytes = x_desc.GetNumBytes();
    const std::size_t y_bytes = y_desc.GetNumBytes();
    const std::size_t w_bytes = w_desc.GetNumBytes();
    const std::size_t h_bytes = std::max(x_bytes, y_bytes);

    const std::size_t raw_mem      = ws_size + x_bytes + y_bytes + w_bytes + 4 * h_bytes;
    const std::size_t headroom     = std::max<std::size_t>(1ULL << 30, raw_mem / 10);
    const std::size_t required_mem = raw_mem + headroom;
    const std::size_t device_mem   = handle.GetGlobalMemorySize();

    return {required_mem, device_mem};
}

} // namespace

#define SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(solver_expr)                                \
    do                                                                                 \
    {                                                                                  \
        miopen::unit_tests::UnitTestConvSolverParams _params;                         \
        miopenConvAlgorithm_t _algo;                                                  \
        ConvTestCase _tc;                                                             \
        std::tie(_params, _algo, _tc) = this->GetParam();                             \
        const auto _mem = EstimateRequiredMemoryFwd(_tc, (solver_expr));              \
        if(_mem.available < _mem.required)                                            \
        {                                                                             \
            GTEST_SKIP() << "Insufficient device memory: need " << _mem.required      \
                         << " bytes, device has " << _mem.available;                  \
        }                                                                             \
    } while(0)

using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP16 =
    GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_BFP16 =
    GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP32 =
    GPU_UnitTestConvSolverFwd_FP32;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP16,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(solver);
    this->RunTest(solver);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_BFP16,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(solver);
    this->RunTest(solver);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP32,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(solver);
    this->RunTest(solver);
};

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetLargeStrideFwdTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetLargeStrideFwdTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_LargeStride_FP32,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetLargeStrideFwdTestCases(miopenFloat))));
