// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Numerical 3D WRW test for ConvHipImplicitGemm3DGroupWrwXdlops on a shape with
// element-strides exceeding INT_MAX. Complements the API-level
// conv_api_solution_count_3d_large_stride.cpp by actually launching the kernel
// and comparing against a CPU reference, catching int32 wraparound that can
// occur inside the CK kernel even after MIOpen's host-side widening. The shape
// is shared with the 3D Fwd test.
//
// Shape: x = (1, 96, 512, 512, 88), w = (16, 96, 1, 1, 1), group=1, pad=0, stride=1.
//   element count of x = 96 * 512 * 512 * 88 = 2.214 B (just above INT_MAX).
//   FP16 footprint of x ~= 4.4 GB; FP32 ~= 8.9 GB. The full test allocates several
//   such tensors (X, W, Y on device plus host-side reference), so heavyweight
//   instances are gated at runtime by an explicit memory estimate.
//
// WRW reduces over N * D * H * W per output element, so the FP32 tolerance is
// widened well beyond the Fwd level to absorb the much larger accumulated RMS
// error.
//
// The 3D large-stride API sweep confirms Wrw applicability at this shape across
// FP16/BFP16, so this test should compile and run cleanly on a 64 GB-class GPU.
// Excluded from the standard test category via test_categories.yaml. Each
// variant dynamically skips when the device cannot fit the estimated working
// set.

#include <algorithm>
#include <cstddef>
#include <utility>

#include "get_handle.hpp"
#include "unit_conv_solver_group_xdlops.hpp"

namespace {

using TestCase     = miopen::unit_tests::GroupXdlopsNumericData;
using TestDataType = miopen::unit_tests::TestDataType;

template <TestDataType type>
std::vector<TestCase> GetLargeStrideWrwTestCases()
{
    return {
        // clang-format off
        TestCase{{1, 96, 512, 512, 88}, {16, 96, 1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1, false, false},
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
    if constexpr(type == TestDataType::FP32)
    {
        // WRW reduces over N*D*H*W per output element (~23M FMAs at this shape) vs.
        // Fwd's C reduction (96 FMAs at the 1x1x1 filter). RMS error scales with
        // reduction size, so a much larger tolerance bump than Fwd's is required.
        p.SetTolerance(supportedDevices, miopenFloat, 80.0f);
    }
    return p;
}

// Conservative working-set estimate for the configured Wrw test. Sums the workspace
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
MemoryEstimate EstimateRequiredMemoryWrw(TestCase tc,
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
        y_desc, w_desc, x_desc, conv_case.GetConv(), miopen::conv::Direction::BackwardWeights);
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
        const auto _mem = EstimateRequiredMemoryWrw<datatype>(_tc, _layout, (solver_expr)); \
        if(_mem.available < _mem.required)                                                  \
        {                                                                                   \
            GTEST_SKIP() << "Insufficient device memory: need " << _mem.required            \
                         << " bytes, device has " << _mem.available;                        \
        }                                                                                   \
    } while(0)

using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenFloat>;

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(miopenHalf, solver);
    this->RunTest(solver);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_BFP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(miopenBFloat16, solver);
    this->RunTest(solver);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP32,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(miopenFloat, solver);
    this->RunTest(solver);
};

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetLargeStrideWrwTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetLargeStrideWrwTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(GetLargeStrideWrwTestCases<TestDataType::FP32>())));
