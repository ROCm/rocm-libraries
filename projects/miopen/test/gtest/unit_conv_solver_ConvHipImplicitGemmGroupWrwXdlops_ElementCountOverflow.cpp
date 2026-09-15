// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Numerical 2D WRW test for ConvHipImplicitGemmGroupWrwXdlops on a shape whose flattened
// input element count exceeds INT_MAX while every individual tensor length/stride still
// fits int32. This complements unit_conv_solver_ConvHipImplicitGemmGroupWrwXdlops_LargeStride.cpp
// (which covers an out-of-range *stride*): RequiresLargeTensorCKInstance() must also catch
// tensors whose per-dimension strides are all small but whose total element count overflows
// a non-large CK instance's int32 element indexing (see implicitgemm_ck_util.hpp).
//
// Shape: x = (9, 256, 1024, 1024), w = (64, 256, 3, 3), group=1, pad=0, stride=1, dilation=1.
//   element count of x = 9 * 256 * 1024 * 1024 = 2.416 B (> INT_MAX = 2.147 B).
//   element count of y (dy, 3x3/pad0/stride1 => Ho=Wo=1022) = 9 * 64 * 1022 * 1022 = 602 M
//   (comfortably under INT_MAX). Max per-dim stride (NHWC, N-dim) = C*H*W ~= 268.4 M, also
//   comfortably under INT_MAX. So this shape trips the *element-count* gate specifically,
//   not the (already-covered) per-stride gate: a 1x1-filter variant with the same X tensor
//   does not reproduce the bug on this CK build (Ho=Wo=H=W means the tuner favors different
//   instances), but the 3x3 filter here routes to an int32-indexing CK instance whose flat
//   element offset overflows once selected, producing a large silent RMS error.
//
// FP16 footprint: x ~= 4.83 GB, y ~= 1.2 GB. The full test allocates several such tensors
// (X, W, Y on device plus host-side reference), so heavyweight instances are gated at
// runtime by an explicit memory estimate, mirroring the LargeStride test.
//
// Heavyweight and slow: the large-tensor (int64) CK instance required post-fix is
// measurably slower than the int32 instance it replaces on this shape. Excluded from the
// standard (per-PR) test category via test_categories.yaml.

#include <algorithm>
#include <cstddef>
#include <utility>

#include "get_handle.hpp"
#include "unit_conv_solver_group_xdlops.hpp"

namespace {

using TestCase     = miopen::unit_tests::GroupXdlopsNumericData;
using TestDataType = miopen::unit_tests::TestDataType;

template <TestDataType type>
std::vector<TestCase> GetElementCountOverflowWrwTestCases()
{
    return {
        // clang-format off
        TestCase{{9, 256, 1024, 1024}, {64, 256, 3, 3}, {0, 0}, {1, 1}, {1, 1}, 1, false, false},
        // clang-format on
    };
}

template <TestDataType type>
miopen::unit_tests::UnitTestConvSolverParams GetTestParams()
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    // Restricted to gfx90A, gfx94X, and gfx950: covered by CI and manually
    // qualified for the large-tensor kernel-launch path on this shape.
    Gpu supportedDevices = Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
#else
    Gpu supportedDevices = Gpu::None;
#endif
    miopen::unit_tests::UnitTestConvSolverParams p(supportedDevices);
    p.Tunable(5);
    p.UsesCKDynamicLib();
    p.UseGpuRef();
    if constexpr(type == TestDataType::FP32)
    {
        // WRW reduces over N*H*W per output element (~2.4M FMAs per output pixel-channel
        // reduction group at this shape). RMS error scales with reduction size, so a much
        // larger tolerance bump than a small-reduction Fwd test is required.
        p.SetTolerance(supportedDevices, miopenFloat, 80.0f);
    }
    return p;
}

// Conservative working-set estimate for the configured Wrw test. Sums the workspace
// (queried from the solver), the X/W/Y device tensors, and 4x the largest tensor for the
// host-side input/weights/output/reference allocations. Adds headroom for runtime/library
// reservations, allocator fragmentation, and (on consumer cards) the display compositor --
// using max(+1 GiB, +10%) to cover both the absolute and the proportional components.
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

using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_ElementCountOverflow_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenHalf>;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_ElementCountOverflow_FP16,
       ConvHipImplicitGemmGroupWrwXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(miopenHalf, solver);
    this->RunTest(solver);
};

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_ElementCountOverflow_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNHWC),
                     testing::ValuesIn(GetElementCountOverflowWrwTestCases<TestDataType::FP16>())));
