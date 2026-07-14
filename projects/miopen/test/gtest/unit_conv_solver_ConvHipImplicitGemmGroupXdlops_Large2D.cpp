// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Numerical (on-device execute + verify) coverage for large 2D grouped
// convolutions in NHWC layout. Shapes span three regimes: (a) sub-INT_MAX
// element counts, (b) tensors above 2 GB whose element count still fits int32,
// and (c) tensors whose element count exceeds INT_MAX (needs the CK
// large-tensor instances). All geometries are 3x3 with SAME padding.
// Complements the API probe in conv_api_solution_count_large2d.cpp; catches any
// silent int32 offset overflow on the regular (non-large-tensor) CK path.
//
// Excluded from the default gtest filter (test/gtest/CMakeLists.txt) as
// multi-GB; run explicitly. A case skips when the solver is inapplicable or the
// device lacks memory.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "unit_conv_solver.hpp"
#include "get_handle.hpp"

namespace {

using miopen::unit_tests::ConvolutionDescriptorParams;
using miopen::unit_tests::ConvTestCase;
using miopen::unit_tests::TensorDescriptorParams;

std::vector<ConvTestCase> GetTestCases(miopenDataType_t dt)
{
    const auto L = miopenTensorNHWC;
    // Lengths are in logical [N, C, H, W] order; L selects NHWC memory layout.
    // Filters are [K, C, 3, 3], SAME padding. Element counts annotated per case.
    return {
        // ---- sub-INT_MAX elements, < 2 GB ----
        {TensorDescriptorParams{dt, L, {48, 2048, 64, 64}},
         TensorDescriptorParams{dt, L, {2048, 2048, 3, 3}},
         dt,
         ConvolutionDescriptorParams{{1, 1}, {1, 1}, {1, 1}, 1}}, // 4.0e8 elem
        // ---- > 2 GB bytes, element count still fits int32 ----
        {TensorDescriptorParams{dt, L, {96, 1024, 112, 112}},
         TensorDescriptorParams{dt, L, {1024, 1024, 3, 3}},
         dt,
         ConvolutionDescriptorParams{{1, 1}, {1, 1}, {1, 1}, 1}}, // 1.23e9 elem
        {TensorDescriptorParams{dt, L, {96, 2048, 64, 64}},
         TensorDescriptorParams{dt, L, {4096, 2048, 3, 3}},
         dt,
         ConvolutionDescriptorParams{{1, 1}, {1, 1}, {1, 1}, 1}}, // 1.61e9 elem, K=2C
        // ---- element count > INT_MAX (needs CK large-tensor instances) ----
        {TensorDescriptorParams{dt, L, {160, 1024, 140, 100}},
         TensorDescriptorParams{dt, L, {1024, 1024, 3, 3}},
         dt,
         ConvolutionDescriptorParams{{1, 1}, {1, 1}, {1, 1}, 1}}, // 2.29e9 elem
        {TensorDescriptorParams{dt, L, {160, 1024, 100, 140}},
         TensorDescriptorParams{dt, L, {1024, 1024, 3, 3}},
         dt,
         ConvolutionDescriptorParams{{1, 1}, {1, 1}, {1, 1}, 1}}, // 2.29e9 elem, H/W swap
        {TensorDescriptorParams{dt, L, {200, 512, 200, 200}},
         TensorDescriptorParams{dt, L, {512, 512, 3, 3}},
         dt,
         ConvolutionDescriptorParams{{1, 1}, {1, 1}, {1, 1}, 1}}, // 4.10e9 elem, large spatial
        {TensorDescriptorParams{dt, L, {160, 1024, 120, 120}},
         TensorDescriptorParams{dt, L, {2048, 1024, 3, 3}},
         dt,
         ConvolutionDescriptorParams{{1, 1}, {1, 1}, {1, 1}, 1}}, // 4.72e9 elem, K=2C
    };
}

const miopen::unit_tests::UnitTestConvSolverParams& GetTestParams()
{
    static const auto params = [] {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        Gpu supportedDevices = Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
#else
        Gpu supportedDevices = Gpu::None;
#endif
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
        p.Tunable(5);
        // Large-tensor wrw reduces over a deep dimension; an FP16 accumulation of that
        // depth cannot meet a 1-ulp RMS threshold and the split-k reduction is
        // order-nondeterministic (worst observed ~9.3x eps). Relax the FP16 tolerance to
        // cover the reduction noise with margin. Applies to fwd/bwd here too (they pass
        // comfortably), matching the shared-params pattern used by peers.
        p.SetTolerance(supportedDevices, miopenHalf, 64.0f);
        return p;
    }();
    return params;
}

enum class Dir
{
    Fwd,
    Bwd,
    Wrw
};

struct RunGate
{
    std::size_t required;
    std::size_t available;
    bool applicable;
};

RunGate GateFor(const ConvTestCase& tc,
                const miopen::solver::conv::ConvSolverInterface& solver,
                Dir dir)
{
    auto&& handle = get_handle();

    const auto x_desc    = tc.GetXTensorDescriptor();
    const auto w_desc    = tc.GetWTensorDescriptor();
    const auto conv_desc = tc.GetConv();
    const auto y_desc = conv_desc.GetForwardOutputTensor(x_desc, w_desc, tc.GetYDataType());

    const auto problem = [&] {
        switch(dir)
        {
        case Dir::Fwd:
            return miopen::conv::ProblemDescription(
                x_desc, w_desc, y_desc, conv_desc, miopen::conv::Direction::Forward);
        case Dir::Bwd:
            return miopen::conv::ProblemDescription(
                y_desc, w_desc, x_desc, conv_desc, miopen::conv::Direction::BackwardData);
        default:
            return miopen::conv::ProblemDescription(
                y_desc, w_desc, x_desc, conv_desc, miopen::conv::Direction::BackwardWeights);
        }
    }();

    auto ctx = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);

    if(!solver.IsApplicable(ctx, problem))
        return {0, 0, false};

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

    return {required_mem, device_mem, true};
}

} // namespace

#define SKIP_IF_NOT_RUNNABLE(solver_expr, dir)                                          \
    do                                                                                  \
    {                                                                                   \
        miopen::unit_tests::UnitTestConvSolverParams _params;                           \
        miopenConvAlgorithm_t _algo;                                                    \
        ConvTestCase _tc;                                                               \
        std::tie(_params, _algo, _tc) = this->GetParam();                               \
        const auto _g = GateFor(_tc, (solver_expr), (dir));                             \
        if(!_g.applicable)                                                              \
        {                                                                               \
            GTEST_SKIP() << "solver not applicable to this shape";                      \
        }                                                                               \
        if(_g.available < _g.required)                                                  \
        {                                                                               \
            GTEST_SKIP() << "Insufficient device memory: need " << _g.required          \
                         << " bytes, device has " << _g.available;                      \
        }                                                                               \
    } while(0)

using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_Large2D_FP16 =
    GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_Large2D_FP16 =
    GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_Large2D_FP16 =
    GPU_UnitTestConvSolverWrw_FP16;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_Large2D_FP16,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{};
    SKIP_IF_NOT_RUNNABLE(solver, Dir::Fwd);
    this->RunTest(solver);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_Large2D_FP16,
       ConvHipImplicitGemmGroupBwdXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{};
    SKIP_IF_NOT_RUNNABLE(solver, Dir::Bwd);
    this->RunTest(solver);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_Large2D_FP16,
       ConvHipImplicitGemmGroupWrwXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{};
    SKIP_IF_NOT_RUNNABLE(solver, Dir::Wrw);
    this->RunTest(solver);
};

INSTANTIATE_TEST_SUITE_P(Large2D,
                         GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_Large2D_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Large2D,
                         GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_Large2D_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Large2D,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_Large2D_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetTestCases(miopenHalf))));
