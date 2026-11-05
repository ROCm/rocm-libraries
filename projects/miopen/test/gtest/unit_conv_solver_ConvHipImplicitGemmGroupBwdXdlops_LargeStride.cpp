// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// 2D BWD applicability test for ConvHipImplicitGemmGroupBwdXdlops on a shape just past
// the 32-bit range.
//
// On this branch the solver rejects any problem with a tensor larger than INT_MAX bytes
// (see IsApplicable in conv_hip_implicit_gemm_grouped_bwd_xdlops.cpp): CK's grouped
// backward-data kernel returned wrong results for these shapes. The shape below crosses
// that bound for every data type, so the solver must be applicable on no device.
//
// Shape: x = (1, 96, 4736, 4736), w = (16, 96, 1, 1), group=1, pad=0, stride=1.
//   x has 96 * 4736 * 4736 = 2,153,316,352 elements: 4.3 GB in FP16/BF16, 8.6 GB in FP32.
//
// This replaces a GPU numerical test that required the solver to be applicable to this
// shape. That test contradicted the gate and could only fail.

#include "unit_conv_solver.hpp"

namespace {

auto GetGatedConvCase(miopenDataType_t type, miopenTensorLayout_t layout)
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{type, layout, {1, 96, 4736, 4736}},
                        {type, layout, {16, 96, 1, 1}},
                        type,
                        {{0, 0}, {1, 1}, {1, 1}}};
}

auto GetGatedConvCases()
{
    std::vector<miopen::unit_tests::ConvTestCase> cases;
    for(const auto type : {miopenHalf, miopenBFloat16, miopenFloat})
        for(const auto layout : {miopenTensorNHWC, miopenTensorNCHW})
            cases.push_back(GetGatedConvCase(type, layout));
    return cases;
}

auto GetGatedParams() { return miopen::unit_tests::UnitTestConvSolverParams(Gpu::None); }

} // namespace

using CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsInt32GateDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsInt32GateDevApplicability_NONE,
       ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsInt32GateDevApplicability_NONE,
    testing::Combine(testing::Values(GetGatedParams()), testing::ValuesIn(GetGatedConvCases())));
