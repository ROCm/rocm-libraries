// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// 2D BWD applicability test for ConvHipImplicitGemmGroupBwdXdlops on shapes around the
// 32-bit range.
//
// On this branch the solver rejects any problem with a tensor larger than INT_MAX bytes
// (see IsApplicable in conv_hip_implicit_gemm_grouped_bwd_xdlops.cpp): CK's grouped
// backward-data kernel returned wrong results for these shapes.
//
// Gated shape: x = (1, 96, 4736, 4736), w = (16, 96, 1, 1), group=1, pad=0, stride=1.
//   x has 96 * 4736 * 4736 = 2,153,250,816 elements: 4.3 GB in FP16/BF16, 8.6 GB in FP32.
//   The shape crosses the bound for every data type, so the solver must be applicable on
//   no device.
//
// In-range shape: x = (1, 96, 3344, 3344), same filter, FP16 only.
//   x has 96 * 3344 * 3344 = 1,073,504,256 elements: 2,147,008,512 bytes in FP16, which is
//   just below INT_MAX (2,147,483,647). The solver must be applicable on the same devices
//   as for the small shapes. This case keeps the suite from passing when the solver rejects
//   every problem, for example a gate that is too tight.
//
// Byte-only gated shape: x = (1, 96, 3345, 3345), same filter, FP16 only.
//   x has 96 * 3345 * 3345 = 1,074,146,400 elements, which fits in int32, but 2,148,292,800
//   bytes in FP16, which is just above INT_MAX. Only a gate on bytes rejects it. A gate that
//   compared the element count would accept it, so this case pins the byte comparison. The
//   solver must be applicable on no device.
//
// The solver also returns false when the CK plugin for the device is not loaded, which
// would make the gated cases pass without exercising the gate. The suite therefore declares
// the CK dynamic library as a dependency and is skipped when the plugin is missing.
//
// This replaces a GPU numerical test that required the solver to be applicable to the gated
// shape. That test contradicted the gate and could only fail.

#include "unit_conv_solver.hpp"

namespace {

using ConvTestCase = miopen::unit_tests::ConvTestCase;
using SolverParams = miopen::unit_tests::UnitTestConvSolverParams;

auto GetConvCase(miopenDataType_t type, miopenTensorLayout_t layout, std::size_t hw)
{
    return ConvTestCase{{type, layout, {1, 96, hw, hw}},
                        {type, layout, {16, 96, 1, 1}},
                        type,
                        {{0, 0}, {1, 1}, {1, 1}}};
}

// No device may accept these shapes.
SolverParams GetGatedParams()
{
    SolverParams p(Gpu::None);
    p.UsesCKDynamicLib();
    return p;
}

// Same device list as the FP16 device applicability test in
// unit_conv_solver_ConvHipImplicitGemmGroupBwdXdlops.cpp.
SolverParams GetInRangeParams()
{
#if MIOPEN_BACKEND_HIP
    Gpu supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950 | Gpu::gfx110X |
                           Gpu::gfx115X | Gpu::gfx120X;
#else
    Gpu supportedDevices = Gpu::None;
#endif
    SolverParams p(supportedDevices);
    p.ExcludeDevice("gfx1103");
    p.UsesCKDynamicLib();
    return p;
}

auto GetTestParams()
{
    std::vector<std::tuple<SolverParams, ConvTestCase>> params;
    for(const auto type : {miopenHalf, miopenBFloat16, miopenFloat})
        for(const auto layout : {miopenTensorNHWC, miopenTensorNCHW})
            params.emplace_back(GetGatedParams(), GetConvCase(type, layout, 4736));
    for(const auto layout : {miopenTensorNHWC, miopenTensorNCHW})
        params.emplace_back(GetGatedParams(), GetConvCase(miopenHalf, layout, 3345));
    for(const auto layout : {miopenTensorNHWC, miopenTensorNCHW})
        params.emplace_back(GetInRangeParams(), GetConvCase(miopenHalf, layout, 3344));
    return params;
}

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
    testing::ValuesIn(GetTestParams()));
