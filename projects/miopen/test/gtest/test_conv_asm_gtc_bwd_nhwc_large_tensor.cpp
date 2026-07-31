/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "unit_conv_solver.hpp"

// ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC (the MISA ASM-GTC backward-data solver)
// indexes global tensor memory with 32-bit element indices and does not implement
// large-tensor support. IsApplicable() therefore gates the solver off any problem
// whose flattened element count exceeds INT_MAX, so it cannot be selected for a
// shape it would silently compute incorrectly.
//
// The two cases below share geometry and differ only in batch size N, so the
// element count crosses the INT_MAX boundary and isolates the gate:
//   C*H*W per sample = 1024*162*92 = 15,261,696 elements
//     N=140 -> 2,136,637,440 <= INT_MAX (2,147,483,647): int32-safe, applicable
//     N=141 -> 2,151,899,136 >  INT_MAX               : gated off, not applicable

namespace {

auto GetInRangeConvCase()
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{miopenHalf, miopenTensorNHWC, {140, 1024, 162, 92}},
                        {miopenHalf, miopenTensorNHWC, {1024, 1024, 3, 3}},
                        miopenHalf,
                        {{1, 1}, {1, 1}, {1, 1}}};
}

auto GetOverInt32ConvCase()
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{miopenHalf, miopenTensorNHWC, {141, 1024, 162, 92}},
                        {miopenHalf, miopenTensorNHWC, {1024, 1024, 3, 3}},
                        miopenHalf,
                        {{1, 1}, {1, 1}, {1, 1}}};
}

// int32-safe shape: the solver keeps its usual device applicability (regression guard).
auto GetInRangeParams()
{
    auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X |
                                                          Gpu::gfx950);
    p.CheckXnackDisabled();
    return p;
}

// element count > INT_MAX: the solver must be applicable on no device (gated off).
auto GetGatedParams()
{
    auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::None);
    p.CheckXnackDisabled();
    return p;
}

} // namespace

using CPU_UnitTestConvSolverAsmGTCBwdNHWCLargeTensorDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(CPU_UnitTestConvSolverAsmGTCBwdNHWCLargeTensorDevApplicability_NONE,
       ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC{});
};

INSTANTIATE_TEST_SUITE_P(SmokeInt32Safe,
                         CPU_UnitTestConvSolverAsmGTCBwdNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetInRangeParams()),
                                          testing::Values(GetInRangeConvCase())));

INSTANTIATE_TEST_SUITE_P(SmokeOverInt32,
                         CPU_UnitTestConvSolverAsmGTCBwdNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetGatedParams()),
                                          testing::Values(GetOverInt32ConvCase())));
