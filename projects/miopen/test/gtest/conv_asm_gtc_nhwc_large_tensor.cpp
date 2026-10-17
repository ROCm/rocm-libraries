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

// The ASM-GTC NHWC solvers ConvAsmImplicitGemmGTCDynamic{Fwd,Bwd,Wrw}XdlopsNHWC index
// global tensor memory with 32-bit BYTE offsets. The bound is on bytes, not elements:
// for fp16 the effective element ceiling is half the byte ceiling.
//
// Forward and backward-data address a full 4 GiB window per dispatch slice and advance
// the base pointer per batch-split, so activations of any size are handled by the split
// mechanism. Only the per-group weight size has to fit the window. Backward-weights never
// splits, because 292 of its 900 kernels ignore the batch-split dimension
// (ROCm/rocm-libraries#11805). On gfx950 its FP32 activations may use the whole 4 GiB
// window; FP16/BF16 and other devices keep the int32 bound.
//
// The first two cases share geometry and differ only in batch size N, so the byte
// count crosses the INT_MAX boundary and isolates the gate. The gate checks the
// (direction-independent) in/out/weights descriptors; for this shape both the input and
// output tensors have C*H*W per sample = 1024*162*92 = 15,261,696 elements, and fp16
// doubles that to bytes:
//     N=70 -> 2,136,637,440 bytes <= INT_MAX (2,147,483,647)
//     N=71 -> 2,167,160,832 bytes >  INT_MAX
// Backward-weights (fp16) stays gated at N=71 on every device; forward and backward-data stay
// applicable, since the weights here are only 18 MiB. GetWeightsOverBoundConvCase covers the
// bound that does still gate forward and backward-data: 16384x16384x3x3 fp16 weights are
// 4.50 GiB per group, over the 4 GiB window, while its activations stay small so the weight
// term is isolated.
//
// The fp32 pair crosses 0xffffffff with the same per-sample geometry (61,046,784 bytes):
//     N=64 -> 3,906,994,176 bytes, inside the window
//     N=71 -> 4,334,321,664 bytes, over it
//
// The backward-weights batch-padding cases cover the hazard behind ROCM-31718. The fp16
// kernels with filter padding read batch rows in packs of 4 or 8 and rely on the buffer range
// check to zero the rows past N, but compute the row offsets in 32 bits. With N=1 and a
// 2048x2048 image of C=K=248 channels (2,080,374,784 bytes per tensor, under INT_MAX) the
// padding rows start at 1x, 2x and 3x that size. Rows 2 and 3 run past 4 GiB, wrap, and read
// real data, so this shape is now gated on every device although it is under the old bound.
// The C=K=256 case is the 2 GiB shape that failed GPU verification. With N=4 and a 1024x1024
// image there are no padding rows and every read stays below 4 GiB.

namespace {

auto GetInRangeConvCase()
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{miopenHalf, miopenTensorNHWC, {70, 1024, 162, 92}},
                        {miopenHalf, miopenTensorNHWC, {1024, 1024, 3, 3}},
                        miopenHalf,
                        {{1, 1}, {1, 1}, {1, 1}}};
}

auto GetOverInt32ConvCase()
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{miopenHalf, miopenTensorNHWC, {71, 1024, 162, 92}},
                        {miopenHalf, miopenTensorNHWC, {1024, 1024, 3, 3}},
                        miopenHalf,
                        {{1, 1}, {1, 1}, {1, 1}}};
}

auto GetWeightsOverBoundConvCase()
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{miopenHalf, miopenTensorNHWC, {1, 16384, 8, 8}},
                        {miopenHalf, miopenTensorNHWC, {16384, 16384, 3, 3}},
                        miopenHalf,
                        {{0, 0}, {1, 1}, {1, 1}}};
}

auto GetFp32InWindowConvCase()
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{miopenFloat, miopenTensorNHWC, {64, 1024, 162, 92}},
                        {miopenFloat, miopenTensorNHWC, {1024, 1024, 3, 3}},
                        miopenFloat,
                        {{1, 1}, {1, 1}, {1, 1}}};
}

auto GetFp32OverWindowConvCase()
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{miopenFloat, miopenTensorNHWC, {71, 1024, 162, 92}},
                        {miopenFloat, miopenTensorNHWC, {1024, 1024, 3, 3}},
                        miopenFloat,
                        {{1, 1}, {1, 1}, {1, 1}}};
}

auto GetBatchPaddingWrapsUnderInt32ConvCase()
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{miopenHalf, miopenTensorNHWC, {1, 248, 2048, 2048}},
                        {miopenHalf, miopenTensorNHWC, {248, 248, 3, 3}},
                        miopenHalf,
                        {{1, 1}, {1, 1}, {1, 1}}};
}

auto GetBatchPaddingWrapsInWindowConvCase()
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{miopenHalf, miopenTensorNHWC, {1, 256, 2048, 2048}},
                        {miopenHalf, miopenTensorNHWC, {256, 256, 3, 3}},
                        miopenHalf,
                        {{1, 1}, {1, 1}, {1, 1}}};
}

auto GetBatchPaddingFitsConvCase()
{
    using miopen::unit_tests::ConvTestCase;
    return ConvTestCase{{miopenHalf, miopenTensorNHWC, {4, 248, 1024, 1024}},
                        {miopenHalf, miopenTensorNHWC, {248, 248, 3, 3}},
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

// byte count > INT_MAX: the solver must be applicable on no device (gated off).
auto GetGatedParams()
{
    auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::None);
    p.CheckXnackDisabled();
    return p;
}

// within the 4 GiB window but above INT_MAX: backward-weights is applicable on gfx950 only.
auto GetGfx950WindowParams()
{
    auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx950);
    p.CheckXnackDisabled();
    return p;
}

} // namespace

// --- Forward -------------------------------------------------------------------------

using CPU_UnitTestConvSolverAsmGTCFwdNHWCLargeTensorDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(CPU_UnitTestConvSolverAsmGTCFwdNHWCLargeTensorDevApplicability_NONE,
       ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC{});
};

INSTANTIATE_TEST_SUITE_P(SmokeInt32Safe,
                         CPU_UnitTestConvSolverAsmGTCFwdNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetInRangeParams()),
                                          testing::Values(GetInRangeConvCase())));

INSTANTIATE_TEST_SUITE_P(SmokeActivationsOverInt32,
                         CPU_UnitTestConvSolverAsmGTCFwdNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetInRangeParams()),
                                          testing::Values(GetOverInt32ConvCase())));

INSTANTIATE_TEST_SUITE_P(SmokeWeightsOverBound,
                         CPU_UnitTestConvSolverAsmGTCFwdNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetGatedParams()),
                                          testing::Values(GetWeightsOverBoundConvCase())));

// --- Backward-data -------------------------------------------------------------------

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

INSTANTIATE_TEST_SUITE_P(SmokeActivationsOverInt32,
                         CPU_UnitTestConvSolverAsmGTCBwdNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetInRangeParams()),
                                          testing::Values(GetOverInt32ConvCase())));

INSTANTIATE_TEST_SUITE_P(SmokeWeightsOverBound,
                         CPU_UnitTestConvSolverAsmGTCBwdNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetGatedParams()),
                                          testing::Values(GetWeightsOverBoundConvCase())));

// --- Backward-weights ----------------------------------------------------------------

using CPU_UnitTestConvSolverAsmGTCWrwNHWCLargeTensorDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(CPU_UnitTestConvSolverAsmGTCWrwNHWCLargeTensorDevApplicability_NONE,
       ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC{});
};

INSTANTIATE_TEST_SUITE_P(SmokeInt32Safe,
                         CPU_UnitTestConvSolverAsmGTCWrwNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetInRangeParams()),
                                          testing::Values(GetInRangeConvCase())));

INSTANTIATE_TEST_SUITE_P(SmokeOverInt32,
                         CPU_UnitTestConvSolverAsmGTCWrwNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetGatedParams()),
                                          testing::Values(GetOverInt32ConvCase())));

INSTANTIATE_TEST_SUITE_P(SmokeFp32InWindow,
                         CPU_UnitTestConvSolverAsmGTCWrwNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetGfx950WindowParams()),
                                          testing::Values(GetFp32InWindowConvCase())));

INSTANTIATE_TEST_SUITE_P(SmokeFp32OverWindow,
                         CPU_UnitTestConvSolverAsmGTCWrwNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetGatedParams()),
                                          testing::Values(GetFp32OverWindowConvCase())));

INSTANTIATE_TEST_SUITE_P(
    SmokeBatchPaddingWrapsUnderInt32,
    CPU_UnitTestConvSolverAsmGTCWrwNHWCLargeTensorDevApplicability_NONE,
    testing::Combine(testing::Values(GetGatedParams()),
                     testing::Values(GetBatchPaddingWrapsUnderInt32ConvCase())));

INSTANTIATE_TEST_SUITE_P(SmokeBatchPaddingWrapsInWindow,
                         CPU_UnitTestConvSolverAsmGTCWrwNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetGatedParams()),
                                          testing::Values(GetBatchPaddingWrapsInWindowConvCase())));

INSTANTIATE_TEST_SUITE_P(SmokeBatchPaddingFits,
                         CPU_UnitTestConvSolverAsmGTCWrwNHWCLargeTensorDevApplicability_NONE,
                         testing::Combine(testing::Values(GetInRangeParams()),
                                          testing::Values(GetBatchPaddingFitsConvCase())));
