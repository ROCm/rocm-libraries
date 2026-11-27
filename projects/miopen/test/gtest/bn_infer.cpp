/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "bn.hpp"

// XDataType       : half_float::half
// YDataYype       : half_float::half
// ScaleDataType   : half_float::half
// BiasDataType    : half_float::half
// MeanVarDataType : float
// AccDataType     : double
struct GPU_BNCKInferLarge2D_FP16 : BNInferTest<half_float::half,
                                               half_float::half,
                                               half_float::half,
                                               half_float::half,
                                               float,
                                               double,
                                               BN2DTestCase>
{
};

struct GPU_BNOCLInferLarge2D_FP16
    : BNInferTest<half_float::half, half_float::half, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLInferLarge3D_FP16
    : BNInferTest<half_float::half, half_float::half, float, float, float, double, BN3DTestCase>
{
};

// Forward inference maps spatial positions to Y workgroups. On gfx12, the 16-bit Y workgroup
// ID limits the grid, so larger shapes require the grid-stride variant.
struct GPU_BNOCLInferYWorkgroupLimit2D_FP16
    : BNInferTest<half_float::half, half_float::half, float, float, float, double, BN2DTestCase>
{
};

// XDataType       : bfloat16
// YDataYype       : bfloat16
// ScaleDataType   : bfloat16
// BiasDataType    : bfloat16
// MeanVarDataType : float
struct GPU_BNCKInferLarge2D_BFP16
    : BNInferTest<bfloat16, bfloat16, bfloat16, bfloat16, float, double, BN2DTestCase>
{
};

// XDataType       : bfloat16
// YDataYype       : bfloat16
// ScaleDataType   : float
// BiasDataType    : float
// MeanVarDataType : float
struct GPU_BNOCLInferLarge2D_BFP16
    : BNInferTest<bfloat16, bfloat16, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLInferLarge3D_BFP16
    : BNInferTest<bfloat16, bfloat16, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNInferSmall2D_FP32
    : BNInferTest<float, float, float, float, float, double, BN2DTestCase>
{
};
struct GPU_BNInferSmall3D_FP32
    : BNInferTest<float, float, float, float, float, double, BN3DTestCase>
{
};
struct GPU_BNInferLarge2D_FP32
    : BNInferTest<float, float, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNInferSmall2D_FP64
    : BNInferTest<double, double, double, double, double, double, BN2DTestCase>
{
};

struct GPU_BNInferLarge2D_FP64
    : BNInferTest<double, double, double, double, double, double, BN2DTestCase>
{
};

// fp16
TEST_P(GPU_BNCKInferLarge2D_FP16, DISABLED_BnV2LargeInferCKfp16_2D) {}
TEST_P(GPU_BNOCLInferLarge2D_FP16, BnV2LargeInferOCLfp16_2D) {}
TEST_P(GPU_BNOCLInferLarge3D_FP16, BnV2LargeInferOCLfp16_3D) {}
TEST_P(GPU_BNOCLInferYWorkgroupLimit2D_FP16, BnV2InferYWorkgroupLimitOCLfp16_2D) {}

// bfp16
TEST_P(GPU_BNCKInferLarge2D_BFP16, DISABLED_BnV2LargeInferCKbfp16_2D) {}
TEST_P(GPU_BNOCLInferLarge2D_BFP16, BnV2LargeInferOCLbfp16_2D) {}
TEST_P(GPU_BNOCLInferLarge3D_BFP16, BnV2LargeInferOCLbfp16_3D) {}

// fp32 (float)
TEST_P(GPU_BNInferSmall2D_FP32, BnV1SmallInferfp32_2D) {}
TEST_P(GPU_BNInferLarge2D_FP32, BnV2LargeInferfp32_2D) {}
TEST_P(GPU_BNInferSmall3D_FP32, BnV1SmallInferfp32_3D) {}

// fp64
TEST_P(GPU_BNInferSmall2D_FP64, DISABLED_BnV1SmallInferfp64_2D) {}
TEST_P(GPU_BNInferLarge2D_FP64, DISABLED_BnV2LargeInferfp64_2D) {}

// fp16
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BNCKInferLarge2D_FP16,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2, testBNAPIInvVar}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    GPU_BNOCLInferLarge2D_FP16,
    testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNSpatial, miopenBNPerActivation}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    GPU_BNOCLInferLarge3D_FP16,
    testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                     testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                     testing::ValuesIn({miopenBNSpatial, miopenBNPerActivation}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN3DTestCase>());

// When the uncapped Y workgroup count exceeds the device limit (65536 for gfx120X, 65535 for
// gfx125X; no limit elsewhere), the solver enables MIO_BN_Y_STRIDE and caps the count at 65535.
// NHWC with C = 257 has vector size 1, xlocalsize = 256 and ylocalsize = 1, so the uncapped
// count is H*W: 255x255 (65025) is a control, 256x256 (65536) strides only on gfx125X, and
// 300x300 (90000) strides on both families. The N = 2 case checks the batch loop at the same
// Y workgroup count. NHWC with C = 2 has vector size 2 and ylocalsize = 256, so 4100x4100
// requires ceil(16810000/256) = 65665 Y workgroups and also strides on both families.
INSTANTIATE_TEST_SUITE_P(
    Standard,
    GPU_BNOCLInferYWorkgroupLimit2D_FP16,
    testing::Combine(testing::ValuesIn(std::vector<BN2DTestCase>{
                         {1, 257, 255, 255, miopen::batchnorm::Direction::ForwardInference, 1, 0},
                         {1, 257, 256, 256, miopen::batchnorm::Direction::ForwardInference, 1, 0},
                         {1, 257, 300, 300, miopen::batchnorm::Direction::ForwardInference, 1, 0},
                         {2, 257, 300, 300, miopen::batchnorm::Direction::ForwardInference, 1, 0},
                         {1, 2, 4100, 4100, miopen::batchnorm::Direction::ForwardInference, 1, 0}}),
                     testing::ValuesIn({miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNSpatial}),
                     testing::ValuesIn({testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

// Per-activation mode reloads its C*H*W parameters at each position in the stride loop. With
// C = 257, 300x300 (90000) and 384x384 (147456) both exceed the gfx12 limits. The N = 2
// case uses 300x300 to cover the batch loop with 23130000 elements per parameter tensor.
INSTANTIATE_TEST_SUITE_P(
    StandardPerActivation,
    GPU_BNOCLInferYWorkgroupLimit2D_FP16,
    testing::Combine(testing::ValuesIn(std::vector<BN2DTestCase>{
                         {1, 257, 300, 300, miopen::batchnorm::Direction::ForwardInference, 1, 0},
                         {2, 257, 300, 300, miopen::batchnorm::Direction::ForwardInference, 1, 0},
                         {1, 257, 384, 384, miopen::batchnorm::Direction::ForwardInference, 1, 0}}),
                     testing::ValuesIn({miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNPerActivation}),
                     testing::ValuesIn({testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

// NCHW with 4097x4097 has odd H*W (16785409), so the vector size is 1. With ylocalsize = 256,
// ceil(16785409/256) = 65569 Y workgroups exceeds both gfx12 limits. C = 1 keeps the
// per-activation parameter tensors to 16785409 elements each.
INSTANTIATE_TEST_SUITE_P(
    StandardNchw,
    GPU_BNOCLInferYWorkgroupLimit2D_FP16,
    testing::Combine(testing::Values(BN2DTestCase{
                         1, 1, 4097, 4097, miopen::batchnorm::Direction::ForwardInference, 1, 0}),
                     testing::ValuesIn({miopenTensorNCHW}),
                     testing::ValuesIn({miopenBNSpatial, miopenBNPerActivation}),
                     testing::ValuesIn({testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

// bfp16
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BNCKInferLarge2D_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2, testBNAPIInvVar}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_BNOCLInferLarge2D_BFP16,
    testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNSpatial, miopenBNPerActivation}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    GPU_BNOCLInferLarge3D_BFP16,
    testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                     testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                     testing::ValuesIn({miopenBNSpatial, miopenBNPerActivation}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN3DTestCase>());

// fp32
INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_BNInferSmall2D_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         GPU_BNInferLarge2D_FP32,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2, testBNAPIInvVar}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_BNInferSmall3D_FP32,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2, testBNAPIInvVar}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());
// fp64
INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_BNInferSmall2D_FP64,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BNInferLarge2D_FP64,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2, testBNAPIInvVar}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());
