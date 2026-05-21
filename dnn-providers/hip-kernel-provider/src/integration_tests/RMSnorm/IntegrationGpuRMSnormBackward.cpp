// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "RMSnormCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::rmsnorm;
using namespace hip_kernel_provider::test_utilities;

namespace hip_kernel_provider::rmsnorm::test
{
using namespace common;

namespace
{

template <typename DyDataType,
          typename XDataType,
          typename ScaleDataType,
          typename DxDataType,
          typename ComputeDataType>
class RMSNormBackward : public IntegrationGraphVerificationHarness<XDataType, RMSnormTestCase>
{
protected:
    void runGraphTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        const RMSnormTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("RMSnormTest");

        auto dyDataType = getDataTypeEnumFromType<DyDataType>();
        auto xDataType = getDataTypeEnumFromType<XDataType>();
        auto computeDataType = getDataTypeEnumFromType<ComputeDataType>();
        graphObj.set_compute_data_type(computeDataType)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

        auto dyAttr = makeTensorAttributes("dy",
                                           dyDataType,
                                           testCase.ioDims,
                                           generateStrides(testCase.ioDims, layout.strideOrder));
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));
        auto xAttr = makeTensorAttributes(
            "x", xDataType, testCase.ioDims, generateStrides(testCase.ioDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto scaleDataType = getDataTypeEnumFromType<ScaleDataType>();
        auto scaleAttr
            = makeTensorAttributes("scale",
                                   scaleDataType,
                                   testCase.scaleDims,
                                   generateStrides(testCase.scaleDims, layout.strideOrder));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));
        auto invRmsDims = testCase.ioDims;
        for(size_t i = 1; i < invRmsDims.size(); ++i)
        {
            if(testCase.scaleDims[i] != 1)
            {
                invRmsDims[i] = 1;
            }
        }
        auto invRmsAttr = makeTensorAttributes("inv_rms",
                                               computeDataType,
                                               invRmsDims,
                                               generateStrides(invRmsDims, layout.strideOrder));
        auto invRmsTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(invRmsAttr));

        graph::RMSNormBackwardAttributes rmsnormBwdAttrs;
        rmsnormBwdAttrs.set_name("rmsnorm_bwd");
        rmsnormBwdAttrs.set_compute_data_type(computeDataType);
        rmsnormBwdAttrs.set_compute_dbias(true);

        auto [dxTensorAttr, dscaleTensorAttr, dbiasTensorAttr] = graphObj.rmsnorm_backward(
            dyTensorAttr, xTensorAttr, scaleTensorAttr, invRmsTensorAttr, rmsnormBwdAttrs);

        dxTensorAttr->set_output(true);
        auto dxDataType = getDataTypeEnumFromType<DxDataType>();
        dxTensorAttr->set_data_type(dxDataType);
        this->registerValidator(dxTensorAttr, getTolerance<DxDataType>());

        dscaleTensorAttr->set_output(true);
        dscaleTensorAttr->set_data_type(scaleDataType);
        this->registerValidator(dscaleTensorAttr, getTolerance<ScaleDataType>());

        dbiasTensorAttr->set_output(true);
        dbiasTensorAttr->set_data_type(scaleDataType);
        this->registerValidator(dbiasTensorAttr, getTolerance<ScaleDataType>());

        this->verifyGraph(graphObj, testCase.seed);
    }
};

// ============================================================================
// Test cases
// ============================================================================

// 1. DyDataType: FP32, XDataType: FP32, ScaleDataType: FP32, DxDataType: FP32, ComputeDataType: FP32
using IntegrationGpuRMSnormBackwardFp32Fp32Fp32Fp32Fp32
    = RMSNormBackward<float, float, float, float, float>;

// 2. DyDataType: FP16, XDataType: FP32, ScaleDataType: FP32, DxDataType: FP32, ComputeDataType: FP32
using IntegrationGpuRMSnormBackwardFp16Fp32Fp32Fp32Fp32
    = RMSNormBackward<half, float, float, float, float>;

// 3. DyDataType: BF16, XDataType: FP32, ScaleDataType: FP32, DxDataType: FP32, ComputeDataType: FP32
using IntegrationGpuRMSnormBackwardBf16Fp32Fp32Fp32Fp32
    = RMSNormBackward<bfloat16, float, float, float, float>;

// 4. DyDataType: FP16, XDataType: FP16, ScaleDataType: FP32, DxDataType: FP16, ComputeDataType: FP32
using IntegrationGpuRMSnormBackwardFp16Fp16Fp32Fp16Fp32
    = RMSNormBackward<half, half, float, half, float>;

// 5. DyDataType: BF16, XDataType: BF16, ScaleDataType: FP32, DxDataType: BF16, ComputeDataType: FP32
using IntegrationGpuRMSnormBackwardBf16Bf16Fp32Bf16Fp32
    = RMSNormBackward<bfloat16, bfloat16, float, bfloat16, float>;
}

// ============================================================================
// Test Registrations
// ============================================================================

// Register tests with different data types and layouts
#define REGISTER_RMS_TEST(TestCase)                                                               \
    /* --- NCHW --- */                                                                            \
    using TestCase##NCHW = TestCase;                                                              \
    TEST_P(TestCase##NCHW, Correctness)                                                           \
    {                                                                                             \
        runGraphTest(TensorLayout::NCHW);                                                         \
    }                                                                                             \
    INSTANTIATE_TEST_SUITE_P(Smoke, TestCase##NCHW, testing::ValuesIn(getRMSnormTestCases()));    \
    INSTANTIATE_TEST_SUITE_P(Full, TestCase##NCHW, testing::ValuesIn(getRMSnormFullTestCases())); \
    /* --- NHWC --- */                                                                            \
    using TestCase##NHWC = TestCase;                                                              \
    TEST_P(TestCase##NHWC, Correctness)                                                           \
    {                                                                                             \
        runGraphTest(TensorLayout::NHWC);                                                         \
    }                                                                                             \
    INSTANTIATE_TEST_SUITE_P(Smoke, TestCase##NHWC, testing::ValuesIn(getRMSnormTestCases()));    \
    INSTANTIATE_TEST_SUITE_P(Full, TestCase##NHWC, testing::ValuesIn(getRMSnormFullTestCases())); \
    /* --- NCDHW --- */                                                                           \
    using TestCase##NCDHW = TestCase;                                                             \
    TEST_P(TestCase##NCDHW, Correctness)                                                          \
    {                                                                                             \
        runGraphTest(TensorLayout::NCDHW);                                                        \
    }                                                                                             \
    INSTANTIATE_TEST_SUITE_P(Smoke, TestCase##NCDHW, testing::ValuesIn(getRMSnorm3dTestCases())); \
    /* --- NDHWC --- */                                                                           \
    using TestCase##NDHWC = TestCase;                                                             \
    TEST_P(TestCase##NDHWC, Correctness)                                                          \
    {                                                                                             \
        runGraphTest(TensorLayout::NDHWC);                                                        \
    }                                                                                             \
    INSTANTIATE_TEST_SUITE_P(Smoke, TestCase##NDHWC, testing::ValuesIn(getRMSnorm3dTestCases()));

REGISTER_RMS_TEST(IntegrationGpuRMSnormBackwardFp32Fp32Fp32Fp32Fp32);
REGISTER_RMS_TEST(IntegrationGpuRMSnormBackwardFp16Fp32Fp32Fp32Fp32);
REGISTER_RMS_TEST(IntegrationGpuRMSnormBackwardBf16Fp32Fp32Fp32Fp32);
REGISTER_RMS_TEST(IntegrationGpuRMSnormBackwardFp16Fp16Fp32Fp16Fp32);
REGISTER_RMS_TEST(IntegrationGpuRMSnormBackwardBf16Bf16Fp32Bf16Fp32);

} // namespace hip_kernel_provider::rmsnorm::test
