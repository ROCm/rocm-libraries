// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../Common/ActivationCommon.hpp"
#include "../IntegrationGraphVerificationHarness.hpp"
#include "BatchnormCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::batchnorm;
using namespace hip_kernel_provider::test_utilities;
using namespace hip_kernel_provider::test_activation_common;

namespace hip_kernel_provider::batchnorm::test
{

using namespace common;

namespace
{

template <typename DataType, typename IntermediateType, typename TestCaseType>
class BatchnormForwardInferenceAndActivation
    : public IntegrationGraphVerificationHarness<DataType, TestCaseType>
{
protected:
    void runGraphTest(float tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        const auto& [testCase, activeCase] = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormInferenceAndActivationTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "X", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto meanAttr = graph::makeTensorAttributes(
            "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = graph::makeTensorAttributes(
            "inv_variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        graph::BatchnormInferenceAttributes bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                        meanTensorAttr,
                                                        invVarianceTensorAttr,
                                                        scaleTensorAttr,
                                                        biasTensorAttr,
                                                        bnAttrs);

        yTensorAttr->set_data_type(intermediateDataType);

        graph::PointwiseAttributes pointwiseAttrs;
        pointwiseAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activeCase.mode));
        if(activeCase.reluLowerClip.has_value())
        {
            pointwiseAttrs.set_relu_lower_clip(activeCase.reluLowerClip.value());
        }
        if(activeCase.reluUpperClip.has_value())
        {
            pointwiseAttrs.set_relu_upper_clip(activeCase.reluUpperClip.value());
        }
        if(activeCase.reluLowerClipSlope.has_value())
        {
            pointwiseAttrs.set_relu_lower_clip_slope(activeCase.reluLowerClipSlope.value());
        }
        if(activeCase.swishBeta.has_value())
        {
            pointwiseAttrs.set_swish_beta(activeCase.swishBeta.value());
        }
        if(activeCase.eluAlpha.has_value())
        {
            pointwiseAttrs.set_elu_alpha(activeCase.eluAlpha.value());
        }
        if(activeCase.softplusBeta.has_value())
        {
            pointwiseAttrs.set_softplus_beta(activeCase.softplusBeta.value());
        }

        auto outTensorAttr = graphObj.pointwise(yTensorAttr, pointwiseAttrs);
        outTensorAttr->set_output(true);

        this->registerValidator(outTensorAttr, tolerance);

        this->verifyGraph(graphObj, testCase.seed);
    }
};

// NCHW layouts
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32
    = BatchnormForwardInferenceAndActivation<float,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16
    = BatchnormForwardInferenceAndActivation<half,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

// NHWC layouts
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32
    = BatchnormForwardInferenceAndActivation<float,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16
    = BatchnormForwardInferenceAndActivation<half,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

// 5D layouts
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp32
    = BatchnormForwardInferenceAndActivation<float,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16
    = BatchnormForwardInferenceAndActivation<half,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp32
    = BatchnormForwardInferenceAndActivation<float,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16
    = BatchnormForwardInferenceAndActivation<half,
                                             float,
                                             std::tuple<BatchnormTestCase, ActivTestCase>>;

} // namespace

// ============================================================================
// NCHW FP32
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32, Correctness)
{
    runGraphTest(getToleranceInference<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NCHW BFP16
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16, Correctness)
{
    runGraphTest(getToleranceInference<bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NCHW FP16
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16, Correctness)
{
    runGraphTest(getToleranceInference<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NHWC FP32
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32, Correctness)
{
    runGraphTest(getToleranceInference<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NHWC BFP16
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16, Correctness)
{
    runGraphTest(getToleranceInference<bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NHWC FP16
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16, Correctness)
{
    runGraphTest(getToleranceInference<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NCDHW FP32 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp32, Correctness)
{
    runGraphTest(getToleranceInference<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

// ============================================================================
// NCDHW BFP16 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16, Correctness)
{
    runGraphTest(getToleranceInference<bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

// ============================================================================
// NCDHW FP16 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16, Correctness)
{
    runGraphTest(getToleranceInference<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

// ============================================================================
// NDHWC FP32 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp32, Correctness)
{
    runGraphTest(getToleranceInference<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

// ============================================================================
// NDHWC BFP16 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16, Correctness)
{
    runGraphTest(getToleranceInference<bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

// ============================================================================
// NDHWC FP16 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16, Correctness)
{
    runGraphTest(getToleranceInference<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

} // namespace hip_kernel_provider::batchnorm::test
