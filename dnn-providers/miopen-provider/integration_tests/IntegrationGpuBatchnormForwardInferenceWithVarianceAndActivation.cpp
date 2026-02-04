// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/BatchnormCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_plugin::test_utilities;
using namespace test_bn_common;

namespace
{

template <typename DataType, typename IntermediateType, typename TestCaseType>
class BatchnormForwardInferenceWithVarianceAndActivation
    : public IntegrationGraphVerificationHarness<DataType, TestCaseType>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        const auto& [testCase, activeCase] = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormInferenceWithVarianceAndActivationTest");

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

        auto varianceAttr = graph::makeTensorAttributes(
            "variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto varianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(varianceAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        // Epsilon (pass-by-value)
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        epsilonTensorAttr->set_name("epsilon").set_value(1e-5);

        graph::BatchnormInferenceAttributesVarianceExt bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference_variance_ext(xTensorAttr,
                                                                     meanTensorAttr,
                                                                     varianceTensorAttr,
                                                                     scaleTensorAttr,
                                                                     biasTensorAttr,
                                                                     epsilonTensorAttr,
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

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        hip_bfloat16,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        hip_bfloat16,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        hip_bfloat16,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        hip_bfloat16,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
