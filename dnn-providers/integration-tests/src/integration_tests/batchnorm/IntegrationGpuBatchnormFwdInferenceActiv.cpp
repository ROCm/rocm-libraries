// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/ActivationCommon.hpp"
#include "common/BatchnormCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_bn_common;

namespace
{

using BnFwdInfActivTestCase = std::
    tuple<TensorLayout, test_bn_common::BatchnormTestCase, test_activation_common::ActivTestCase>;

template <typename DataType>
class BatchnormFwdInferenceActiv
    : public IntegrationGraphVerificationHarness<DataType, BnFwdInfActivTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> out;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const BnFwdInfActivTestCase& tc)
    {
        const auto& [layout, testCase, activTestCase] = tc;

        auto derivedDims = getDerivedShape(testCase.dims);

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;

        graph::Graph graphObj;
        graphObj.set_name("BatchnormFwd+ActivTest");
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
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

        const graph::BatchnormInferenceAttributes bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                        meanTensorAttr,
                                                        invVarianceTensorAttr,
                                                        scaleTensorAttr,
                                                        biasTensorAttr,
                                                        bnAttrs);
        yTensorAttr->set_data_type(intermediateDataType);

        graph::PointwiseAttributes pointwiseAttrs;
        pointwiseAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activTestCase.mode));
        if(activTestCase.reluLowerClip.has_value())
        {
            pointwiseAttrs.set_relu_lower_clip(activTestCase.reluLowerClip.value());
        }
        if(activTestCase.reluUpperClip.has_value())
        {
            pointwiseAttrs.set_relu_upper_clip(activTestCase.reluUpperClip.value());
        }
        if(activTestCase.reluLowerClipSlope.has_value())
        {
            pointwiseAttrs.set_relu_lower_clip_slope(activTestCase.reluLowerClipSlope.value());
        }
        if(activTestCase.swishBeta.has_value())
        {
            pointwiseAttrs.set_swish_beta(activTestCase.swishBeta.value());
        }
        if(activTestCase.eluAlpha.has_value())
        {
            pointwiseAttrs.set_elu_alpha(activTestCase.eluAlpha.value());
        }
        if(activTestCase.softplusBeta.has_value())
        {
            pointwiseAttrs.set_softplus_beta(activTestCase.softplusBeta.value());
        }

        auto outTensorAttr = graphObj.pointwise(yTensorAttr, pointwiseAttrs);
        outTensorAttr->set_output(true);

        auto validateResult = graphObj.validate();
        if(validateResult.is_bad())
        {
            throw std::runtime_error("Failed to validate graph: " + validateResult.get_message());
        }

        auto buildResult = graphObj.build_operation_graph(handle);
        if(buildResult.is_bad())
        {
            throw std::runtime_error("Failed to build operation graph: "
                                     + buildResult.get_message());
        }

        return {std::move(graphObj), GraphOutputs{outTensorAttr}};
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, bnTestCase, activTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.out, this->getTolerance(graphObj, outputs.out));

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(bnTestCase.note);
        this->verifyGraph(graphObj, bnTestCase.seed);
    }
};

// 1D layout tests (NCL, NLC)
using IntegrationGpuBatchnormFwdInferenceActiv1dFp32 = BatchnormFwdInferenceActiv<float>;
using IntegrationGpuBatchnormFwdInferenceActiv1dBfp16 = BatchnormFwdInferenceActiv<bfloat16>;
using IntegrationGpuBatchnormFwdInferenceActiv1dFp16 = BatchnormFwdInferenceActiv<half>;

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuBatchnormFwdInferenceActiv2dFp32 = BatchnormFwdInferenceActiv<float>;
using IntegrationGpuBatchnormFwdInferenceActiv2dBfp16 = BatchnormFwdInferenceActiv<bfloat16>;
using IntegrationGpuBatchnormFwdInferenceActiv2dFp16 = BatchnormFwdInferenceActiv<half>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuBatchnormFwdInferenceActiv3dFp32 = BatchnormFwdInferenceActiv<float>;
using IntegrationGpuBatchnormFwdInferenceActiv3dBfp16 = BatchnormFwdInferenceActiv<bfloat16>;
using IntegrationGpuBatchnormFwdInferenceActiv3dFp16 = BatchnormFwdInferenceActiv<half>;

} // namespace

// ============================================================================
// 1D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceActiv1dFp32);
TEST_P(IntegrationGpuBatchnormFwdInferenceActiv1dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceActiv1dFp32,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceActiv1dFp32,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceActiv1dBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceActiv1dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceActiv1dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceActiv1dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceActiv1dFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceActiv1dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceActiv1dFp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceActiv1dFp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// ============================================================================
// 2D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceActiv2dFp32);
TEST_P(IntegrationGpuBatchnormFwdInferenceActiv2dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceActiv2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceActiv2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceActiv2dBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceActiv2dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceActiv2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceActiv2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceActiv2dFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceActiv2dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceActiv2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceActiv2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// ============================================================================
// 3D Tests (Smoke only)
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceActiv3dFp32);
TEST_P(IntegrationGpuBatchnormFwdInferenceActiv3dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceActiv3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceActiv3dBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceActiv3dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceActiv3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceActiv3dFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceActiv3dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceActiv3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
