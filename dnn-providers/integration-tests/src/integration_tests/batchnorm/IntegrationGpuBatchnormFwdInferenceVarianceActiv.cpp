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

struct BnInfVarActivTensorIds
{
    static constexpr int64_t X_UID = 1;
    static constexpr int64_t MEAN_UID = 2;
    static constexpr int64_t VARIANCE_UID = 3;
    static constexpr int64_t SCALE_UID = 4;
    static constexpr int64_t BIAS_UID = 5;
};

using BnFwdInfVarActivTestCase = std::
    tuple<TensorLayout, test_bn_common::BatchnormTestCase, test_activation_common::ActivTestCase>;

template <typename DataType>
class BatchnormFwdInferenceVarianceActiv
    : public IntegrationGraphVerificationHarness<DataType, BnFwdInfVarActivTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> out;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const BnFwdInfVarActivTestCase& tc)
    {
        const auto& [layout, testCase, activTestCase] = tc;

        auto derivedDims = getDerivedShape(testCase.dims);

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;

        graph::Graph graphObj;
        graphObj.set_name("BatchnormInferenceWithVarianceAndActivationTest");
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "X", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        xAttr.set_uid(BnInfVarActivTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto meanAttr = graph::makeTensorAttributes(
            "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
        meanAttr.set_uid(BnInfVarActivTensorIds::MEAN_UID);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto varianceAttr = graph::makeTensorAttributes(
            "variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
        varianceAttr.set_uid(BnInfVarActivTensorIds::VARIANCE_UID);
        auto varianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(varianceAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        scaleAttr.set_uid(BnInfVarActivTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        biasAttr.set_uid(BnInfVarActivTensorIds::BIAS_UID);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        // Epsilon (pass-by-value)
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        epsilonTensorAttr->set_name("epsilon").set_value(1e-5);

        const graph::BatchnormInferenceAttributesVarianceExt bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference_variance_ext(xTensorAttr,
                                                                     meanTensorAttr,
                                                                     varianceTensorAttr,
                                                                     scaleTensorAttr,
                                                                     biasTensorAttr,
                                                                     epsilonTensorAttr,
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
    void initializeBundle([[maybe_unused]] const graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.tensors.at(BnInfVarActivTensorIds::X_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
        bundle.tensors.at(BnInfVarActivTensorIds::MEAN_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
        // Variance must be non-negative; use positive range
        bundle.tensors.at(BnInfVarActivTensorIds::VARIANCE_UID)
            ->fillTensorWithRandomValues(0.1f, 1.0f, seed);
        bundle.tensors.at(BnInfVarActivTensorIds::SCALE_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
        bundle.tensors.at(BnInfVarActivTensorIds::BIAS_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
    }

    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, bnTestCase, activTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.out, this->getTolerance(graphObj, outputs.out));

        this->verifyGraph(graphObj, bnTestCase.seed);
    }
};

// 1D layout tests (NCL, NLC)
using IntegrationGpuBatchnormFwdInferenceVarianceActiv1dFp32
    = BatchnormFwdInferenceVarianceActiv<float>;
using IntegrationGpuBatchnormFwdInferenceVarianceActiv1dBfp16
    = BatchnormFwdInferenceVarianceActiv<bfloat16>;
using IntegrationGpuBatchnormFwdInferenceVarianceActiv1dFp16
    = BatchnormFwdInferenceVarianceActiv<half>;

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuBatchnormFwdInferenceVarianceActiv2dFp32
    = BatchnormFwdInferenceVarianceActiv<float>;
using IntegrationGpuBatchnormFwdInferenceVarianceActiv2dBfp16
    = BatchnormFwdInferenceVarianceActiv<bfloat16>;
using IntegrationGpuBatchnormFwdInferenceVarianceActiv2dFp16
    = BatchnormFwdInferenceVarianceActiv<half>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuBatchnormFwdInferenceVarianceActiv3dFp32
    = BatchnormFwdInferenceVarianceActiv<float>;
using IntegrationGpuBatchnormFwdInferenceVarianceActiv3dBfp16
    = BatchnormFwdInferenceVarianceActiv<bfloat16>;
using IntegrationGpuBatchnormFwdInferenceVarianceActiv3dFp16
    = BatchnormFwdInferenceVarianceActiv<half>;

} // namespace

// ============================================================================
// 1D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVarianceActiv1dFp32);
TEST_P(IntegrationGpuBatchnormFwdInferenceVarianceActiv1dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv1dFp32,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv1dFp32,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVarianceActiv1dBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVarianceActiv1dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv1dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv1dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVarianceActiv1dFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVarianceActiv1dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv1dFp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv1dFp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// ============================================================================
// 2D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVarianceActiv2dFp32);
TEST_P(IntegrationGpuBatchnormFwdInferenceVarianceActiv2dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVarianceActiv2dBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVarianceActiv2dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVarianceActiv2dFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVarianceActiv2dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// ============================================================================
// 3D Tests (Smoke only)
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVarianceActiv3dFp32);
TEST_P(IntegrationGpuBatchnormFwdInferenceVarianceActiv3dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVarianceActiv3dBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVarianceActiv3dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVarianceActiv3dFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVarianceActiv3dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdInferenceVarianceActiv3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
