// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/ActivationCommon.hpp"
#include "common/RMSNormCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_rmsnorm_common;
using namespace test_activation_common;

namespace
{

using RMSNormActivationTestCaseType = std::tuple<TensorLayout, RMSNormTestCase, ActivTestCase>;

template <typename XType, typename ScaleType, typename YType>
class RMSNormActivation
    : public IntegrationGraphVerificationHarness<YType, RMSNormActivationTestCaseType>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> yActiv;
        std::shared_ptr<graph::TensorAttributes> invRms; // nullptr in inference mode
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const RMSNormActivationTestCaseType& tc)
    {
        auto& [layout, testCase, activTestCase] = tc;

        graph::Graph graphObj;
        graphObj.set_name("RMSNormActivationTest");

        const auto xType = getDataTypeEnumFromType<XType>();
        const auto scaleType = getDataTypeEnumFromType<ScaleType>();
        const auto yType = getDataTypeEnumFromType<YType>();
        const auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(yType);

        auto xAttr = graph::makeTensorAttributes(
            "x", xType, testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr
            = graph::makeTensorAttributes("scale",
                                          scaleType,
                                          testCase.scaleDims,
                                          generateStrides(testCase.scaleDims, layout.strideOrder));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        epsilonTensorAttr->set_name("epsilon").set_value(testCase.epsilon);

        graph::RMSNormAttributes rmsnormAttrs;
        rmsnormAttrs.set_epsilon(epsilonTensorAttr)
            .set_forward_phase(testCase.isTraining ? NormFwdPhase::TRAINING
                                                   : NormFwdPhase::INFERENCE);

        if(testCase.biasDims.has_value())
        {
            auto biasAttr = graph::makeTensorAttributes(
                "bias",
                scaleType,
                *testCase.biasDims,
                generateStrides(*testCase.biasDims, layout.strideOrder));
            auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));
            rmsnormAttrs.set_bias(biasTensorAttr);
        }

        auto [yTensorAttr, invRmsOut]
            = graphObj.rmsnorm(xTensorAttr, scaleTensorAttr, rmsnormAttrs);

        if(testCase.isTraining)
        {
            invRmsOut->set_output(true).set_data_type(intermediateDataType);
        }

        graph::PointwiseAttributes activAttrs;
        activAttrs.set_mode(sdkToFrontendPointwiseMode(activTestCase.mode));

        if(activTestCase.reluLowerClip.has_value())
        {
            activAttrs.set_relu_lower_clip(activTestCase.reluLowerClip.value());
        }
        if(activTestCase.reluUpperClip.has_value())
        {
            activAttrs.set_relu_upper_clip(activTestCase.reluUpperClip.value());
        }
        if(activTestCase.reluLowerClipSlope.has_value())
        {
            activAttrs.set_relu_lower_clip_slope(activTestCase.reluLowerClipSlope.value());
        }
        if(activTestCase.swishBeta.has_value())
        {
            activAttrs.set_swish_beta(activTestCase.swishBeta.value());
        }
        if(activTestCase.eluAlpha.has_value())
        {
            activAttrs.set_elu_alpha(activTestCase.eluAlpha.value());
        }
        if(activTestCase.softplusBeta.has_value())
        {
            activAttrs.set_softplus_beta(activTestCase.softplusBeta.value());
        }

        auto yActivOut = graphObj.pointwise(yTensorAttr, activAttrs);

        yActivOut->set_data_type(yType).set_output(true);

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

        return std::make_pair(std::move(graphObj), GraphOutputs{yActivOut, invRmsOut});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, rmsnormTestCase, activTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.yActiv, this->getTolerance(graphObj, outputs.yActiv));
        if(outputs.invRms)
        {
            this->registerValidator(outputs.invRms, this->getTolerance(graphObj, outputs.invRms));
        }

        this->inputFillRecipes().setGlobalSeed(rmsnormTestCase.seed);
        this->verifyGraph(graphObj);
    }
};

using IntegrationGpuRMSNormActivationPureFp32 = RMSNormActivation<float, float, float>;
using IntegrationGpuRMSNormActivationPureBfp16 = RMSNormActivation<bfloat16, bfloat16, bfloat16>;
using IntegrationGpuRMSNormActivationPureFp16 = RMSNormActivation<half, half, half>;

// "Mixed" = input (X) is lower precision while scale/output stay FP32, exercising the
// upcast IO path that the pure same-precision configs above don't cover.
using IntegrationGpuRMSNormActivationMixedFp16 = RMSNormActivation<half, float, float>;
using IntegrationGpuRMSNormActivationMixedBfp16 = RMSNormActivation<bfloat16, float, float>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormActivationPureFp32);
TEST_P(IntegrationGpuRMSNormActivationPureFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormActivationPureBfp16);
TEST_P(IntegrationGpuRMSNormActivationPureBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormActivationPureFp16);
TEST_P(IntegrationGpuRMSNormActivationPureFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormActivationPureFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormActivationPureBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormActivationPureFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormActivationPureFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormActivationPureBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormActivationPureFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormActivationMixedFp16);
TEST_P(IntegrationGpuRMSNormActivationMixedFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormActivationMixedBfp16);
TEST_P(IntegrationGpuRMSNormActivationMixedBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormActivationMixedFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormActivationMixedBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormActivationMixedFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormActivationMixedBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuRMSNormActivation3dFp32 = IntegrationGpuRMSNormActivationPureFp32;
using IntegrationGpuRMSNormActivation3dBfp16 = IntegrationGpuRMSNormActivationPureBfp16;
using IntegrationGpuRMSNormActivation3dFp16 = IntegrationGpuRMSNormActivationPureFp16;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormActivation3dFp32);
TEST_P(IntegrationGpuRMSNormActivation3dFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormActivation3dBfp16);
TEST_P(IntegrationGpuRMSNormActivation3dBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormActivation3dFp16);
TEST_P(IntegrationGpuRMSNormActivation3dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormActivation3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormActivation3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormActivation3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

using IntegrationGpuRMSNormActivation3dMixedFp16 = IntegrationGpuRMSNormActivationMixedFp16;
using IntegrationGpuRMSNormActivation3dMixedBfp16 = IntegrationGpuRMSNormActivationMixedBfp16;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormActivation3dMixedFp16);
TEST_P(IntegrationGpuRMSNormActivation3dMixedFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormActivation3dMixedBfp16);
TEST_P(IntegrationGpuRMSNormActivation3dMixedBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormActivation3dMixedFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormActivation3dMixedBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
