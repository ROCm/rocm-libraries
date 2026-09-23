// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/ActivationCommon.hpp"
#include "common/PointwiseCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_pointwise_common;

namespace
{

using ActivationForwardTestCase
    = std::tuple<TensorLayout, PointwiseTestCase, test_activation_common::ActivTestCase>;

template <typename DataType>
class ActivationForward
    : public IntegrationGraphVerificationHarness<DataType, ActivationForwardTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const ActivationForwardTestCase& tc)
    {
        const auto& [layout, pwTestCase, activTestCase] = tc;

        graph::Graph graphObj;
        graphObj.set_name("ActivationForwardTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", pwTestCase.dims, generateStrides(pwTestCase.dims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        graph::PointwiseAttributes pwAttrs;
        pwAttrs.set_mode(sdkToFrontendPointwiseMode(activTestCase.mode));

        if(activTestCase.reluLowerClip.has_value())
        {
            pwAttrs.set_relu_lower_clip(activTestCase.reluLowerClip.value());
        }
        if(activTestCase.reluUpperClip.has_value())
        {
            pwAttrs.set_relu_upper_clip(activTestCase.reluUpperClip.value());
        }
        if(activTestCase.reluLowerClipSlope.has_value())
        {
            pwAttrs.set_relu_lower_clip_slope(activTestCase.reluLowerClipSlope.value());
        }

        auto yTensorAttr = graphObj.pointwise(xTensorAttr, pwAttrs);
        yTensorAttr->set_output(true);

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

        return std::make_pair(std::move(graphObj), GraphOutputs{yTensorAttr});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, pwTestCase, activTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.y, this->getTolerance(graphObj, outputs.y));

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(activTestCase.note);
        this->inputFillRecipes().setGlobalSeed(pwTestCase.seed);
        this->verifyGraph(graphObj);
    }
};

using IntegrationGpuActivationForwardFp32 = ActivationForward<float>;
using IntegrationGpuActivationForwardFp16 = ActivationForward<half>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuActivationForwardFp32);
TEST_P(IntegrationGpuActivationForwardFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuActivationForwardFp16);
TEST_P(IntegrationGpuActivationForwardFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuActivationForwardFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getPointwiseTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdUnaryActivationCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuActivationForwardFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getPointwiseTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdUnaryActivationCases())));
