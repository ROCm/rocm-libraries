// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
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

using PointwiseForwardTestCase
    = std::tuple<TensorLayout, PointwiseTestCase, test_activation_common::ActivTestCase>;

template <typename DataType>
class PointwiseForward
    : public IntegrationGraphVerificationHarness<DataType, PointwiseForwardTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const PointwiseForwardTestCase& tc)
    {
        const auto& [layout, pwTestCase, activTestCase] = tc;

        graph::Graph graphObj;
        graphObj.set_name("PointwiseForwardTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", pwTestCase.dims, generateStrides(pwTestCase.dims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        graph::PointwiseAttributes pwAttrs;
        pwAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activTestCase.mode));

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
        runGraphTest(1e-5f);
    }

    // Tolerance is a fixed per-dtype constant rather than resolved via
    // this->getTolerance(). That helper determines tolerance by walking the graph
    // for a "root op" other than PointwiseNode (so that, e.g., a Conv+ReLU fused
    // graph is toleranced based on Conv, not the fused activation). These tests
    // build graphs whose only node is the PointwiseNode itself, so no such root op
    // exists and getTolerance() would fail to resolve a tolerance. Once the harness
    // supports a standalone PointwiseNode as its own root, this should switch to
    // this->getTolerance(graphObj, outputs.y) like the other suites.
    void runGraphTest(float tolerance)
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, pwTestCase, activTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.y, tolerance);

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(activTestCase.note);
        this->verifyGraph(graphObj, pwTestCase.seed);
    }
};

using IntegrationGpuPointwiseForwardFp32 = PointwiseForward<float>;
using IntegrationGpuPointwiseForwardFp16 = PointwiseForward<half>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuPointwiseForwardFp32);
TEST_P(IntegrationGpuPointwiseForwardFp32, Correctness)
{
    runGraphTest(1e-5f);
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuPointwiseForwardFp16);
TEST_P(IntegrationGpuPointwiseForwardFp16, Correctness)
{
    runGraphTest(1e-3f);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuPointwiseForwardFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getPointwiseTestCases()),
                     testing::ValuesIn(test_activation_common::createPointwiseFwdSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuPointwiseForwardFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(getPointwiseTestCases()),
                     testing::ValuesIn(test_activation_common::createPointwiseFwdSmokeCases())));
