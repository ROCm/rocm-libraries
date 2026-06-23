// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/ConvolutionCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_conv_common;

namespace
{

using ConvFwdTestCase = std::tuple<TensorLayout, test_conv_common::ConvTestCase>;

template <typename DataType>
class ConvForward : public IntegrationGraphVerificationHarness<DataType, ConvFwdTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const ConvFwdTestCase& tc)
    {
        const auto& [layout, testCase] = tc;

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ConvolutionForwardTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto wAttr = graph::makeTensorAttributes(
            "w", testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto yAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);
        yAttr->set_output(true);

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

        return std::make_pair(std::move(graphObj), GraphOutputs{yAttr});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, convTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.y, this->getTolerance(graphObj, outputs.y));

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(convTestCase.note);
        this->verifyGraph(graphObj, convTestCase.seed);
    }
};

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuConvFwd2dFp32 = ConvForward<float>;
using IntegrationGpuConvFwd2dBfp16 = ConvForward<bfloat16>;
using IntegrationGpuConvFwd2dFp16 = ConvForward<half>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuConvFwd3dFp32 = ConvForward<float>;
using IntegrationGpuConvFwd3dBfp16 = ConvForward<bfloat16>;
using IntegrationGpuConvFwd3dFp16 = ConvForward<half>;

} // namespace

// 2D tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwd2dFp32);
TEST_P(IntegrationGpuConvFwd2dFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwd2dBfp16);
TEST_P(IntegrationGpuConvFwd2dBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwd2dFp16);
TEST_P(IntegrationGpuConvFwd2dFp16, Correctness)
{
    runGraphTest();
}

// 3D tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwd3dFp32);
TEST_P(IntegrationGpuConvFwd3dFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwd3dBfp16);
TEST_P(IntegrationGpuConvFwd3dBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwd3dFp16);
TEST_P(IntegrationGpuConvFwd3dFp16, Correctness)
{
    runGraphTest();
}

// 2D instantiations
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwd2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwd2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwd2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));

// 3D instantiations
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwd3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwd3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwd3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D())));
