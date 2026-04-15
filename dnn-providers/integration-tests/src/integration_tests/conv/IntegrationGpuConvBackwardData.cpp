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

using ConvBwdDataTestCase = std::tuple<TensorLayout, test_conv_common::ConvTestCase>;

template <typename DataType>
class ConvBackwardData : public IntegrationGraphVerificationHarness<DataType, ConvBwdDataTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> dx;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const ConvBwdDataTestCase& tc)
    {
        const auto& [layout, testCase] = tc;

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ConvolutionBackwardDataTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto dyAttr = graph::makeTensorAttributes(
            "dy", testCase.yDims, generateStrides(testCase.yDims, layout.strideOrder));
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto wAttr = graph::makeTensorAttributes(
            "w", testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvDgradAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto dxAttr = graphObj.conv_dgrad(dyTensorAttr, wTensorAttr, convAttrs);
        dxAttr->set_output(true);

        // Set these explicitly since grouped convs cannot infer tensor shape.
        // Infer behavior will assume groups == 1, but some cases have groups > 1.
        dxAttr->set_dim(testCase.xDims);
        dxAttr->set_stride(generateStrides(testCase.xDims, layout.strideOrder));

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

        return std::make_pair(std::move(graphObj), GraphOutputs{dxAttr});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, convTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.dx, this->getTolerance(graphObj, outputs.dx));

        this->verifyGraph(graphObj, convTestCase.seed);
    }
};

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuConvBwdData2dFp32 = ConvBackwardData<float>;
using IntegrationGpuConvBwdData2dBfp16 = ConvBackwardData<bfloat16>;
using IntegrationGpuConvBwdData2dFp16 = ConvBackwardData<half>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuConvBwdData3dFp32 = ConvBackwardData<float>;
using IntegrationGpuConvBwdData3dBfp16 = ConvBackwardData<bfloat16>;
using IntegrationGpuConvBwdData3dFp16 = ConvBackwardData<half>;

} // namespace

// 2D tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvBwdData2dFp32);
TEST_P(IntegrationGpuConvBwdData2dFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvBwdData2dBfp16);
TEST_P(IntegrationGpuConvBwdData2dBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvBwdData2dFp16);
TEST_P(IntegrationGpuConvBwdData2dFp16, Correctness)
{
    runGraphTest();
}

// 3D tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvBwdData3dFp32);
TEST_P(IntegrationGpuConvBwdData3dFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvBwdData3dBfp16);
TEST_P(IntegrationGpuConvBwdData3dBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvBwdData3dFp16);
TEST_P(IntegrationGpuConvBwdData3dFp16, Correctness)
{
    runGraphTest();
}

// 2D instantiations
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvBwdData2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvBwdData2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvBwdData2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));

// 3D instantiations
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvBwdData3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvBwdData3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvBwdData3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D())));
