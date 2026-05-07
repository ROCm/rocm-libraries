// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/ActivationCommon.hpp"
#include "common/ConvolutionCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;

namespace
{

using ConvFwdBiasActivTestCase = std::tuple<TensorLayout,
                                            test_conv_common::ConvTestCase,
                                            bool,
                                            test_activation_common::ActivTestCase>;

template <typename DataType>
class ConvFwdBiasActiv
    : public IntegrationGraphVerificationHarness<DataType, ConvFwdBiasActivTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const ConvFwdBiasActivTestCase& tc)
    {
        const auto& [layout, convTestCase, doBias, activTestCase] = tc;

        graph::Graph graphObj;
        graphObj.set_name(doBias ? "ConvFwdBiasActivTest" : "ConvFwdActivTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", convTestCase.xDims, generateStrides(convTestCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto wAttr = graph::makeTensorAttributes(
            "w", convTestCase.wDims, generateStrides(convTestCase.wDims, layout.strideOrder));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding(convTestCase.convPrePadding);
        convAttrs.set_post_padding(convTestCase.convPostPadding);
        convAttrs.set_stride(convTestCase.convStride);
        convAttrs.set_dilation(convTestCase.convDilation);

        auto yConvTensorAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);

        std::shared_ptr<graph::TensorAttributes> yBiasTensorAttr;
        if(doBias)
        {
            const auto biasDims = getDerivedShape(convTestCase.yDims);

            auto biasAttr = graph::makeTensorAttributes(
                "bias", biasDims, generateStrides(biasDims, layout.strideOrder));
            auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

            graph::PointwiseAttributes biasAttrs;
            biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
            biasAttrs.set_compute_data_type(dataType);

            yBiasTensorAttr = graphObj.pointwise(yConvTensorAttr, biasTensorAttr, biasAttrs);
        }

        graph::PointwiseAttributes activAttrs;
        activAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activTestCase.mode));
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

        auto yTensorAttr
            = graphObj.pointwise(doBias ? yBiasTensorAttr : yConvTensorAttr, activAttrs);
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
        const auto& [layout, convTestCase, doBias, activTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.y, this->getTolerance(graphObj, outputs.y));

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(convTestCase.note);
        this->verifyGraph(graphObj, convTestCase.seed);
    }
};

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuConvFwdBiasActiv2dFp32 = ConvFwdBiasActiv<float>;
using IntegrationGpuConvFwdBiasActiv2dBfp16 = ConvFwdBiasActiv<bfloat16>;
using IntegrationGpuConvFwdBiasActiv2dFp16 = ConvFwdBiasActiv<half>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuConvFwdBiasActiv3dFp32 = ConvFwdBiasActiv<float>;
using IntegrationGpuConvFwdBiasActiv3dBfp16 = ConvFwdBiasActiv<bfloat16>;
using IntegrationGpuConvFwdBiasActiv3dFp16 = ConvFwdBiasActiv<half>;

} // namespace

// 2D tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwdBiasActiv2dFp32);
TEST_P(IntegrationGpuConvFwdBiasActiv2dFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwdBiasActiv2dBfp16);
TEST_P(IntegrationGpuConvFwdBiasActiv2dBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwdBiasActiv2dFp16);
TEST_P(IntegrationGpuConvFwdBiasActiv2dFp16, Correctness)
{
    runGraphTest();
}

// 3D tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwdBiasActiv3dFp32);
TEST_P(IntegrationGpuConvFwdBiasActiv3dFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwdBiasActiv3dBfp16);
TEST_P(IntegrationGpuConvFwdBiasActiv3dBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwdBiasActiv3dFp16);
TEST_P(IntegrationGpuConvFwdBiasActiv3dFp16, Correctness)
{
    runGraphTest();
}

// 2D Smoke test cases
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActiv2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActiv2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActiv2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

// 3D Smoke test cases
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActiv3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActiv3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActiv3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

// 2D Full test cases
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActiv2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActiv2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActiv2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// 3D Full test cases
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActiv3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActiv3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActiv3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));
