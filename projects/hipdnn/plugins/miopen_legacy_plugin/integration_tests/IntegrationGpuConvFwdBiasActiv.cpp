// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/ConvolutionCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace miopen_legacy_plugin::test_utilities;

namespace
{

template <typename DataType>
class ConvFwdBiasActiv
    : public IntegrationGraphVerificationHarness<
          DataType,
          std::tuple<test_conv_common::ConvTestCase, bool, test_activation_common::ActivTestCase>>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        // Skipping until CK is working on Windows
        SKIP_IF_WINDOWS();

        const auto& [convTestCase, doBias, activTestCase] = this->GetParam();

        graph::Graph graphObj;

        graphObj.set_name(doBias ? "ConvFwdBiasActivTest" : "ConvFwdActivTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        int64_t uid = 1;

        auto dataType = getDataTypeEnumFromType<DataType>();

        auto xAttr
            = graph::makeTensorAttributes("x",
                                          dataType,
                                          convTestCase.xDims,
                                          generateStrides(convTestCase.xDims, layout.strideOrder));
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto wAttr
            = graph::makeTensorAttributes("w",
                                          dataType,
                                          convTestCase.wDims,
                                          generateStrides(convTestCase.wDims, layout.strideOrder));
        wAttr.set_uid(uid++);
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_name("convolution_forward");
        convAttrs.set_pre_padding(convTestCase.convPrePadding);
        convAttrs.set_post_padding(convTestCase.convPostPadding);
        convAttrs.set_stride(convTestCase.convStride);
        convAttrs.set_dilation(convTestCase.convDilation);

        auto yConvTensorAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);
        yConvTensorAttr->set_name("y_conv");
        yConvTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);
        yConvTensorAttr->set_dim(convTestCase.yDims);
        yConvTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
        yConvTensorAttr->set_uid(uid++);

        std::shared_ptr<graph::TensorAttributes> yBiasTensorAttr;
        if(doBias)
        {
            const auto biasDims = getDerivedShape(convTestCase.yDims);

            auto biasAttr = graph::makeTensorAttributes(
                "bias", dataType, biasDims, generateStrides(biasDims, layout.strideOrder));
            biasAttr.set_uid(uid++);
            auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

            graph::PointwiseAttributes biasAttrs;
            biasAttrs.set_name("bias");
            biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
            biasAttrs.set_compute_data_type(dataType);

            yBiasTensorAttr = graphObj.pointwise(yConvTensorAttr, biasTensorAttr, biasAttrs);
            yBiasTensorAttr->set_name("y_bias");
            yBiasTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);
            yBiasTensorAttr->set_dim(convTestCase.yDims);
            yBiasTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
            yBiasTensorAttr->set_uid(uid++);
        }

        graph::PointwiseAttributes activAttrs;
        activAttrs.set_name("activation_forward");
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
        yTensorAttr->set_name("y");
        yTensorAttr->set_data_type(dataType);
        yTensorAttr->set_dim(convTestCase.yDims);
        yTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
        yTensorAttr->set_output(true);
        yTensorAttr->set_uid(uid);

        this->registerValidator(yTensorAttr, tolerance);

        ASSERT_NO_FATAL_FAILURE(this->verifyGraph(graphObj, convTestCase.seed));
    }
};

using IntegrationGpuConvFwdBiasActivNchwFp32 = ConvFwdBiasActiv<float>;
using IntegrationGpuConvFwdBiasActivNcdhwFp32 = ConvFwdBiasActiv<float>;

using IntegrationGpuConvFwdBiasActivNchwBfp16 = ConvFwdBiasActiv<hip_bfloat16>;
using IntegrationGpuConvFwdBiasActivNcdhwBfp16 = ConvFwdBiasActiv<hip_bfloat16>;

using IntegrationGpuConvFwdBiasActivNchwFp16 = ConvFwdBiasActiv<half>;
using IntegrationGpuConvFwdBiasActivNcdhwFp16 = ConvFwdBiasActiv<half>;

using IntegrationGpuConvFwdBiasActivNhwcFp32 = ConvFwdBiasActiv<float>;
using IntegrationGpuConvFwdBiasActivNdhwcFp32 = ConvFwdBiasActiv<float>;

using IntegrationGpuConvFwdBiasActivNhwcBfp16 = ConvFwdBiasActiv<hip_bfloat16>;
using IntegrationGpuConvFwdBiasActivNdhwcBfp16 = ConvFwdBiasActiv<hip_bfloat16>;

using IntegrationGpuConvFwdBiasActivNhwcFp16 = ConvFwdBiasActiv<half>;
using IntegrationGpuConvFwdBiasActivNdhwcFp16 = ConvFwdBiasActiv<half>;

} // namespace

TEST_P(IntegrationGpuConvFwdBiasActivNchwFp32, Correctness)
{
    runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdBiasActivNcdhwFp32, Correctness)
{
    runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdBiasActivNchwBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdBiasActivNcdhwBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdBiasActivNchwFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdBiasActivNcdhwFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdBiasActivNhwcFp32, Correctness)
{
    runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdBiasActivNdhwcFp32, Correctness)
{
    runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvFwdBiasActivNhwcBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdBiasActivNdhwcBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvFwdBiasActivNhwcFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdBiasActivNdhwcFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NDHWC);
}

// Smoke test cases
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNchwFp32,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNchwBfp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNchwFp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNhwcFp32,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNhwcBfp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNhwcFp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNcdhwFp32,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNcdhwBfp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNcdhwFp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNdhwcFp32,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNdhwcBfp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdBiasActivNdhwcFp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

// Full test cases
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNchwFp32,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNchwBfp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNchwFp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNhwcFp32,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNhwcBfp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNhwcFp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases4D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNcdhwFp32,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNcdhwBfp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNcdhwFp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNdhwcFp32,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNdhwcBfp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuConvFwdBiasActivNdhwcFp16,
    testing::Combine(testing::ValuesIn(test_conv_common::getConvTestCases5D()),
                     testing::Bool(),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));
