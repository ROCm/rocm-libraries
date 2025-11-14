// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/BatchnormCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace test_bn_common;

namespace
{

struct BatchnormActivationTensorIds
{
    static constexpr int64_t X_UID = 1;
    static constexpr int64_t SCALE_UID = 2;
    static constexpr int64_t BIAS_UID = 3;
    static constexpr int64_t MEAN_UID = 4;
    static constexpr int64_t INV_VARIANCE_UID = 5;
    static constexpr int64_t BN_Y_UID = 6;
    static constexpr int64_t DY_UID = 7;
    static constexpr int64_t DX_DRELU_UID = 8;
    static constexpr int64_t DX_OUT_UID = 9;
    static constexpr int64_t DSCALE_OUT_UID = 10;
    static constexpr int64_t DBIAS_OUT_UID = 11;
};

template <typename DataType>
class BatchnormBackwardActivation
    : public IntegrationGraphVerificationHarness<
          DataType,
          std::tuple<test_bn_common::BatchnormTestCase, test_activation_common::ActivTestCase>>
{
protected:
    void initializeBundle([[maybe_unused]] const graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.tensors.at(BatchnormActivationTensorIds::X_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
        bundle.tensors.at(BatchnormActivationTensorIds::DY_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
        bundle.tensors.at(BatchnormActivationTensorIds::SCALE_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);
        bundle.tensors.at(BatchnormActivationTensorIds::BIAS_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);
        bundle.tensors.at(BatchnormActivationTensorIds::MEAN_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);
        bundle.tensors.at(BatchnormActivationTensorIds::INV_VARIANCE_UID)
            ->fillTensorWithRandomValues(1.9f, 2.0f, seed);
    }

    void runGraphTest([[maybe_unused]] DataType tolerance, const TensorLayout& layout) override
    {
        namespace fe = hipdnn_frontend;

        const auto& [bnTestCase, activTestCase] = this->GetParam();
        auto dims = bnTestCase.dims;

        std::vector<int64_t> channelDims = getDerivedShape(dims);

        graph::Graph graphObj;
        graphObj.set_name("BatchnormBackwardActivationTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = fe::DataType::FLOAT;
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(fe::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", dataType, dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(BatchnormActivationTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr
            = graph::makeTensorAttributes("scale",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        scaleAttr.set_uid(BatchnormActivationTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr
            = graph::makeTensorAttributes("bias",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        biasAttr.set_uid(BatchnormActivationTensorIds::BIAS_UID);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto meanAttr
            = graph::makeTensorAttributes("mean",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        meanAttr.set_uid(BatchnormActivationTensorIds::MEAN_UID);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarAttr
            = graph::makeTensorAttributes("inv_variance",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        invVarAttr.set_uid(BatchnormActivationTensorIds::INV_VARIANCE_UID);
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarAttr));

        // BN_Y = batchnorm_inference(X, mean, inv_variance, scale, bias)
        graph::BatchnormInferenceAttributes bnInfAttrs;
        bnInfAttrs.set_name("batchnorm_inference");

        auto bnY = graphObj.batchnorm_inference(xTensorAttr,
                                                meanTensorAttr,
                                                invVarianceTensorAttr,
                                                scaleTensorAttr,
                                                biasTensorAttr,
                                                bnInfAttrs);

        bnY->set_name("BN_Y");
        bnY->set_dim(dims);
        bnY->set_stride(generateStrides(dims, layout.strideOrder));
        bnY->set_uid(BatchnormActivationTensorIds::BN_Y_UID);

        auto dyAttr = graph::makeTensorAttributes(
            "dy", dataType, dims, generateStrides(dims, layout.strideOrder));
        dyAttr.set_uid(BatchnormActivationTensorIds::DY_UID);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        // DX_dactiv = pointwise(DY, BN_Y, activation_mode)
        graph::PointwiseAttributes activBwdAttrs;
        activBwdAttrs.set_name("activation_bwd");
        activBwdAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activTestCase.mode));
        if(activTestCase.reluLowerClip.has_value())
        {
            activBwdAttrs.set_relu_lower_clip(activTestCase.reluLowerClip.value());
        }
        if(activTestCase.reluUpperClip.has_value())
        {
            activBwdAttrs.set_relu_upper_clip(activTestCase.reluUpperClip.value());
        }
        if(activTestCase.reluLowerClipSlope.has_value())
        {
            activBwdAttrs.set_relu_lower_clip_slope(activTestCase.reluLowerClipSlope.value());
        }
        if(activTestCase.swishBeta.has_value())
        {
            activBwdAttrs.set_swish_beta(activTestCase.swishBeta.value());
        }
        if(activTestCase.eluAlpha.has_value())
        {
            activBwdAttrs.set_elu_alpha(activTestCase.eluAlpha.value());
        }
        if(activTestCase.softplusBeta.has_value())
        {
            activBwdAttrs.set_softplus_beta(activTestCase.softplusBeta.value());
        }

        auto dxDrelu = graphObj.pointwise(bnY, dyTensorAttr, activBwdAttrs);
        dxDrelu->set_name("DX_drelu");
        dxDrelu->set_dim(dims);
        dxDrelu->set_stride(generateStrides(dims, layout.strideOrder));
        dxDrelu->set_uid(BatchnormActivationTensorIds::DX_DRELU_UID);

        graph::BatchnormBackwardAttributes bnBwdAttrs;
        bnBwdAttrs.set_name("batchnorm_backward");
        bnBwdAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);

        // [DX, dscale, dbias] = batchnorm_backward(DX_drelu, X, scale, saved_mean_inv_var)
        auto bnBwdOuts
            = graphObj.batchnorm_backward(dxDrelu, xTensorAttr, scaleTensorAttr, bnBwdAttrs);

        auto& dxOut = bnBwdOuts[0];
        dxOut->set_name("dx");
        dxOut->set_dim(dims);
        dxOut->set_data_type(dataType);
        dxOut->set_stride(generateStrides(dims, layout.strideOrder));
        dxOut->set_output(true);
        dxOut->set_uid(BatchnormActivationTensorIds::DX_OUT_UID);

        auto& dscaleOut = bnBwdOuts[1];
        dscaleOut->set_name("dscale");
        dscaleOut->set_data_type(intermediateDataType);
        dscaleOut->set_dim(channelDims);
        dscaleOut->set_stride(generateStrides(channelDims, layout.strideOrder));
        dscaleOut->set_output(true);
        dscaleOut->set_uid(BatchnormActivationTensorIds::DSCALE_OUT_UID);

        auto& dbiasOut = bnBwdOuts[2];
        dbiasOut->set_name("dbias");
        dbiasOut->set_data_type(intermediateDataType);
        dbiasOut->set_dim(channelDims);
        dbiasOut->set_stride(generateStrides(channelDims, layout.strideOrder));
        dbiasOut->set_output(true);
        dbiasOut->set_uid(BatchnormActivationTensorIds::DBIAS_OUT_UID);

        auto intermediateTolerance = batchnorm::getToleranceBackward<float>();

        this->registerValidator(dxOut, static_cast<float>(tolerance));
        this->registerValidator(dscaleOut, intermediateTolerance);
        this->registerValidator(dbiasOut, intermediateTolerance);

        this->verifyGraph(graphObj, bnTestCase.seed);
    }
};

using IntegrationGpuBatchnormBackwardActivationNchwFp32 = BatchnormBackwardActivation<float>;

using IntegrationGpuBatchnormBackwardActivationNchwBfp16
    = BatchnormBackwardActivation<hip_bfloat16>;

using IntegrationGpuBatchnormBackwardActivationNchwFp16 = BatchnormBackwardActivation<half>;

using IntegrationGpuBatchnormBackwardActivationNhwcFp32 = BatchnormBackwardActivation<float>;

using IntegrationGpuBatchnormBackwardActivationNhwcBfp16
    = BatchnormBackwardActivation<hip_bfloat16>;

using IntegrationGpuBatchnormBackwardActivationNhwcFp16 = BatchnormBackwardActivation<half>;

using IntegrationGpuBatchnormBackwardActivationNcdhwFp32 = BatchnormBackwardActivation<float>;

using IntegrationGpuBatchnormBackwardActivationNcdhwBfp16
    = BatchnormBackwardActivation<hip_bfloat16>;

using IntegrationGpuBatchnormBackwardActivationNcdhwFp16 = BatchnormBackwardActivation<half>;

using IntegrationGpuBatchnormBackwardActivationNdhwcFp32 = BatchnormBackwardActivation<float>;

using IntegrationGpuBatchnormBackwardActivationNdhwcBfp16
    = BatchnormBackwardActivation<hip_bfloat16>;

using IntegrationGpuBatchnormBackwardActivationNdhwcFp16 = BatchnormBackwardActivation<half>;

} // namespace

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNchwFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNchwBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNchwFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNhwcFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNhwcBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNhwcFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNcdhwFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNcdhwBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNcdhwFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNdhwcFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNdhwcBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationTestCases())));
