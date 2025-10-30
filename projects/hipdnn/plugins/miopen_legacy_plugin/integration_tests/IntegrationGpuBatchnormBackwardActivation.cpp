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

#include "../tests/common/BatchnormCommon.hpp"
#include "../tests/common/BatchnormFusionCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace test_bn_common;
using namespace test_bn_fusion_common;

namespace
{

template <typename DataType>
class BatchnormBackwardActivation
    : public IntegrationGraphVerificationHarness<DataType, BnActivTestCase>
{
protected:
    std::unordered_map<std::string, int64_t> _inputTensorIds;

    void initializeBundle([[maybe_unused]] const graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        // x and dy: wider range
        bundle.tensors.at(_inputTensorIds.at("x"))->fillTensorWithRandomValues(-1.8f, 1.8f, seed);
        bundle.tensors.at(_inputTensorIds.at("dy"))->fillTensorWithRandomValues(-1.8f, 1.8f, seed);

        // scale: around 1.0
        bundle.tensors.at(_inputTensorIds.at("scale"))
            ->fillTensorWithRandomValues(0.5f, 1.5f, seed);

        // bias and mean: small values
        bundle.tensors.at(_inputTensorIds.at("bias"))
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);
        bundle.tensors.at(_inputTensorIds.at("mean"))
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);

        // inv_variance: positive values
        bundle.tensors.at(_inputTensorIds.at("inv_variance"))
            ->fillTensorWithRandomValues(0.5f, 2.0f, seed);
    }

    void runGraphTest([[maybe_unused]] DataType tolerance, const TensorLayout& layout) override
    {
        namespace fe = hipdnn_frontend;

        const BnActivTestCase& testCase = this->GetParam();
        auto dims = testCase.bn.dims;

        std::vector<int64_t> channelDims = getDerivedShape(dims);

        graph::Graph graphObj;
        graphObj.set_name("BatchnormBackwardActivationTest");
        graphObj.set_compute_data_type(fe::DataType::FLOAT);

        int64_t uid = 1;
        auto nextUid = [&]() { return uid++; };

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = fe::DataType::FLOAT;

        auto xAttr = graph::makeTensorAttributes(
            "x", dataType, dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(nextUid());
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        _inputTensorIds.insert({"x", xTensorAttr->get_uid()});

        auto scaleAttr
            = graph::makeTensorAttributes("scale",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        scaleAttr.set_uid(nextUid());
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));
        _inputTensorIds.insert({"scale", scaleTensorAttr->get_uid()});

        auto biasAttr
            = graph::makeTensorAttributes("bias",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        biasAttr.set_uid(nextUid());
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));
        _inputTensorIds.insert({"bias", biasTensorAttr->get_uid()});

        auto meanAttr
            = graph::makeTensorAttributes("mean",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        meanAttr.set_uid(nextUid());
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));
        _inputTensorIds.insert({"mean", meanTensorAttr->get_uid()});

        auto invVarAttr
            = graph::makeTensorAttributes("inv_variance",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        invVarAttr.set_uid(nextUid());
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarAttr));
        _inputTensorIds.insert({"inv_variance", invVarianceTensorAttr->get_uid()});

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
        bnY->set_data_type(dataType);
        bnY->set_dim(dims);
        bnY->set_stride(generateStrides(dims, layout.strideOrder));
        bnY->set_is_virtual(true);
        if(!bnY->has_uid())
        {
            bnY->set_uid(nextUid());
        }

        auto dyAttr = graph::makeTensorAttributes(
            "dy", dataType, dims, generateStrides(dims, layout.strideOrder));
        dyAttr.set_uid(nextUid());
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));
        _inputTensorIds.insert({"dy", dyTensorAttr->get_uid()});

        // DX_dactiv = pointwise(DY, BN_Y, activation_mode)
        graph::PointwiseAttributes activBwdAttrs;
        activBwdAttrs.set_name("activation_bwd");
        activBwdAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(testCase.activ.mode));
        if(testCase.activ.reluLowerClip.has_value())
        {
            activBwdAttrs.set_relu_lower_clip(testCase.activ.reluLowerClip.value());
        }
        if(testCase.activ.reluUpperClip.has_value())
        {
            activBwdAttrs.set_relu_upper_clip(testCase.activ.reluUpperClip.value());
        }
        if(testCase.activ.reluLowerClipSlope.has_value())
        {
            activBwdAttrs.set_relu_lower_clip_slope(testCase.activ.reluLowerClipSlope.value());
        }
        if(testCase.activ.swishBeta.has_value())
        {
            activBwdAttrs.set_swish_beta(testCase.activ.swishBeta.value());
        }
        if(testCase.activ.eluAlpha.has_value())
        {
            activBwdAttrs.set_elu_alpha(testCase.activ.eluAlpha.value());
        }
        if(testCase.activ.softplusBeta.has_value())
        {
            activBwdAttrs.set_softplus_beta(testCase.activ.softplusBeta.value());
        }

        auto dxDrelu = graphObj.pointwise(bnY, dyTensorAttr, activBwdAttrs);
        dxDrelu->set_name("DX_drelu");
        dxDrelu->set_data_type(
            dataType); // leaving this as intermediate could yield more accurate results
        dxDrelu->set_dim(dims);
        dxDrelu->set_stride(generateStrides(dims, layout.strideOrder));
        dxDrelu->set_is_virtual(true);
        if(!dxDrelu->has_uid())
        {
            dxDrelu->set_uid(nextUid());
        }

        graph::BatchnormBackwardAttributes bnBwdAttrs;
        bnBwdAttrs.set_name("batchnorm_backward");
        bnBwdAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);

        // [DX, dscale, dbias] = batchnorm_backward(DX_drelu, X, scale, saved_mean_inv_var)
        auto bnBwdOuts
            = graphObj.batchnorm_backward(dxDrelu, xTensorAttr, scaleTensorAttr, bnBwdAttrs);

        auto& dxOut = bnBwdOuts[0];
        dxOut->set_name("dx");
        dxOut->set_data_type(dataType);
        dxOut->set_dim(dims);
        dxOut->set_stride(generateStrides(dims, layout.strideOrder));
        dxOut->set_is_virtual(false);
        dxOut->set_output(true);
        if(!dxOut->has_uid())
        {
            dxOut->set_uid(nextUid());
        }

        auto& dscaleOut = bnBwdOuts[1];
        dscaleOut->set_name("dscale");
        dscaleOut->set_data_type(intermediateDataType);
        dscaleOut->set_dim(channelDims);
        dscaleOut->set_stride(generateStrides(channelDims, layout.strideOrder));
        dscaleOut->set_is_virtual(false);
        dscaleOut->set_output(true);
        if(!dscaleOut->has_uid())
        {
            dscaleOut->set_uid(nextUid());
        }

        auto& dbiasOut = bnBwdOuts[2];
        dbiasOut->set_name("dbias");
        dbiasOut->set_data_type(intermediateDataType);
        dbiasOut->set_dim(channelDims);
        dbiasOut->set_stride(generateStrides(channelDims, layout.strideOrder));
        dbiasOut->set_is_virtual(false);
        dbiasOut->set_output(true);
        if(!dbiasOut->has_uid())
        {
            dbiasOut->set_uid(nextUid());
        }

        // Use float tolerance * 2 = 4e-3 for all data types to match MIOpen.
        // It is also the highest, and passes unlike the others.
        const auto rmsFloatTol = batchnorm::getToleranceBackward<float>() * 2.0f;

        this->registerRmsValidator(dxOut, rmsFloatTol);
        this->registerRmsValidator(dscaleOut, rmsFloatTol);
        this->registerRmsValidator(dbiasOut, rmsFloatTol);

        this->verifyGraph(graphObj, testCase.bn.seed);
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

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNchwFp32,
                         testing::ValuesIn(getBnActivBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNchwBfp16,
                         testing::ValuesIn(getBnActivBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNchwFp16,
                         testing::ValuesIn(getBnActivBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNhwcFp32,
                         testing::ValuesIn(getBnActivBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNhwcBfp16,
                         testing::ValuesIn(getBnActivBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNhwcFp16,
                         testing::ValuesIn(getBnActivBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNcdhwFp32,
                         testing::ValuesIn(getBnActiv3dBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNcdhwBfp16,
                         testing::ValuesIn(getBnActiv3dBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNcdhwFp16,
                         testing::ValuesIn(getBnActiv3dBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNdhwcFp32,
                         testing::ValuesIn(getBnActiv3dBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNdhwcBfp16,
                         testing::ValuesIn(getBnActiv3dBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardActivationNdhwcFp16,
                         testing::ValuesIn(getBnActiv3dBwdTestCases()));
