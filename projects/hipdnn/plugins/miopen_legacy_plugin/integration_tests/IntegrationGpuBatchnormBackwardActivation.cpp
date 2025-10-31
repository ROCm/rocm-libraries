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
    int64_t xUid;
    int64_t dyUid;
    int64_t scaleUid;
    int64_t biasUid;
    int64_t meanUid;
    int64_t invVarianceUid;
};

template <typename DataType>
class BatchnormBackwardActivation
    : public IntegrationGraphVerificationHarness<
          DataType,
          std::tuple<test_bn_common::BatchnormTestCase, test_activation_common::ActivTestCase>>
{
protected:
    BatchnormActivationTensorIds _tensorIds;

    void initializeBundle([[maybe_unused]] const graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.tensors.at(_tensorIds.xUid)->fillTensorWithRandomValues(-1.8f, 1.8f, seed);
        bundle.tensors.at(_tensorIds.dyUid)->fillTensorWithRandomValues(-1.8f, 1.8f, seed);

        bundle.tensors.at(_tensorIds.scaleUid)->fillTensorWithRandomValues(0.5f, 1.5f, seed);

        bundle.tensors.at(_tensorIds.biasUid)->fillTensorWithRandomValues(-0.1f, 0.1f, seed);
        bundle.tensors.at(_tensorIds.meanUid)->fillTensorWithRandomValues(-0.1f, 0.1f, seed);

        bundle.tensors.at(_tensorIds.invVarianceUid)->fillTensorWithRandomValues(0.5f, 2.0f, seed);
    }

    void runGraphTest([[maybe_unused]] DataType tolerance, const TensorLayout& layout) override
    {
        namespace fe = hipdnn_frontend;

        const auto& [bnTestCase, activTestCase] = this->GetParam();
        auto dims = bnTestCase.dims;

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
        _tensorIds.xUid = xTensorAttr->get_uid();

        auto scaleAttr
            = graph::makeTensorAttributes("scale",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        scaleAttr.set_uid(nextUid());
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));
        _tensorIds.scaleUid = scaleTensorAttr->get_uid();

        auto biasAttr
            = graph::makeTensorAttributes("bias",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        biasAttr.set_uid(nextUid());
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));
        _tensorIds.biasUid = biasTensorAttr->get_uid();

        auto meanAttr
            = graph::makeTensorAttributes("mean",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        meanAttr.set_uid(nextUid());
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));
        _tensorIds.meanUid = meanTensorAttr->get_uid();

        auto invVarAttr
            = graph::makeTensorAttributes("inv_variance",
                                          intermediateDataType,
                                          channelDims,
                                          generateStrides(channelDims, layout.strideOrder));
        invVarAttr.set_uid(nextUid());
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarAttr));
        _tensorIds.invVarianceUid = invVarianceTensorAttr->get_uid();

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
        _tensorIds.dyUid = dyTensorAttr->get_uid();

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

        // Use 4e-3 float tolerance for all data types to match MIOpen.
        // https://github.com/ROCm/rocm-libraries/blob/develop/projects/miopen/test/gtest/bn.hpp#L484
        // It is also the highest, and unlike the others tols; it passes.
        const auto rmsFloatTol = 4e-3f;

        this->registerRmsValidator(dxOut, rmsFloatTol);
        this->registerRmsValidator(dscaleOut, rmsFloatTol);
        this->registerRmsValidator(dbiasOut, rmsFloatTol);

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
