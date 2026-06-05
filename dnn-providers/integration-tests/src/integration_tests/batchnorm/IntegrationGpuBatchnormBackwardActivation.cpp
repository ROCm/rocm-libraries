// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <iostream>

#include "common/ActivationCommon.hpp"
#include "common/BatchnormCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
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
    static constexpr int64_t DY_UID = 6;
};

using BnBwdActivTestCase = std::
    tuple<TensorLayout, test_bn_common::BatchnormTestCase, test_activation_common::ActivTestCase>;

template <typename DataType>
class BatchnormBackwardActivation
    : public IntegrationGraphVerificationHarness<DataType, BnBwdActivTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> dx;
        std::shared_ptr<graph::TensorAttributes> dscale;
        std::shared_ptr<graph::TensorAttributes> dbias;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const BnBwdActivTestCase& tc)
    {
        const auto& [layout, bnTestCase, activTestCase] = tc;
        auto dims = bnTestCase.dims;
        const std::vector<int64_t> channelDims = getDerivedShape(dims);

        graph::Graph graphObj;
        graphObj.set_name("BatchnormBackwardActivationTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr
            = graph::makeTensorAttributes("x", dims, generateStrides(dims, layout.strideOrder));
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
        const graph::BatchnormInferenceAttributes bnInfAttrs;

        auto bnY = graphObj.batchnorm_inference(xTensorAttr,
                                                meanTensorAttr,
                                                invVarianceTensorAttr,
                                                scaleTensorAttr,
                                                biasTensorAttr,
                                                bnInfAttrs);

        auto dyAttr
            = graph::makeTensorAttributes("dy", dims, generateStrides(dims, layout.strideOrder));
        dyAttr.set_uid(BatchnormActivationTensorIds::DY_UID);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        // DX_dactiv = pointwise(DY, BN_Y, activation_mode)
        graph::PointwiseAttributes activBwdAttrs;
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

        auto dxDrelu = graphObj.pointwise(dyTensorAttr, bnY, activBwdAttrs);

        graph::BatchnormBackwardAttributes bnBwdAttrs;
        bnBwdAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);

        // [DX, dscale, dbias] = batchnorm_backward(DX_drelu, X, scale, saved_mean_inv_var)
        auto bnBwdOuts
            = graphObj.batchnorm_backward(dxDrelu, xTensorAttr, scaleTensorAttr, bnBwdAttrs);

        auto& dxOut = bnBwdOuts[0];
        dxOut->set_output(true);

        auto& dscaleOut = bnBwdOuts[1];
        dscaleOut->set_data_type(intermediateDataType);
        dscaleOut->set_output(true);

        auto& dbiasOut = bnBwdOuts[2];
        dbiasOut->set_data_type(intermediateDataType);
        dbiasOut->set_output(true);

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

        return {std::move(graphObj), GraphOutputs{dxOut, dscaleOut, dbiasOut}};
    }

protected:
    void initializeBundle([[maybe_unused]] const graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.sentinelFillOutputTensors();

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

    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, bnTestCase, activTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        // Register validators
        this->registerValidator(outputs.dx, this->getTolerance(graphObj, outputs.dx));
        this->registerValidator(outputs.dscale, this->getTolerance(graphObj, outputs.dscale));
        this->registerValidator(outputs.dbias, this->getTolerance(graphObj, outputs.dbias));

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(bnTestCase.note);
        this->verifyGraph(graphObj, bnTestCase.seed);
    }
};

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuBatchnormBackwardActivation2dFp32 = BatchnormBackwardActivation<float>;
using IntegrationGpuBatchnormBackwardActivation2dBfp16 = BatchnormBackwardActivation<bfloat16>;
using IntegrationGpuBatchnormBackwardActivation2dFp16 = BatchnormBackwardActivation<half>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuBatchnormBackwardActivation3dFp32 = BatchnormBackwardActivation<float>;
using IntegrationGpuBatchnormBackwardActivation3dBfp16 = BatchnormBackwardActivation<bfloat16>;
using IntegrationGpuBatchnormBackwardActivation3dFp16 = BatchnormBackwardActivation<half>;

} // namespace

// 2D tests - Fp32
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivation2dFp32);
TEST_P(IntegrationGpuBatchnormBackwardActivation2dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivation2dFp32,
    testing::Combine(
        testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
        testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivation2dFp32,
    testing::Combine(
        testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
        testing::ValuesIn(test_bn_common::getBnBwdFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

// 2D tests - Bfp16
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivation2dBfp16);
TEST_P(IntegrationGpuBatchnormBackwardActivation2dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivation2dBfp16,
    testing::Combine(
        testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
        testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivation2dBfp16,
    testing::Combine(
        testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
        testing::ValuesIn(test_bn_common::getBnBwdFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

// 2D tests - Fp16
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivation2dFp16);
TEST_P(IntegrationGpuBatchnormBackwardActivation2dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivation2dFp16,
    testing::Combine(
        testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
        testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivation2dFp16,
    testing::Combine(
        testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
        testing::ValuesIn(test_bn_common::getBnBwdFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

// 3D tests - Fp32
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivation3dFp32);
TEST_P(IntegrationGpuBatchnormBackwardActivation3dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivation3dFp32,
    testing::Combine(
        testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
        testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

// 3D tests - Bfp16
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivation3dBfp16);
TEST_P(IntegrationGpuBatchnormBackwardActivation3dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivation3dBfp16,
    testing::Combine(
        testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
        testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

// 3D tests - Fp16
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivation3dFp16);
TEST_P(IntegrationGpuBatchnormBackwardActivation3dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivation3dFp16,
    testing::Combine(
        testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
        testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));
