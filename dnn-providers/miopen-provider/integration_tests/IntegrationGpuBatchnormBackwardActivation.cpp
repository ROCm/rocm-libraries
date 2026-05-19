// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/BatchnormCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_plugin::test_utilities;
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
        // Fill output tensors with sentinel values
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

    void runGraphTest([[maybe_unused]] float tolerance, const TensorLayout& layout)
    {
        // Known failures under MIOPEN_FIND_ENFORCE=4 (exhaustive tuning) on degenerate spatial
        // dims (e.g. [2,3,1,1], [32,3,1,14]): exhaustive search selects a kernel with numerical
        // accuracy issues on 1-element spatial dimensions. Not an ASAN error. Root cause is in
        // MIOpen kernel selection for degenerate shapes.
        namespace fe = hipdnn_frontend;

        const auto& [bnTestCase, activTestCase] = this->GetParam();
        auto dims = bnTestCase.dims;

        const std::vector<int64_t> channelDims = getDerivedShape(dims);

        graph::Graph graphObj;
        graphObj.set_name("BatchnormBackwardActivationTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = fe::DataType::FLOAT;
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(fe::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = makeTensorAttributes("x", dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(BatchnormActivationTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr = makeTensorAttributes("scale",
                                              intermediateDataType,
                                              channelDims,
                                              generateStrides(channelDims, layout.strideOrder));
        scaleAttr.set_uid(BatchnormActivationTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = makeTensorAttributes("bias",
                                             intermediateDataType,
                                             channelDims,
                                             generateStrides(channelDims, layout.strideOrder));
        biasAttr.set_uid(BatchnormActivationTensorIds::BIAS_UID);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto meanAttr = makeTensorAttributes("mean",
                                             intermediateDataType,
                                             channelDims,
                                             generateStrides(channelDims, layout.strideOrder));
        meanAttr.set_uid(BatchnormActivationTensorIds::MEAN_UID);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarAttr = makeTensorAttributes("inv_variance",
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

        auto dyAttr = makeTensorAttributes("dy", dims, generateStrides(dims, layout.strideOrder));
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

        auto dxDrelu = graphObj.pointwise(bnY, dyTensorAttr, activBwdAttrs);

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

        auto intermediateTolerance = batchnorm::getToleranceBackward<float>();

        this->registerValidator(dxOut, tolerance);
        this->registerValidator(dscaleOut, intermediateTolerance);
        this->registerValidator(dbiasOut, intermediateTolerance);

        this->verifyGraph(graphObj, bnTestCase.seed);
    }
};

using IntegrationGpuBatchnormBackwardActivationNclFp32 = BatchnormBackwardActivation<float>;

using IntegrationGpuBatchnormBackwardActivationNclBfp16 = BatchnormBackwardActivation<bfloat16>;

using IntegrationGpuBatchnormBackwardActivationNclFp16 = BatchnormBackwardActivation<half>;

using IntegrationGpuBatchnormBackwardActivationNlcFp32 = BatchnormBackwardActivation<float>;

using IntegrationGpuBatchnormBackwardActivationNlcBfp16 = BatchnormBackwardActivation<bfloat16>;

using IntegrationGpuBatchnormBackwardActivationNlcFp16 = BatchnormBackwardActivation<half>;

using IntegrationGpuBatchnormBackwardActivationNchwFp32 = BatchnormBackwardActivation<float>;

using IntegrationGpuBatchnormBackwardActivationNchwBfp16 = BatchnormBackwardActivation<bfloat16>;

using IntegrationGpuBatchnormBackwardActivationNchwFp16 = BatchnormBackwardActivation<half>;

using IntegrationGpuBatchnormBackwardActivationNhwcFp32 = BatchnormBackwardActivation<float>;

using IntegrationGpuBatchnormBackwardActivationNhwcBfp16 = BatchnormBackwardActivation<bfloat16>;

using IntegrationGpuBatchnormBackwardActivationNhwcFp16 = BatchnormBackwardActivation<half>;

using IntegrationGpuBatchnormBackwardActivationNcdhwFp32 = BatchnormBackwardActivation<float>;

using IntegrationGpuBatchnormBackwardActivationNcdhwBfp16 = BatchnormBackwardActivation<bfloat16>;

using IntegrationGpuBatchnormBackwardActivationNcdhwFp16 = BatchnormBackwardActivation<half>;

using IntegrationGpuBatchnormBackwardActivationNdhwcFp32 = BatchnormBackwardActivation<float>;

using IntegrationGpuBatchnormBackwardActivationNdhwcBfp16 = BatchnormBackwardActivation<bfloat16>;

using IntegrationGpuBatchnormBackwardActivationNdhwcFp16 = BatchnormBackwardActivation<half>;

} // namespace

TEST_P(IntegrationGpuBatchnormBackwardActivationNclFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCL);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNclFp32,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNclFp32,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNclBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<bfloat16>(), TensorLayout::NCL);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNclBfp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNclBfp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNclFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCL);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNclFp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNclFp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNlcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NLC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNlcFp32,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNlcFp32,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNlcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<bfloat16>(), TensorLayout::NLC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNlcBfp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNlcBfp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNlcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NLC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNlcFp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNlcFp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd1dFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNchwFp32,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNchwFp32,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNchwBfp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNchwBfp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNchwFp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNchwFp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNhwcFp32,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNhwcFp32,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNhwcBfp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNhwcBfp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNhwcFp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormBackwardActivationNhwcFp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwdFullTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNcdhwFp32,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNcdhwBfp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNcdhwFp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNdhwcFp32,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNdhwcBfp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));

TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormBackwardActivationNdhwcFp16,
    testing::Combine(
        testing::ValuesIn(test_bn_common::getBnBwd3dTestCases()),
        testing::ValuesIn(test_activation_common::createBatchnormBwdActivationTestCases())));
