// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/ActivationCommon.hpp"
#include "common/RMSNormCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_rmsnorm_common;
using namespace test_activation_common;

namespace
{

// bias/isTraining only affect the corresponding forward graph (IntegrationGpuRMSNorm.cpp);
// rmsnorm_backward() takes no bias input and doesn't branch on training phase, so this
// suite reuses the same shape/case generator as forward for both.
struct RMSNormBackwardTensorIds
{
    // inv_rms defaults to a DERIVED fill (recompute-from-x), which the synthesis
    // pipeline hasn't implemented yet (SynthesizeInputs.cpp: "DERIVED fill not yet
    // implemented" -> every case SKIPs). Give it a stable uid so the constructor
    // below can override it with a FREE range instead. inv_rms = 1/rms(x) is
    // always strictly positive, so keep it positive and away from zero
    // (mirroring BatchnormBwdTensorIds::INV_VARIANCE_UID's narrow, positive
    // range) — a range straddling/near zero, like the legacy provider-local
    // harness's blanket [-1, 1] fill, occasionally produces near-zero/negative
    // draws that fail bf16 tolerance at large reductions (observed at
    // x:[4096,128,...]: PureBfp16 shape #4 across every layout/bias/phase
    // combination, with the exact failing draws shifting as the range moved).
    static constexpr int64_t INV_RMS_UID = 1;
};

using RMSNormBackwardActivationTestCaseType
    = std::tuple<TensorLayout, RMSNormTestCase, ActivTestCase>;

template <typename DyType,
          typename XType,
          typename ScaleType,
          typename DxType,
          typename ComputeType>
class RMSNormBackwardActivation
    : public IntegrationGraphVerificationHarness<DxType, RMSNormBackwardActivationTestCaseType>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> dx;
        std::shared_ptr<graph::TensorAttributes> dscale;
        std::shared_ptr<graph::TensorAttributes> dbias;
    };

    static std::pair<graph::Graph, GraphOutputs>
        buildGraph(hipdnnHandle_t handle, const RMSNormBackwardActivationTestCaseType& tc)
    {
        const auto& [layout, testCase, activTestCase] = tc;

        graph::Graph graphObj;
        graphObj.set_name("RMSNormBackwardActivationTest");

        const auto dyType = getDataTypeEnumFromType<DyType>();
        const auto xType = getDataTypeEnumFromType<XType>();
        const auto scaleType = getDataTypeEnumFromType<ScaleType>();
        const auto computeType = getDataTypeEnumFromType<ComputeType>();
        graphObj.set_compute_data_type(computeType)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

        auto dyAttr = graph::makeTensorAttributes(
            "dy", dyType, testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto xAttr = graph::makeTensorAttributes(
            "x", xType, testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto yAttr = graph::makeTensorAttributes(
            "y", dyType, testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto yTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(yAttr));

        auto scaleAttr
            = graph::makeTensorAttributes("scale",
                                          scaleType,
                                          testCase.scaleDims,
                                          generateStrides(testCase.scaleDims, layout.strideOrder));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        // inv_rms is broadcast over every dim the scale doesn't normalize.
        auto invRmsDims = testCase.xDims;
        for(size_t i = 1; i < invRmsDims.size(); ++i)
        {
            if(testCase.scaleDims[i] != 1)
            {
                invRmsDims[i] = 1;
            }
        }
        auto invRmsAttr = graph::makeTensorAttributes(
            "inv_rms", computeType, invRmsDims, generateStrides(invRmsDims, layout.strideOrder));
        invRmsAttr.set_uid(RMSNormBackwardTensorIds::INV_RMS_UID);
        auto invRmsTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(invRmsAttr));

        graph::PointwiseAttributes activBwdAttrs;
        activBwdAttrs.set_mode(sdkToFrontendPointwiseMode(activTestCase.mode));

        // Set activation-specific parameters
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

        auto dyDreluTensorAttr = graphObj.pointwise(dyTensorAttr, yTensorAttr, activBwdAttrs);

        graph::RMSNormBackwardAttributes rmsnormBwdAttrs;
        rmsnormBwdAttrs.set_compute_data_type(computeType);
        rmsnormBwdAttrs.set_compute_dbias(true);

        auto [dxTensorAttr, dscaleTensorAttr, dbiasTensorAttr] = graphObj.rmsnorm_backward(
            dyDreluTensorAttr, xTensorAttr, scaleTensorAttr, invRmsTensorAttr, rmsnormBwdAttrs);

        const auto dxType = getDataTypeEnumFromType<DxType>();
        dxTensorAttr->set_output(true).set_data_type(dxType);
        dscaleTensorAttr->set_output(true).set_data_type(scaleType);
        dbiasTensorAttr->set_output(true).set_data_type(scaleType);

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

        return std::make_pair(std::move(graphObj),
                              GraphOutputs{dxTensorAttr, dscaleTensorAttr, dbiasTensorAttr});
    }

    RMSNormBackwardActivation()
    {
        this->inputFillRecipes().setRange(RMSNormBackwardTensorIds::INV_RMS_UID, 0.9f, 1.5f);
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& rmsnormTestCase = std::get<1>(testCase);

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.dx, this->getTolerance(graphObj, outputs.dx));
        this->registerValidator(outputs.dscale, this->getTolerance(graphObj, outputs.dscale));
        this->registerValidator(outputs.dbias, this->getTolerance(graphObj, outputs.dbias));

        this->inputFillRecipes().setGlobalSeed(rmsnormTestCase.seed);
        this->verifyGraph(graphObj);
    }
};

// "Pure" = forward input (X) and backward gradients (Dy, Dx) share precision (scale/compute
// stay FP32). "Mixed" = gradient (Dy) is lower precision while X stays FP32.
using IntegrationGpuRMSNormBackwardActivationPureFp32
    = RMSNormBackwardActivation<float, float, float, float, float>;
using IntegrationGpuRMSNormBackwardActivationPureFp16
    = RMSNormBackwardActivation<half, half, float, half, float>;
using IntegrationGpuRMSNormBackwardActivationPureBfp16
    = RMSNormBackwardActivation<bfloat16, bfloat16, float, bfloat16, float>;
using IntegrationGpuRMSNormBackwardActivationMixedFp16
    = RMSNormBackwardActivation<half, float, float, float, float>;
using IntegrationGpuRMSNormBackwardActivationMixedBfp16
    = RMSNormBackwardActivation<bfloat16, float, float, float, float>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardActivationPureFp32);
TEST_P(IntegrationGpuRMSNormBackwardActivationPureFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardActivationPureFp16);
TEST_P(IntegrationGpuRMSNormBackwardActivationPureFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardActivationPureBfp16);
TEST_P(IntegrationGpuRMSNormBackwardActivationPureBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardActivationMixedFp16);
TEST_P(IntegrationGpuRMSNormBackwardActivationMixedFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardActivationMixedBfp16);
TEST_P(IntegrationGpuRMSNormBackwardActivationMixedBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardActivationPureFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardActivationPureFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardActivationPureBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardActivationMixedFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardActivationMixedBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormBackwardActivationPureFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormBackwardActivationPureFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormBackwardActivationPureBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormBackwardActivationMixedFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormBackwardActivationMixedBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationFullCases())));

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuRMSNormBackwardActivation3dPureFp32
    = IntegrationGpuRMSNormBackwardActivationPureFp32;
using IntegrationGpuRMSNormBackwardActivation3dPureFp16
    = IntegrationGpuRMSNormBackwardActivationPureFp16;
using IntegrationGpuRMSNormBackwardActivation3dPureBfp16
    = IntegrationGpuRMSNormBackwardActivationPureBfp16;
using IntegrationGpuRMSNormBackwardActivation3dMixedFp16
    = IntegrationGpuRMSNormBackwardActivationMixedFp16;
using IntegrationGpuRMSNormBackwardActivation3dMixedBfp16
    = IntegrationGpuRMSNormBackwardActivationMixedBfp16;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardActivation3dPureFp32);
TEST_P(IntegrationGpuRMSNormBackwardActivation3dPureFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardActivation3dPureFp16);
TEST_P(IntegrationGpuRMSNormBackwardActivation3dPureFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardActivation3dPureBfp16);
TEST_P(IntegrationGpuRMSNormBackwardActivation3dPureBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardActivation3dMixedFp16);
TEST_P(IntegrationGpuRMSNormBackwardActivation3dMixedFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardActivation3dMixedBfp16);
TEST_P(IntegrationGpuRMSNormBackwardActivation3dMixedBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardActivation3dPureFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardActivation3dPureFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardActivation3dPureBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardActivation3dMixedFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardActivation3dMixedBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases()),
                     testing::ValuesIn(test_activation_common::createBwdActivationSmokeCases())));
