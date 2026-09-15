// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/RMSNormCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_rmsnorm_common;

namespace
{

// bias/isTraining only affect the corresponding forward graph (IntegrationGpuRMSNorm.cpp);
// rmsnorm_backward() takes no bias input and doesn't branch on training phase, so this
// suite reuses the same shape/case generator as forward for both.
struct RMSNormBackwardTensorIds
{
    // inv_rms defaults to a DERIVED fill (recompute-from-x), which the fill
    // pipeline hasn't implemented yet (FillInputs.cpp: "DERIVED fill not yet
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

using RMSNormBackwardTestCaseType = std::tuple<TensorLayout, RMSNormTestCase>;

template <typename DyType,
          typename XType,
          typename ScaleType,
          typename DxType,
          typename ComputeType>
class RMSNormBackward
    : public IntegrationGraphVerificationHarness<DxType, RMSNormBackwardTestCaseType>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> dx;
        std::shared_ptr<graph::TensorAttributes> dscale;
        std::shared_ptr<graph::TensorAttributes> dbias;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const RMSNormBackwardTestCaseType& tc)
    {
        const auto& [layout, testCase] = tc;

        graph::Graph graphObj;
        graphObj.set_name("RMSNormBackwardTest");

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

        graph::RMSNormBackwardAttributes rmsnormBwdAttrs;
        rmsnormBwdAttrs.set_compute_data_type(computeType);
        rmsnormBwdAttrs.set_compute_dbias(true);

        auto [dxTensorAttr, dscaleTensorAttr, dbiasTensorAttr] = graphObj.rmsnorm_backward(
            dyTensorAttr, xTensorAttr, scaleTensorAttr, invRmsTensorAttr, rmsnormBwdAttrs);

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

    RMSNormBackward()
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
using IntegrationGpuRMSNormBackwardPureFp32 = RMSNormBackward<float, float, float, float, float>;
using IntegrationGpuRMSNormBackwardPureFp16 = RMSNormBackward<half, half, float, half, float>;
using IntegrationGpuRMSNormBackwardPureBfp16
    = RMSNormBackward<bfloat16, bfloat16, float, bfloat16, float>;
using IntegrationGpuRMSNormBackwardMixedFp16 = RMSNormBackward<half, float, float, float, float>;
using IntegrationGpuRMSNormBackwardMixedBfp16
    = RMSNormBackward<bfloat16, float, float, float, float>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardPureFp32);
TEST_P(IntegrationGpuRMSNormBackwardPureFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardPureFp16);
TEST_P(IntegrationGpuRMSNormBackwardPureFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardPureBfp16);
TEST_P(IntegrationGpuRMSNormBackwardPureBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardMixedFp16);
TEST_P(IntegrationGpuRMSNormBackwardMixedFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackwardMixedBfp16);
TEST_P(IntegrationGpuRMSNormBackwardMixedBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardPureFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardPureFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardPureBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardMixedFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackwardMixedBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormBackwardPureFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormBackwardPureFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormBackwardPureBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormBackwardMixedFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuRMSNormBackwardMixedBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormFullTestCases())));

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuRMSNormBackward3dPureFp32 = IntegrationGpuRMSNormBackwardPureFp32;
using IntegrationGpuRMSNormBackward3dPureFp16 = IntegrationGpuRMSNormBackwardPureFp16;
using IntegrationGpuRMSNormBackward3dPureBfp16 = IntegrationGpuRMSNormBackwardPureBfp16;
using IntegrationGpuRMSNormBackward3dMixedFp16 = IntegrationGpuRMSNormBackwardMixedFp16;
using IntegrationGpuRMSNormBackward3dMixedBfp16 = IntegrationGpuRMSNormBackwardMixedBfp16;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackward3dPureFp32);
TEST_P(IntegrationGpuRMSNormBackward3dPureFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackward3dPureFp16);
TEST_P(IntegrationGpuRMSNormBackward3dPureFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackward3dPureBfp16);
TEST_P(IntegrationGpuRMSNormBackward3dPureBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackward3dMixedFp16);
TEST_P(IntegrationGpuRMSNormBackward3dMixedFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBackward3dMixedBfp16);
TEST_P(IntegrationGpuRMSNormBackward3dMixedBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackward3dPureFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackward3dPureFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackward3dPureBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackward3dMixedFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBackward3dMixedBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNorm3dTestCases())));
