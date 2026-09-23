// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/BatchnormCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_bn_common;

namespace
{

struct BnInfVarTensorIds
{
    static constexpr int64_t X_UID = 1;
    static constexpr int64_t MEAN_UID = 2;
    static constexpr int64_t VARIANCE_UID = 3;
    static constexpr int64_t SCALE_UID = 4;
    static constexpr int64_t BIAS_UID = 5;
};

using BnFwdInfVarTestCase = std::tuple<TensorLayout, test_bn_common::BatchnormTestCase>;

template <typename DataType, typename OutputDataType = DataType>
class BatchnormForwardInferenceWithVariance
    : public IntegrationGraphVerificationHarness<DataType, BnFwdInfVarTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const BnFwdInfVarTestCase& tc)
    {
        const auto& [layout, testCase] = tc;

        auto derivedDims = getDerivedShape(testCase.dims);

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;

        graph::Graph graphObj;
        graphObj.set_name("BatchnormInferenceWithVarianceTest");
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "X", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        xAttr.set_uid(BnInfVarTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto meanAttr = graph::makeTensorAttributes(
            "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
        meanAttr.set_uid(BnInfVarTensorIds::MEAN_UID);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto varianceAttr = graph::makeTensorAttributes(
            "variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
        varianceAttr.set_uid(BnInfVarTensorIds::VARIANCE_UID);
        auto varianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(varianceAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        scaleAttr.set_uid(BnInfVarTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        biasAttr.set_uid(BnInfVarTensorIds::BIAS_UID);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        // Epsilon (pass-by-value)
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        epsilonTensorAttr->set_name("epsilon").set_value(1e-5);

        const graph::BatchnormInferenceAttributesVarianceExt bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference_variance_ext(xTensorAttr,
                                                                     meanTensorAttr,
                                                                     varianceTensorAttr,
                                                                     scaleTensorAttr,
                                                                     biasTensorAttr,
                                                                     epsilonTensorAttr,
                                                                     bnAttrs);
        yTensorAttr->set_output(true);
        yTensorAttr->set_data_type(getDataTypeEnumFromType<OutputDataType>());

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

        return {std::move(graphObj), GraphOutputs{yTensorAttr}};
    }

    BatchnormForwardInferenceWithVariance()
    {
        this->inputFillRecipes()
            .setRange(BnInfVarTensorIds::X_UID, -1.0f, 1.0f)
            .setRange(BnInfVarTensorIds::MEAN_UID, -1.0f, 1.0f)
            .setRange(BnInfVarTensorIds::VARIANCE_UID, 0.1f, 1.0f)
            .setRange(BnInfVarTensorIds::SCALE_UID, -1.0f, 1.0f)
            .setRange(BnInfVarTensorIds::BIAS_UID, -1.0f, 1.0f);
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, bnTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.y, this->getTolerance(graphObj, outputs.y));

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(bnTestCase.note);
        this->inputFillRecipes().setGlobalSeed(bnTestCase.seed);
        this->verifyGraph(graphObj);
    }
};

// 1D layout tests (NCL, NLC)
using IntegrationGpuBatchnormFwdInferenceVariance1dFp32
    = BatchnormForwardInferenceWithVariance<float>;
using IntegrationGpuBatchnormFwdInferenceVariance1dBfp16
    = BatchnormForwardInferenceWithVariance<bfloat16>;
using IntegrationGpuBatchnormFwdInferenceVariance1dFp16
    = BatchnormForwardInferenceWithVariance<half>;
using IntegrationGpuBatchnormFwdInferenceVariance1dUpcastBfp16
    = BatchnormForwardInferenceWithVariance<bfloat16, float>;
using IntegrationGpuBatchnormFwdInferenceVariance1dUpcastFp16
    = BatchnormForwardInferenceWithVariance<half, float>;

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuBatchnormFwdInferenceVariance2dFp32
    = BatchnormForwardInferenceWithVariance<float>;
using IntegrationGpuBatchnormFwdInferenceVariance2dBfp16
    = BatchnormForwardInferenceWithVariance<bfloat16>;
using IntegrationGpuBatchnormFwdInferenceVariance2dFp16
    = BatchnormForwardInferenceWithVariance<half>;
using IntegrationGpuBatchnormFwdInferenceVariance2dUpcastBfp16
    = BatchnormForwardInferenceWithVariance<bfloat16, float>;
using IntegrationGpuBatchnormFwdInferenceVariance2dUpcastFp16
    = BatchnormForwardInferenceWithVariance<half, float>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuBatchnormFwdInferenceVariance3dFp32
    = BatchnormForwardInferenceWithVariance<float>;
using IntegrationGpuBatchnormFwdInferenceVariance3dBfp16
    = BatchnormForwardInferenceWithVariance<bfloat16>;
using IntegrationGpuBatchnormFwdInferenceVariance3dFp16
    = BatchnormForwardInferenceWithVariance<half>;
using IntegrationGpuBatchnormFwdInferenceVariance3dUpcastBfp16
    = BatchnormForwardInferenceWithVariance<bfloat16, float>;
using IntegrationGpuBatchnormFwdInferenceVariance3dUpcastFp16
    = BatchnormForwardInferenceWithVariance<half, float>;

} // namespace

// ============================================================================
// 1D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceVariance1dFp32);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance1dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance1dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdInferenceVariance1dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceVariance1dBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance1dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance1dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdInferenceVariance1dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVariance1dUpcastBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance1dUpcastBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance1dUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdInferenceVariance1dUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceVariance1dFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance1dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance1dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdInferenceVariance1dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVariance1dUpcastFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance1dUpcastFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance1dUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdInferenceVariance1dUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dFullTestCases())));

// ============================================================================
// 2D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceVariance2dFp32);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance2dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance2dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdInferenceVariance2dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceVariance2dBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance2dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance2dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdInferenceVariance2dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVariance2dUpcastBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance2dUpcastBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance2dUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdInferenceVariance2dUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceVariance2dFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance2dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance2dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdInferenceVariance2dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVariance2dUpcastFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance2dUpcastFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance2dUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdInferenceVariance2dUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceFullTestCases())));

// ============================================================================
// 3D Tests (Smoke only)
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceVariance3dFp32);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance3dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance3dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceVariance3dBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance3dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance3dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVariance3dUpcastBfp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance3dUpcastBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance3dUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdInferenceVariance3dFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance3dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance3dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdInferenceVariance3dUpcastFp16);
TEST_P(IntegrationGpuBatchnormFwdInferenceVariance3dUpcastFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdInferenceVariance3dUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dTestCases())));
