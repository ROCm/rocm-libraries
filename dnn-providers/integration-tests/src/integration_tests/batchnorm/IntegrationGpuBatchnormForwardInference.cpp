// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
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

using BnFwdInfTestCase = std::tuple<TensorLayout, test_bn_common::BatchnormTestCase>;

template <typename DataType>
class BatchnormForwardInference
    : public IntegrationGraphVerificationHarness<DataType, BnFwdInfTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const BnFwdInfTestCase& tc)
    {
        const auto& [layout, bnTestCase] = tc;
        auto dims = bnTestCase.dims;
        auto derivedDims = getDerivedShape(dims);

        graph::Graph graphObj;
        graphObj.set_name("BatchnormInferenceTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr
            = graph::makeTensorAttributes("X", dims, generateStrides(dims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto meanAttr = graph::makeTensorAttributes(
            "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = graph::makeTensorAttributes(
            "inv_variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        const graph::BatchnormInferenceAttributes bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                        meanTensorAttr,
                                                        invVarianceTensorAttr,
                                                        scaleTensorAttr,
                                                        biasTensorAttr,
                                                        bnAttrs);

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

        return {std::move(graphObj), GraphOutputs{yTensorAttr}};
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, bnTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.y, this->getTolerance(graphObj, outputs.y));

        this->verifyGraph(graphObj, bnTestCase.seed);
    }
};

// 1D layout tests (NCL, NLC)
using IntegrationGpuBatchnormForwardInference1dFp32 = BatchnormForwardInference<float>;
using IntegrationGpuBatchnormForwardInference1dBfp16 = BatchnormForwardInference<bfloat16>;
using IntegrationGpuBatchnormForwardInference1dFp16 = BatchnormForwardInference<half>;

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuBatchnormForwardInference2dFp32 = BatchnormForwardInference<float>;
using IntegrationGpuBatchnormForwardInference2dBfp16 = BatchnormForwardInference<bfloat16>;
using IntegrationGpuBatchnormForwardInference2dFp16 = BatchnormForwardInference<half>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuBatchnormForwardInference3dFp32 = BatchnormForwardInference<float>;
using IntegrationGpuBatchnormForwardInference3dBfp16 = BatchnormForwardInference<bfloat16>;
using IntegrationGpuBatchnormForwardInference3dFp16 = BatchnormForwardInference<half>;

} // namespace

// ============================================================================
// 1D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormForwardInference1dFp32);
TEST_P(IntegrationGpuBatchnormForwardInference1dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInference1dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInference1dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormForwardInference1dBfp16);
TEST_P(IntegrationGpuBatchnormForwardInference1dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInference1dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInference1dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormForwardInference1dFp16);
TEST_P(IntegrationGpuBatchnormForwardInference1dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInference1dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInference1dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnFwdInference1dFullTestCases())));

// ============================================================================
// 2D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormForwardInference2dFp32);
TEST_P(IntegrationGpuBatchnormForwardInference2dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInference2dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInference2dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormForwardInference2dBfp16);
TEST_P(IntegrationGpuBatchnormForwardInference2dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInference2dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInference2dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormForwardInference2dFp16);
TEST_P(IntegrationGpuBatchnormForwardInference2dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInference2dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInference2dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnFwdInferenceFullTestCases())));

// ============================================================================
// 3D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormForwardInference3dFp32);
TEST_P(IntegrationGpuBatchnormForwardInference3dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInference3dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInference3dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormForwardInference3dBfp16);
TEST_P(IntegrationGpuBatchnormForwardInference3dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInference3dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInference3dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormForwardInference3dFp16);
TEST_P(IntegrationGpuBatchnormForwardInference3dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInference3dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInference3dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnFwdInference3dFullTestCases())));
