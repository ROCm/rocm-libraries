// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
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

using RMSNormTestCaseType = std::tuple<TensorLayout, RMSNormTestCase>;

template <typename XType, typename ScaleType, typename YType>
class RMSNorm : public IntegrationGraphVerificationHarness<YType, RMSNormTestCaseType>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
        std::shared_ptr<graph::TensorAttributes> invRms; // nullptr in inference mode
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const RMSNormTestCaseType& tc)
    {
        const auto& [layout, testCase] = tc;

        graph::Graph graphObj;
        graphObj.set_name("RMSNormTest");

        const auto xType = getDataTypeEnumFromType<XType>();
        const auto scaleType = getDataTypeEnumFromType<ScaleType>();
        const auto yType = getDataTypeEnumFromType<YType>();
        const auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(yType);

        auto xAttr = graph::makeTensorAttributes(
            "x", xType, testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr
            = graph::makeTensorAttributes("scale",
                                          scaleType,
                                          testCase.scaleDims,
                                          generateStrides(testCase.scaleDims, layout.strideOrder));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        epsilonTensorAttr->set_name("epsilon").set_value(testCase.epsilon);

        graph::RMSNormAttributes rmsAttrs;
        rmsAttrs.set_epsilon(epsilonTensorAttr)
            .set_forward_phase(testCase.isTraining ? NormFwdPhase::TRAINING
                                                   : NormFwdPhase::INFERENCE);

        if(testCase.biasDims.has_value())
        {
            auto biasAttr = graph::makeTensorAttributes(
                "bias",
                scaleType,
                *testCase.biasDims,
                generateStrides(*testCase.biasDims, layout.strideOrder));
            auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));
            rmsAttrs.set_bias(biasTensorAttr);
        }

        auto [yOut, invRmsOut] = graphObj.rmsnorm(xTensorAttr, scaleTensorAttr, rmsAttrs);
        yOut->set_output(true);

        if(testCase.isTraining)
        {
            invRmsOut->set_output(true).set_data_type(intermediateDataType);
        }

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

        return std::make_pair(std::move(graphObj), GraphOutputs{yOut, invRmsOut});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, rmsnormTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.y, this->getTolerance(graphObj, outputs.y));
        if(outputs.invRms)
        {
            this->registerValidator(outputs.invRms, this->getTolerance(graphObj, outputs.invRms));
        }

        this->verifyGraph(graphObj, rmsnormTestCase.seed);
    }
};

using IntegrationGpuRMSNormFp32 = RMSNorm<float, float, float>;
using IntegrationGpuRMSNormBfp16 = RMSNorm<bfloat16, bfloat16, bfloat16>;
using IntegrationGpuRMSNormFp16 = RMSNorm<half, half, half>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormFp32);
TEST_P(IntegrationGpuRMSNormFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormBfp16);
TEST_P(IntegrationGpuRMSNormBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuRMSNormFp16);
TEST_P(IntegrationGpuRMSNormFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuRMSNormFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_rmsnorm_common::getRMSNormTestCases())));
