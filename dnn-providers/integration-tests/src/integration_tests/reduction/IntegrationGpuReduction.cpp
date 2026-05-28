// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/ReductionCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_reduction_common;

namespace
{

using ReductionTestCaseType = std::tuple<TensorLayout, ReductionTestCase>;

template <typename DataType>
class Reduction : public IntegrationGraphVerificationHarness<DataType, ReductionTestCaseType>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const ReductionTestCaseType& tc)
    {
        const auto& [layout, testCase] = tc;

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ReductionTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto yAttr = graph::makeTensorAttributes(
            "y", testCase.yDims, generateStrides(testCase.yDims, layout.strideOrder));
        auto yTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(yAttr));

        graph::ReductionAttributes redAttrs;
        redAttrs.set_mode(testCase.mode);

        auto yOut = graphObj.reduction(xTensorAttr, yTensorAttr, redAttrs);
        yOut->set_output(true);

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

        return std::make_pair(std::move(graphObj), GraphOutputs{yOut});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, reductionTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.y, this->getTolerance(graphObj, outputs.y));

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(reductionTestCase.note);
        this->verifyGraph(graphObj, reductionTestCase.seed);
    }
};

using IntegrationGpuReductionFp32 = Reduction<float>;
using IntegrationGpuReductionBfp16 = Reduction<bfloat16>;
using IntegrationGpuReductionFp16 = Reduction<half>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuReductionFp32);
TEST_P(IntegrationGpuReductionFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuReductionBfp16);
TEST_P(IntegrationGpuReductionBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuReductionFp16);
TEST_P(IntegrationGpuReductionFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuReductionFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_reduction_common::getReductionTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuReductionBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_reduction_common::getReductionTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuReductionFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_reduction_common::getReductionTestCases())));
