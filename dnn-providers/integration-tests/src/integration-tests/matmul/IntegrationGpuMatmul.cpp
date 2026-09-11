// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/MatmulCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_matmul_common;

namespace
{

template <typename DataType>
class Matmul : public IntegrationGraphVerificationHarness<DataType, MatmulTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> c;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const MatmulTestCase& tc)
    {
        graph::Graph graphObj;
        graphObj.set_name("MatmulTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto aAttr = graph::makeTensorAttributes(
            "a", tc.aDims, generateInputStrideOrder(tc.aDims, tc.transA));
        auto aTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(aAttr));

        auto bAttr = graph::makeTensorAttributes(
            "b", tc.bDims, generateInputStrideOrder(tc.bDims, tc.transB));
        auto bTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(bAttr));

        const graph::MatmulAttributes matmulAttrs;
        auto cAttr = graphObj.matmul(aTensorAttr, bTensorAttr, matmulAttrs);
        cAttr->set_output(true);

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

        return std::make_pair(std::move(graphObj), GraphOutputs{cAttr});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.c, this->getTolerance(graphObj, outputs.c));

        this->inputFillRecipes().setGlobalSeed(testCase.seed);
        this->verifyGraph(graphObj);
    }
};

using IntegrationGpuMatmulFp32 = Matmul<float>;
using IntegrationGpuMatmulFp16 = Matmul<half>;
using IntegrationGpuMatmulBf16 = Matmul<bfloat16>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMatmulFp32);
TEST_P(IntegrationGpuMatmulFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMatmulFp16);
TEST_P(IntegrationGpuMatmulFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMatmulBf16);
TEST_P(IntegrationGpuMatmulBf16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke, IntegrationGpuMatmulFp32, testing::ValuesIn(getMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke, IntegrationGpuMatmulFp16, testing::ValuesIn(getMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke, IntegrationGpuMatmulBf16, testing::ValuesIn(getMatmulTestCases()));
