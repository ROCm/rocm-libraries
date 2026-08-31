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
class MatmulBias : public IntegrationGraphVerificationHarness<DataType, MatmulTestCase>
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
        graphObj.set_name("MatmulBiasTest");

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

        std::vector<int64_t> biasDims(tc.cDims.size(), 1);
        biasDims.back() = tc.cDims.back();

        auto biasAttr = graph::makeTensorAttributes(
            "bias", biasDims, generateInputStrideOrder(biasDims, false));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        graph::PointwiseAttributes biasAttrs;
        biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);

        auto cBiasAttr = graphObj.pointwise(cAttr, biasTensorAttr, biasAttrs);
        cBiasAttr->set_output(true);

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

        return std::make_pair(std::move(graphObj), GraphOutputs{cBiasAttr});
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

using IntegrationGpuMatmulBiasFp32 = MatmulBias<float>;
using IntegrationGpuMatmulBiasFp16 = MatmulBias<half>;
using IntegrationGpuMatmulBiasBf16 = MatmulBias<bfloat16>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMatmulBiasFp32);
TEST_P(IntegrationGpuMatmulBiasFp32, Correctness)
{
    runGraphTest();
}

// hipBLASLt gfx12 FP16 T-T + bias epilogue gap (issue #8033) is exempted via
// the engine's [[test_skips]] TOML config, not a code-level workaround here.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMatmulBiasFp16);
TEST_P(IntegrationGpuMatmulBiasFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMatmulBiasBf16);
TEST_P(IntegrationGpuMatmulBiasBf16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMatmulBiasFp32,
                         testing::ValuesIn(getMatmulBiasActivTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMatmulBiasFp16,
                         testing::ValuesIn(getMatmulBiasActivTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMatmulBiasBf16,
                         testing::ValuesIn(getMatmulBiasActivTestCases()));
