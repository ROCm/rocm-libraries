// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/ActivationCommon.hpp"
#include "common/MatmulCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_matmul_common;

namespace
{

using MatmulActivTestCase = std::tuple<MatmulTestCase, test_activation_common::ActivTestCase>;

template <typename DataType>
class MatmulActiv : public IntegrationGraphVerificationHarness<DataType, MatmulActivTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> c;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const MatmulActivTestCase& tc)
    {
        const auto& [matmulTestCase, activTestCase] = tc;

        graph::Graph graphObj;
        graphObj.set_name("MatmulActivTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto aAttr = graph::makeTensorAttributes(
            "a",
            matmulTestCase.aDims,
            generateInputStrideOrder(matmulTestCase.aDims, matmulTestCase.transA));
        auto aTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(aAttr));

        auto bAttr = graph::makeTensorAttributes(
            "b",
            matmulTestCase.bDims,
            generateInputStrideOrder(matmulTestCase.bDims, matmulTestCase.transB));
        auto bTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(bAttr));

        const graph::MatmulAttributes matmulAttrs;
        auto cAttr = graphObj.matmul(aTensorAttr, bTensorAttr, matmulAttrs);

        graph::PointwiseAttributes activAttrs;
        activAttrs.set_mode(sdkToFrontendPointwiseMode(activTestCase.mode));
        if(activTestCase.reluLowerClip.has_value())
        {
            activAttrs.set_relu_lower_clip(activTestCase.reluLowerClip.value());
        }
        if(activTestCase.reluUpperClip.has_value())
        {
            activAttrs.set_relu_upper_clip(activTestCase.reluUpperClip.value());
        }
        if(activTestCase.reluLowerClipSlope.has_value())
        {
            activAttrs.set_relu_lower_clip_slope(activTestCase.reluLowerClipSlope.value());
        }
        if(activTestCase.swishBeta.has_value())
        {
            activAttrs.set_swish_beta(activTestCase.swishBeta.value());
        }
        if(activTestCase.eluAlpha.has_value())
        {
            activAttrs.set_elu_alpha(activTestCase.eluAlpha.value());
        }
        if(activTestCase.softplusBeta.has_value())
        {
            activAttrs.set_softplus_beta(activTestCase.softplusBeta.value());
        }

        auto cActivAttr = graphObj.pointwise(cAttr, activAttrs);
        cActivAttr->set_output(true);

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

        return std::make_pair(std::move(graphObj), GraphOutputs{cActivAttr});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [matmulTestCase, activTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.c, this->getTolerance(graphObj, outputs.c));

        this->setTestCaseNote(activTestCase.note);
        this->inputFillRecipes().setGlobalSeed(matmulTestCase.seed);
        this->verifyGraph(graphObj);
    }
};

using IntegrationGpuMatmulActivFp32 = MatmulActiv<float>;
using IntegrationGpuMatmulActivFp16 = MatmulActiv<half>;
using IntegrationGpuMatmulActivBf16 = MatmulActiv<bfloat16>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMatmulActivFp32);
TEST_P(IntegrationGpuMatmulActivFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMatmulActivFp16);
TEST_P(IntegrationGpuMatmulActivFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMatmulActivBf16);
TEST_P(IntegrationGpuMatmulActivBf16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuMatmulActivFp32,
    testing::Combine(testing::ValuesIn(getMatmulBiasActivTestCases()),
                     testing::ValuesIn(test_activation_common::createMatmulActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuMatmulActivFp16,
    testing::Combine(testing::ValuesIn(getMatmulBiasActivTestCases()),
                     testing::ValuesIn(test_activation_common::createMatmulActivationTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuMatmulActivBf16,
    testing::Combine(testing::ValuesIn(getMatmulBiasActivTestCases()),
                     testing::ValuesIn(test_activation_common::createMatmulActivationTestCases())));
