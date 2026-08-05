// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/LayernormCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_layernorm_common;

namespace
{

using LayernormTestCaseType = std::tuple<TensorLayout, LayernormTestCase>;

// "Pure" = input, output, scale/bias, and mean/inv-variance all share precision. "Mixed" =
// input/output share a lower precision while scale/bias and mean/inv-variance stay FP32.
// "Upcast" = input is lower precision but output widens to FP32.
template <typename InputType,
          typename OutputType,
          typename ScaleBiasType,
          typename MeanInvVarianceType>
class Layernorm : public IntegrationGraphVerificationHarness<OutputType, LayernormTestCaseType>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
        std::shared_ptr<graph::TensorAttributes> mean; // nullptr in inference mode
        std::shared_ptr<graph::TensorAttributes> invVariance; // nullptr in inference mode
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const LayernormTestCaseType& tc)
    {
        const auto& [layout, testCase] = tc;

        std::vector<int64_t> affineDims(testCase.dims.size(), 1);
        for(size_t i = testCase.normalizedDim; i < testCase.dims.size(); ++i)
        {
            affineDims[i] = testCase.dims[i];
        }

        graph::Graph graphObj;
        graphObj.set_name("LayernormTest");
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        const auto inputType = getDataTypeEnumFromType<InputType>();
        const auto scaleBiasType = getDataTypeEnumFromType<ScaleBiasType>();

        auto ioStrides = generateStrides(testCase.dims, layout.strideOrder);
        auto affineStrides = generateStrides(affineDims, layout.strideOrder);

        auto xAttr = graph::makeTensorAttributes("x", inputType, testCase.dims, ioStrides);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr
            = graph::makeTensorAttributes("scale", scaleBiasType, affineDims, affineStrides);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr
            = graph::makeTensorAttributes("bias", scaleBiasType, affineDims, affineStrides);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto epsilonAttr = graph::makeTensorAttributes("epsilon", 1e-5f);
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(epsilonAttr));

        graph::LayernormAttributes lnAttrs;
        lnAttrs.set_epsilon(std::move(epsilonTensorAttr));
        lnAttrs.set_forward_phase(testCase.isTraining ? NormFwdPhase::TRAINING
                                                      : NormFwdPhase::INFERENCE);

        auto results = graphObj.layernorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, lnAttrs);
        const auto& yTensorAttr = results[0];
        const auto& meanTensorAttr = results[1];
        const auto& invVarianceTensorAttr = results[2];

        const auto outputType = getDataTypeEnumFromType<OutputType>();
        yTensorAttr->set_output(true).set_data_type(outputType);

        if(testCase.isTraining)
        {
            const auto meanInvVarianceType = getDataTypeEnumFromType<MeanInvVarianceType>();
            meanTensorAttr->set_output(true).set_data_type(meanInvVarianceType);
            invVarianceTensorAttr->set_output(true).set_data_type(meanInvVarianceType);
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

        return std::make_pair(std::move(graphObj),
                              GraphOutputs{yTensorAttr, meanTensorAttr, invVarianceTensorAttr});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& layernormTestCase = std::get<1>(testCase);

        // Inference-mode CPU reference only lines up with the GPU graph when mean/inv-variance
        // would share the input's own precision (the graph omits both tensors in inference
        // mode, but the reference executor still infers their precision from the graph).
        if(!layernormTestCase.isTraining && !std::is_same_v<InputType, MeanInvVarianceType>)
        {
            GTEST_SKIP() << "Skipping since the CPU reference implementation does not work "
                            "properly for this inference-mode mixed-precision case.";
        }

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.y, this->getTolerance(graphObj, outputs.y));
        if(outputs.mean)
        {
            this->registerValidator(outputs.mean, this->getTolerance(graphObj, outputs.mean));
            this->registerValidator(outputs.invVariance,
                                    this->getTolerance(graphObj, outputs.invVariance));
        }

        this->inputFillRecipes().setGlobalSeed(layernormTestCase.seed);
        this->verifyGraph(graphObj);
    }
};

using IntegrationGpuLayernormPureFp32 = Layernorm<float, float, float, float>;
using IntegrationGpuLayernormMixedFp16 = Layernorm<half, half, float, float>;
using IntegrationGpuLayernormMixedBfp16 = Layernorm<bfloat16, bfloat16, float, float>;
using IntegrationGpuLayernormUpcastFp16 = Layernorm<half, float, float, float>;
using IntegrationGpuLayernormUpcastBfp16 = Layernorm<bfloat16, float, float, float>;
using IntegrationGpuLayernormPureFp16 = Layernorm<half, half, half, half>;
using IntegrationGpuLayernormPureBfp16 = Layernorm<bfloat16, bfloat16, bfloat16, bfloat16>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormPureFp32);
TEST_P(IntegrationGpuLayernormPureFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormMixedFp16);
TEST_P(IntegrationGpuLayernormMixedFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormMixedBfp16);
TEST_P(IntegrationGpuLayernormMixedBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormUpcastFp16);
TEST_P(IntegrationGpuLayernormUpcastFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormUpcastBfp16);
TEST_P(IntegrationGpuLayernormUpcastBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormPureFp16);
TEST_P(IntegrationGpuLayernormPureFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormPureBfp16);
TEST_P(IntegrationGpuLayernormPureBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormPureFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormPureFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormMixedFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormMixedFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormMixedBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormMixedBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormPureFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormPureFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormPureBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormPureBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormPureFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormPureFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormMixedFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormMixedFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormMixedBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormMixedBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormPureFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormPureFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormPureBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernormFwd4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormPureBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernormFwd5DFullTestCases())));

// Heavy batch-256/512 volumetric shapes. Registered under a dedicated Full5dLargeBatch prefix
// (still a Full* prefix, so excluded from quick/standard/comprehensive) and skipped via each
// engine's test-config TOML until per-test tier filtering is fully wired.
INSTANTIATE_TEST_SUITE_P(
    Full5dLargeBatch,
    IntegrationGpuLayernormPureFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernormFwd5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full5dLargeBatch,
    IntegrationGpuLayernormMixedFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernormFwd5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full5dLargeBatch,
    IntegrationGpuLayernormMixedBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernormFwd5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full5dLargeBatch,
    IntegrationGpuLayernormUpcastFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernormFwd5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full5dLargeBatch,
    IntegrationGpuLayernormUpcastBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernormFwd5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full5dLargeBatch,
    IntegrationGpuLayernormPureFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernormFwd5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full5dLargeBatch,
    IntegrationGpuLayernormPureBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernormFwd5DLargeBatchTestCases())));
