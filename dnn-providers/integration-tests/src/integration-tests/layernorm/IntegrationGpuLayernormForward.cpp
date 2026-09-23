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

        auto epsilonAttr
            = graph::makeTensorAttributes("epsilon", static_cast<float>(LAYERNORM_DEFAULT_EPSILON));
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(epsilonAttr));

        graph::LayernormAttributes lnAttrs;
        lnAttrs.set_epsilon(std::move(epsilonTensorAttr));
        lnAttrs.set_forward_phase(testCase.optionalTensors ? NormFwdPhase::TRAINING
                                                           : NormFwdPhase::INFERENCE);

        auto results = graphObj.layernorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, lnAttrs);
        const auto& yTensorAttr = results[0];
        const auto& meanTensorAttr = results[1];
        const auto& invVarianceTensorAttr = results[2];

        const auto outputType = getDataTypeEnumFromType<OutputType>();
        yTensorAttr->set_output(true).set_data_type(outputType);

        if(testCase.optionalTensors)
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
        if(!layernormTestCase.optionalTensors && !std::is_same_v<InputType, MeanInvVarianceType>)
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

// One alias per rank so each INSTANTIATE_TEST_SUITE_P can use a plain tier prefix:
// GTest keys instantiations on (prefix, fixture), so 4D and 5D need distinct
// fixtures to share the prefix. Matches IntegrationGpuLayernormBackward.cpp.
using IntegrationGpuLayernormPure4DFp32 = Layernorm<float, float, float, float>;
using IntegrationGpuLayernormMixed4DFp16 = Layernorm<half, half, float, float>;
using IntegrationGpuLayernormMixed4DBfp16 = Layernorm<bfloat16, bfloat16, float, float>;
using IntegrationGpuLayernormUpcast4DFp16 = Layernorm<half, float, float, float>;
using IntegrationGpuLayernormUpcast4DBfp16 = Layernorm<bfloat16, float, float, float>;
using IntegrationGpuLayernormPure4DFp16 = Layernorm<half, half, half, half>;
using IntegrationGpuLayernormPure4DBfp16 = Layernorm<bfloat16, bfloat16, bfloat16, bfloat16>;

using IntegrationGpuLayernormPure5DFp32 = Layernorm<float, float, float, float>;
using IntegrationGpuLayernormMixed5DFp16 = Layernorm<half, half, float, float>;
using IntegrationGpuLayernormMixed5DBfp16 = Layernorm<bfloat16, bfloat16, float, float>;
using IntegrationGpuLayernormUpcast5DFp16 = Layernorm<half, float, float, float>;
using IntegrationGpuLayernormUpcast5DBfp16 = Layernorm<bfloat16, float, float, float>;
using IntegrationGpuLayernormPure5DFp16 = Layernorm<half, half, half, half>;
using IntegrationGpuLayernormPure5DBfp16 = Layernorm<bfloat16, bfloat16, bfloat16, bfloat16>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormPure4DFp32);
TEST_P(IntegrationGpuLayernormPure4DFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormMixed4DFp16);
TEST_P(IntegrationGpuLayernormMixed4DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormMixed4DBfp16);
TEST_P(IntegrationGpuLayernormMixed4DBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormUpcast4DFp16);
TEST_P(IntegrationGpuLayernormUpcast4DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormUpcast4DBfp16);
TEST_P(IntegrationGpuLayernormUpcast4DBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormPure4DFp16);
TEST_P(IntegrationGpuLayernormPure4DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormPure4DBfp16);
TEST_P(IntegrationGpuLayernormPure4DBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormPure5DFp32);
TEST_P(IntegrationGpuLayernormPure5DFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormMixed5DFp16);
TEST_P(IntegrationGpuLayernormMixed5DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormMixed5DBfp16);
TEST_P(IntegrationGpuLayernormMixed5DBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormUpcast5DFp16);
TEST_P(IntegrationGpuLayernormUpcast5DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormUpcast5DBfp16);
TEST_P(IntegrationGpuLayernormUpcast5DBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormPure5DFp16);
TEST_P(IntegrationGpuLayernormPure5DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormPure5DBfp16);
TEST_P(IntegrationGpuLayernormPure5DBfp16, Correctness)
{
    runGraphTest();
}

// Tier prefixes match the CTest categories in each provider's
// test_categories_integration.yaml. The tiers are cumulative there and disjoint
// here, so each case runs in exactly one tier and nothing is repeated.

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormPure4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormMixed4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormMixed4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormUpcast4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormUpcast4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormPure4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormPure4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormPure5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormMixed5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormMixed5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormUpcast5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormUpcast5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormPure5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormPure5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));

INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormPure4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormMixed4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormMixed4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormUpcast4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormUpcast4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormPure4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormPure4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));

INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormPure5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormMixed5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormMixed5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormUpcast5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormUpcast5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormPure5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormPure5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormPure5DFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormMixed5DFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormMixed5DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormUpcast5DFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormUpcast5DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormPure5DFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormPure5DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormPure5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormMixed5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormMixed5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormUpcast5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormUpcast5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormPure5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormPure5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
