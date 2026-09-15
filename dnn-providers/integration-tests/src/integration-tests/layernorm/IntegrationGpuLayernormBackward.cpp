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

// "Pure" = input, output, scale/bias, and mean/inv-variance all share precision.
// "Mixed" = input/output share a lower precision while scale/bias and mean/inv-variance stay FP32.
// "Upcast" = input is lower precision but output widens to FP32.
template <typename InputDataType,
          typename OutputDataType,
          typename ScaleBiasDataType,
          typename MeanInvVarianceDataType>
class LayernormBackward
    : public IntegrationGraphVerificationHarness<OutputDataType, LayernormTestCaseType>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> dx;
        std::shared_ptr<graph::TensorAttributes> dscale;
        std::shared_ptr<graph::TensorAttributes> dbias;
    };

    struct LayernormBwdTensorIds
    {
        static constexpr int64_t DY_UID = 1;
        static constexpr int64_t X_UID = 2;
        static constexpr int64_t SCALE_UID = 3;
        static constexpr int64_t MEAN_UID = 4;
        static constexpr int64_t INV_VARIANCE_UID = 5;
        static constexpr int64_t EPSILON_UID = 6;
        static constexpr int64_t DX_UID = 7;
        static constexpr int64_t DSCALE_UID = 8;
        static constexpr int64_t DBIAS_UID = 9;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const LayernormTestCaseType& tc)
    {
        const auto& [layout, testCase] = tc;

        std::vector<int64_t> statDims(testCase.dims.size(), 1);
        std::vector<int64_t> affineDims(testCase.dims.size(), 1);
        for(size_t i = 0; i < testCase.dims.size(); ++i)
        {
            if(i < testCase.normalizedDim)
            {
                statDims[i] = testCase.dims[i];
            }
            else
            {
                affineDims[i] = testCase.dims[i];
            }
        }

        graph::Graph graphObj;
        graphObj.set_name("LayernormBwdTest");
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        auto inputDataType = getDataTypeEnumFromType<InputDataType>();
        auto outputDataType = getDataTypeEnumFromType<OutputDataType>();
        auto scaleBiasDataType = getDataTypeEnumFromType<ScaleBiasDataType>();
        auto meanInvVarianceDataType = getDataTypeEnumFromType<MeanInvVarianceDataType>();

        auto ioStrides = generateStrides(testCase.dims, layout.strideOrder);
        auto statStrides = generateStrides(statDims, layout.strideOrder);
        auto affineStrides = generateStrides(affineDims, layout.strideOrder);

        auto dyAttr = graph::makeTensorAttributes("dY", outputDataType, testCase.dims, ioStrides);
        dyAttr.set_uid(LayernormBwdTensorIds::DY_UID);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto xAttr = graph::makeTensorAttributes("X", inputDataType, testCase.dims, ioStrides);
        xAttr.set_uid(LayernormBwdTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr
            = graph::makeTensorAttributes("scale", scaleBiasDataType, affineDims, affineStrides);
        scaleAttr.set_uid(LayernormBwdTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        graph::LayernormBackwardAttributes lnAttrs;
        if(testCase.optionalTensors)
        {
            auto meanAttr = graph::makeTensorAttributes(
                "mean", meanInvVarianceDataType, statDims, statStrides);
            meanAttr.set_uid(LayernormBwdTensorIds::MEAN_UID);
            auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

            auto rstdAttr = graph::makeTensorAttributes(
                "rstd", meanInvVarianceDataType, statDims, statStrides);
            rstdAttr.set_uid(LayernormBwdTensorIds::INV_VARIANCE_UID);
            auto rstdTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(rstdAttr));

            lnAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, rstdTensorAttr);
        }

        auto epsilonAttr
            = graph::makeTensorAttributes("epsilon", static_cast<float>(LAYERNORM_DEFAULT_EPSILON));
        epsilonAttr.set_uid(LayernormBwdTensorIds::EPSILON_UID);
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(epsilonAttr));
        lnAttrs.set_epsilon(std::move(epsilonTensorAttr));

        auto results
            = graphObj.layernorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, lnAttrs);
        const auto& dxTensorAttr = results[0];
        dxTensorAttr->set_uid(LayernormBwdTensorIds::DX_UID);
        const auto& dscaleTensorAttr = results[1];
        dscaleTensorAttr->set_uid(LayernormBwdTensorIds::DSCALE_UID);
        const auto& dbiasTensorAttr = results[2];
        dbiasTensorAttr->set_uid(LayernormBwdTensorIds::DBIAS_UID);

        dxTensorAttr->set_output(true).set_data_type(inputDataType);
        dscaleTensorAttr->set_output(true).set_data_type(scaleBiasDataType);
        dbiasTensorAttr->set_output(true).set_data_type(scaleBiasDataType);

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

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& layernormTestCase = std::get<1>(testCase);

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.dx, this->getTolerance(graphObj, outputs.dx));
        // RMS validator as the standard validator breaks down for resulting elements that happen to be near zero after summing hundreds of thousands of floating point values
        this->registerRmsValidator(outputs.dscale, this->getTolerance(graphObj, outputs.dscale));
        this->registerRmsValidator(outputs.dbias, this->getTolerance(graphObj, outputs.dbias));

        this->inputFillRecipes().setGlobalSeed(layernormTestCase.seed);
        this->verifyGraph(graphObj);
    }
};

using IntegrationGpuLayernormBackwardPure4DFp32 = LayernormBackward<float, float, float, float>;
using IntegrationGpuLayernormBackwardMixed4DFp16 = LayernormBackward<half, half, float, float>;
using IntegrationGpuLayernormBackwardMixed4DBfp16
    = LayernormBackward<bfloat16, bfloat16, float, float>;
using IntegrationGpuLayernormBackwardUpcast4DFp16 = LayernormBackward<half, float, float, float>;
using IntegrationGpuLayernormBackwardUpcast4DBfp16
    = LayernormBackward<bfloat16, float, float, float>;
using IntegrationGpuLayernormBackwardPure4DFp16 = LayernormBackward<half, half, half, half>;
using IntegrationGpuLayernormBackwardPure4DBfp16
    = LayernormBackward<bfloat16, bfloat16, bfloat16, bfloat16>;

using IntegrationGpuLayernormBackwardPure5DFp32 = LayernormBackward<float, float, float, float>;
using IntegrationGpuLayernormBackwardMixed5DFp16 = LayernormBackward<half, half, float, float>;
using IntegrationGpuLayernormBackwardMixed5DBfp16
    = LayernormBackward<bfloat16, bfloat16, float, float>;
using IntegrationGpuLayernormBackwardUpcast5DFp16 = LayernormBackward<half, float, float, float>;
using IntegrationGpuLayernormBackwardUpcast5DBfp16
    = LayernormBackward<bfloat16, float, float, float>;
using IntegrationGpuLayernormBackwardPure5DFp16 = LayernormBackward<half, half, half, half>;
using IntegrationGpuLayernormBackwardPure5DBfp16
    = LayernormBackward<bfloat16, bfloat16, bfloat16, bfloat16>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardPure4DFp32);
TEST_P(IntegrationGpuLayernormBackwardPure4DFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardMixed4DFp16);
TEST_P(IntegrationGpuLayernormBackwardMixed4DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardMixed4DBfp16);
TEST_P(IntegrationGpuLayernormBackwardMixed4DBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardUpcast4DFp16);
TEST_P(IntegrationGpuLayernormBackwardUpcast4DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardUpcast4DBfp16);
TEST_P(IntegrationGpuLayernormBackwardUpcast4DBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardPure4DFp16);
TEST_P(IntegrationGpuLayernormBackwardPure4DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardPure4DBfp16);
TEST_P(IntegrationGpuLayernormBackwardPure4DBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardPure5DFp32);
TEST_P(IntegrationGpuLayernormBackwardPure5DFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardMixed5DFp16);
TEST_P(IntegrationGpuLayernormBackwardMixed5DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardMixed5DBfp16);
TEST_P(IntegrationGpuLayernormBackwardMixed5DBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardUpcast5DFp16);
TEST_P(IntegrationGpuLayernormBackwardUpcast5DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardUpcast5DBfp16);
TEST_P(IntegrationGpuLayernormBackwardUpcast5DBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardPure5DFp16);
TEST_P(IntegrationGpuLayernormBackwardPure5DFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardPure5DBfp16);
TEST_P(IntegrationGpuLayernormBackwardPure5DBfp16, Correctness)
{
    runGraphTest();
}

// Tier prefixes match the CTest categories in each provider's
// test_categories_integration.yaml. The tiers are cumulative there and disjoint
// here, so each case runs in exactly one tier and nothing is repeated.

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardPure4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardMixed4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardMixed4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardUpcast4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardUpcast4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardPure4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardPure4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DQuickTestCases())));

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardPure5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardMixed5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardMixed5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardUpcast5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardUpcast5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardPure5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuLayernormBackwardPure5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DQuickTestCases())));

INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardPure4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardMixed4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardMixed4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardUpcast4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardUpcast4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardPure4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardPure4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DStandardTestCases())));

INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardPure5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardMixed5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardMixed5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardUpcast5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardUpcast5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardPure5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuLayernormBackwardPure5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DStandardTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormBackwardPure5DFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormBackwardMixed5DFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormBackwardMixed5DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormBackwardUpcast5DFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormBackwardUpcast5DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormBackwardPure5DFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Comprehensive,
    IntegrationGpuLayernormBackwardPure5DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(getLayernorm5DComprehensiveTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardPure5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardMixed5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardMixed5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardUpcast5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardUpcast5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardPure5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardPure5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));
