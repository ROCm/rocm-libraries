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

        this->synthesis().setGlobalSeed(layernormTestCase.seed);
        this->verifyGraph(graphObj);
    }
};

using IntegrationGpuLayernormBackwardPureFp32 = LayernormBackward<float, float, float, float>;
using IntegrationGpuLayernormBackwardMixedFp16 = LayernormBackward<half, half, float, float>;
using IntegrationGpuLayernormBackwardMixedBfp16
    = LayernormBackward<bfloat16, bfloat16, float, float>;
using IntegrationGpuLayernormBackwardUpcastFp16 = LayernormBackward<half, float, float, float>;
using IntegrationGpuLayernormBackwardUpcastBfp16 = LayernormBackward<bfloat16, float, float, float>;
using IntegrationGpuLayernormBackwardPureFp16 = LayernormBackward<half, half, half, half>;
using IntegrationGpuLayernormBackwardPureBfp16
    = LayernormBackward<bfloat16, bfloat16, bfloat16, bfloat16>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardPureFp32);
TEST_P(IntegrationGpuLayernormBackwardPureFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardMixedFp16);
TEST_P(IntegrationGpuLayernormBackwardMixedFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardMixedBfp16);
TEST_P(IntegrationGpuLayernormBackwardMixedBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardUpcastFp16);
TEST_P(IntegrationGpuLayernormBackwardUpcastFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardUpcastBfp16);
TEST_P(IntegrationGpuLayernormBackwardUpcastBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardPureFp16);
TEST_P(IntegrationGpuLayernormBackwardPureFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuLayernormBackwardPureBfp16);
TEST_P(IntegrationGpuLayernormBackwardPureBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormBackwardPureFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormBackwardPureFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormBackwardMixedFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormBackwardMixedFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormBackwardMixedBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormBackwardMixedBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormBackwardUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormBackwardUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormBackwardUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormBackwardUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormBackwardPureFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormBackwardPureFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke4d,
                         IntegrationGpuLayernormBackwardPureBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Smoke5d,
                         IntegrationGpuLayernormBackwardPureBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormBackwardPureFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormBackwardPureFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormBackwardMixedFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormBackwardMixedFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormBackwardMixedBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormBackwardMixedBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormBackwardUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormBackwardUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormBackwardUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormBackwardUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormBackwardPureFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormBackwardPureFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full4d,
                         IntegrationGpuLayernormBackwardPureBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getLayernorm4DFullTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5d,
                         IntegrationGpuLayernormBackwardPureBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DFullTestCases())));

// Heavy batch-256/512 volumetric shapes. Registered under a dedicated Full5dLargeBatch prefix
// (still a Full* prefix, so excluded from quick/standard/comprehensive) and skipped via each
// engine's test-config TOML until per-test tier filtering is fully wired.
INSTANTIATE_TEST_SUITE_P(Full5dLargeBatch,
                         IntegrationGpuLayernormBackwardPureFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5dLargeBatch,
                         IntegrationGpuLayernormBackwardMixedFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5dLargeBatch,
                         IntegrationGpuLayernormBackwardMixedBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5dLargeBatch,
                         IntegrationGpuLayernormBackwardUpcastFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5dLargeBatch,
                         IntegrationGpuLayernormBackwardUpcastBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5dLargeBatch,
                         IntegrationGpuLayernormBackwardPureFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DLargeBatchTestCases())));
INSTANTIATE_TEST_SUITE_P(Full5dLargeBatch,
                         IntegrationGpuLayernormBackwardPureBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getLayernorm5DLargeBatchTestCases())));
