// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <random>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/ActivationCommon.hpp"
#include "common/BatchnormCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;

namespace
{

using test_activation_common::ActivTestCase;
using test_bn_common::AffineShapeMode;
using test_bn_common::BatchnormTestCase;

struct BatchnormFwdTrainingActivTensorIds
{
    static constexpr int64_t X_UID = 1;
    static constexpr int64_t SCALE_UID = 2;
    static constexpr int64_t BIAS_UID = 3;
    static constexpr int64_t PREV_RUNNING_MEAN_UID = 4;
    static constexpr int64_t PREV_RUNNING_VARIANCE_UID = 5;
    static constexpr int64_t NEXT_RUNNING_MEAN_UID = 6;
    static constexpr int64_t NEXT_RUNNING_VARIANCE_UID = 7;
};

// Note: hipDNN BatchNorm implements Spatial normalization only (miopenBNSpatial).
// The mode is hardcoded in the MIOpen plugin (see MiopenBatchnormFwdTrainingActivPlan.cpp).
// Per-activation normalization would require LayerNorm or InstanceNorm operations.
//
// These scenarios test different output combinations in forward training:
// - WITH_BATCH_STATS: Computes batch statistics (mean/invVariance) without updating running stats
// - FULL_TRAINING: Computes batch statistics AND updates running mean/variance via EMA
enum class BatchnormTrainingScenario
{
    WITH_BATCH_STATS, // Batch stats only (no running stats update)
    FULL_TRAINING // Batch stats + running stats update (canonical training)
};

using BnFwdTrainingActivTestCase = std::tuple<TensorLayout,
                                              BatchnormTrainingScenario,
                                              test_bn_common::BatchnormTestCase,
                                              test_activation_common::ActivTestCase>;

template <typename InputType, AffineShapeMode Mode = AffineShapeMode::FULL_RANK>
class BatchnormFwdTrainingActivation
    : public IntegrationGraphVerificationHarness<InputType, BnFwdTrainingActivTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> yActiv;
        std::shared_ptr<graph::TensorAttributes> mean;
        std::shared_ptr<graph::TensorAttributes> invVariance;
        std::shared_ptr<graph::TensorAttributes> nextRunningMean;
        std::shared_ptr<graph::TensorAttributes> nextRunningVariance;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const BnFwdTrainingActivTestCase& tc)
    {
        const auto& [layout, scenario, bnTestCase, activTestCase] = tc;
        auto dims = bnTestCase.dims;
        auto derivedDims = getDerivedShape(dims);

        // Full-rank {1, C, 1, ...} or reduced-rank per-channel scale/bias (PR #7566);
        // running statistics remain full-rank as required by all providers.
        const auto affineDims = Mode == AffineShapeMode::REDUCED_RANK
                                    ? test_bn_common::getReducedAffineShape(layout, dims)
                                    : derivedDims;

        auto inputDataType = getDataTypeEnumFromType<InputType>();
        auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;

        graph::Graph graphObj;
        graphObj.set_name("BatchnormFwdTrainingActivTest");
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(inputDataType);

        // Create input tensor attributes
        auto xAttr
            = graph::makeTensorAttributes("X", dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(BatchnormFwdTrainingActivTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, affineDims, generateStrides(affineDims));
        scaleAttr.set_uid(BatchnormFwdTrainingActivTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, affineDims, generateStrides(affineDims));
        biasAttr.set_uid(BatchnormFwdTrainingActivTensorIds::BIAS_UID);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        // Epsilon: use pass-by-value with double (matches MIOpen API)
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        std::mt19937 gen(bnTestCase.seed);
        std::uniform_real_distribution<double> epsilonDist(1e-6, 1e-4);
        epsilonTensorAttr->set_value(epsilonDist(gen)).set_name("epsilon");

        // Conditionally setup running statistics based on scenario
        std::shared_ptr<graph::TensorAttributes> prevRunningMeanTensorAttr;
        std::shared_ptr<graph::TensorAttributes> prevRunningVarianceTensorAttr;
        std::shared_ptr<graph::TensorAttributes> momentumTensorAttr;

        if(scenario == BatchnormTrainingScenario::FULL_TRAINING)
        {
            auto prevRunningMeanAttr = graph::makeTensorAttributes("prev_running_mean",
                                                                   intermediateDataType,
                                                                   derivedDims,
                                                                   generateStrides(derivedDims));
            prevRunningMeanAttr.set_uid(BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_MEAN_UID);
            prevRunningMeanTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(prevRunningMeanAttr));

            auto prevRunningVarianceAttr
                = graph::makeTensorAttributes("prev_running_variance",
                                              intermediateDataType,
                                              derivedDims,
                                              generateStrides(derivedDims));
            prevRunningVarianceAttr.set_uid(
                BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_VARIANCE_UID);
            prevRunningVarianceTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(prevRunningVarianceAttr));

            // Momentum: use pass-by-value with double (matches MIOpen API)
            momentumTensorAttr = std::make_shared<graph::TensorAttributes>();
            std::uniform_real_distribution<double> momentumDist(0.05, 0.15);
            momentumTensorAttr->set_value(momentumDist(gen)).set_name("momentum");
        }

        // Create batchnorm attributes
        graph::BatchnormAttributes bnAttrs;

        if(prevRunningMeanTensorAttr && prevRunningVarianceTensorAttr && momentumTensorAttr)
        {
            bnAttrs.set_previous_running_stats(
                prevRunningMeanTensorAttr, prevRunningVarianceTensorAttr, momentumTensorAttr);
        }

        bnAttrs.set_epsilon(epsilonTensorAttr);

        auto [yBnTensorAttr,
              meanTensorAttr,
              invVarianceTensorAttr,
              nextRunningMeanTensorAttr,
              nextRunningVarianceTensorAttr]
            = graphObj.batchnorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

        yBnTensorAttr->set_data_type(intermediateDataType);

        // Add activation node with parameters from test case
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

        auto yActivTensorAttr = graphObj.pointwise(yBnTensorAttr, activAttrs);
        yActivTensorAttr->set_output(true);

        // Configure batch statistics outputs
        if(meanTensorAttr)
        {
            meanTensorAttr->set_output(true);
            meanTensorAttr->set_data_type(intermediateDataType);
        }

        if(invVarianceTensorAttr)
        {
            invVarianceTensorAttr->set_output(true);
            invVarianceTensorAttr->set_data_type(intermediateDataType);
        }

        // Configure running statistics outputs if they exist
        if(nextRunningMeanTensorAttr)
        {
            nextRunningMeanTensorAttr->set_uid(
                BatchnormFwdTrainingActivTensorIds::NEXT_RUNNING_MEAN_UID);
            nextRunningMeanTensorAttr->set_name("next_running_mean");
            nextRunningMeanTensorAttr->set_output(true);
            nextRunningMeanTensorAttr->set_data_type(intermediateDataType);
        }

        if(nextRunningVarianceTensorAttr)
        {
            nextRunningVarianceTensorAttr->set_uid(
                BatchnormFwdTrainingActivTensorIds::NEXT_RUNNING_VARIANCE_UID);
            nextRunningVarianceTensorAttr->set_name("next_running_variance");
            nextRunningVarianceTensorAttr->set_output(true);
            nextRunningVarianceTensorAttr->set_data_type(intermediateDataType);
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

        return {std::move(graphObj),
                GraphOutputs{yActivTensorAttr,
                             meanTensorAttr,
                             invVarianceTensorAttr,
                             nextRunningMeanTensorAttr,
                             nextRunningVarianceTensorAttr}};
    }

    BatchnormFwdTrainingActivation()
    {
        this->inputFillRecipes()
            .setRange(BatchnormFwdTrainingActivTensorIds::X_UID, -1.0f, 1.0f)
            .setRange(BatchnormFwdTrainingActivTensorIds::SCALE_UID, -2.0f, 2.0f)
            .setRange(BatchnormFwdTrainingActivTensorIds::BIAS_UID, -2.0f, 2.0f)
            .setRange(BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_MEAN_UID, -2.0f, 2.0f)
            .setRange(BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_VARIANCE_UID, 0.1f, 2.0f);
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, scenario, bnTestCase, activTestCase] = testCase;

        HIPDNN_PLUGIN_LOG_INFO("Test is using " << bnTestCase.seed << " for its random seed");

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        // Register validators for all output tensors
        this->registerValidator(outputs.yActiv, this->getTolerance(graphObj, outputs.yActiv));
        this->registerValidator(outputs.mean, this->getTolerance(graphObj, outputs.mean));
        this->registerValidator(outputs.invVariance,
                                this->getTolerance(graphObj, outputs.invVariance));
        if(outputs.nextRunningMean)
        {
            this->registerValidator(outputs.nextRunningMean,
                                    this->getTolerance(graphObj, outputs.nextRunningMean));
        }
        if(outputs.nextRunningVariance)
        {
            this->registerValidator(outputs.nextRunningVariance,
                                    this->getTolerance(graphObj, outputs.nextRunningVariance));
        }

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(bnTestCase.note);
        this->inputFillRecipes()
            .setSeed(BatchnormFwdTrainingActivTensorIds::X_UID, bnTestCase.seed)
            .setSeed(BatchnormFwdTrainingActivTensorIds::SCALE_UID, bnTestCase.seed + 1)
            .setSeed(BatchnormFwdTrainingActivTensorIds::BIAS_UID, bnTestCase.seed + 2)
            .setSeed(BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_MEAN_UID,
                     bnTestCase.seed + 1000)
            .setSeed(BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_VARIANCE_UID,
                     bnTestCase.seed + 2000);
        this->verifyGraph(graphObj);
    }
};

// 1D layout tests (NCL, NLC)
using IntegrationGpuBatchnormFwdTrainingActiv1dFp32 = BatchnormFwdTrainingActivation<float>;
using IntegrationGpuBatchnormFwdTrainingActiv1dBfp16 = BatchnormFwdTrainingActivation<bfloat16>;
using IntegrationGpuBatchnormFwdTrainingActiv1dFp16 = BatchnormFwdTrainingActivation<half>;

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuBatchnormFwdTrainingActiv2dFp32 = BatchnormFwdTrainingActivation<float>;
using IntegrationGpuBatchnormFwdTrainingActiv2dBfp16 = BatchnormFwdTrainingActivation<bfloat16>;
using IntegrationGpuBatchnormFwdTrainingActiv2dFp16 = BatchnormFwdTrainingActivation<half>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuBatchnormFwdTrainingActiv3dFp32 = BatchnormFwdTrainingActivation<float>;
using IntegrationGpuBatchnormFwdTrainingActiv3dBfp16 = BatchnormFwdTrainingActivation<bfloat16>;
using IntegrationGpuBatchnormFwdTrainingActiv3dFp16 = BatchnormFwdTrainingActivation<half>;

// Reduced-rank affine (scale/bias) coverage (PR #7566). Runs against every
// engine; providers that only accept full-rank affine tensors skip these.
// 1D is omitted: no provider currently runs reduced-affine 1D batchnorm.
using IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dFp32
    = BatchnormFwdTrainingActivation<float, AffineShapeMode::REDUCED_RANK>;
using IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dBfp16
    = BatchnormFwdTrainingActivation<bfloat16, AffineShapeMode::REDUCED_RANK>;
using IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dFp16
    = BatchnormFwdTrainingActivation<half, AffineShapeMode::REDUCED_RANK>;
using IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dFp32
    = BatchnormFwdTrainingActivation<float, AffineShapeMode::REDUCED_RANK>;
using IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dBfp16
    = BatchnormFwdTrainingActivation<bfloat16, AffineShapeMode::REDUCED_RANK>;
using IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dFp16
    = BatchnormFwdTrainingActivation<half, AffineShapeMode::REDUCED_RANK>;

} // namespace

// ============================================================================
// 1D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdTrainingActiv1dFp32);
TEST_P(IntegrationGpuBatchnormFwdTrainingActiv1dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActiv1dFp32,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActiv1dFp32,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingFull1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdTrainingActiv1dBfp16);
TEST_P(IntegrationGpuBatchnormFwdTrainingActiv1dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActiv1dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActiv1dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingFull1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdTrainingActiv1dFp16);
TEST_P(IntegrationGpuBatchnormFwdTrainingActiv1dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActiv1dFp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActiv1dFp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingFull1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// ============================================================================
// 2D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdTrainingActiv2dFp32);
TEST_P(IntegrationGpuBatchnormFwdTrainingActiv2dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActiv2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActiv2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdTrainingActiv2dBfp16);
TEST_P(IntegrationGpuBatchnormFwdTrainingActiv2dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActiv2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActiv2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdTrainingActiv2dFp16);
TEST_P(IntegrationGpuBatchnormFwdTrainingActiv2dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActiv2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActiv2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// ============================================================================
// 3D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdTrainingActiv3dFp32);
TEST_P(IntegrationGpuBatchnormFwdTrainingActiv3dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActiv3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActiv3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdTrainingActiv3dBfp16);
TEST_P(IntegrationGpuBatchnormFwdTrainingActiv3dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActiv3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActiv3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormFwdTrainingActiv3dFp16);
TEST_P(IntegrationGpuBatchnormFwdTrainingActiv3dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActiv3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActiv3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// ============================================================================
// Reduced-rank affine (scale/bias) tests (PR #7566)
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dFp32);
TEST_P(IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dBfp16);
TEST_P(IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dFp16);
TEST_P(IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dFp32);
TEST_P(IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dBfp16);
TEST_P(IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dFp16);
TEST_P(IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivReducedAffine3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::Values(BatchnormTrainingScenario::FULL_TRAINING,
                                     BatchnormTrainingScenario::WITH_BATCH_STATS),
                     testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
