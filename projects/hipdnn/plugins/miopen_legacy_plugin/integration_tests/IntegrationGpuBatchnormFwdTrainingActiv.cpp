// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/BatchnormCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

namespace
{

using test_activation_common::ActivTestCase;
using test_bn_common::BatchnormTestCase;

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

template <typename InputType, typename IntermediateType>
class BatchnormFwdTrainingActivation
    : public IntegrationGraphVerificationHarness<
          InputType,
          std::tuple<test_bn_common::BatchnormTestCase, test_activation_common::ActivTestCase>>
{
protected:
    void runGraphTest(InputType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        runGraphTestWithScenario(tolerance, BatchnormTrainingScenario::WITH_BATCH_STATS, layout);
    }

    void runGraphTestWithScenario(InputType tolerance,
                                  BatchnormTrainingScenario scenario,
                                  const TensorLayout& layout = TensorLayout::NCHW)
    {
        const auto& [bnTestCase, activTestCase] = this->GetParam();

        auto inputDataType = getDataTypeEnumFromType<InputType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        HIPDNN_LOG_INFO("Test is using {} for its random seed", bnTestCase.seed);

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("BatchnormFwdTrainingActivTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        int64_t uid = 1;
        auto dims = bnTestCase.dims;
        auto derivedDims = getDerivedShape(dims);

        // Create input tensor attributes
        auto xAttr = graph::makeTensorAttributes(
            "X", inputDataType, dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        biasAttr.set_uid(uid++);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        // Epsilon: use pass-by-value with double (matches MIOpen API)
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        std::mt19937 gen(bnTestCase.seed);
        std::uniform_real_distribution<double> epsilonDist(1e-6, 1e-4);
        epsilonTensorAttr->set_value(epsilonDist(gen)).set_name("epsilon").set_uid(uid++);

        // Store tensor IDs for initialization
        _inputTensorIds[graph::BatchnormAttributes::InputNames::X] = xTensorAttr->get_uid();
        _inputTensorIds[graph::BatchnormAttributes::InputNames::SCALE] = scaleTensorAttr->get_uid();
        _inputTensorIds[graph::BatchnormAttributes::InputNames::BIAS] = biasTensorAttr->get_uid();
        _inputTensorIds[graph::BatchnormAttributes::InputNames::EPSILON]
            = epsilonTensorAttr->get_uid();

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
            prevRunningMeanAttr.set_uid(uid++);
            prevRunningMeanTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(prevRunningMeanAttr));

            auto prevRunningVarianceAttr
                = graph::makeTensorAttributes("prev_running_variance",
                                              intermediateDataType,
                                              derivedDims,
                                              generateStrides(derivedDims));
            prevRunningVarianceAttr.set_uid(uid++);
            prevRunningVarianceTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(prevRunningVarianceAttr));

            // Momentum: use pass-by-value with double (matches MIOpen API)
            momentumTensorAttr = std::make_shared<graph::TensorAttributes>();
            std::uniform_real_distribution<double> momentumDist(0.05, 0.15);
            momentumTensorAttr->set_value(momentumDist(gen)).set_name("momentum").set_uid(uid++);

            // Store running stats tensor IDs
            _inputTensorIds[graph::BatchnormAttributes::InputNames::MOMENTUM]
                = momentumTensorAttr->get_uid();
            _inputTensorIds[graph::BatchnormAttributes::InputNames::PREV_RUNNING_MEAN]
                = prevRunningMeanTensorAttr->get_uid();
            _inputTensorIds[graph::BatchnormAttributes::InputNames::PREV_RUNNING_VARIANCE]
                = prevRunningVarianceTensorAttr->get_uid();
        }

        // Create batchnorm attributes
        graph::BatchnormAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_training");

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

        // Set BN output tensor as virtual (intermediate between BN and activation)
        if(!yBnTensorAttr->has_uid())
        {
            yBnTensorAttr->set_uid(uid++);
        }
        yBnTensorAttr->set_is_virtual(true); // VIRTUAL - fusion key point
        yBnTensorAttr->set_data_type(inputDataType);
        yBnTensorAttr->set_dim(dims);
        yBnTensorAttr->set_stride(generateStrides(dims, layout.strideOrder));

        // Add activation node with parameters from test case
        graph::PointwiseAttributes activAttrs;
        activAttrs.set_name("activation");
        activAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activTestCase.mode));

        // Set activation-specific parameters
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

        // Set final activation output tensor
        if(!yActivTensorAttr->has_uid())
        {
            yActivTensorAttr->set_uid(uid++);
        }
        yActivTensorAttr->set_output(true);
        yActivTensorAttr->set_data_type(inputDataType);
        yActivTensorAttr->set_dim(dims);
        yActivTensorAttr->set_stride(generateStrides(dims, layout.strideOrder));

        // Configure batch statistics outputs
        if(meanTensorAttr)
        {
            if(!meanTensorAttr->has_uid())
            {
                meanTensorAttr->set_uid(uid++);
            }
            meanTensorAttr->set_output(true);
            meanTensorAttr->set_data_type(intermediateDataType);
            meanTensorAttr->set_dim(derivedDims);
            meanTensorAttr->set_stride(generateStrides(derivedDims));
        }

        if(invVarianceTensorAttr)
        {
            if(!invVarianceTensorAttr->has_uid())
            {
                invVarianceTensorAttr->set_uid(uid++);
            }
            invVarianceTensorAttr->set_output(true);
            invVarianceTensorAttr->set_data_type(intermediateDataType);
            invVarianceTensorAttr->set_dim(derivedDims);
            invVarianceTensorAttr->set_stride(generateStrides(derivedDims));
        }

        // Configure running statistics outputs if they exist
        if(nextRunningMeanTensorAttr)
        {
            if(!nextRunningMeanTensorAttr->has_uid())
            {
                nextRunningMeanTensorAttr->set_uid(uid++);
            }
            nextRunningMeanTensorAttr->set_name("next_running_mean");
            nextRunningMeanTensorAttr->set_output(true);
            nextRunningMeanTensorAttr->set_data_type(intermediateDataType);
            nextRunningMeanTensorAttr->set_dim(derivedDims);
            nextRunningMeanTensorAttr->set_stride(generateStrides(derivedDims));
        }

        if(nextRunningVarianceTensorAttr)
        {
            if(!nextRunningVarianceTensorAttr->has_uid())
            {
                nextRunningVarianceTensorAttr->set_uid(uid++);
            }
            nextRunningVarianceTensorAttr->set_name("next_running_variance");
            nextRunningVarianceTensorAttr->set_output(true);
            nextRunningVarianceTensorAttr->set_data_type(intermediateDataType);
            nextRunningVarianceTensorAttr->set_dim(derivedDims);
            nextRunningVarianceTensorAttr->set_stride(generateStrides(derivedDims));
        }

        // Store next running stats tensor IDs if they exist
        if(nextRunningMeanTensorAttr)
        {
            _outputTensorIds[graph::BatchnormAttributes::OutputNames::NEXT_RUNNING_MEAN]
                = nextRunningMeanTensorAttr->get_uid();
        }

        if(nextRunningVarianceTensorAttr)
        {
            _outputTensorIds[graph::BatchnormAttributes::OutputNames::NEXT_RUNNING_VARIANCE]
                = nextRunningVarianceTensorAttr->get_uid();
        }

        // Register validators for all output tensors
        this->registerValidator(yActivTensorAttr, tolerance);
        this->registerValidator(meanTensorAttr, tolerance);
        this->registerValidator(invVarianceTensorAttr, tolerance);
        if(nextRunningMeanTensorAttr)
        {
            this->registerValidator(nextRunningMeanTensorAttr, tolerance);
        }
        if(nextRunningVarianceTensorAttr)
        {
            this->registerValidator(nextRunningVarianceTensorAttr, tolerance);
        }

        this->verifyGraph(graphObj, bnTestCase.seed);
    }

    void initializeBundle([[maybe_unused]] const hipdnn_frontend::graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        // Scale and bias: -2.0 to 2.0 to match MIOpen
        bundle.tensors.at(_inputTensorIds.at(graph::BatchnormAttributes::InputNames::SCALE))
            ->fillTensorWithRandomValues(-2.0f, 2.0f, seed + 1);
        bundle.tensors.at(_inputTensorIds.at(graph::BatchnormAttributes::InputNames::BIAS))
            ->fillTensorWithRandomValues(-2.0f, 2.0f, seed + 2);

        // Running mean: prev and next must start with SAME values
        // because MIOpen's API uses IN/OUT parameter semantics
        auto prevMeanIt
            = _inputTensorIds.find(graph::BatchnormAttributes::InputNames::PREV_RUNNING_MEAN);
        auto nextMeanIt
            = _outputTensorIds.find(graph::BatchnormAttributes::OutputNames::NEXT_RUNNING_MEAN);
        if(prevMeanIt != _inputTensorIds.end() && nextMeanIt != _outputTensorIds.end())
        {
            unsigned runningMeanSeed = seed + 1000;
            bundle.tensors.at(prevMeanIt->second)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningMeanSeed);
            bundle.tensors.at(nextMeanIt->second)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningMeanSeed);
        }

        // Running variance: prev and next must start with SAME values
        auto prevVarIt
            = _inputTensorIds.find(graph::BatchnormAttributes::InputNames::PREV_RUNNING_VARIANCE);
        auto nextVarIt
            = _outputTensorIds.find(graph::BatchnormAttributes::OutputNames::NEXT_RUNNING_VARIANCE);
        if(prevVarIt != _inputTensorIds.end() && nextVarIt != _outputTensorIds.end())
        {
            unsigned runningVarianceSeed = seed + 2000;
            bundle.tensors.at(prevVarIt->second)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningVarianceSeed);
            bundle.tensors.at(nextVarIt->second)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningVarianceSeed);
        }

        // X input: default range
        bundle.tensors.at(_inputTensorIds.at(graph::BatchnormAttributes::InputNames::X))
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
    }

private:
    std::unordered_map<graph::BatchnormAttributes::InputNames, int64_t> _inputTensorIds;
    std::unordered_map<graph::BatchnormAttributes::OutputNames, int64_t> _outputTensorIds;
};

// NCHW 2D
using IntegrationGpuBatchnormFwdTrainingActivNchwFp32
    = BatchnormFwdTrainingActivation<float, float>;
using IntegrationGpuBatchnormFwdTrainingActivNchwFp16 = BatchnormFwdTrainingActivation<half, float>;
using IntegrationGpuBatchnormFwdTrainingActivNchwBfp16
    = BatchnormFwdTrainingActivation<hip_bfloat16, float>;

// NHWC 2D
using IntegrationGpuBatchnormFwdTrainingActivNhwcFp32
    = BatchnormFwdTrainingActivation<float, float>;
using IntegrationGpuBatchnormFwdTrainingActivNhwcFp16 = BatchnormFwdTrainingActivation<half, float>;
using IntegrationGpuBatchnormFwdTrainingActivNhwcBfp16
    = BatchnormFwdTrainingActivation<hip_bfloat16, float>;

// NCDHW 3D
using IntegrationGpuBatchnormFwdTrainingActivNcdhwFp32
    = BatchnormFwdTrainingActivation<float, float>;
using IntegrationGpuBatchnormFwdTrainingActivNcdhwFp16
    = BatchnormFwdTrainingActivation<half, float>;
using IntegrationGpuBatchnormFwdTrainingActivNcdhwBfp16
    = BatchnormFwdTrainingActivation<hip_bfloat16, float>;

// NDHWC 3D
using IntegrationGpuBatchnormFwdTrainingActivNdhwcFp32
    = BatchnormFwdTrainingActivation<float, float>;
using IntegrationGpuBatchnormFwdTrainingActivNdhwcFp16
    = BatchnormFwdTrainingActivation<half, float>;
using IntegrationGpuBatchnormFwdTrainingActivNdhwcBfp16
    = BatchnormFwdTrainingActivation<hip_bfloat16, float>;

} // namespace

// ============================================================================
// NCHW 2D Tests
// ============================================================================

// NOTE: FullTraining tests are disabled due to API mismatch between hipDNN and MIOpen.
// hipDNN's graph API uses separate prev_running_mean/variance (input) and next_running_mean/variance (output)
// buffers, but MIOpen's API requires single IN/OUT buffers for running statistics.
// This cannot be correctly bridged without either:
// 1. Updating MIOpen API to support separate input/output buffers, or
// 2. Implementing buffer copy operations (with performance overhead)
// Until MIOpen is updated, batchnorm training with running statistics is not supported.
// BatchStatsOnly tests (without running statistics) continue to work correctly.

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCHW);
}

// ============================================================================
// NHWC 2D Tests
// ============================================================================

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NHWC);
}

// ============================================================================
// NCDHW 3D Tests
// ============================================================================

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCDHW);
}

// ============================================================================
// NDHWC 3D Tests
// ============================================================================

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NDHWC);
}

// ============================================================================
// Test Instantiation
// ============================================================================

// 2D NCHW Tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNchwFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNchwFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNchwFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNchwFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNchwBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNchwBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// 2D NHWC Tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNhwcFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNhwcFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNhwcFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNhwcFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNhwcBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNhwcBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// 3D NCDHW Tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// 3D NDHWC Tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));
