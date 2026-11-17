// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesBatchNorm.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferenceWithVariancePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferenceWithVarianceSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestBatchnormFwdWithVariancePlan : public ::testing::Test
{
};

TEST_F(TestBatchnormFwdWithVariancePlan, ExecutePlan)
{
    // Tensor ranges from BatchnormFwdWithVarianceTensorBundle:
    // x=[0,1], scale=[0,1], bias=[0,1], mean=[0,1], var=[0.1,1.0]
    // epsilon = BATCHNORM_DEFAULT_EPSILON = 1e-5
    auto tolerance = batchnorm::calculateBatchnormInferenceWithVarianceTolerance<float, float>(
        0.0,
        1.0,
        0.0,
        1.0,
        0.1,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        hipdnn_data_sdk::utilities::BATCHNORM_DEFAULT_EPSILON);
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildBatchnormFwdInferenceWithVarianceGraph(DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             dims,
                                                             TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    BatchnormFwdWithVarianceTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    BatchnormFwdWithVarianceTensorBundle directTensorBundle(
        node, graphWrapper.getTensorMap(), seed);

    const auto& attributes = node.attributesAs<
        hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt>();
    const auto& tensorMap = graphWrapper.getTensorMap();
    BatchnormFwdInferenceWithVarianceParams params(*tensorMap.at(attributes.x_tensor_uid()),
                                                   *tensorMap.at(attributes.y_tensor_uid()),
                                                   *tensorMap.at(attributes.scale_tensor_uid()),
                                                   *tensorMap.at(attributes.bias_tensor_uid()),
                                                   *tensorMap.at(attributes.mean_tensor_uid()),
                                                   *tensorMap.at(attributes.variance_tensor_uid()),
                                                   *tensorMap.at(attributes.epsilon_tensor_uid()));

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.tensors[attributes.x_tensor_uid()]->rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.tensors[attributes.scale_tensor_uid()]->rawHostData());
    auto shallowBiasTensor = createShallowTensor<float>(
        params.biasTensor, directTensorBundle.tensors[attributes.bias_tensor_uid()]->rawHostData());
    auto shallowMeanTensor = createShallowTensor<float>(
        params.meanTensor, directTensorBundle.tensors[attributes.mean_tensor_uid()]->rawHostData());
    auto shallowVarianceTensor = createShallowTensor<float>(
        params.varianceTensor,
        directTensorBundle.tensors[attributes.variance_tensor_uid()]->rawHostData());
    auto shallowYTensor = createShallowTensor<float>(
        params.yTensor, directTensorBundle.tensors[attributes.y_tensor_uid()]->rawHostData());

    const double epsilon = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(
        params.epsilonTensor, "Epsilon");

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(*shallowXTensor,
                                                      *shallowScaleTensor,
                                                      *shallowBiasTensor,
                                                      *shallowMeanTensor,
                                                      *shallowVarianceTensor,
                                                      *shallowYTensor,
                                                      epsilon);

    BatchnormFwdInferenceWithVariancePlan<float, float, float, float, float> fwdPlan(
        std::move(params));
    fwdPlan.execute(variantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        *directTensorBundle.tensors[attributes.y_tensor_uid()].get(),
        *planTensorBundle.tensors[attributes.y_tensor_uid()].get()));
}

TEST_F(TestBatchnormFwdWithVariancePlan, ExecutePlanWithRuntimeEpsilonFromPack)
{
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const unsigned int seed = getGlobalTestSeed();

    // Pure runtime pass-by-value epsilon delivered through the variant pack.
    auto runtimeGraph = buildBatchnormFwdInferenceWithVarianceGraph(DataType::FLOAT,
                                                                    DataType::FLOAT,
                                                                    DataType::FLOAT,
                                                                    DataType::FLOAT,
                                                                    dims,
                                                                    TensorLayout::NHWC,
                                                                    /*isOutputVirtual=*/false,
                                                                    /*runtimeEpsilon=*/true);
    auto [runtimeSerialized, runtimeSerErr] = runtimeGraph->to_binary();
    ASSERT_TRUE(runtimeSerErr.is_good()) << runtimeSerErr.get_message();
    const GraphWrapper runtimeWrapper(runtimeSerialized.data(), runtimeSerialized.size());
    const INodeWrapper& runtimeNode = runtimeWrapper.getNodeWrapper(0);
    BatchnormFwdWithVarianceTensorBundle runtimeBundle(
        runtimeNode, runtimeWrapper.getTensorMap(), seed);

    const auto& runtimeAttrs = runtimeNode.attributesAs<
        hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt>();

    const auto epsilonHostValue = static_cast<float>(BATCHNORM_DEFAULT_EPSILON);
    *static_cast<float*>(runtimeBundle.tensors[runtimeAttrs.epsilon_tensor_uid()]->rawHostData())
        = epsilonHostValue;

    BatchnormFwdInferenceWithVarianceParams runtimeParams(
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.x_tensor_uid()),
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.y_tensor_uid()),
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.scale_tensor_uid()),
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.bias_tensor_uid()),
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.mean_tensor_uid()),
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.variance_tensor_uid()),
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.epsilon_tensor_uid()));
    const std::unordered_map<int64_t, void*> runtimeVariantPack = runtimeBundle.toHostVariantPack();
    BatchnormFwdInferenceWithVariancePlan<float, float, float, float, float> runtimePlan(
        std::move(runtimeParams));
    runtimePlan.execute(runtimeVariantPack);

    // Baked-epsilon reference graph with the same seed and equal epsilon value.
    auto bakedGraph = buildBatchnormFwdInferenceWithVarianceGraph(DataType::FLOAT,
                                                                  DataType::FLOAT,
                                                                  DataType::FLOAT,
                                                                  DataType::FLOAT,
                                                                  dims,
                                                                  TensorLayout::NHWC);
    auto [bakedSerialized, bakedSerErr] = bakedGraph->to_binary();
    ASSERT_TRUE(bakedSerErr.is_good()) << bakedSerErr.get_message();
    const GraphWrapper bakedWrapper(bakedSerialized.data(), bakedSerialized.size());
    const INodeWrapper& bakedNode = bakedWrapper.getNodeWrapper(0);
    BatchnormFwdWithVarianceTensorBundle bakedBundle(bakedNode, bakedWrapper.getTensorMap(), seed);

    const auto& bakedAttrs = bakedNode.attributesAs<
        hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt>();
    BatchnormFwdInferenceWithVarianceParams bakedParams(
        *bakedWrapper.getTensorMap().at(bakedAttrs.x_tensor_uid()),
        *bakedWrapper.getTensorMap().at(bakedAttrs.y_tensor_uid()),
        *bakedWrapper.getTensorMap().at(bakedAttrs.scale_tensor_uid()),
        *bakedWrapper.getTensorMap().at(bakedAttrs.bias_tensor_uid()),
        *bakedWrapper.getTensorMap().at(bakedAttrs.mean_tensor_uid()),
        *bakedWrapper.getTensorMap().at(bakedAttrs.variance_tensor_uid()),
        *bakedWrapper.getTensorMap().at(bakedAttrs.epsilon_tensor_uid()));
    const std::unordered_map<int64_t, void*> bakedVariantPack = bakedBundle.toHostVariantPack();
    BatchnormFwdInferenceWithVariancePlan<float, float, float, float, float> bakedPlan(
        std::move(bakedParams));
    bakedPlan.execute(bakedVariantPack);

    auto tolerance = batchnorm::calculateBatchnormInferenceWithVarianceTolerance<float, float>(
        0.0, 1.0, 0.0, 1.0, 0.1, 1.0, 0.0, 1.0, 0.0, 1.0, BATCHNORM_DEFAULT_EPSILON);
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(*bakedBundle.tensors[bakedAttrs.y_tensor_uid()].get(),
                                        *runtimeBundle.tensors[runtimeAttrs.y_tensor_uid()].get()));
}

TEST(TestBatchnormFwdWithVariancePlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildBatchnormFwdInferenceWithVarianceGraph(DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             dims,
                                                             TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const BatchnormFwdInferenceWithVariancePlanBuilder<DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<BatchnormFwdInferenceWithVariancePlan<float, float, float, float, float>*>(
              builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBatchnormFwdWithVariancePlanBuilder, IsApplicable)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildBatchnormFwdInferenceWithVarianceGraph(DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             dims,
                                                             TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const BatchnormFwdInferenceWithVariancePlanBuilder<DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const BatchnormFwdInferenceWithVariancePlanBuilder<DataType::FLOAT,
                                                       DataType::HALF,
                                                       DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    auto tensorMapCopy = graphWrapper.getTensorMap();
    const auto& node = graphWrapper.getNode(0);
    const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    tensorMapCopy.erase(nodeAttributes->epsilon_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(node, tensorMapCopy));
}

TEST(TestBatchnormFwdInferenceWithVariancePlan, PlanBuilderMapContainsExpectedKeys)
{
    auto planBuilders = BatchnormFwdInferenceWithVarianceSignatureKey::getPlanBuilders();

    // Verify we have builders for common type combinations
    EXPECT_GT(planBuilders.size(), 0);

    // FP32 case
    const BatchnormFwdInferenceWithVarianceSignatureKey fp32Key(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    EXPECT_TRUE(planBuilders.find(fp32Key) != planBuilders.end());

    // FP16 case with FP32 params
    const BatchnormFwdInferenceWithVarianceSignatureKey fp16Key(
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT);
    EXPECT_TRUE(planBuilders.find(fp16Key) != planBuilders.end());

    // BFP16 case with FP32 params
    const BatchnormFwdInferenceWithVarianceSignatureKey bfp16Key(
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT);
    EXPECT_TRUE(planBuilders.find(bfp16Key) != planBuilders.end());
}

TEST(TestBatchnormFwdInferenceWithVariancePlan, SignatureKeyHashingWorks)
{
    const BatchnormFwdInferenceWithVarianceSignatureKey key1(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    const BatchnormFwdInferenceWithVarianceSignatureKey key2(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    const BatchnormFwdInferenceWithVarianceSignatureKey key3(
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT);

    // Same keys should be equal
    EXPECT_TRUE(key1 == key2);
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    // Different keys should not be equal
    EXPECT_FALSE(key1 == key3);
    // Hash collision is possible but unlikely for these specific cases
    EXPECT_NE(key1.hashSelf(), key3.hashSelf());
}

TEST(TestBatchnormFwdInferenceWithVariancePlan, NodeTypeIsCorrect)
{
    BatchnormFwdInferenceWithVarianceSignatureKey key; // NOLINT(misc-const-correctness)
    EXPECT_EQ(key.nodeType, NodeAttributes::BatchnormInferenceAttributesVarianceExt);
}
