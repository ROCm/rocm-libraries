// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "LayernormGraphUtils.hpp"
#include "LayernormTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormFpropPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestLayernormFpropPlan : public ::testing::Test
{
};

TEST_F(TestLayernormFpropPlan, ExecutePlan)
{
    auto tolerance = layernorm::getTolerance<float>();
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const int64_t normalizedDimCount = 3;
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildLayernormFpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    LayernormFpropTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    LayernormFpropTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();
    LayernormFpropParams params(*tensorMap.at(attributes.x_tensor_uid()),
                                *tensorMap.at(attributes.y_tensor_uid()),
                                *tensorMap.at(attributes.epsilon_tensor_uid()),
                                *tensorMap.at(attributes.scale_tensor_uid()),
                                *tensorMap.at(attributes.bias_tensor_uid()),
                                normalizedDimCount);

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.getTensor(attributes.x_tensor_uid()).rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.getTensor(attributes.scale_tensor_uid()).rawHostData());
    auto shallowBiasTensor = createShallowTensor<float>(
        params.biasTensor,
        directTensorBundle.getTensor(attributes.bias_tensor_uid()).rawHostData());
    auto shallowYTensor = createShallowTensor<float>(
        params.yTensor, directTensorBundle.getTensor(attributes.y_tensor_uid()).rawHostData());

    CpuFpReferenceLayernorm::fprop(*shallowXTensor,
                                   shallowScaleTensor.get(),
                                   shallowBiasTensor.get(),
                                   *shallowYTensor,
                                   hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON,
                                   normalizedDimCount);

    LayernormFpropPlan<float, float, float, float, float> fpropPlan(std::move(params));
    fpropPlan.execute(variantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.y_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.y_tensor_uid())));
}

TEST_F(TestLayernormFpropPlan, ExecutePlanWithRuntimeEpsilonFromPack)
{
    auto tolerance = layernorm::getTolerance<float>();
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const int64_t normalizedDimCount = 3;
    const unsigned int seed = getGlobalTestSeed();

    // Pure runtime pass-by-value epsilon delivered through the variant pack.
    auto runtimeGraph = buildLayernormFpropGraph(DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 dims,
                                                 normalizedDimCount,
                                                 TensorLayout::NHWC,
                                                 /*useTrainingPhase=*/false,
                                                 /*onePadded=*/false,
                                                 /*runtimeEpsilon=*/true);
    auto [runtimeSerialized, runtimeSerErr] = runtimeGraph->to_binary();
    ASSERT_TRUE(runtimeSerErr.is_good()) << runtimeSerErr.get_message();
    const GraphWrapper runtimeWrapper(runtimeSerialized.data(), runtimeSerialized.size());
    const INodeWrapper& runtimeNode = runtimeWrapper.getNodeWrapper(0);
    LayernormFpropTensorBundle runtimeBundle(runtimeNode, runtimeWrapper.getTensorMap(), seed);

    const auto& runtimeAttrs
        = runtimeNode.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes>();
    const auto epsilonHostValue = static_cast<float>(LAYERNORM_DEFAULT_EPSILON);
    *static_cast<float*>(runtimeBundle.tensors[runtimeAttrs.epsilon_tensor_uid()]->rawHostData())
        = epsilonHostValue;

    LayernormFpropParams runtimeParams(
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.x_tensor_uid()),
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.y_tensor_uid()),
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.epsilon_tensor_uid()),
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.scale_tensor_uid()),
        *runtimeWrapper.getTensorMap().at(runtimeAttrs.bias_tensor_uid()),
        normalizedDimCount);
    const std::unordered_map<int64_t, void*> runtimeVariantPack = runtimeBundle.toHostVariantPack();
    LayernormFpropPlan<float, float, float, float, float> runtimePlan(std::move(runtimeParams));
    runtimePlan.execute(runtimeVariantPack);

    // Baked-epsilon reference graph with the same seed and equal epsilon value.
    auto bakedGraph = buildLayernormFpropGraph(DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               dims,
                                               normalizedDimCount,
                                               TensorLayout::NHWC);
    auto [bakedSerialized, bakedSerErr] = bakedGraph->to_binary();
    ASSERT_TRUE(bakedSerErr.is_good()) << bakedSerErr.get_message();
    const GraphWrapper bakedWrapper(bakedSerialized.data(), bakedSerialized.size());
    const INodeWrapper& bakedNode = bakedWrapper.getNodeWrapper(0);
    LayernormFpropTensorBundle bakedBundle(bakedNode, bakedWrapper.getTensorMap(), seed);

    const auto& bakedAttrs
        = bakedNode.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes>();
    LayernormFpropParams bakedParams(
        *bakedWrapper.getTensorMap().at(bakedAttrs.x_tensor_uid()),
        *bakedWrapper.getTensorMap().at(bakedAttrs.y_tensor_uid()),
        *bakedWrapper.getTensorMap().at(bakedAttrs.epsilon_tensor_uid()),
        *bakedWrapper.getTensorMap().at(bakedAttrs.scale_tensor_uid()),
        *bakedWrapper.getTensorMap().at(bakedAttrs.bias_tensor_uid()),
        normalizedDimCount);
    const std::unordered_map<int64_t, void*> bakedVariantPack = bakedBundle.toHostVariantPack();
    LayernormFpropPlan<float, float, float, float, float> bakedPlan(std::move(bakedParams));
    bakedPlan.execute(bakedVariantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(bakedBundle.getTensor(bakedAttrs.y_tensor_uid()),
                                        runtimeBundle.getTensor(runtimeAttrs.y_tensor_uid())));
}

TEST_F(TestLayernormFpropPlan, ExecutePlanOnePaddedNormalizedDimCount2)
{
    auto tolerance = layernorm::getTolerance<float>();
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const int64_t normalizedDimCount = 2;
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildLayernormFpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          false,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    LayernormFpropTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    LayernormFpropTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();
    LayernormFpropParams params(*tensorMap.at(attributes.x_tensor_uid()),
                                *tensorMap.at(attributes.y_tensor_uid()),
                                *tensorMap.at(attributes.epsilon_tensor_uid()),
                                *tensorMap.at(attributes.scale_tensor_uid()),
                                *tensorMap.at(attributes.bias_tensor_uid()),
                                normalizedDimCount);

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.getTensor(attributes.x_tensor_uid()).rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.getTensor(attributes.scale_tensor_uid()).rawHostData());
    auto shallowBiasTensor = createShallowTensor<float>(
        params.biasTensor,
        directTensorBundle.getTensor(attributes.bias_tensor_uid()).rawHostData());
    auto shallowYTensor = createShallowTensor<float>(
        params.yTensor, directTensorBundle.getTensor(attributes.y_tensor_uid()).rawHostData());

    CpuFpReferenceLayernorm::fprop(*shallowXTensor,
                                   shallowScaleTensor.get(),
                                   shallowBiasTensor.get(),
                                   *shallowYTensor,
                                   hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON,
                                   normalizedDimCount);

    LayernormFpropPlan<float, float, float, float, float> fpropPlan(std::move(params));
    fpropPlan.execute(variantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.y_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.y_tensor_uid())));
}

TEST_F(TestLayernormFpropPlan, ExecutePlanTrainingPhase)
{
    auto tolerance = layernorm::getTolerance<float>();
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const int64_t normalizedDimCount = 3;
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildLayernormFpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    LayernormFpropTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    LayernormFpropTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttr = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarianceAttr = nullptr;
    if(attributes.mean_tensor_uid().has_value())
    {
        meanAttr = tensorMap.at(attributes.mean_tensor_uid().value());
    }
    if(attributes.inv_variance_tensor_uid().has_value())
    {
        invVarianceAttr = tensorMap.at(attributes.inv_variance_tensor_uid().value());
    }

    LayernormFpropParams params(*tensorMap.at(attributes.x_tensor_uid()),
                                *tensorMap.at(attributes.y_tensor_uid()),
                                *tensorMap.at(attributes.epsilon_tensor_uid()),
                                *tensorMap.at(attributes.scale_tensor_uid()),
                                *tensorMap.at(attributes.bias_tensor_uid()),
                                normalizedDimCount,
                                meanAttr,
                                invVarianceAttr);

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.getTensor(attributes.x_tensor_uid()).rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.getTensor(attributes.scale_tensor_uid()).rawHostData());
    auto shallowBiasTensor = createShallowTensor<float>(
        params.biasTensor,
        directTensorBundle.getTensor(attributes.bias_tensor_uid()).rawHostData());
    auto shallowYTensor = createShallowTensor<float>(
        params.yTensor, directTensorBundle.getTensor(attributes.y_tensor_uid()).rawHostData());

    std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowMeanTensor;
    hipdnn_data_sdk::utilities::TensorBase<float>* meanPtr = nullptr;
    if(attributes.mean_tensor_uid().has_value())
    {
        shallowMeanTensor = createShallowTensor<float>(
            params.meanTensor.value(),
            directTensorBundle.getTensor(attributes.mean_tensor_uid().value()).rawHostData());
        meanPtr = shallowMeanTensor.get();
    }

    std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowInvVarianceTensor;
    hipdnn_data_sdk::utilities::TensorBase<float>* invVariancePtr = nullptr;
    if(attributes.inv_variance_tensor_uid().has_value())
    {
        shallowInvVarianceTensor = createShallowTensor<float>(
            params.invVarianceTensor.value(),
            directTensorBundle.getTensor(attributes.inv_variance_tensor_uid().value())
                .rawHostData());
        invVariancePtr = shallowInvVarianceTensor.get();
    }

    CpuFpReferenceLayernorm::fprop(*shallowXTensor,
                                   shallowScaleTensor.get(),
                                   shallowBiasTensor.get(),
                                   *shallowYTensor,
                                   hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON,
                                   normalizedDimCount,
                                   meanPtr,
                                   invVariancePtr);

    LayernormFpropPlan<float, float, float, float, float> fpropPlan(std::move(params));
    fpropPlan.execute(variantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.y_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.y_tensor_uid())));

    if(attributes.mean_tensor_uid().has_value())
    {
        EXPECT_TRUE(cpuRefOutputValidation.allClose(
            directTensorBundle.getTensor(attributes.mean_tensor_uid().value()),
            planTensorBundle.getTensor(attributes.mean_tensor_uid().value())));
    }

    if(attributes.inv_variance_tensor_uid().has_value())
    {
        EXPECT_TRUE(cpuRefOutputValidation.allClose(
            directTensorBundle.getTensor(attributes.inv_variance_tensor_uid().value()),
            planTensorBundle.getTensor(attributes.inv_variance_tensor_uid().value())));
    }
}

TEST(TestLayernormFpropPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const int64_t normalizedDimCount = 3;
    auto graph = buildLayernormFpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const LayernormFpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<LayernormFpropPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestLayernormFpropPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const int64_t normalizedDimCount = 3;
    auto graph = buildLayernormFpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const LayernormFpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const LayernormFpropPlanBuilder<DataType::FLOAT,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    auto tensorMapCopy = graphWrapper.getTensorMap();
    tensorMapCopy.erase(5);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));
}

TEST(TestLayernormFpropPlanBuilder, PlanConstructionTrainingPhase)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const int64_t normalizedDimCount = 3;
    auto graph = buildLayernormFpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const LayernormFpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<LayernormFpropPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestLayernormFpropPlanBuilder, IsApplicableTrainingPhase)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const int64_t normalizedDimCount = 3;
    auto graph = buildLayernormFpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const LayernormFpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const LayernormFpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        badMeanTypePlanBuilder;
    EXPECT_FALSE(
        badMeanTypePlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}
