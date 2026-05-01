// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "RMSNormGraphUtils.hpp"
#include "RMSNormTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRMSNorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormFwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;
namespace rmsnorm = hipdnn_test_sdk::utilities::rmsnorm;

TEST(TestRMSNormFwdPlan, ExecutePlan)
{
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildRMSNormFwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    RMSNormFwdTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    RMSNormFwdTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();

    const auto* invRmsPtr = attributes.inv_rms_tensor_uid().has_value()
                                ? tensorMap.at(attributes.inv_rms_tensor_uid().value())
                                : nullptr;
    RMSNormFwdParams params(*tensorMap.at(attributes.x_tensor_uid()),
                            *tensorMap.at(attributes.scale_tensor_uid()),
                            *tensorMap.at(attributes.epsilon_tensor_uid()),
                            *tensorMap.at(attributes.y_tensor_uid()),
                            invRmsPtr);

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    const double epsilon = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(
        params.epsilonTensor, "Epsilon");

    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.tensors[attributes.x_tensor_uid()]->rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.tensors[attributes.scale_tensor_uid()]->rawHostData());
    auto shallowYTensor = createShallowTensor<float>(
        params.yTensor, directTensorBundle.tensors[attributes.y_tensor_uid()]->rawHostData());

    CpuFpReferenceRMSNorm::forward(*shallowXTensor, *shallowScaleTensor, *shallowYTensor, epsilon);

    RMSNormFwdPlan<float, float, float, float> fwdPlan(std::move(params));
    fwdPlan.execute(variantPack);

    // x in [0, 1], scale in [0, 1], C=3, no bias
    const float tolerance
        = rmsnorm::calculateRMSNormFwdTolerance<float, float, float>(0.0, 1.0, 0.0, 1.0, dims[1]);
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        *directTensorBundle.tensors[attributes.y_tensor_uid()].get(),
        *planTensorBundle.tensors[attributes.y_tensor_uid()].get()));
}

TEST(TestRMSNormFwdPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> dims = {1, 2, 1, 1};
    auto graph = buildRMSNormFwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const RMSNormFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<RMSNormFwdPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestRMSNormFwdPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> dims = {1, 2, 1, 1};
    auto graph = buildRMSNormFwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const RMSNormFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const RMSNormFwdPlanBuilder<DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestRMSNormFwdPlan, ExecutePlanWithBias)
{
    const std::vector<int64_t> dims = {4, 3, 16, 16};
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildRMSNormFwdGraphWithBias(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    RMSNormFwdWithBiasTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    RMSNormFwdWithBiasTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();

    const auto* invRmsBiasPtr = attributes.inv_rms_tensor_uid().has_value()
                                    ? tensorMap.at(attributes.inv_rms_tensor_uid().value())
                                    : nullptr;
    ASSERT_TRUE(attributes.bias_tensor_uid().has_value());
    const auto* biasPtr = tensorMap.at(attributes.bias_tensor_uid().value());

    RMSNormFwdParams params(*tensorMap.at(attributes.x_tensor_uid()),
                            *tensorMap.at(attributes.scale_tensor_uid()),
                            *tensorMap.at(attributes.epsilon_tensor_uid()),
                            *tensorMap.at(attributes.y_tensor_uid()),
                            invRmsBiasPtr,
                            biasPtr);

    const double epsilon = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(
        params.epsilonTensor, "Epsilon");

    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.tensors[attributes.x_tensor_uid()]->rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.tensors[attributes.scale_tensor_uid()]->rawHostData());
    auto shallowBiasTensor = createShallowTensor<float>(
        params.biasTensor.value(),
        directTensorBundle.tensors[attributes.bias_tensor_uid().value()]->rawHostData());
    auto shallowYTensor = createShallowTensor<float>(
        params.yTensor, directTensorBundle.tensors[attributes.y_tensor_uid()]->rawHostData());

    CpuFpReferenceRMSNorm::forward<float, float, float, float>(*shallowXTensor,
                                                               *shallowScaleTensor,
                                                               *shallowYTensor,
                                                               epsilon,
                                                               nullptr,
                                                               shallowBiasTensor.get());

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();
    RMSNormFwdPlan<float, float, float, float> fwdPlan(std::move(params));
    fwdPlan.execute(variantPack);

    // x in [0, 1], scale in [0, 1], C=3, bias in [-0.5, 0.5]
    const float tolerance = rmsnorm::calculateRMSNormFwdTolerance<float, float, float>(
        0.0, 1.0, 0.0, 1.0, dims[1], -0.5, 0.5);
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        *directTensorBundle.tensors[attributes.y_tensor_uid()].get(),
        *planTensorBundle.tensors[attributes.y_tensor_uid()].get()));
}

TEST(TestRMSNormFwdPlanBuilder, PlanConstructionWithBias)
{
    const std::vector<int64_t> dims = {1, 2, 1, 1};
    auto graph = buildRMSNormFwdGraphWithBias(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const RMSNormFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<RMSNormFwdPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestRMSNormFwdPlanBuilder, IsApplicableWithBias)
{
    const std::vector<int64_t> dims = {1, 2, 1, 1};
    auto graph = buildRMSNormFwdGraphWithBias(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const RMSNormFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}
