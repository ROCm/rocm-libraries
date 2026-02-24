// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "RmsnormGraphUtils.hpp"
#include "RmsnormTensorBundles.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRmsnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RmsnormFwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

TEST(TestRmsnormFwdPlan, ExecutePlan)
{
    std::vector<int64_t> dims = {6, 3, 32, 32};
    unsigned int seed = getGlobalTestSeed();
    auto graph = buildRmsnormFwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    RmsnormFwdTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    RmsnormFwdTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_data_sdk::data_objects::RMSNormAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();

    RmsnormFwdParams params;
    if(attributes.inv_rms_tensor_uid().has_value())
    {
        params = RmsnormFwdParams(*tensorMap.at(attributes.x_tensor_uid()),
                                  *tensorMap.at(attributes.scale_tensor_uid()),
                                  *tensorMap.at(attributes.epsilon_tensor_uid()),
                                  *tensorMap.at(attributes.y_tensor_uid()),
                                  *tensorMap.at(attributes.inv_rms_tensor_uid().value()));
    }
    else
    {
        params = RmsnormFwdParams(*tensorMap.at(attributes.x_tensor_uid()),
                                  *tensorMap.at(attributes.scale_tensor_uid()),
                                  *tensorMap.at(attributes.epsilon_tensor_uid()),
                                  *tensorMap.at(attributes.y_tensor_uid()));
    }

    std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    double epsilon = hipdnn_data_sdk::utilities::extractDoubleFromTensorValue(
        params.epsilonTensor, "Epsilon");

    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.tensors[attributes.x_tensor_uid()]->rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.tensors[attributes.scale_tensor_uid()]->rawHostData());
    auto shallowYTensor = createShallowTensor<float>(
        params.yTensor, directTensorBundle.tensors[attributes.y_tensor_uid()]->rawHostData());

    CpuFpReferenceRmsnorm::forward(*shallowXTensor,
                                   *shallowScaleTensor,
                                   *shallowYTensor,
                                   epsilon);

    RmsnormFwdPlan<float, float, float, float> fwdPlan(std::move(params));
    fwdPlan.execute(variantPack);

    float tolerance = 1e-5f;
    CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        *directTensorBundle.tensors[attributes.y_tensor_uid()].get(),
        *planTensorBundle.tensors[attributes.y_tensor_uid()].get()));
}

TEST(TestRmsnormFwdPlanBuilder, PlanConstruction)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildRmsnormFwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    RmsnormFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    bool result
        = dynamic_cast<RmsnormFwdPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestRmsnormFwdPlanBuilder, IsApplicable)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildRmsnormFwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    RmsnormFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    RmsnormFwdPlanBuilder<DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestRmsnormFwdPlan, ExecutePlanWithBias)
{
    std::vector<int64_t> dims = {4, 3, 16, 16};
    unsigned int seed         = getGlobalTestSeed();
    auto graph = buildRmsnormFwdGraphWithBias(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    RmsnormFwdWithBiasTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    RmsnormFwdWithBiasTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_data_sdk::data_objects::RMSNormAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();

    RmsnormFwdParams params(*tensorMap.at(attributes.x_tensor_uid()),
                            *tensorMap.at(attributes.scale_tensor_uid()),
                            *tensorMap.at(attributes.epsilon_tensor_uid()),
                            *tensorMap.at(attributes.y_tensor_uid()));
    ASSERT_TRUE(attributes.bias_tensor_uid().has_value());
    params.biasTensor = unpackTensorAttributes(*tensorMap.at(attributes.bias_tensor_uid().value()));
    params.hasBias    = true;

    double epsilon = extractDoubleFromTensorValue(params.epsilonTensor, "Epsilon");

    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.tensors[attributes.x_tensor_uid()]->rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.tensors[attributes.scale_tensor_uid()]->rawHostData());
    auto shallowBiasTensor = createShallowTensor<float>(
        params.biasTensor,
        directTensorBundle.tensors[attributes.bias_tensor_uid().value()]->rawHostData());
    auto shallowYTensor = createShallowTensor<float>(
        params.yTensor, directTensorBundle.tensors[attributes.y_tensor_uid()]->rawHostData());

    CpuFpReferenceRmsnorm::forward<float, float, float, float>(*shallowXTensor,
                                                               *shallowScaleTensor,
                                                               *shallowYTensor,
                                                               epsilon,
                                                               nullptr,
                                                               shallowBiasTensor.get());

    std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();
    RmsnormFwdPlan<float, float, float, float> fwdPlan(std::move(params));
    fwdPlan.execute(variantPack);

    float tolerance = 1e-5f;
    CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        *directTensorBundle.tensors[attributes.y_tensor_uid()].get(),
        *planTensorBundle.tensors[attributes.y_tensor_uid()].get()));
}

TEST(TestRmsnormFwdPlanBuilder, PlanConstructionWithBias)
{
    std::vector<int64_t> dims = {1, 2, 1, 1};
    auto graph = buildRmsnormFwdGraphWithBias(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    RmsnormFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    bool result
        = dynamic_cast<RmsnormFwdPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestRmsnormFwdPlanBuilder, IsApplicableWithBias)
{
    std::vector<int64_t> dims = {1, 2, 1, 1};
    auto graph = buildRmsnormFwdGraphWithBias(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    RmsnormFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}
