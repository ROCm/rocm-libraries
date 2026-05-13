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
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormBwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

TEST(TestRMSNormBwdPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> dims = {1, 2, 3, 4};
    auto graph = buildRMSNormBwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NCHW);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const RMSNormBwdPlanBuilder<DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT>
        builder;

    auto builtPlan = builder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<RMSNormBwdPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestRMSNormBwdPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> dims = {1, 2, 3, 4};
    auto graph = buildRMSNormBwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const RMSNormBwdPlanBuilder<DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const RMSNormBwdPlanBuilder<DataType::FLOAT,
                                DataType::HALF,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT>
        badInputTypePlanBuilder;
    EXPECT_FALSE(
        badInputTypePlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const RMSNormBwdPlanBuilder<DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::HALF,
                                DataType::FLOAT,
                                DataType::FLOAT>
        badScaleTypePlanBuilder;
    EXPECT_FALSE(
        badScaleTypePlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const RMSNormBwdPlanBuilder<DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::HALF>
        badComputeTypePlanBuilder;
    EXPECT_FALSE(badComputeTypePlanBuilder.isApplicable(graphWrapper.getNode(0),
                                                        graphWrapper.getTensorMap()));
}

TEST(TestRMSNormBwdPlan, ExecutePlan)
{
    const std::vector<int64_t> dims = {4, 8, 16, 32};
    const unsigned int seed = getGlobalTestSeed();

    auto graph = buildRMSNormBwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);

    RMSNormBwdTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    RMSNormBwdTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();

    RMSNormBwdParams params(*tensorMap.at(attributes.dy_tensor_uid()),
                            *tensorMap.at(attributes.x_tensor_uid()),
                            *tensorMap.at(attributes.scale_tensor_uid()),
                            *tensorMap.at(attributes.inv_rms_tensor_uid()),
                            *tensorMap.at(attributes.dx_tensor_uid()),
                            *tensorMap.at(attributes.dscale_tensor_uid()));

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowDyTensor = createShallowTensor<float>(
        params.dyTensor, directTensorBundle.tensors[attributes.dy_tensor_uid()]->rawHostData());
    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.tensors[attributes.x_tensor_uid()]->rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.tensors[attributes.scale_tensor_uid()]->rawHostData());
    auto shallowInvRmsTensor = createShallowTensor<float>(
        params.invRmsTensor,
        directTensorBundle.tensors[attributes.inv_rms_tensor_uid()]->rawHostData());

    auto shallowDxTensor = createShallowTensor<float>(
        params.dxTensor, directTensorBundle.tensors[attributes.dx_tensor_uid()]->rawHostData());
    auto shallowDScaleTensor = createShallowTensor<float>(
        params.dscaleTensor,
        directTensorBundle.tensors[attributes.dscale_tensor_uid()]->rawHostData());

    CpuFpReferenceRMSNorm::backward<float, float, float, float, float>(*shallowDyTensor,
                                                                       *shallowXTensor,
                                                                       *shallowScaleTensor,
                                                                       *shallowInvRmsTensor,
                                                                       *shallowDxTensor,
                                                                       *shallowDScaleTensor);

    RMSNormBwdPlan<float, float, float, float, float> bwdPlan(std::move(params));
    bwdPlan.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);

    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        *directTensorBundle.tensors[attributes.dx_tensor_uid()].get(),
        *planTensorBundle.tensors[attributes.dx_tensor_uid()].get()));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        *directTensorBundle.tensors[attributes.dscale_tensor_uid()].get(),
        *planTensorBundle.tensors[attributes.dscale_tensor_uid()].get()));
}
