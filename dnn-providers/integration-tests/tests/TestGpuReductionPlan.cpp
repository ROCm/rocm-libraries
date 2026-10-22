// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "RaggedGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuPlanBuilderRegistry.hpp"
#include "harness/gpu-graph-executor/detail/GpuReductionPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuReductionSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuReductionPlanBuilder, PlanConstruction)
{
    auto builder = hipdnn_test_sdk::utilities::createReductionGraph(
        hipdnn_test_sdk::utilities::ReductionGraphSpec{});
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuReductionPlan<float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuReductionPlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createReductionGraph(
        hipdnn_test_sdk::utilities::ReductionGraphSpec{});
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuReductionPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF> halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_ReductionAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->in_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuReductionPlanBuilder, IsNotApplicableForRaggedTensors)
{
    auto builder = hipdnn_test_sdk::utilities::createReductionGraph(
        hipdnn_test_sdk::utilities::ReductionGraphSpec{});

    const GpuReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto denseWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(patient.isApplicable(denseWrap.getNode(0), denseWrap.getTensorMap()));

    auto ragged
        = hipdnn_integration_tests::test_utils::markFirstTensorRagged(builder.GetBufferPointer());
    auto raggedWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(ragged.data(), ragged.size());
    EXPECT_FALSE(patient.isApplicable(raggedWrap.getNode(0), raggedWrap.getTensorMap()));
}

TEST(TestGpuReductionPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuReductionPlanBuilder, IsApplicableReturnsFalseForWrongOutputDataType)
{
    auto builder = hipdnn_test_sdk::utilities::createReductionGraph(
        hipdnn_test_sdk::utilities::ReductionGraphSpec{});
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuReductionPlanBuilder<DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT> patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuReductionPlanBuilder, IsApplicableReturnsFalseForWrongComputeDataType)
{
    auto builder = hipdnn_test_sdk::utilities::createReductionGraph(
        hipdnn_test_sdk::utilities::ReductionGraphSpec{});
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::HALF> patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuReductionPlanBuilder, IsApplicableReturnsFalseForIncorrectReductionMode)
{
    auto builder = hipdnn_test_sdk::utilities::createReductionGraph(
        hipdnn_test_sdk::utilities::ReductionGraphSpec{});
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto nodeAttributes
        = const_cast<ReductionAttributes*>(graph.getNode(0).attributes_as_ReductionAttributes());
    ASSERT_NE(nodeAttributes, nullptr);
    ASSERT_TRUE(nodeAttributes->mutate_mode(static_cast<ReductionMode>(99))); // Invalid mode

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuReductionPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::invalid_argument);
}

TEST(TestGpuReductionPlanBuilder, UnregisteredSignatureThrows)
{
    GpuPlanBuilderRegistry registry;

    const GpuReductionSignatureKey unregisteredKey{DataType::INT8, DataType::INT8, DataType::FLOAT};

    EXPECT_THROW(registry.getPlanBuilder(unregisteredKey), std::runtime_error);
}

TEST(TestGpuReductionPlanBuilder, IsApplicableReturnsFalseAboveMaxRank)
{
    const GpuReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    // An fp32 ADD over the last axis of a packed tensor of `rank` dims. It passes every other
    // isApplicable check, so the rank alone decides.
    const auto createAddOverLastAxisGraph = [](size_t rank) {
        const std::vector<int64_t> xDims(rank, 2);
        auto yDims = xDims;
        yDims.back() = 1;
        return hipdnn_test_sdk::utilities::createReductionGraph(
            hipdnn_test_sdk::utilities::ReductionGraphSpec::packed(
                ReductionMode::ADD, xDims, yDims));
    };

    auto maxRankBuilder
        = createAddOverLastAxisGraph(hipdnn_gpu_ref::GpuFpReferenceReduction::MAX_DIMS);
    auto maxRankGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        maxRankBuilder.GetBufferPointer(), maxRankBuilder.GetSize());
    EXPECT_TRUE(patient.isApplicable(maxRankGraph.getNode(0), maxRankGraph.getTensorMap()));

    // GpuFpReferenceReduction::reduce() would throw for a higher rank at execute time, so the
    // builder must decline it up front.
    auto tooHighRankBuilder
        = createAddOverLastAxisGraph(hipdnn_gpu_ref::GpuFpReferenceReduction::MAX_DIMS + 1);
    auto tooHighRankGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        tooHighRankBuilder.GetBufferPointer(), tooHighRankBuilder.GetSize());
    EXPECT_FALSE(
        patient.isApplicable(tooHighRankGraph.getNode(0), tooHighRankGraph.getTensorMap()));
}
