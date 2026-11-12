// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/detail/GpuBatchnormBwdSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using hipdnn_data_sdk::types::half;

TEST(TestGpuBatchnormBwdSignatureKey, EqualityAndHash)
{
    const GpuBatchnormBwdSignatureKey key1{DataType::HALF,
                                           DataType::HALF,
                                           DataType::FLOAT,
                                           DataType::FLOAT,
                                           DataType::HALF,
                                           DataType::FLOAT};
    const GpuBatchnormBwdSignatureKey key2{key1};
    const GpuBatchnormBwdSignatureKey different{DataType::FLOAT,
                                                DataType::HALF,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::HALF,
                                                DataType::FLOAT};

    EXPECT_EQ(key1, key2);
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());
    EXPECT_FALSE(key1 == different);
    EXPECT_NE(key1.hashSelf(), different.hashSelf());
}

TEST(TestGpuBatchnormBwdSignatureKey, CreateFromNodeWithSavedStatistics)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormBwdSignatureKey key(graph.getNode(0), graph.getTensorMap());
    const GpuBatchnormBwdSignatureKey expected{DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT};

    EXPECT_EQ(key, expected);
}

TEST(TestGpuBatchnormBwdSignatureKey, CreateFromNodeWithoutSavedStatistics)
{
    const auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {12, 4, 2, 1}, {2, 3, 2, 2}, false, DataType::FLOAT, DataType::HALF, DataType::DOUBLE);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormBwdSignatureKey key(graph.getNode(0), graph.getTensorMap());

    const GpuBatchnormBwdSignatureKey expected{DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::HALF,
                                               DataType::HALF,
                                               DataType::FLOAT,
                                               DataType::FLOAT};
    EXPECT_EQ(key, expected);
}

class TestGpuBatchnormBwdSignatureKeySavedStatistic : public testing::TestWithParam<int64_t>
{
};

TEST_P(TestGpuBatchnormBwdSignatureKeySavedStatistic, MismatchedSavedStatisticsTypesThrow)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph({12, 4, 2, 1}, {2, 3, 2, 2});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper matchingGraph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_NO_THROW(
        GpuBatchnormBwdSignatureKey(matchingGraph.getNode(0), matchingGraph.getTensorMap()));

    auto graph = UnPackGraph(builder.GetBufferPointer());
    for(auto& tensor : graph->tensors)
    {
        if(tensor->uid == GetParam())
        {
            tensor->data_type = DataType::HALF;
        }
    }
    builder.Clear();
    builder.Finish(Graph::Pack(builder, graph.get()));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper mismatchedGraph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(
        GpuBatchnormBwdSignatureKey(mismatchedGraph.getNode(0), mismatchedGraph.getTensorMap()),
        std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(BothStatistics,
                         TestGpuBatchnormBwdSignatureKeySavedStatistic,
                         testing::Values(7, 8));

TEST(TestGpuBatchnormBwdSignatureKey, RegistersFallbackWithoutSavedStatistics)
{
    std::unordered_map<GpuBatchnormBwdSignatureKey,
                       std::unique_ptr<IGpuGraphNodePlanBuilder>,
                       GpuBatchnormBwdSignatureKey>
        builders;
    GpuBatchnormBwdSignatureKey::addPlanBuilder<DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::HALF,
                                                DataType::DOUBLE,
                                                DataType::FLOAT,
                                                DataType::FLOAT>(builders);

    EXPECT_EQ(builders.size(), 2);
    for(const bool useSavedStats : {false, true})
    {
        SCOPED_TRACE(useSavedStats);
        const auto builder
            = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph({12, 4, 2, 1},
                                                                       {2, 3, 2, 2},
                                                                       useSavedStats,
                                                                       DataType::FLOAT,
                                                                       DataType::HALF,
                                                                       DataType::DOUBLE);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());
        const GpuBatchnormBwdSignatureKey key(graph.getNode(0), graph.getTensorMap());
        const auto it = builders.find(key);
        ASSERT_NE(it, builders.end());
        EXPECT_TRUE(it->second->isApplicable(graph.getNode(0), graph.getTensorMap()));
        const auto plan = it->second->buildNodePlan(graph, graph.getNode(0));
        if(useSavedStats)
        {
            const bool hasExpectedType
                = dynamic_cast<GpuBatchnormBwdPlan<float, float, half, double, float, float>*>(
                      plan.get())
                  != nullptr;
            EXPECT_TRUE(hasExpectedType);
        }
        else
        {
            const bool hasExpectedType
                = dynamic_cast<GpuBatchnormBwdPlan<float, float, half, half, float, float>*>(
                      plan.get())
                  != nullptr;
            EXPECT_TRUE(hasExpectedType);
        }
    }
}

TEST(TestGpuBatchnormBwdSignatureKey, RegistersSupportedComputeTypesOnly)
{
    const auto builders = GpuBatchnormBwdSignatureKey::getPlanBuilders();

    EXPECT_EQ(builders.size(), 7);
    const GpuBatchnormBwdSignatureKey supported{DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT};
    const GpuBatchnormBwdSignatureKey unsupported{DataType::HALF,
                                                  DataType::HALF,
                                                  DataType::HALF,
                                                  DataType::HALF,
                                                  DataType::HALF,
                                                  DataType::HALF};
    EXPECT_NE(builders.find(supported), builders.end());
    EXPECT_EQ(builders.find(unsupported), builders.end());
}
