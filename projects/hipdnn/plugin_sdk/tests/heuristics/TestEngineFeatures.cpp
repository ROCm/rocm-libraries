// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <utility>

#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>

namespace
{
using namespace hipdnn_flatbuffers_sdk::data_objects;

struct Device
{
    int multiProcessorCount = 64;
    int warpSize = 32;
    size_t totalGlobalMem = 1024;
    int memoryBusWidth = 256;
    int memoryClockRate = 1000;
    size_t sharedMemPerBlock = 65536;
};

void addTensor(GraphT& graph, int64_t uid, std::vector<int64_t> dims)
{
    auto tensor = std::make_unique<TensorAttributesT>();
    tensor->uid = uid;
    tensor->dims = std::move(dims);
    tensor->data_type = DataType::HALF;
    graph.tensors.push_back(std::move(tensor));
}

template <typename TAttributes>
void addNode(GraphT& graph, TAttributes attributes)
{
    auto node = std::make_unique<NodeT>();
    node->attributes.Set(std::move(attributes));
    graph.nodes.push_back(std::move(node));
}

nlohmann::json features(const GraphT& graph)
{
    flatbuffers::FlatBufferBuilder graphBuffer;
    graphBuffer.Finish(Graph::Pack(graphBuffer, &graph));
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        graphBuffer.GetBufferPointer(), graphBuffer.GetSize());
    flatbuffers::FlatBufferBuilder configBuffer;
    configBuffer.Finish(CreateEngineConfig(configBuffer, 1));
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
        configBuffer.GetBufferPointer(), configBuffer.GetSize());
    return hipdnn_plugin_sdk::heuristics::engineFeatures(wrapper, config, Device{}).toJson();
}

TEST(TestEngineFeatures, BatchedMatmulUsesBroadcastedOutputBatch)
{
    GraphT graph;
    addTensor(graph, 1, {1, 3, 4});
    addTensor(graph, 2, {5, 4, 7});
    addTensor(graph, 3, {5, 3, 7});
    MatmulAttributesT matmul;
    matmul.a_tensor_uid = 1;
    matmul.b_tensor_uid = 2;
    matmul.c_tensor_uid = 3;
    addNode(graph, matmul);
    EXPECT_DOUBLE_EQ(features(graph).at("graph.flops").get<double>(), 2.0 * 5 * 3 * 7 * 4);
    graph.tensors[0]->dims[0] = 2;
    EXPECT_FALSE(features(graph).contains("graph.flops"));
}

TEST(TestEngineFeatures, GroupedConvolutionUsesChannelsPerGroup)
{
    GraphT graph;
    addTensor(graph, 1, {2, 8, 7, 7});
    addTensor(graph, 2, {12, 2, 3, 3});
    addTensor(graph, 3, {2, 12, 5, 5});
    ConvolutionFwdAttributesT conv;
    conv.x_tensor_uid = 1;
    conv.w_tensor_uid = 2;
    conv.y_tensor_uid = 3;
    addNode(graph, conv);
    EXPECT_DOUBLE_EQ(features(graph).at("graph.flops").get<double>(),
                     2.0 * 2 * 12 * 5 * 5 * 2 * 3 * 3);
}

TEST(TestEngineFeatures, RectangularCausalAttentionCountsItsActualMask)
{
    GraphT graph;
    addTensor(graph, 1, {2, 8, 3, 16});
    addTensor(graph, 2, {2, 2, 7, 16});
    addTensor(graph, 3, {2, 2, 7, 32});
    addTensor(graph, 4, {2, 8, 3, 32});
    SdpaAttributesT sdpa;
    sdpa.q_tensor_uid = 1;
    sdpa.k_tensor_uid = 2;
    sdpa.v_tensor_uid = 3;
    sdpa.o_tensor_uid = 4;
    sdpa.causal_mask_bottom_right = true;
    addNode(graph, sdpa);
    EXPECT_DOUBLE_EQ(features(graph).at("graph.flops").get<double>(), 2.0 * 2 * 8 * 18 * (16 + 32));
    auto* attention = graph.nodes[0]->attributes.AsSdpaAttributes();
    attention->causal_mask_bottom_right = false;
    attention->causal_mask = true;
    EXPECT_DOUBLE_EQ(features(graph).at("graph.flops").get<double>(), 2.0 * 2 * 8 * 6 * (16 + 32));
    attention->seq_len_q_tensor_uid = 5;
    EXPECT_FALSE(features(graph).contains("graph.flops"));
}

TEST(TestEngineFeatures, UnsupportedFusedWorkDoesNotPublishAPartialGraphRate)
{
    GraphT graph;
    addTensor(graph, 1, {3, 4});
    addTensor(graph, 2, {4, 7});
    addTensor(graph, 3, {3, 7});
    MatmulAttributesT matmul;
    matmul.a_tensor_uid = 1;
    matmul.b_tensor_uid = 2;
    matmul.c_tensor_uid = 3;
    addNode(graph, matmul);
    addNode(graph, PointwiseAttributesT{});
    const auto published = features(graph);
    EXPECT_FALSE(published.contains("graph.flops"));
    EXPECT_DOUBLE_EQ(published.at("graph.nodes[0].flops").get<double>(), 2.0 * 3 * 7 * 4);
}
} // namespace
