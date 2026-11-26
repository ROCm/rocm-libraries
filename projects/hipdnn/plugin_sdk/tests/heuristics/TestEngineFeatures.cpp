// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <atomic>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/FeatureSemantics.hpp>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>

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

void addTensor(GraphT& graph,
               int64_t uid,
               std::vector<int64_t> dims,
               std::vector<int64_t> strides = {},
               bool isVirtual = false)
{
    auto tensor = std::make_unique<TensorAttributesT>();
    tensor->uid = uid;
    tensor->dims = std::move(dims);
    tensor->strides = std::move(strides);
    tensor->virtual_ = isVirtual;
    tensor->data_type = DataType::HALF;
    graph.tensors.push_back(std::move(tensor));
}

template <typename TAttributes>
void addNode(GraphT& graph, TAttributes attributes, DataType compute = DataType::UNSET)
{
    auto node = std::make_unique<NodeT>();
    node->compute_data_type = compute;
    node->attributes.Set(std::move(attributes));
    graph.nodes.push_back(std::move(node));
}

nlohmann::json features(const flatbuffers::FlatBufferBuilder& graphBuffer)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        graphBuffer.GetBufferPointer(), graphBuffer.GetSize());
    flatbuffers::FlatBufferBuilder configBuffer;
    configBuffer.Finish(CreateEngineConfig(configBuffer, 1));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
        configBuffer.GetBufferPointer(), configBuffer.GetSize());
    return hipdnn_plugin_sdk::heuristics::engineFeatures(wrapper, config, Device{}).toJson();
}

nlohmann::json features(const GraphT& graph)
{
    flatbuffers::FlatBufferBuilder graphBuffer;
    graphBuffer.Finish(Graph::Pack(graphBuffer, &graph));
    return features(graphBuffer);
}

// Representative graphs whose published features are pinned by the parity golden below.
GraphT matmulBroadcastGraph()
{
    GraphT graph;
    addTensor(graph, 1, {1, 3, 4}, {12, 4, 1});
    addTensor(graph, 2, {5, 4, 7}, {28, 7, 1});
    addTensor(graph, 3, {5, 3, 7}, {21, 7, 1});
    MatmulAttributesT matmul;
    matmul.a_tensor_uid = 1;
    matmul.b_tensor_uid = 2;
    matmul.c_tensor_uid = 3;
    addNode(graph, matmul, DataType::FLOAT);
    return graph;
}

ConvolutionFwdAttributesT convolution(const std::vector<int64_t>& padding,
                                      std::vector<int64_t> stride)
{
    ConvolutionFwdAttributesT conv;
    conv.x_tensor_uid = 1;
    conv.w_tensor_uid = 2;
    conv.y_tensor_uid = 3;
    conv.pre_padding = padding;
    conv.post_padding = padding;
    conv.stride = std::move(stride);
    conv.dilation = {1, 1};
    conv.conv_mode = ConvMode::CROSS_CORRELATION;
    return conv;
}

GraphT convFwdGraph()
{
    GraphT graph;
    addTensor(graph, 1, {2, 8, 7, 7}, {392, 49, 7, 1});
    addTensor(graph, 2, {12, 8, 3, 3}, {72, 9, 3, 1});
    addTensor(graph, 3, {2, 12, 5, 5}, {300, 25, 5, 1});
    addNode(graph, convolution({0, 0}, {1, 1}), DataType::FLOAT);
    return graph;
}

GraphT groupedConvFwdGraph()
{
    GraphT graph;
    // Channels-last strides: the layout must reach the features unchanged.
    addTensor(graph, 1, {2, 8, 7, 7}, {392, 1, 56, 8});
    addTensor(graph, 2, {12, 2, 3, 3}, {18, 1, 6, 2});
    addTensor(graph, 3, {2, 12, 4, 4}, {192, 1, 48, 12});
    addNode(graph, convolution({1, 1}, {2, 2}), DataType::FLOAT);
    return graph;
}

// attention() and attentionGraph() (tensor shapes and strides for uids 1-8) are copied into
// tools/common/tests/TestOptionalFeatures.cpp, which cannot include this file: keep them equal.
SdpaAttributesT attention()
{
    SdpaAttributesT sdpa;
    sdpa.q_tensor_uid = 1;
    sdpa.k_tensor_uid = 2;
    sdpa.v_tensor_uid = 3;
    sdpa.o_tensor_uid = 4;
    return sdpa;
}

GraphT attentionGraph(const SdpaAttributesT& sdpa)
{
    GraphT graph;
    addTensor(graph, 1, {2, 8, 3, 16}, {384, 48, 16, 1});
    addTensor(graph, 2, {2, 2, 7, 16}, {224, 112, 16, 1});
    addTensor(graph, 3, {2, 2, 7, 32}, {448, 224, 32, 1});
    addTensor(graph, 4, {2, 8, 3, 32}, {768, 96, 32, 1});
    for(int64_t uid = 5; uid <= 8; ++uid)
    {
        addTensor(graph, uid, {2, 1, 1, 1}, {1, 1, 1, 1});
    }
    addNode(graph, sdpa, DataType::FLOAT);
    return graph;
}

GraphT causalTopLeftGraph()
{
    auto sdpa = attention();
    sdpa.causal_mask = true;
    return attentionGraph(sdpa);
}

GraphT causalBottomRightGraph()
{
    auto sdpa = attention();
    sdpa.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    sdpa.right_bound = 0;
    return attentionGraph(sdpa);
}

GraphT boundedAttentionGraph()
{
    auto sdpa = attention();
    sdpa.left_bound = 4;
    sdpa.right_bound = 2;
    return attentionGraph(sdpa);
}

GraphT dropoutAttentionGraph()
{
    auto sdpa = attention();
    sdpa.dropout_probability = 0.1F;
    sdpa.seed_tensor_uid = 5;
    sdpa.offset_tensor_uid = 6;
    sdpa.attn_scale_value = 0.125F;
    return attentionGraph(sdpa);
}

GraphT pagedVariableLengthAttentionGraph()
{
    auto sdpa = attention();
    sdpa.seq_len_q_tensor_uid = 5;
    sdpa.seq_len_kv_tensor_uid = 6;
    sdpa.page_table_k_tensor_uid = 7;
    sdpa.page_table_v_tensor_uid = 8;
    sdpa.padding_mask = true;
    return attentionGraph(sdpa);
}

GraphT convBiasReluGraph()
{
    GraphT graph;
    addTensor(graph, 1, {2, 8, 7, 7}, {392, 49, 7, 1});
    addTensor(graph, 2, {12, 8, 3, 3}, {72, 9, 3, 1});
    addTensor(graph, 3, {2, 12, 5, 5}, {300, 25, 5, 1}, true);
    addTensor(graph, 4, {1, 12, 1, 1}, {12, 1, 1, 1});
    addTensor(graph, 5, {2, 12, 5, 5}, {300, 25, 5, 1}, true);
    addTensor(graph, 6, {2, 12, 5, 5}, {300, 25, 5, 1});
    addNode(graph, convolution({0, 0}, {1, 1}), DataType::FLOAT);
    PointwiseAttributesT bias;
    bias.operation = PointwiseMode::ADD;
    bias.in_0_tensor_uid = 3;
    bias.in_1_tensor_uid = 4;
    bias.out_0_tensor_uid = 5;
    addNode(graph, bias, DataType::FLOAT);
    PointwiseAttributesT relu;
    relu.operation = PointwiseMode::RELU_FWD;
    relu.in_0_tensor_uid = 5;
    relu.out_0_tensor_uid = 6;
    addNode(graph, relu, DataType::FLOAT);
    return graph;
}

// Features published for these graphs. Shipped models read these names, so each must keep
// its name, value and JSON kind.
constexpr const char* GOLDEN_DEVICE = R"json({
    "device.cu_count": 64, "device.lds_size": 65536, "device.memory_bus_width": 256,
    "device.memory_clock_rate": 1000, "device.multi_processor_count": 64,
    "device.peak_memory_bandwidth": 64000000.0, "device.total_global_mem": 1024,
    "device.warp_size": 32
})json";

constexpr const char* GOLDEN_MATMUL_BROADCAST = R"json({
    "graph.flops": 840.0, "graph.node_count": 1, "graph.nodes[0].a.data_type": 2,
    "graph.nodes[0].a.dims[0]": 1, "graph.nodes[0].a.dims[1]": 3, "graph.nodes[0].a.dims[2]": 4,
    "graph.nodes[0].a.strides[0]": 12, "graph.nodes[0].a.strides[1]": 4,
    "graph.nodes[0].a.strides[2]": 1, "graph.nodes[0].b.data_type": 2,
    "graph.nodes[0].b.dims[0]": 5, "graph.nodes[0].b.dims[1]": 4, "graph.nodes[0].b.dims[2]": 7,
    "graph.nodes[0].b.strides[0]": 28, "graph.nodes[0].b.strides[1]": 7,
    "graph.nodes[0].b.strides[2]": 1, "graph.nodes[0].c.data_type": 2,
    "graph.nodes[0].c.dims[0]": 5, "graph.nodes[0].c.dims[1]": 3, "graph.nodes[0].c.dims[2]": 7,
    "graph.nodes[0].c.strides[0]": 21, "graph.nodes[0].c.strides[1]": 7,
    "graph.nodes[0].c.strides[2]": 1, "graph.nodes[0].compute_data_type": 1,
    "graph.nodes[0].flops": 840.0, "graph.nodes[0].type": 9, "graph.tensor_count": 3,
    "graph.tensors[0].data_type": 2, "graph.tensors[0].dims[0]": 1,
    "graph.tensors[0].dims[1]": 3, "graph.tensors[0].dims[2]": 4, "graph.tensors[0].rank": 3,
    "graph.tensors[0].strides[0]": 12, "graph.tensors[0].strides[1]": 4,
    "graph.tensors[0].strides[2]": 1, "graph.tensors[1].data_type": 2,
    "graph.tensors[1].dims[0]": 5, "graph.tensors[1].dims[1]": 4, "graph.tensors[1].dims[2]": 7,
    "graph.tensors[1].rank": 3, "graph.tensors[1].strides[0]": 28,
    "graph.tensors[1].strides[1]": 7, "graph.tensors[1].strides[2]": 1,
    "graph.tensors[2].data_type": 2, "graph.tensors[2].dims[0]": 5,
    "graph.tensors[2].dims[1]": 3, "graph.tensors[2].dims[2]": 7, "graph.tensors[2].rank": 3,
    "graph.tensors[2].strides[0]": 21, "graph.tensors[2].strides[1]": 7,
    "graph.tensors[2].strides[2]": 1
})json";

constexpr const char* GOLDEN_CONV_FWD = R"json({
    "graph.flops": 86400.0, "graph.node_count": 1, "graph.nodes[0].compute_data_type": 1,
    "graph.nodes[0].conv_mode": 2, "graph.nodes[0].dilation[0]": 1,
    "graph.nodes[0].dilation[1]": 1, "graph.nodes[0].flops": 86400.0,
    "graph.nodes[0].post_padding[0]": 0, "graph.nodes[0].post_padding[1]": 0,
    "graph.nodes[0].pre_padding[0]": 0, "graph.nodes[0].pre_padding[1]": 0,
    "graph.nodes[0].stride[0]": 1, "graph.nodes[0].stride[1]": 1, "graph.nodes[0].type": 5,
    "graph.nodes[0].w.data_type": 2, "graph.nodes[0].w.dims[0]": 12,
    "graph.nodes[0].w.dims[1]": 8, "graph.nodes[0].w.dims[2]": 3, "graph.nodes[0].w.dims[3]": 3,
    "graph.nodes[0].w.strides[0]": 72, "graph.nodes[0].w.strides[1]": 9,
    "graph.nodes[0].w.strides[2]": 3, "graph.nodes[0].w.strides[3]": 1,
    "graph.nodes[0].x.data_type": 2, "graph.nodes[0].x.dims[0]": 2,
    "graph.nodes[0].x.dims[1]": 8, "graph.nodes[0].x.dims[2]": 7, "graph.nodes[0].x.dims[3]": 7,
    "graph.nodes[0].x.strides[0]": 392, "graph.nodes[0].x.strides[1]": 49,
    "graph.nodes[0].x.strides[2]": 7, "graph.nodes[0].x.strides[3]": 1,
    "graph.nodes[0].y.data_type": 2, "graph.nodes[0].y.dims[0]": 2,
    "graph.nodes[0].y.dims[1]": 12, "graph.nodes[0].y.dims[2]": 5,
    "graph.nodes[0].y.dims[3]": 5, "graph.nodes[0].y.strides[0]": 300,
    "graph.nodes[0].y.strides[1]": 25, "graph.nodes[0].y.strides[2]": 5,
    "graph.nodes[0].y.strides[3]": 1, "graph.tensor_count": 3, "graph.tensors[0].data_type": 2,
    "graph.tensors[0].dims[0]": 2, "graph.tensors[0].dims[1]": 8, "graph.tensors[0].dims[2]": 7,
    "graph.tensors[0].dims[3]": 7, "graph.tensors[0].rank": 4,
    "graph.tensors[0].strides[0]": 392, "graph.tensors[0].strides[1]": 49,
    "graph.tensors[0].strides[2]": 7, "graph.tensors[0].strides[3]": 1,
    "graph.tensors[1].data_type": 2, "graph.tensors[1].dims[0]": 12,
    "graph.tensors[1].dims[1]": 8, "graph.tensors[1].dims[2]": 3, "graph.tensors[1].dims[3]": 3,
    "graph.tensors[1].rank": 4, "graph.tensors[1].strides[0]": 72,
    "graph.tensors[1].strides[1]": 9, "graph.tensors[1].strides[2]": 3,
    "graph.tensors[1].strides[3]": 1, "graph.tensors[2].data_type": 2,
    "graph.tensors[2].dims[0]": 2, "graph.tensors[2].dims[1]": 12,
    "graph.tensors[2].dims[2]": 5, "graph.tensors[2].dims[3]": 5, "graph.tensors[2].rank": 4,
    "graph.tensors[2].strides[0]": 300, "graph.tensors[2].strides[1]": 25,
    "graph.tensors[2].strides[2]": 5, "graph.tensors[2].strides[3]": 1
})json";

constexpr const char* GOLDEN_CONV_FWD_GROUPED = R"json({
    "graph.flops": 13824.0, "graph.node_count": 1, "graph.nodes[0].compute_data_type": 1,
    "graph.nodes[0].conv_mode": 2, "graph.nodes[0].dilation[0]": 1,
    "graph.nodes[0].dilation[1]": 1, "graph.nodes[0].flops": 13824.0,
    "graph.nodes[0].post_padding[0]": 1, "graph.nodes[0].post_padding[1]": 1,
    "graph.nodes[0].pre_padding[0]": 1, "graph.nodes[0].pre_padding[1]": 1,
    "graph.nodes[0].stride[0]": 2, "graph.nodes[0].stride[1]": 2, "graph.nodes[0].type": 5,
    "graph.nodes[0].w.data_type": 2, "graph.nodes[0].w.dims[0]": 12,
    "graph.nodes[0].w.dims[1]": 2, "graph.nodes[0].w.dims[2]": 3, "graph.nodes[0].w.dims[3]": 3,
    "graph.nodes[0].w.strides[0]": 18, "graph.nodes[0].w.strides[1]": 1,
    "graph.nodes[0].w.strides[2]": 6, "graph.nodes[0].w.strides[3]": 2,
    "graph.nodes[0].x.data_type": 2, "graph.nodes[0].x.dims[0]": 2,
    "graph.nodes[0].x.dims[1]": 8, "graph.nodes[0].x.dims[2]": 7, "graph.nodes[0].x.dims[3]": 7,
    "graph.nodes[0].x.strides[0]": 392, "graph.nodes[0].x.strides[1]": 1,
    "graph.nodes[0].x.strides[2]": 56, "graph.nodes[0].x.strides[3]": 8,
    "graph.nodes[0].y.data_type": 2, "graph.nodes[0].y.dims[0]": 2,
    "graph.nodes[0].y.dims[1]": 12, "graph.nodes[0].y.dims[2]": 4,
    "graph.nodes[0].y.dims[3]": 4, "graph.nodes[0].y.strides[0]": 192,
    "graph.nodes[0].y.strides[1]": 1, "graph.nodes[0].y.strides[2]": 48,
    "graph.nodes[0].y.strides[3]": 12, "graph.tensor_count": 3, "graph.tensors[0].data_type": 2,
    "graph.tensors[0].dims[0]": 2, "graph.tensors[0].dims[1]": 8, "graph.tensors[0].dims[2]": 7,
    "graph.tensors[0].dims[3]": 7, "graph.tensors[0].rank": 4,
    "graph.tensors[0].strides[0]": 392, "graph.tensors[0].strides[1]": 1,
    "graph.tensors[0].strides[2]": 56, "graph.tensors[0].strides[3]": 8,
    "graph.tensors[1].data_type": 2, "graph.tensors[1].dims[0]": 12,
    "graph.tensors[1].dims[1]": 2, "graph.tensors[1].dims[2]": 3, "graph.tensors[1].dims[3]": 3,
    "graph.tensors[1].rank": 4, "graph.tensors[1].strides[0]": 18,
    "graph.tensors[1].strides[1]": 1, "graph.tensors[1].strides[2]": 6,
    "graph.tensors[1].strides[3]": 2, "graph.tensors[2].data_type": 2,
    "graph.tensors[2].dims[0]": 2, "graph.tensors[2].dims[1]": 12,
    "graph.tensors[2].dims[2]": 4, "graph.tensors[2].dims[3]": 4, "graph.tensors[2].rank": 4,
    "graph.tensors[2].strides[0]": 192, "graph.tensors[2].strides[1]": 1,
    "graph.tensors[2].strides[2]": 48, "graph.tensors[2].strides[3]": 12
})json";

constexpr const char* GOLDEN_SDPA_CAUSAL_TOP_LEFT = R"json({
    "graph.flops": 9216.0, "graph.node_count": 1, "graph.nodes[0].alibi_mask": false,
    "graph.nodes[0].causal_mask": true, "graph.nodes[0].causal_mask_bottom_right": false,
    "graph.nodes[0].compute_data_type": 1, "graph.nodes[0].diagonal_alignment": 0,
    "graph.nodes[0].flops": 9216.0, "graph.nodes[0].has_attention_mask": false,
    "graph.nodes[0].has_variable_lengths": false, "graph.nodes[0].k.data_type": 2,
    "graph.nodes[0].k.dims[0]": 2, "graph.nodes[0].k.dims[1]": 2, "graph.nodes[0].k.dims[2]": 7,
    "graph.nodes[0].k.dims[3]": 16, "graph.nodes[0].k.strides[0]": 224,
    "graph.nodes[0].k.strides[1]": 112, "graph.nodes[0].k.strides[2]": 16,
    "graph.nodes[0].k.strides[3]": 1, "graph.nodes[0].o.data_type": 2,
    "graph.nodes[0].o.dims[0]": 2, "graph.nodes[0].o.dims[1]": 8, "graph.nodes[0].o.dims[2]": 3,
    "graph.nodes[0].o.dims[3]": 32, "graph.nodes[0].o.strides[0]": 768,
    "graph.nodes[0].o.strides[1]": 96, "graph.nodes[0].o.strides[2]": 32,
    "graph.nodes[0].o.strides[3]": 1, "graph.nodes[0].padding_mask": false,
    "graph.nodes[0].q.data_type": 2, "graph.nodes[0].q.dims[0]": 2,
    "graph.nodes[0].q.dims[1]": 8, "graph.nodes[0].q.dims[2]": 3,
    "graph.nodes[0].q.dims[3]": 16, "graph.nodes[0].q.strides[0]": 384,
    "graph.nodes[0].q.strides[1]": 48, "graph.nodes[0].q.strides[2]": 16,
    "graph.nodes[0].q.strides[3]": 1, "graph.nodes[0].type": 12,
    "graph.nodes[0].v.data_type": 2, "graph.nodes[0].v.dims[0]": 2,
    "graph.nodes[0].v.dims[1]": 2, "graph.nodes[0].v.dims[2]": 7,
    "graph.nodes[0].v.dims[3]": 32, "graph.nodes[0].v.strides[0]": 448,
    "graph.nodes[0].v.strides[1]": 224, "graph.nodes[0].v.strides[2]": 32,
    "graph.nodes[0].v.strides[3]": 1, "graph.tensor_count": 8, "graph.tensors[0].data_type": 2,
    "graph.tensors[0].dims[0]": 2, "graph.tensors[0].dims[1]": 8, "graph.tensors[0].dims[2]": 3,
    "graph.tensors[0].dims[3]": 16, "graph.tensors[0].rank": 4,
    "graph.tensors[0].strides[0]": 384, "graph.tensors[0].strides[1]": 48,
    "graph.tensors[0].strides[2]": 16, "graph.tensors[0].strides[3]": 1,
    "graph.tensors[1].data_type": 2, "graph.tensors[1].dims[0]": 2,
    "graph.tensors[1].dims[1]": 2, "graph.tensors[1].dims[2]": 7,
    "graph.tensors[1].dims[3]": 16, "graph.tensors[1].rank": 4,
    "graph.tensors[1].strides[0]": 224, "graph.tensors[1].strides[1]": 112,
    "graph.tensors[1].strides[2]": 16, "graph.tensors[1].strides[3]": 1,
    "graph.tensors[2].data_type": 2, "graph.tensors[2].dims[0]": 2,
    "graph.tensors[2].dims[1]": 2, "graph.tensors[2].dims[2]": 7,
    "graph.tensors[2].dims[3]": 32, "graph.tensors[2].rank": 4,
    "graph.tensors[2].strides[0]": 448, "graph.tensors[2].strides[1]": 224,
    "graph.tensors[2].strides[2]": 32, "graph.tensors[2].strides[3]": 1,
    "graph.tensors[3].data_type": 2, "graph.tensors[3].dims[0]": 2,
    "graph.tensors[3].dims[1]": 8, "graph.tensors[3].dims[2]": 3,
    "graph.tensors[3].dims[3]": 32, "graph.tensors[3].rank": 4,
    "graph.tensors[3].strides[0]": 768, "graph.tensors[3].strides[1]": 96,
    "graph.tensors[3].strides[2]": 32, "graph.tensors[3].strides[3]": 1,
    "graph.tensors[4].data_type": 2, "graph.tensors[4].dims[0]": 2,
    "graph.tensors[4].dims[1]": 1, "graph.tensors[4].dims[2]": 1, "graph.tensors[4].dims[3]": 1,
    "graph.tensors[4].rank": 4, "graph.tensors[4].strides[0]": 1,
    "graph.tensors[4].strides[1]": 1, "graph.tensors[4].strides[2]": 1,
    "graph.tensors[4].strides[3]": 1, "graph.tensors[5].data_type": 2,
    "graph.tensors[5].dims[0]": 2, "graph.tensors[5].dims[1]": 1, "graph.tensors[5].dims[2]": 1,
    "graph.tensors[5].dims[3]": 1, "graph.tensors[5].rank": 4, "graph.tensors[5].strides[0]": 1,
    "graph.tensors[5].strides[1]": 1, "graph.tensors[5].strides[2]": 1,
    "graph.tensors[5].strides[3]": 1, "graph.tensors[6].data_type": 2,
    "graph.tensors[6].dims[0]": 2, "graph.tensors[6].dims[1]": 1, "graph.tensors[6].dims[2]": 1,
    "graph.tensors[6].dims[3]": 1, "graph.tensors[6].rank": 4, "graph.tensors[6].strides[0]": 1,
    "graph.tensors[6].strides[1]": 1, "graph.tensors[6].strides[2]": 1,
    "graph.tensors[6].strides[3]": 1, "graph.tensors[7].data_type": 2,
    "graph.tensors[7].dims[0]": 2, "graph.tensors[7].dims[1]": 1, "graph.tensors[7].dims[2]": 1,
    "graph.tensors[7].dims[3]": 1, "graph.tensors[7].rank": 4, "graph.tensors[7].strides[0]": 1,
    "graph.tensors[7].strides[1]": 1, "graph.tensors[7].strides[2]": 1,
    "graph.tensors[7].strides[3]": 1
})json";

constexpr const char* GOLDEN_SDPA_CAUSAL_BOTTOM_RIGHT = R"json({
    "graph.flops": 27648.0, "graph.node_count": 1, "graph.nodes[0].alibi_mask": false,
    "graph.nodes[0].causal_mask": false, "graph.nodes[0].causal_mask_bottom_right": false,
    "graph.nodes[0].compute_data_type": 1, "graph.nodes[0].diagonal_alignment": 1,
    "graph.nodes[0].flops": 27648.0, "graph.nodes[0].has_attention_mask": false,
    "graph.nodes[0].has_variable_lengths": false, "graph.nodes[0].k.data_type": 2,
    "graph.nodes[0].k.dims[0]": 2, "graph.nodes[0].k.dims[1]": 2, "graph.nodes[0].k.dims[2]": 7,
    "graph.nodes[0].k.dims[3]": 16, "graph.nodes[0].k.strides[0]": 224,
    "graph.nodes[0].k.strides[1]": 112, "graph.nodes[0].k.strides[2]": 16,
    "graph.nodes[0].k.strides[3]": 1, "graph.nodes[0].o.data_type": 2,
    "graph.nodes[0].o.dims[0]": 2, "graph.nodes[0].o.dims[1]": 8, "graph.nodes[0].o.dims[2]": 3,
    "graph.nodes[0].o.dims[3]": 32, "graph.nodes[0].o.strides[0]": 768,
    "graph.nodes[0].o.strides[1]": 96, "graph.nodes[0].o.strides[2]": 32,
    "graph.nodes[0].o.strides[3]": 1, "graph.nodes[0].padding_mask": false,
    "graph.nodes[0].q.data_type": 2, "graph.nodes[0].q.dims[0]": 2,
    "graph.nodes[0].q.dims[1]": 8, "graph.nodes[0].q.dims[2]": 3,
    "graph.nodes[0].q.dims[3]": 16, "graph.nodes[0].q.strides[0]": 384,
    "graph.nodes[0].q.strides[1]": 48, "graph.nodes[0].q.strides[2]": 16,
    "graph.nodes[0].q.strides[3]": 1, "graph.nodes[0].right_bound": 0,
    "graph.nodes[0].type": 12, "graph.nodes[0].v.data_type": 2, "graph.nodes[0].v.dims[0]": 2,
    "graph.nodes[0].v.dims[1]": 2, "graph.nodes[0].v.dims[2]": 7,
    "graph.nodes[0].v.dims[3]": 32, "graph.nodes[0].v.strides[0]": 448,
    "graph.nodes[0].v.strides[1]": 224, "graph.nodes[0].v.strides[2]": 32,
    "graph.nodes[0].v.strides[3]": 1, "graph.tensor_count": 8, "graph.tensors[0].data_type": 2,
    "graph.tensors[0].dims[0]": 2, "graph.tensors[0].dims[1]": 8, "graph.tensors[0].dims[2]": 3,
    "graph.tensors[0].dims[3]": 16, "graph.tensors[0].rank": 4,
    "graph.tensors[0].strides[0]": 384, "graph.tensors[0].strides[1]": 48,
    "graph.tensors[0].strides[2]": 16, "graph.tensors[0].strides[3]": 1,
    "graph.tensors[1].data_type": 2, "graph.tensors[1].dims[0]": 2,
    "graph.tensors[1].dims[1]": 2, "graph.tensors[1].dims[2]": 7,
    "graph.tensors[1].dims[3]": 16, "graph.tensors[1].rank": 4,
    "graph.tensors[1].strides[0]": 224, "graph.tensors[1].strides[1]": 112,
    "graph.tensors[1].strides[2]": 16, "graph.tensors[1].strides[3]": 1,
    "graph.tensors[2].data_type": 2, "graph.tensors[2].dims[0]": 2,
    "graph.tensors[2].dims[1]": 2, "graph.tensors[2].dims[2]": 7,
    "graph.tensors[2].dims[3]": 32, "graph.tensors[2].rank": 4,
    "graph.tensors[2].strides[0]": 448, "graph.tensors[2].strides[1]": 224,
    "graph.tensors[2].strides[2]": 32, "graph.tensors[2].strides[3]": 1,
    "graph.tensors[3].data_type": 2, "graph.tensors[3].dims[0]": 2,
    "graph.tensors[3].dims[1]": 8, "graph.tensors[3].dims[2]": 3,
    "graph.tensors[3].dims[3]": 32, "graph.tensors[3].rank": 4,
    "graph.tensors[3].strides[0]": 768, "graph.tensors[3].strides[1]": 96,
    "graph.tensors[3].strides[2]": 32, "graph.tensors[3].strides[3]": 1,
    "graph.tensors[4].data_type": 2, "graph.tensors[4].dims[0]": 2,
    "graph.tensors[4].dims[1]": 1, "graph.tensors[4].dims[2]": 1, "graph.tensors[4].dims[3]": 1,
    "graph.tensors[4].rank": 4, "graph.tensors[4].strides[0]": 1,
    "graph.tensors[4].strides[1]": 1, "graph.tensors[4].strides[2]": 1,
    "graph.tensors[4].strides[3]": 1, "graph.tensors[5].data_type": 2,
    "graph.tensors[5].dims[0]": 2, "graph.tensors[5].dims[1]": 1, "graph.tensors[5].dims[2]": 1,
    "graph.tensors[5].dims[3]": 1, "graph.tensors[5].rank": 4, "graph.tensors[5].strides[0]": 1,
    "graph.tensors[5].strides[1]": 1, "graph.tensors[5].strides[2]": 1,
    "graph.tensors[5].strides[3]": 1, "graph.tensors[6].data_type": 2,
    "graph.tensors[6].dims[0]": 2, "graph.tensors[6].dims[1]": 1, "graph.tensors[6].dims[2]": 1,
    "graph.tensors[6].dims[3]": 1, "graph.tensors[6].rank": 4, "graph.tensors[6].strides[0]": 1,
    "graph.tensors[6].strides[1]": 1, "graph.tensors[6].strides[2]": 1,
    "graph.tensors[6].strides[3]": 1, "graph.tensors[7].data_type": 2,
    "graph.tensors[7].dims[0]": 2, "graph.tensors[7].dims[1]": 1, "graph.tensors[7].dims[2]": 1,
    "graph.tensors[7].dims[3]": 1, "graph.tensors[7].rank": 4, "graph.tensors[7].strides[0]": 1,
    "graph.tensors[7].strides[1]": 1, "graph.tensors[7].strides[2]": 1,
    "graph.tensors[7].strides[3]": 1
})json";

constexpr const char* GOLDEN_SDPA_BOUNDS = R"json({
    "graph.node_count": 1, "graph.nodes[0].alibi_mask": false,
    "graph.nodes[0].causal_mask": false, "graph.nodes[0].causal_mask_bottom_right": false,
    "graph.nodes[0].compute_data_type": 1, "graph.nodes[0].diagonal_alignment": 0,
    "graph.nodes[0].has_attention_mask": false, "graph.nodes[0].has_variable_lengths": false,
    "graph.nodes[0].k.data_type": 2, "graph.nodes[0].k.dims[0]": 2,
    "graph.nodes[0].k.dims[1]": 2, "graph.nodes[0].k.dims[2]": 7,
    "graph.nodes[0].k.dims[3]": 16, "graph.nodes[0].k.strides[0]": 224,
    "graph.nodes[0].k.strides[1]": 112, "graph.nodes[0].k.strides[2]": 16,
    "graph.nodes[0].k.strides[3]": 1, "graph.nodes[0].left_bound": 4,
    "graph.nodes[0].o.data_type": 2, "graph.nodes[0].o.dims[0]": 2,
    "graph.nodes[0].o.dims[1]": 8, "graph.nodes[0].o.dims[2]": 3,
    "graph.nodes[0].o.dims[3]": 32, "graph.nodes[0].o.strides[0]": 768,
    "graph.nodes[0].o.strides[1]": 96, "graph.nodes[0].o.strides[2]": 32,
    "graph.nodes[0].o.strides[3]": 1, "graph.nodes[0].padding_mask": false,
    "graph.nodes[0].q.data_type": 2, "graph.nodes[0].q.dims[0]": 2,
    "graph.nodes[0].q.dims[1]": 8, "graph.nodes[0].q.dims[2]": 3,
    "graph.nodes[0].q.dims[3]": 16, "graph.nodes[0].q.strides[0]": 384,
    "graph.nodes[0].q.strides[1]": 48, "graph.nodes[0].q.strides[2]": 16,
    "graph.nodes[0].q.strides[3]": 1, "graph.nodes[0].right_bound": 2,
    "graph.nodes[0].type": 12, "graph.nodes[0].v.data_type": 2, "graph.nodes[0].v.dims[0]": 2,
    "graph.nodes[0].v.dims[1]": 2, "graph.nodes[0].v.dims[2]": 7,
    "graph.nodes[0].v.dims[3]": 32, "graph.nodes[0].v.strides[0]": 448,
    "graph.nodes[0].v.strides[1]": 224, "graph.nodes[0].v.strides[2]": 32,
    "graph.nodes[0].v.strides[3]": 1, "graph.tensor_count": 8, "graph.tensors[0].data_type": 2,
    "graph.tensors[0].dims[0]": 2, "graph.tensors[0].dims[1]": 8, "graph.tensors[0].dims[2]": 3,
    "graph.tensors[0].dims[3]": 16, "graph.tensors[0].rank": 4,
    "graph.tensors[0].strides[0]": 384, "graph.tensors[0].strides[1]": 48,
    "graph.tensors[0].strides[2]": 16, "graph.tensors[0].strides[3]": 1,
    "graph.tensors[1].data_type": 2, "graph.tensors[1].dims[0]": 2,
    "graph.tensors[1].dims[1]": 2, "graph.tensors[1].dims[2]": 7,
    "graph.tensors[1].dims[3]": 16, "graph.tensors[1].rank": 4,
    "graph.tensors[1].strides[0]": 224, "graph.tensors[1].strides[1]": 112,
    "graph.tensors[1].strides[2]": 16, "graph.tensors[1].strides[3]": 1,
    "graph.tensors[2].data_type": 2, "graph.tensors[2].dims[0]": 2,
    "graph.tensors[2].dims[1]": 2, "graph.tensors[2].dims[2]": 7,
    "graph.tensors[2].dims[3]": 32, "graph.tensors[2].rank": 4,
    "graph.tensors[2].strides[0]": 448, "graph.tensors[2].strides[1]": 224,
    "graph.tensors[2].strides[2]": 32, "graph.tensors[2].strides[3]": 1,
    "graph.tensors[3].data_type": 2, "graph.tensors[3].dims[0]": 2,
    "graph.tensors[3].dims[1]": 8, "graph.tensors[3].dims[2]": 3,
    "graph.tensors[3].dims[3]": 32, "graph.tensors[3].rank": 4,
    "graph.tensors[3].strides[0]": 768, "graph.tensors[3].strides[1]": 96,
    "graph.tensors[3].strides[2]": 32, "graph.tensors[3].strides[3]": 1,
    "graph.tensors[4].data_type": 2, "graph.tensors[4].dims[0]": 2,
    "graph.tensors[4].dims[1]": 1, "graph.tensors[4].dims[2]": 1, "graph.tensors[4].dims[3]": 1,
    "graph.tensors[4].rank": 4, "graph.tensors[4].strides[0]": 1,
    "graph.tensors[4].strides[1]": 1, "graph.tensors[4].strides[2]": 1,
    "graph.tensors[4].strides[3]": 1, "graph.tensors[5].data_type": 2,
    "graph.tensors[5].dims[0]": 2, "graph.tensors[5].dims[1]": 1, "graph.tensors[5].dims[2]": 1,
    "graph.tensors[5].dims[3]": 1, "graph.tensors[5].rank": 4, "graph.tensors[5].strides[0]": 1,
    "graph.tensors[5].strides[1]": 1, "graph.tensors[5].strides[2]": 1,
    "graph.tensors[5].strides[3]": 1, "graph.tensors[6].data_type": 2,
    "graph.tensors[6].dims[0]": 2, "graph.tensors[6].dims[1]": 1, "graph.tensors[6].dims[2]": 1,
    "graph.tensors[6].dims[3]": 1, "graph.tensors[6].rank": 4, "graph.tensors[6].strides[0]": 1,
    "graph.tensors[6].strides[1]": 1, "graph.tensors[6].strides[2]": 1,
    "graph.tensors[6].strides[3]": 1, "graph.tensors[7].data_type": 2,
    "graph.tensors[7].dims[0]": 2, "graph.tensors[7].dims[1]": 1, "graph.tensors[7].dims[2]": 1,
    "graph.tensors[7].dims[3]": 1, "graph.tensors[7].rank": 4, "graph.tensors[7].strides[0]": 1,
    "graph.tensors[7].strides[1]": 1, "graph.tensors[7].strides[2]": 1,
    "graph.tensors[7].strides[3]": 1
})json";

constexpr const char* GOLDEN_SDPA_DROPOUT = R"json({
    "graph.node_count": 1, "graph.nodes[0].alibi_mask": false,
    "graph.nodes[0].causal_mask": false, "graph.nodes[0].causal_mask_bottom_right": false,
    "graph.nodes[0].compute_data_type": 1, "graph.nodes[0].diagonal_alignment": 0,
    "graph.nodes[0].dropout_probability": 0.10000000149011612,
    "graph.nodes[0].has_attention_mask": false, "graph.nodes[0].has_variable_lengths": false,
    "graph.nodes[0].k.data_type": 2, "graph.nodes[0].k.dims[0]": 2,
    "graph.nodes[0].k.dims[1]": 2, "graph.nodes[0].k.dims[2]": 7,
    "graph.nodes[0].k.dims[3]": 16, "graph.nodes[0].k.strides[0]": 224,
    "graph.nodes[0].k.strides[1]": 112, "graph.nodes[0].k.strides[2]": 16,
    "graph.nodes[0].k.strides[3]": 1, "graph.nodes[0].o.data_type": 2,
    "graph.nodes[0].o.dims[0]": 2, "graph.nodes[0].o.dims[1]": 8, "graph.nodes[0].o.dims[2]": 3,
    "graph.nodes[0].o.dims[3]": 32, "graph.nodes[0].o.strides[0]": 768,
    "graph.nodes[0].o.strides[1]": 96, "graph.nodes[0].o.strides[2]": 32,
    "graph.nodes[0].o.strides[3]": 1, "graph.nodes[0].padding_mask": false,
    "graph.nodes[0].q.data_type": 2, "graph.nodes[0].q.dims[0]": 2,
    "graph.nodes[0].q.dims[1]": 8, "graph.nodes[0].q.dims[2]": 3,
    "graph.nodes[0].q.dims[3]": 16, "graph.nodes[0].q.strides[0]": 384,
    "graph.nodes[0].q.strides[1]": 48, "graph.nodes[0].q.strides[2]": 16,
    "graph.nodes[0].q.strides[3]": 1, "graph.nodes[0].type": 12,
    "graph.nodes[0].v.data_type": 2, "graph.nodes[0].v.dims[0]": 2,
    "graph.nodes[0].v.dims[1]": 2, "graph.nodes[0].v.dims[2]": 7,
    "graph.nodes[0].v.dims[3]": 32, "graph.nodes[0].v.strides[0]": 448,
    "graph.nodes[0].v.strides[1]": 224, "graph.nodes[0].v.strides[2]": 32,
    "graph.nodes[0].v.strides[3]": 1, "graph.tensor_count": 8, "graph.tensors[0].data_type": 2,
    "graph.tensors[0].dims[0]": 2, "graph.tensors[0].dims[1]": 8, "graph.tensors[0].dims[2]": 3,
    "graph.tensors[0].dims[3]": 16, "graph.tensors[0].rank": 4,
    "graph.tensors[0].strides[0]": 384, "graph.tensors[0].strides[1]": 48,
    "graph.tensors[0].strides[2]": 16, "graph.tensors[0].strides[3]": 1,
    "graph.tensors[1].data_type": 2, "graph.tensors[1].dims[0]": 2,
    "graph.tensors[1].dims[1]": 2, "graph.tensors[1].dims[2]": 7,
    "graph.tensors[1].dims[3]": 16, "graph.tensors[1].rank": 4,
    "graph.tensors[1].strides[0]": 224, "graph.tensors[1].strides[1]": 112,
    "graph.tensors[1].strides[2]": 16, "graph.tensors[1].strides[3]": 1,
    "graph.tensors[2].data_type": 2, "graph.tensors[2].dims[0]": 2,
    "graph.tensors[2].dims[1]": 2, "graph.tensors[2].dims[2]": 7,
    "graph.tensors[2].dims[3]": 32, "graph.tensors[2].rank": 4,
    "graph.tensors[2].strides[0]": 448, "graph.tensors[2].strides[1]": 224,
    "graph.tensors[2].strides[2]": 32, "graph.tensors[2].strides[3]": 1,
    "graph.tensors[3].data_type": 2, "graph.tensors[3].dims[0]": 2,
    "graph.tensors[3].dims[1]": 8, "graph.tensors[3].dims[2]": 3,
    "graph.tensors[3].dims[3]": 32, "graph.tensors[3].rank": 4,
    "graph.tensors[3].strides[0]": 768, "graph.tensors[3].strides[1]": 96,
    "graph.tensors[3].strides[2]": 32, "graph.tensors[3].strides[3]": 1,
    "graph.tensors[4].data_type": 2, "graph.tensors[4].dims[0]": 2,
    "graph.tensors[4].dims[1]": 1, "graph.tensors[4].dims[2]": 1, "graph.tensors[4].dims[3]": 1,
    "graph.tensors[4].rank": 4, "graph.tensors[4].strides[0]": 1,
    "graph.tensors[4].strides[1]": 1, "graph.tensors[4].strides[2]": 1,
    "graph.tensors[4].strides[3]": 1, "graph.tensors[5].data_type": 2,
    "graph.tensors[5].dims[0]": 2, "graph.tensors[5].dims[1]": 1, "graph.tensors[5].dims[2]": 1,
    "graph.tensors[5].dims[3]": 1, "graph.tensors[5].rank": 4, "graph.tensors[5].strides[0]": 1,
    "graph.tensors[5].strides[1]": 1, "graph.tensors[5].strides[2]": 1,
    "graph.tensors[5].strides[3]": 1, "graph.tensors[6].data_type": 2,
    "graph.tensors[6].dims[0]": 2, "graph.tensors[6].dims[1]": 1, "graph.tensors[6].dims[2]": 1,
    "graph.tensors[6].dims[3]": 1, "graph.tensors[6].rank": 4, "graph.tensors[6].strides[0]": 1,
    "graph.tensors[6].strides[1]": 1, "graph.tensors[6].strides[2]": 1,
    "graph.tensors[6].strides[3]": 1, "graph.tensors[7].data_type": 2,
    "graph.tensors[7].dims[0]": 2, "graph.tensors[7].dims[1]": 1, "graph.tensors[7].dims[2]": 1,
    "graph.tensors[7].dims[3]": 1, "graph.tensors[7].rank": 4, "graph.tensors[7].strides[0]": 1,
    "graph.tensors[7].strides[1]": 1, "graph.tensors[7].strides[2]": 1,
    "graph.tensors[7].strides[3]": 1
})json";

constexpr const char* GOLDEN_SDPA_SEQ_LEN_PAGED = R"json({
    "graph.node_count": 1, "graph.nodes[0].alibi_mask": false,
    "graph.nodes[0].causal_mask": false, "graph.nodes[0].causal_mask_bottom_right": false,
    "graph.nodes[0].compute_data_type": 1, "graph.nodes[0].diagonal_alignment": 0,
    "graph.nodes[0].has_attention_mask": false, "graph.nodes[0].has_variable_lengths": true,
    "graph.nodes[0].k.data_type": 2, "graph.nodes[0].k.dims[0]": 2,
    "graph.nodes[0].k.dims[1]": 2, "graph.nodes[0].k.dims[2]": 7,
    "graph.nodes[0].k.dims[3]": 16, "graph.nodes[0].k.strides[0]": 224,
    "graph.nodes[0].k.strides[1]": 112, "graph.nodes[0].k.strides[2]": 16,
    "graph.nodes[0].k.strides[3]": 1, "graph.nodes[0].o.data_type": 2,
    "graph.nodes[0].o.dims[0]": 2, "graph.nodes[0].o.dims[1]": 8, "graph.nodes[0].o.dims[2]": 3,
    "graph.nodes[0].o.dims[3]": 32, "graph.nodes[0].o.strides[0]": 768,
    "graph.nodes[0].o.strides[1]": 96, "graph.nodes[0].o.strides[2]": 32,
    "graph.nodes[0].o.strides[3]": 1, "graph.nodes[0].padding_mask": true,
    "graph.nodes[0].q.data_type": 2, "graph.nodes[0].q.dims[0]": 2,
    "graph.nodes[0].q.dims[1]": 8, "graph.nodes[0].q.dims[2]": 3,
    "graph.nodes[0].q.dims[3]": 16, "graph.nodes[0].q.strides[0]": 384,
    "graph.nodes[0].q.strides[1]": 48, "graph.nodes[0].q.strides[2]": 16,
    "graph.nodes[0].q.strides[3]": 1, "graph.nodes[0].type": 12,
    "graph.nodes[0].v.data_type": 2, "graph.nodes[0].v.dims[0]": 2,
    "graph.nodes[0].v.dims[1]": 2, "graph.nodes[0].v.dims[2]": 7,
    "graph.nodes[0].v.dims[3]": 32, "graph.nodes[0].v.strides[0]": 448,
    "graph.nodes[0].v.strides[1]": 224, "graph.nodes[0].v.strides[2]": 32,
    "graph.nodes[0].v.strides[3]": 1, "graph.tensor_count": 8, "graph.tensors[0].data_type": 2,
    "graph.tensors[0].dims[0]": 2, "graph.tensors[0].dims[1]": 8, "graph.tensors[0].dims[2]": 3,
    "graph.tensors[0].dims[3]": 16, "graph.tensors[0].rank": 4,
    "graph.tensors[0].strides[0]": 384, "graph.tensors[0].strides[1]": 48,
    "graph.tensors[0].strides[2]": 16, "graph.tensors[0].strides[3]": 1,
    "graph.tensors[1].data_type": 2, "graph.tensors[1].dims[0]": 2,
    "graph.tensors[1].dims[1]": 2, "graph.tensors[1].dims[2]": 7,
    "graph.tensors[1].dims[3]": 16, "graph.tensors[1].rank": 4,
    "graph.tensors[1].strides[0]": 224, "graph.tensors[1].strides[1]": 112,
    "graph.tensors[1].strides[2]": 16, "graph.tensors[1].strides[3]": 1,
    "graph.tensors[2].data_type": 2, "graph.tensors[2].dims[0]": 2,
    "graph.tensors[2].dims[1]": 2, "graph.tensors[2].dims[2]": 7,
    "graph.tensors[2].dims[3]": 32, "graph.tensors[2].rank": 4,
    "graph.tensors[2].strides[0]": 448, "graph.tensors[2].strides[1]": 224,
    "graph.tensors[2].strides[2]": 32, "graph.tensors[2].strides[3]": 1,
    "graph.tensors[3].data_type": 2, "graph.tensors[3].dims[0]": 2,
    "graph.tensors[3].dims[1]": 8, "graph.tensors[3].dims[2]": 3,
    "graph.tensors[3].dims[3]": 32, "graph.tensors[3].rank": 4,
    "graph.tensors[3].strides[0]": 768, "graph.tensors[3].strides[1]": 96,
    "graph.tensors[3].strides[2]": 32, "graph.tensors[3].strides[3]": 1,
    "graph.tensors[4].data_type": 2, "graph.tensors[4].dims[0]": 2,
    "graph.tensors[4].dims[1]": 1, "graph.tensors[4].dims[2]": 1, "graph.tensors[4].dims[3]": 1,
    "graph.tensors[4].rank": 4, "graph.tensors[4].strides[0]": 1,
    "graph.tensors[4].strides[1]": 1, "graph.tensors[4].strides[2]": 1,
    "graph.tensors[4].strides[3]": 1, "graph.tensors[5].data_type": 2,
    "graph.tensors[5].dims[0]": 2, "graph.tensors[5].dims[1]": 1, "graph.tensors[5].dims[2]": 1,
    "graph.tensors[5].dims[3]": 1, "graph.tensors[5].rank": 4, "graph.tensors[5].strides[0]": 1,
    "graph.tensors[5].strides[1]": 1, "graph.tensors[5].strides[2]": 1,
    "graph.tensors[5].strides[3]": 1, "graph.tensors[6].data_type": 2,
    "graph.tensors[6].dims[0]": 2, "graph.tensors[6].dims[1]": 1, "graph.tensors[6].dims[2]": 1,
    "graph.tensors[6].dims[3]": 1, "graph.tensors[6].rank": 4, "graph.tensors[6].strides[0]": 1,
    "graph.tensors[6].strides[1]": 1, "graph.tensors[6].strides[2]": 1,
    "graph.tensors[6].strides[3]": 1, "graph.tensors[7].data_type": 2,
    "graph.tensors[7].dims[0]": 2, "graph.tensors[7].dims[1]": 1, "graph.tensors[7].dims[2]": 1,
    "graph.tensors[7].dims[3]": 1, "graph.tensors[7].rank": 4, "graph.tensors[7].strides[0]": 1,
    "graph.tensors[7].strides[1]": 1, "graph.tensors[7].strides[2]": 1,
    "graph.tensors[7].strides[3]": 1
})json";

constexpr const char* GOLDEN_CONV_BIAS_RELU = R"json({
    "graph.node_count": 3, "graph.nodes[0].compute_data_type": 1, "graph.nodes[0].conv_mode": 2,
    "graph.nodes[0].dilation[0]": 1, "graph.nodes[0].dilation[1]": 1,
    "graph.nodes[0].flops": 86400.0, "graph.nodes[0].post_padding[0]": 0,
    "graph.nodes[0].post_padding[1]": 0, "graph.nodes[0].pre_padding[0]": 0,
    "graph.nodes[0].pre_padding[1]": 0, "graph.nodes[0].stride[0]": 1,
    "graph.nodes[0].stride[1]": 1, "graph.nodes[0].type": 5, "graph.nodes[0].w.data_type": 2,
    "graph.nodes[0].w.dims[0]": 12, "graph.nodes[0].w.dims[1]": 8,
    "graph.nodes[0].w.dims[2]": 3, "graph.nodes[0].w.dims[3]": 3,
    "graph.nodes[0].w.strides[0]": 72, "graph.nodes[0].w.strides[1]": 9,
    "graph.nodes[0].w.strides[2]": 3, "graph.nodes[0].w.strides[3]": 1,
    "graph.nodes[0].x.data_type": 2, "graph.nodes[0].x.dims[0]": 2,
    "graph.nodes[0].x.dims[1]": 8, "graph.nodes[0].x.dims[2]": 7, "graph.nodes[0].x.dims[3]": 7,
    "graph.nodes[0].x.strides[0]": 392, "graph.nodes[0].x.strides[1]": 49,
    "graph.nodes[0].x.strides[2]": 7, "graph.nodes[0].x.strides[3]": 1,
    "graph.nodes[0].y.data_type": 2, "graph.nodes[0].y.dims[0]": 2,
    "graph.nodes[0].y.dims[1]": 12, "graph.nodes[0].y.dims[2]": 5,
    "graph.nodes[0].y.dims[3]": 5, "graph.nodes[0].y.strides[0]": 300,
    "graph.nodes[0].y.strides[1]": 25, "graph.nodes[0].y.strides[2]": 5,
    "graph.nodes[0].y.strides[3]": 1, "graph.nodes[1].compute_data_type": 1,
    "graph.nodes[1].type": 2, "graph.nodes[2].compute_data_type": 1, "graph.nodes[2].type": 2,
    "graph.tensor_count": 6, "graph.tensors[0].data_type": 2, "graph.tensors[0].dims[0]": 2,
    "graph.tensors[0].dims[1]": 8, "graph.tensors[0].dims[2]": 7, "graph.tensors[0].dims[3]": 7,
    "graph.tensors[0].rank": 4, "graph.tensors[0].strides[0]": 392,
    "graph.tensors[0].strides[1]": 49, "graph.tensors[0].strides[2]": 7,
    "graph.tensors[0].strides[3]": 1, "graph.tensors[1].data_type": 2,
    "graph.tensors[1].dims[0]": 12, "graph.tensors[1].dims[1]": 8,
    "graph.tensors[1].dims[2]": 3, "graph.tensors[1].dims[3]": 3, "graph.tensors[1].rank": 4,
    "graph.tensors[1].strides[0]": 72, "graph.tensors[1].strides[1]": 9,
    "graph.tensors[1].strides[2]": 3, "graph.tensors[1].strides[3]": 1,
    "graph.tensors[2].data_type": 2, "graph.tensors[2].dims[0]": 2,
    "graph.tensors[2].dims[1]": 12, "graph.tensors[2].dims[2]": 5,
    "graph.tensors[2].dims[3]": 5, "graph.tensors[2].rank": 4,
    "graph.tensors[2].strides[0]": 300, "graph.tensors[2].strides[1]": 25,
    "graph.tensors[2].strides[2]": 5, "graph.tensors[2].strides[3]": 1,
    "graph.tensors[3].data_type": 2, "graph.tensors[3].dims[0]": 1,
    "graph.tensors[3].dims[1]": 12, "graph.tensors[3].dims[2]": 1,
    "graph.tensors[3].dims[3]": 1, "graph.tensors[3].rank": 4,
    "graph.tensors[3].strides[0]": 12, "graph.tensors[3].strides[1]": 1,
    "graph.tensors[3].strides[2]": 1, "graph.tensors[3].strides[3]": 1,
    "graph.tensors[4].data_type": 2, "graph.tensors[4].dims[0]": 2,
    "graph.tensors[4].dims[1]": 12, "graph.tensors[4].dims[2]": 5,
    "graph.tensors[4].dims[3]": 5, "graph.tensors[4].rank": 4,
    "graph.tensors[4].strides[0]": 300, "graph.tensors[4].strides[1]": 25,
    "graph.tensors[4].strides[2]": 5, "graph.tensors[4].strides[3]": 1,
    "graph.tensors[5].data_type": 2, "graph.tensors[5].dims[0]": 2,
    "graph.tensors[5].dims[1]": 12, "graph.tensors[5].dims[2]": 5,
    "graph.tensors[5].dims[3]": 5, "graph.tensors[5].rank": 4,
    "graph.tensors[5].strides[0]": 300, "graph.tensors[5].strides[1]": 25,
    "graph.tensors[5].strides[2]": 5, "graph.tensors[5].strides[3]": 1
})json";

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
}

TEST(TestEngineFeatures, UnsupportedFusedWorkDoesNotPublishAPartialGraphRate)
{
    GraphT graph;
    addTensor(graph, 1, {3, 4});
    addTensor(graph, 2, {4, 7});
    addTensor(graph, 3, {3, 7}, {}, true);
    addTensor(graph, 4, {3, 7});
    MatmulAttributesT matmul;
    matmul.a_tensor_uid = 1;
    matmul.b_tensor_uid = 2;
    matmul.c_tensor_uid = 3;
    addNode(graph, matmul);
    CustomOpAttributesT custom;
    custom.custom_op_id = "example.opaque";
    custom.input_tensor_uids = {3};
    custom.output_tensor_uids = {4};
    addNode(graph, custom);
    const auto published = features(graph);
    EXPECT_FALSE(published.contains("graph.flops"));
    EXPECT_FALSE(published.contains("graph.arithmetic_intensity"));
    EXPECT_DOUBLE_EQ(published.at("graph.nodes[0].flops").get<double>(), 2.0 * 3 * 7 * 4);
    // Only the unknown type loses its total; the known one and absent ones still aggregate.
    EXPECT_FALSE(published.contains("graph.flops_by_type.CustomOpAttributes"));
    EXPECT_DOUBLE_EQ(published.at("graph.flops_by_type.MatmulAttributes").get<double>(),
                     2.0 * 3 * 7 * 4);
    EXPECT_DOUBLE_EQ(published.at("graph.flops_by_type.ConvolutionFwdAttributes").get<double>(),
                     0.0);
}

// nlohmann compares 1 and 1.0 equal; a model reading a feature can still tell them apart.
std::string kind(const nlohmann::json& value)
{
    if(value.is_boolean())
    {
        return "bool";
    }
    if(value.is_number_integer())
    {
        return "integer";
    }
    if(value.is_number_float())
    {
        return "float";
    }
    return value.type_name();
}

TEST(TestEngineFeatures, GenericOperandsPublishEveryGoldenFeature)
{
    const std::vector<std::pair<const char*, GraphT (*)()>> cases = {
        {GOLDEN_MATMUL_BROADCAST, matmulBroadcastGraph},
        {GOLDEN_CONV_FWD, convFwdGraph},
        {GOLDEN_CONV_FWD_GROUPED, groupedConvFwdGraph},
        {GOLDEN_SDPA_CAUSAL_TOP_LEFT, causalTopLeftGraph},
        {GOLDEN_SDPA_CAUSAL_BOTTOM_RIGHT, causalBottomRightGraph},
        {GOLDEN_SDPA_BOUNDS, boundedAttentionGraph},
        {GOLDEN_SDPA_DROPOUT, dropoutAttentionGraph},
        {GOLDEN_SDPA_SEQ_LEN_PAGED, pagedVariableLengthAttentionGraph},
        {GOLDEN_CONV_BIAS_RELU, convBiasReluGraph},
    };
    const auto device = nlohmann::json::parse(GOLDEN_DEVICE);
    for(const auto& [golden, make] : cases)
    {
        const auto published = features(make());
        auto expected = nlohmann::json::parse(golden);
        expected.update(device);
        for(const auto& [name, value] : expected.items())
        {
            SCOPED_TRACE(name);
            ASSERT_TRUE(published.contains(name));
            EXPECT_EQ(kind(published.at(name)), kind(value));
            EXPECT_EQ(published.at(name), value);
        }
    }
}

TEST(TestEngineFeatures, EveryNodeTypePublishesItsOperands)
{
    size_t visited = 0;
    for(const auto type : EnumValuesNodeAttributes())
    {
        SCOPED_TRACE(EnumNameNodeAttributes(type));
        flatbuffers::FlatBufferBuilder builder;
        const std::vector<int64_t> dims{4};
        const auto tensor
            = CreateTensorAttributesDirect(builder, 0, nullptr, DataType::HALF, nullptr, &dims);
        // An empty table reads as every field's default, so each type's required tensor
        // operands reference uid 0, which the graph holds.
        const auto start = builder.StartTable();
        const flatbuffers::Offset<void> attributes(builder.EndTable(start));
        const auto node = CreateNode(
            builder, 0, DataType::FLOAT, type, type == NodeAttributes::NONE ? 0 : attributes);
        const auto tensors = builder.CreateVector(&tensor, 1);
        const auto nodes = builder.CreateVector(&node, 1);
        GraphBuilder graph(builder);
        graph.add_tensors(tensors);
        graph.add_nodes(nodes);
        builder.Finish(graph.Finish());
        const auto published = features(builder);

        if(type == NodeAttributes::NONE)
        {
            EXPECT_FALSE(published.contains("graph.nodes[0].data_dependent"));
            continue;
        }
        ++visited;
        // Required routing offsets are the only required operands whose contents decide
        // the work.
        const bool routed = type == NodeAttributes::MoeGroupedMatmulAttributes
                            || type == NodeAttributes::MoeGroupedMatmulBwdAttributes;
        EXPECT_EQ(published.at("graph.nodes[0].data_dependent"), routed);
        // Tensor operands resolve to uid 0, whose single dimension every one publishes.
        const std::string suffix = ".dims[0]";
        size_t operands = 0;
        for(const auto& entry : published.items())
        {
            const auto& name = entry.key();
            if(name.rfind("graph.nodes[0].", 0) == 0 && name.size() > suffix.size()
               && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            {
                ++operands;
            }
        }
        // Custom ops reference tensors only through (here empty) uid vectors.
        if(type == NodeAttributes::CustomOpAttributes)
        {
            EXPECT_EQ(operands, 0U);
        }
        else
        {
            EXPECT_GT(operands, 0U);
        }
    }
    EXPECT_EQ(visited, 25U);
}

TEST(TestEngineFeatures, ConvolutionBackwardDataPublishesItsOperands)
{
    GraphT graph;
    addTensor(graph, 1, {2, 12, 4, 4}, {192, 16, 4, 1});
    addTensor(graph, 2, {12, 8, 3, 3});
    addTensor(graph, 3, {2, 8, 7, 7});
    ConvolutionBwdAttributesT conv;
    conv.dy_tensor_uid = 1;
    conv.w_tensor_uid = 2;
    conv.dx_tensor_uid = 3;
    conv.pre_padding = {1, 1};
    conv.stride = {2, 2};
    conv.conv_mode = ConvMode::CONVOLUTION;
    addNode(graph, conv);
    const auto published = features(graph);
    EXPECT_EQ(published.at("graph.nodes[0].dy.strides[1]"), 16);
    EXPECT_EQ(published.at("graph.nodes[0].w.rank"), 4);
    EXPECT_EQ(published.at("graph.nodes[0].dx.numel"), 2 * 8 * 7 * 7);
    EXPECT_EQ(published.at("graph.nodes[0].dx.virtual"), false);
    EXPECT_EQ(published.at("graph.nodes[0].pre_padding[1]"), 1);
    EXPECT_EQ(published.at("graph.nodes[0].stride[0]"), 2);
    EXPECT_FALSE(published.contains("graph.nodes[0].dilation[0]"));
    EXPECT_EQ(published.at("graph.nodes[0].conv_mode"),
              static_cast<int64_t>(ConvMode::CONVOLUTION));
    EXPECT_EQ(published.at("graph.nodes[0].data_dependent"), false);
}

TEST(TestEngineFeatures, ConvolutionWeightGradientPublishesItsOperands)
{
    GraphT graph;
    addTensor(graph, 1, {2, 8, 7, 7});
    addTensor(graph, 2, {2, 12, 5, 5});
    addTensor(graph, 3, {12, 8, 3, 3});
    ConvolutionWrwAttributesT conv;
    conv.x_tensor_uid = 1;
    conv.dy_tensor_uid = 2;
    conv.dw_tensor_uid = 3;
    conv.dilation = {2, 2};
    addNode(graph, conv);
    const auto published = features(graph);
    EXPECT_EQ(published.at("graph.nodes[0].x.dims[1]"), 8);
    EXPECT_EQ(published.at("graph.nodes[0].dy.dims[1]"), 12);
    EXPECT_EQ(published.at("graph.nodes[0].dw.numel"), 12 * 8 * 3 * 3);
    EXPECT_EQ(published.at("graph.nodes[0].dilation[1]"), 2);
    EXPECT_EQ(published.at("graph.nodes[0].data_dependent"), false);
}

TEST(TestEngineFeatures, PointwiseAxisIsAnAttributeNotATensor)
{
    GraphT graph;
    addTensor(graph, 1, {2, 3}, {3, 1}, true);
    addTensor(graph, 2, {2, 3}, {3, 1});
    PointwiseAttributesT index;
    index.operation = PointwiseMode::GEN_INDEX;
    index.axis_tensor_uid = 1;
    index.in_0_tensor_uid = 1;
    index.out_0_tensor_uid = 2;
    index.relu_lower_clip = 0.5F;
    addNode(graph, index);
    const auto published = features(graph);
    EXPECT_EQ(published.at("graph.nodes[0].operation"),
              static_cast<int64_t>(PointwiseMode::GEN_INDEX));
    EXPECT_EQ(published.at("graph.nodes[0].axis_tensor_uid"), 1);
    EXPECT_FALSE(published.contains("graph.nodes[0].axis.dims[0]"));
    EXPECT_EQ(published.at("graph.nodes[0].in_0.virtual"), true);
    EXPECT_EQ(published.at("graph.nodes[0].out_0.numel"), 6);
    EXPECT_FALSE(published.contains("graph.nodes[0].in_1.dims[0]"));
    EXPECT_DOUBLE_EQ(published.at("graph.nodes[0].relu_lower_clip").get<double>(), 0.5);
    EXPECT_FALSE(published.contains("graph.nodes[0].swish_beta"));
}

TEST(TestEngineFeatures, ReductionPublishesItsOperands)
{
    GraphT graph;
    addTensor(graph, 1, {4, 16});
    addTensor(graph, 2, {4, 1});
    ReductionAttributesT reduction;
    reduction.mode = ReductionMode::AMAX;
    reduction.in_tensor_uid = 1;
    reduction.out_tensor_uid = 2;
    reduction.is_deterministic = true;
    addNode(graph, reduction);
    const auto published = features(graph);
    EXPECT_EQ(published.at("graph.nodes[0].mode"), static_cast<int64_t>(ReductionMode::AMAX));
    EXPECT_EQ(published.at("graph.nodes[0].in.numel"), 64);
    EXPECT_EQ(published.at("graph.nodes[0].out.dims[1]"), 1);
    EXPECT_EQ(published.at("graph.nodes[0].is_deterministic"), true);
}

TEST(TestEngineFeatures, LayernormSkipsAbsentOptionalOperands)
{
    GraphT graph;
    for(int64_t uid = 1; uid <= 5; ++uid)
    {
        addTensor(graph, uid, {8, 64});
    }
    LayernormAttributesT norm;
    norm.x_tensor_uid = 1;
    norm.scale_tensor_uid = 2;
    norm.bias_tensor_uid = 3;
    norm.epsilon_tensor_uid = 4;
    norm.y_tensor_uid = 5;
    norm.normalized_dim_count = 1;
    norm.forward_phase = NormFwdPhase::INFERENCE;
    addNode(graph, norm);
    const auto published = features(graph);
    EXPECT_EQ(published.at("graph.nodes[0].epsilon.numel"), 8 * 64);
    EXPECT_EQ(published.at("graph.nodes[0].normalized_dim_count"), 1);
    EXPECT_EQ(published.at("graph.nodes[0].forward_phase"),
              static_cast<int64_t>(NormFwdPhase::INFERENCE));
    EXPECT_FALSE(published.contains("graph.nodes[0].mean.rank"));
    EXPECT_FALSE(published.contains("graph.nodes[0].inv_variance.rank"));
}

TEST(TestEngineFeatures, ResamplePublishesEveryWindowElement)
{
    GraphT graph;
    addTensor(graph, 1, {2, 8, 8, 8});
    addTensor(graph, 2, {2, 8, 4, 4});
    ResampleFwdAttributesT pool;
    pool.x_tensor_uid = 1;
    pool.y_tensor_uid = 2;
    pool.window = {3, 2};
    pool.stride = {2, 2};
    pool.resample_mode = ResampleMode::MAXPOOL;
    pool.generate_index = false;
    addNode(graph, pool);
    const auto published = features(graph);
    EXPECT_EQ(published.at("graph.nodes[0].window[0]"), 3);
    EXPECT_EQ(published.at("graph.nodes[0].window[1]"), 2);
    EXPECT_FALSE(published.contains("graph.nodes[0].window[2]"));
    EXPECT_EQ(published.at("graph.nodes[0].resample_mode"),
              static_cast<int64_t>(ResampleMode::MAXPOOL));
    EXPECT_EQ(published.at("graph.nodes[0].generate_index"), false);
    EXPECT_FALSE(published.contains("graph.nodes[0].index.rank"));
}

TEST(TestEngineFeatures, UidVectorsPublishOneRolePerElement)
{
    GraphT graph;
    for(int64_t uid = 1; uid <= 6; ++uid)
    {
        addTensor(graph, uid, {1, 16});
    }
    addTensor(graph, 7, {2, 16});
    BatchnormAttributesT norm;
    norm.x_tensor_uid = 1;
    norm.scale_tensor_uid = 2;
    norm.bias_tensor_uid = 3;
    norm.epsilon_tensor_uid = 4;
    norm.y_tensor_uid = 5;
    norm.peer_stats_tensor_uid = {6, 7};
    addNode(graph, norm);
    const auto published = features(graph);
    EXPECT_EQ(published.at("graph.nodes[0].peer_stats[0].dims[0]"), 1);
    EXPECT_EQ(published.at("graph.nodes[0].peer_stats[1].dims[0]"), 2);
    EXPECT_FALSE(published.contains("graph.nodes[0].peer_stats[2].rank"));
}

// Which rows each expert multiplies is the routing offsets' contents; the work counts the
// declared rows: the output's when gathering, the token's otherwise.
TEST(TestEngineFeatures, MoeRoutingCountsTheDeclaredRows)
{
    GraphT graph;
    addTensor(graph, 1, {1, 32, 64});
    addTensor(graph, 2, {4, 64, 128});
    addTensor(graph, 3, {4, 1, 1});
    addTensor(graph, 4, {1, 64, 1});
    addTensor(graph, 5, {1, 64, 128});
    MoeGroupedMatmulAttributesT moe;
    moe.token_tensor_uid = 1;
    moe.weight_tensor_uid = 2;
    moe.first_token_offset_tensor_uid = 3;
    moe.token_index_tensor_uid = 4;
    moe.output_tensor_uid = 5;
    moe.mode = MoeGroupedMatmulMode::GATHER;
    moe.top_k = 2;
    addNode(graph, moe);
    const auto published = features(graph);
    EXPECT_EQ(published.at("graph.nodes[0].data_dependent"), true);
    EXPECT_EQ(published.at("graph.nodes[0].first_token_offset.dims[0]"), 4);
    EXPECT_EQ(published.at("graph.nodes[0].token_index.numel"), 64);
    EXPECT_EQ(published.at("graph.nodes[0].mode"),
              static_cast<int64_t>(MoeGroupedMatmulMode::GATHER));
    EXPECT_EQ(published.at("graph.nodes[0].top_k"), 2);
    // 2 * 64 gathered rows * K (64) * N (128), not the token's 32 rows.
    EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), 1048576.0);
}

TEST(TestEngineFeatures, VariableLengthAttentionIsDataDependent)
{
    auto dense = features(causalTopLeftGraph());
    EXPECT_EQ(dense.at("graph.nodes[0].data_dependent"), false);
    EXPECT_TRUE(dense.contains("graph.flops"));

    auto sdpa = attention();
    sdpa.seq_len_kv_tensor_uid = 6;
    const auto published = features(attentionGraph(sdpa));
    EXPECT_EQ(published.at("graph.nodes[0].data_dependent"), true);
    EXPECT_EQ(published.at("graph.nodes[0].has_variable_lengths"), true);
    EXPECT_EQ(published.at("graph.nodes[0].seq_len_kv.dims[0]"), 2);
    // The declared extent: the padded shapes' dense count.
    EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), 32256.0);

    // An additive bias is applied to every score whatever its values.
    sdpa = attention();
    sdpa.attn_mask_tensor_uid = 5;
    EXPECT_EQ(features(attentionGraph(sdpa)).at("graph.nodes[0].data_dependent"), false);
}

// Cumulative lengths carry the same information as seq_len_*: a graph carrying either side is
// data dependent and counts its padded shapes, even with padding_mask unset (reachable via JSON,
// flatbuffer or the backend API).
TEST(TestEngineFeatures, CumulativeSequenceLengthsMakeAttentionDataDependent)
{
    using Setter = void (*)(SdpaAttributesT&);
    const std::vector<std::pair<const char*, Setter>> sides = {
        {"cu_seq_len_q", [](SdpaAttributesT& op) { op.cu_seq_len_q_tensor_uid = 6; }},
        {"cu_seq_len_kv", [](SdpaAttributesT& op) { op.cu_seq_len_kv_tensor_uid = 6; }},
    };
    for(const auto& [name, set] : sides)
    {
        SCOPED_TRACE(name);
        auto sdpa = attention();
        set(sdpa);
        const auto published = features(attentionGraph(sdpa));
        EXPECT_EQ(published.at("graph.nodes[0].data_dependent"), true);
        EXPECT_DOUBLE_EQ(published.at("graph.nodes[0].flops").get<double>(), 32256.0);
        EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), 32256.0);
    }
}

// Operands whose contents decide the score count (RFC 0019 section 6.8) leave the work at the
// declared extent, which data_dependent marks; a paged cache declares its keys per sequence.
TEST(TestEngineFeatures, AttentionCountsTheDeclaredExtentWhenOperandContentsDecideIt)
{
    EXPECT_TRUE(features(attentionGraph(attention())).contains("graph.flops"));

    auto ragged = attentionGraph(attention());
    ragged.tensors[0]->ragged_offset_tensor_uid = 5;

    auto pagedAttributes = attention();
    pagedAttributes.page_table_k_tensor_uid = 6;
    pagedAttributes.page_table_v_tensor_uid = 7;
    const auto paged = attentionGraph(pagedAttributes);

    auto blockAttributes = attention();
    blockAttributes.block_mask_tensor_uid = 8;
    const auto blockSparse = attentionGraph(blockAttributes);

    for(const auto& [name, graph] :
        {std::pair<const char*, const GraphT*>{"ragged q", &ragged},
         std::pair<const char*, const GraphT*>{"paged kv", &paged},
         std::pair<const char*, const GraphT*>{"block mask", &blockSparse}})
    {
        SCOPED_TRACE(name);
        const auto published = features(*graph);
        EXPECT_EQ(published.at("graph.nodes[0].data_dependent"), true);
        EXPECT_DOUBLE_EQ(published.at("graph.nodes[0].flops").get<double>(), 32256.0);
        EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), 32256.0);
        EXPECT_DOUBLE_EQ(published.at("graph.flops_by_type.SdpaAttributes").get<double>(), 32256.0);
    }

    // Two pages of 7 per sequence on each side: 14 keys for each of the 3 queries.
    auto wider = attentionGraph(pagedAttributes);
    wider.tensors[5]->dims = {2, 1, 2, 1};
    wider.tensors[6]->dims = {2, 1, 2, 1};
    EXPECT_DOUBLE_EQ(features(wider).at("graph.flops").get<double>(), 2.0 * 2 * 8 * 42 * 48);
    // Only K's table is wider: V holds 7 keys per sequence, so 7 are declared.
    wider.tensors[6]->dims = {2, 1, 1, 1};
    EXPECT_DOUBLE_EQ(features(wider).at("graph.flops").get<double>(), 32256.0);
    // A stated max_seq_len_kv is the declared keys: 5 for each of the 3 queries.
    pagedAttributes.max_seq_len_kv = 5;
    EXPECT_DOUBLE_EQ(features(attentionGraph(pagedAttributes)).at("graph.flops").get<double>(),
                     2.0 * 2 * 8 * 15 * 48);
}

/// Q [2, 8, 3, 16] and O [2, 8, 3, 32] (uids 1 and 4) around K (uid 2) and V (uid 3); a
/// table of K (uid 5) and of V (uid 6) when given, and @p mask (uid 7) when given.
GraphT pagedGraph(const SdpaAttributesT& sdpa,
                  std::vector<int64_t> k,
                  std::vector<int64_t> v,
                  std::vector<int64_t> tableK = {2, 1, 1, 1},
                  std::vector<int64_t> tableV = {2, 1, 1, 1},
                  std::vector<int64_t> mask = {2, 8, 3, 1})
{
    GraphT graph;
    addTensor(graph, 1, {2, 8, 3, 16});
    addTensor(graph, 2, std::move(k));
    addTensor(graph, 3, std::move(v));
    addTensor(graph, 4, {2, 8, 3, 32});
    addTensor(graph, 5, std::move(tableK));
    addTensor(graph, 6, std::move(tableV));
    addTensor(graph, 7, std::move(mask));
    addNode(graph, sdpa, DataType::FLOAT);
    return graph;
}

// The keys of a paged node are inferred as cuDNN and the references (sdpaPagedMaxSeqKv) do:
// either side may be paged alone with its own block size, and the keys never exceed what a
// paged side holds. The work model counts a superset of the layouts the references accept; the
// last two cases are layouts they decline, pinned so that the superset stays a decision.
TEST(TestEngineFeatures, PagedKeysAreInferredAsTheReferencesDo)
{
    const auto paged = [](bool pageK, bool pageV) {
        auto sdpa = attention();
        if(pageK)
        {
            sdpa.page_table_k_tensor_uid = 5;
        }
        if(pageV)
        {
            sdpa.page_table_v_tensor_uid = 6;
        }
        return sdpa;
    };
    // 2 * B (2) * H (8) * pairs * (Dq + Dv) (48), every query seeing every key.
    const auto flopsFor
        = [](int64_t keys) { return 2.0 * 2 * 8 * 3 * static_cast<double>(keys) * 48; };
    struct Case
    {
        const char* name;
        GraphT graph;
        std::optional<double> flops;
    };
    auto bothWithMax = paged(true, true);
    bothWithMax.max_seq_len_kv = 20;
    auto bothWithMask = paged(true, true);
    bothWithMask.attn_mask_tensor_uid = 7;
    auto unreadable = paged(true, true);
    unreadable.page_table_k_tensor_uid = 99;
    auto bottomRight = paged(true, true);
    bottomRight.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    bottomRight.right_bound = 0;
    std::vector<Case> cases;
    // K: 3 pages of 4 (12 keys); V: 5 pages of 2 (10 keys): the smaller.
    cases.push_back(
        {"unequal block sizes",
         pagedGraph(paged(true, true), {6, 2, 4, 16}, {10, 2, 2, 32}, {2, 1, 3, 1}, {2, 1, 5, 1}),
         flopsFor(10)});
    cases.push_back({"V alone, K's 8 keys fit",
                     pagedGraph(paged(false, true), {2, 2, 8, 16}, {4, 2, 4, 32}, {}, {2, 1, 2, 1}),
                     flopsFor(8)});
    cases.push_back({"V alone, K's 9 keys do not fit",
                     pagedGraph(paged(false, true), {2, 2, 9, 16}, {4, 2, 4, 32}, {}, {2, 1, 2, 1}),
                     std::nullopt});
    cases.push_back({"K alone, V's 10 keys",
                     pagedGraph(paged(true, false), {6, 2, 4, 16}, {2, 2, 10, 32}, {2, 1, 3, 1}),
                     flopsFor(10)});
    cases.push_back(
        {"the mask's keys",
         pagedGraph(
             bothWithMask, {6, 2, 4, 16}, {6, 2, 4, 32}, {2, 1, 3, 1}, {2, 1, 3, 1}, {2, 8, 3, 10}),
         flopsFor(10)});
    cases.push_back(
        {"a maximum beyond the cache",
         pagedGraph(bothWithMax, {6, 2, 4, 16}, {6, 2, 4, 32}, {2, 1, 3, 1}, {2, 1, 3, 1}),
         std::nullopt});
    cases.push_back(
        {"an unreadable table",
         pagedGraph(unreadable, {6, 2, 4, 16}, {6, 2, 4, 32}, {2, 1, 3, 1}, {2, 1, 3, 1}),
         std::nullopt});
    // 12 keys, bottom-right causal: rows keep 10, 11 and 12 keys.
    cases.push_back(
        {"bottom-right over the paged keys",
         pagedGraph(bottomRight, {6, 2, 4, 16}, {6, 2, 4, 32}, {2, 1, 3, 1}, {2, 1, 3, 1}),
         2.0 * 2 * 8 * 33 * 48});
    auto unreadableV = paged(true, true);
    unreadableV.page_table_v_tensor_uid = 99;
    cases.push_back(
        {"an unreadable V table",
         pagedGraph(unreadableV, {6, 2, 4, 16}, {6, 2, 4, 32}, {2, 1, 3, 1}, {2, 1, 3, 1}),
         std::nullopt});
    auto zeroMax = paged(true, true);
    zeroMax.max_seq_len_kv = 0;
    cases.push_back({"a maximum of 0",
                     pagedGraph(zeroMax, {6, 2, 4, 16}, {6, 2, 4, 32}, {2, 1, 3, 1}, {2, 1, 3, 1}),
                     std::nullopt});
    // An unpaged side's keys come before the mask's.
    auto vWithMask = paged(false, true);
    vWithMask.attn_mask_tensor_uid = 7;
    cases.push_back(
        {"an unpaged K before the mask",
         pagedGraph(vWithMask, {2, 2, 8, 16}, {4, 2, 4, 32}, {}, {2, 1, 2, 1}, {2, 8, 3, 6}),
         flopsFor(8)});
    auto bothWithRng = paged(true, true);
    bothWithRng.rng_dump_tensor_uid = 7;
    cases.push_back(
        {"rng_dump's keys",
         pagedGraph(
             bothWithRng, {6, 2, 4, 16}, {6, 2, 4, 32}, {2, 1, 3, 1}, {2, 1, 3, 1}, {2, 8, 3, 10}),
         flopsFor(10)});
    // K holds 2 pages of 4: a maximum of 10 fits V's 12 but not K's 8.
    auto beyondK = paged(true, true);
    beyondK.max_seq_len_kv = 10;
    cases.push_back({"a maximum beyond K's cache",
                     pagedGraph(beyondK, {4, 2, 4, 16}, {6, 2, 4, 32}, {2, 1, 2, 1}, {2, 1, 3, 1}),
                     std::nullopt});
    // With a stated maximum, an unpaged side must hold exactly that many keys.
    auto kWithMax = paged(true, false);
    kWithMax.max_seq_len_kv = 12;
    cases.push_back({"a maximum an unpaged V holds",
                     pagedGraph(kWithMax, {6, 2, 4, 16}, {2, 2, 12, 32}, {2, 1, 3, 1}),
                     flopsFor(12)});
    cases.push_back({"a maximum an unpaged V does not hold",
                     pagedGraph(kWithMax, {6, 2, 4, 16}, {2, 2, 10, 32}, {2, 1, 3, 1}),
                     std::nullopt});
    // K's table has 3 pages of 4 where V's 6 keys need ceil(6 / 4) = 2: the references
    // decline a table wider than the keys, the work model counts V's keys.
    cases.push_back({"a table wider than the keys",
                     pagedGraph(paged(true, false), {6, 2, 4, 16}, {2, 2, 6, 32}, {2, 1, 3, 1}),
                     flopsFor(6)});
    // A mask broadcast over the keys (last extent 1) on two paged sides is one key, as cuDNN's
    // infer_s_kv reads it; the references decline its 3-page tables.
    cases.push_back(
        {"a broadcast mask over several pages",
         pagedGraph(bothWithMask, {6, 2, 4, 16}, {6, 2, 4, 32}, {2, 1, 3, 1}, {2, 1, 3, 1}),
         flopsFor(1)});
    for(const auto& [name, graph, flops] : cases)
    {
        SCOPED_TRACE(name);
        const auto published = features(graph);
        if(flops)
        {
            EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), *flops);
        }
        else
        {
            EXPECT_FALSE(published.contains("graph.nodes[0].flops"));
            EXPECT_FALSE(published.contains("graph.flops"));
        }
    }
}

TEST(TestEngineFeatures, RaggedOperandCountsItsDeclaredExtent)
{
    auto graph = matmulBroadcastGraph();
    EXPECT_EQ(features(graph).at("graph.nodes[0].data_dependent"), false);
    addTensor(graph, 4, {6, 1, 1});
    graph.tensors[0]->ragged_offset_tensor_uid = 4;
    const auto published = features(graph);
    EXPECT_EQ(published.at("graph.nodes[0].data_dependent"), true);
    // The padded dims, as declared: the offsets' contents are not read.
    EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), 840.0);
    // a (12), b (140), c (105) and the offsets (6), in half precision.
    EXPECT_DOUBLE_EQ(published.at("graph.logical_bytes").get<double>(), 263.0 * 2.0);
}

// A one-node graph whose tensors take uids 1, 2, ... in @p shapes order.
template <typename TAttributes>
GraphT single(std::vector<std::vector<int64_t>> shapes, TAttributes attributes)
{
    GraphT graph;
    int64_t uid = 0;
    for(auto& dims : shapes)
    {
        addTensor(graph, ++uid, std::move(dims));
    }
    addNode(graph, std::move(attributes));
    return graph;
}

/// The causal top-left backward of RectangularCausalAttention's shapes, with operands 10-13
/// ([2, 1, 1, 1]) for the optional ones a variation sets.
GraphT backwardGraph(void (*vary)(SdpaBackwardAttributesT&))
{
    SdpaBackwardAttributesT sdpa;
    sdpa.q_tensor_uid = 1;
    sdpa.k_tensor_uid = 2;
    sdpa.v_tensor_uid = 3;
    sdpa.o_tensor_uid = 4;
    sdpa.do_tensor_uid = 5;
    sdpa.stats_tensor_uid = 6;
    sdpa.dq_tensor_uid = 7;
    sdpa.dk_tensor_uid = 8;
    sdpa.dv_tensor_uid = 9;
    sdpa.causal_mask = true;
    vary(sdpa);
    return single({{2, 8, 3, 16},
                   {2, 2, 7, 16},
                   {2, 2, 7, 32},
                   {2, 8, 3, 32},
                   {2, 8, 3, 32},
                   {2, 8, 3, 1},
                   {2, 8, 3, 16},
                   {2, 2, 7, 16},
                   {2, 2, 7, 32},
                   {2, 1, 1, 1},
                   {2, 1, 1, 1},
                   {2, 1, 1, 1},
                   {2, 1, 1, 1}},
                  sdpa);
}

// The backward counts 2.5x the forward over the same band; a bias gradient and inverse-scaled
// dropout are elementwise, and valid lengths leave the declared extent.
TEST(TestEngineFeatures, AttentionBackwardCountsItsBandAndNotItsElementwiseOperands)
{
    using Variation = void (*)(SdpaBackwardAttributesT&);
    const std::vector<std::tuple<const char*, Variation, double, bool>> variations = {
        {"causal", [](SdpaBackwardAttributesT&) {}, 23040.0, false},
        {"bias gradient and dropout",
         [](SdpaBackwardAttributesT& op) {
             op.dbias_tensor_uid = 10;
             op.dropout_scale_inv_tensor_uid = 11;
         },
         23040.0,
         false},
        // Rows 0-2 keep keys [0, 0], [0, 1], [1, 2]: 5 pairs, 2.5 x 7680.
        {"causal window", [](SdpaBackwardAttributesT& op) { op.left_bound = 2; }, 19200.0, false},
        {"valid lengths",
         [](SdpaBackwardAttributesT& op) {
             op.seq_len_q_tensor_uid = 12;
             op.seq_len_kv_tensor_uid = 13;
         },
         23040.0,
         true},
    };
    for(const auto& [name, vary, flops, dataDependent] : variations)
    {
        SCOPED_TRACE(name);
        const auto published = features(backwardGraph(vary));
        EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), flops);
        EXPECT_EQ(published.at("graph.nodes[0].data_dependent"), dataDependent);
    }
}

/// The key a reference keeps for query row @p row, as CpuFpReferenceSdpa masks (SEM-001).
bool referenceKeeps(
    const DiagonalBandParams& band, int64_t row, int64_t key, int64_t sq, int64_t sk)
{
    const int64_t offset = band.topLeftAlignment ? 0 : sk - sq;
    return (band.rightBound < 0 || key < row + 1 + offset + band.rightBound)
           && (band.leftBound < 0 || key > row + offset - band.leftBound);
}

const SdpaAttributes& packedAttention(flatbuffers::FlatBufferBuilder& buffer, const GraphT& graph)
{
    buffer.Finish(Graph::Pack(buffer, &graph));
    return *GetGraph(buffer.GetBufferPointer())->nodes()->Get(0)->attributes_as_SdpaAttributes();
}

// The work model's band composition and paged keys are a second copy of the references'
// (extractDiagonalBandParams, sdpaPagedMaxSeqKv): every graph a reference accepts is counted
// over the band and keys it uses, and every band it rejects is uncounted.
TEST(TestEngineFeatures, BandAndPagedKeysAgreeWithTheReferences)
{
    // UNSET leaves the bound absent.
    constexpr int64_t UNSET = std::numeric_limits<int64_t>::min();
    const std::vector<int64_t> lefts = {UNSET, -2, -1, 0, 1, 2, 3};
    const std::vector<int64_t> rights = {UNSET, -2, -1, 0, 1, 2};
    for(int64_t sq = 1; sq <= 4; ++sq)
    {
        for(int64_t sk = 1; sk <= 4; ++sk)
        {
            for(const int64_t left : lefts)
            {
                for(const int64_t right : rights)
                {
                    for(int flags = 0; flags < 4; ++flags)
                    {
                        for(const bool bottomRight : {false, true})
                        {
                            auto sdpa = attention();
                            if(left != UNSET)
                            {
                                sdpa.left_bound = left;
                            }
                            if(right != UNSET)
                            {
                                sdpa.right_bound = right;
                            }
                            sdpa.causal_mask = (flags & 1) != 0;
                            sdpa.causal_mask_bottom_right = (flags & 2) != 0;
                            sdpa.diagonal_alignment = bottomRight ? DiagonalAlignment::BOTTOM_RIGHT
                                                                  : DiagonalAlignment::TOP_LEFT;
                            const auto graph = single(
                                {{1, 1, sq, 4}, {1, 1, sk, 4}, {1, 1, sk, 4}, {1, 1, sq, 4}}, sdpa);
                            flatbuffers::FlatBufferBuilder buffer;
                            const auto& packed = packedAttention(buffer, graph);
                            SCOPED_TRACE(testing::Message()
                                         << "sq " << sq << " sk " << sk << " flags " << flags
                                         << " br " << bottomRight);
                            const auto published = features(buffer);
                            std::optional<DiagonalBandParams> band;
                            try
                            {
                                band = extractDiagonalBandParams(packed, "test");
                            }
                            catch(const std::invalid_argument&)
                            {
                                EXPECT_FALSE(published.contains("graph.flops"));
                                continue;
                            }
                            int64_t pairs = 0;
                            for(int64_t row = 0; row < sq; ++row)
                            {
                                for(int64_t key = 0; key < sk; ++key)
                                {
                                    pairs += referenceKeeps(*band, row, key, sq, sk) ? 1 : 0;
                                }
                            }
                            if(pairs == 0)
                            {
                                EXPECT_FALSE(published.contains("graph.flops"));
                            }
                            else
                            {
                                EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(),
                                                 16.0 * static_cast<double>(pairs));
                            }
                        }
                    }
                }
            }
        }
    }

    // Paged keys: wherever the reference accepts the layout, its keys are the counted ones.
    int compared = 0;
    for(const auto& [pageK, pageV] :
        {std::pair{true, false}, std::pair{false, true}, std::pair{true, true}})
    {
        for(const int64_t blockK : {2, 3})
        {
            for(const int64_t blockV : {2, 3})
            {
                for(const int64_t pages : {2, 3})
                {
                    for(const int32_t maximum : {0, 4, 5, 6})
                    {
                        for(const int64_t mask : {int64_t{0}, int64_t{5}})
                        {
                            for(const int64_t dense : {4, 5, 6})
                            {
                                auto sdpa = attention();
                                sdpa.seq_len_q_tensor_uid = 8;
                                sdpa.seq_len_kv_tensor_uid = 9;
                                if(pageK)
                                {
                                    sdpa.page_table_k_tensor_uid = 5;
                                }
                                if(pageV)
                                {
                                    sdpa.page_table_v_tensor_uid = 6;
                                }
                                if(maximum > 0)
                                {
                                    sdpa.max_seq_len_kv = maximum;
                                }
                                if(mask > 0)
                                {
                                    sdpa.attn_mask_tensor_uid = 7;
                                }
                                auto graph = single({{2, 1, 3, 4},
                                                     pageK ? std::vector<int64_t>{8, 1, blockK, 4}
                                                           : std::vector<int64_t>{2, 1, dense, 4},
                                                     pageV ? std::vector<int64_t>{8, 1, blockV, 4}
                                                           : std::vector<int64_t>{2, 1, dense, 4},
                                                     {2, 1, 3, 4},
                                                     {2, 1, pages, 1},
                                                     {2, 1, pages, 1},
                                                     {2, 1, 3, mask > 0 ? mask : 1},
                                                     {2, 1, 1, 1},
                                                     {2, 1, 1, 1}},
                                                    sdpa);
                                for(const size_t index : {4U, 5U, 7U, 8U})
                                {
                                    graph.tensors[index]->data_type = DataType::INT32;
                                }
                                flatbuffers::FlatBufferBuilder buffer;
                                const auto& packed = packedAttention(buffer, graph);
                                std::unordered_map<int64_t, const TensorAttributes*> tensors;
                                for(const auto* value :
                                    *GetGraph(buffer.GetBufferPointer())->tensors())
                                {
                                    tensors.emplace(value->uid(), value);
                                }
                                const auto keys = sdpaPagedMaxSeqKv(packed, tensors);
                                if(keys <= 0)
                                {
                                    continue;
                                }
                                SCOPED_TRACE(testing::Message()
                                             << "K " << pageK << " V " << pageV << " blocks "
                                             << blockK << "/" << blockV << " pages " << pages
                                             << " max " << maximum << " mask " << mask << " dense "
                                             << dense);
                                ++compared;
                                EXPECT_DOUBLE_EQ(features(buffer).at("graph.flops").get<double>(),
                                                 2.0 * 2 * 3 * static_cast<double>(keys) * 8);
                            }
                        }
                    }
                }
            }
        }
    }
    EXPECT_GT(compared, 20);
}

TEST(TestEngineFeatures, WorkOfLaterTypesCountsTheDeclaredExtent)
{
    PointwiseAttributesT relu;
    relu.operation = PointwiseMode::RELU_FWD;
    relu.in_0_tensor_uid = 1;
    relu.out_0_tensor_uid = 2;
    auto graph = single({{6, 4}, {6, 4}, {7}}, relu);
    EXPECT_DOUBLE_EQ(features(graph).at("graph.flops").get<double>(), 24.0);
    graph.tensors[0]->ragged_offset_tensor_uid = 3;
    const auto published = features(graph);
    EXPECT_DOUBLE_EQ(published.at("graph.nodes[0].flops").get<double>(), 24.0);
    EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), 24.0);
    EXPECT_DOUBLE_EQ(published.at("graph.flops_by_type.PointwiseAttributes").get<double>(), 24.0);
}

TEST(TestEngineFeatures, WorkModelMatchesEveryGoldenCount)
{
    // Shipped models read these values; changing any requires bumping
    // FEATURE_SEMANTICS_REVISION. Publishing a value where none was (a band, dropout, a paged
    // cache) does not (RFC 0019 section 6.9). Bands outside hipDNN's domain, which the work
    // model once counted, are no longer counted without a bump: no engine serves them, and
    // their training rows are re-collected (RFC 0019 section 6.8).
    EXPECT_EQ(hipdnn_plugin_sdk::heuristics::FEATURE_SEMANTICS_REVISION, 1);
    struct Case
    {
        const char* name;
        GraphT (*make)();
        double flops;
    };
    const std::vector<Case> cases = {
        {"matmul broadcast", matmulBroadcastGraph, 840.0},
        {"convolution", convFwdGraph, 86400.0},
        {"grouped convolution", groupedConvFwdGraph, 13824.0},
        {"dense attention", [] { return attentionGraph(attention()); }, 32256.0},
        {"causal top-left", causalTopLeftGraph, 9216.0},
        {"causal bottom-right", causalBottomRightGraph, 27648.0},
        {"causal bottom-right flag",
         [] {
             auto sdpa = attention();
             sdpa.causal_mask_bottom_right = true;
             return attentionGraph(sdpa);
         },
         27648.0},
        {"unbounded right",
         [] {
             auto sdpa = attention();
             sdpa.right_bound = -1;
             return attentionGraph(sdpa);
         },
         32256.0},
        {"unbounded left",
         [] {
             auto sdpa = attention();
             sdpa.left_bound = -1;
             return attentionGraph(sdpa);
         },
         32256.0},
        {"zero dropout",
         [] {
             auto sdpa = attention();
             sdpa.dropout_probability = 0.0F;
             return attentionGraph(sdpa);
         },
         32256.0},
        // Rows 0-2 keep keys [0, 2], [0, 3], [0, 4]: 12 pairs.
        {"diagonal band", boundedAttentionGraph, 18432.0},
        // Dropout is elementwise on the scores, as softmax is.
        {"dropout", dropoutAttentionGraph, 32256.0},
        // One page of K's 7 keys per sequence, all 3 queries attending to each.
        {"paged variable length", pagedVariableLengthAttentionGraph, 32256.0},
        // Rows 4-6 of 7 reach keys 0, 0-1, 0-2: 6 pairs.
        {"causal bottom-right with more queries than keys",
         [] {
             auto sdpa = attention();
             sdpa.causal_mask_bottom_right = true;
             return single({{2, 8, 7, 16}, {2, 2, 3, 16}, {2, 2, 3, 32}, {2, 8, 7, 32}}, sdpa);
         },
         9216.0},
        // Rows 0-6 of 7 reach keys up to their own, never past key 2: 1 + 2 + 5 x 3 pairs.
        {"causal top-left with more queries than keys",
         [] {
             auto sdpa = attention();
             sdpa.causal_mask = true;
             return single({{2, 8, 7, 16}, {2, 2, 3, 16}, {2, 2, 3, 32}, {2, 8, 7, 32}}, sdpa);
         },
         27648.0},
        // Diagonals 4-6, two keys up to each and every key past it: [3, 6], [4, 6], [5, 6].
        {"bottom-right window",
         [] {
             auto sdpa = attention();
             sdpa.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
             sdpa.left_bound = 2;
             return attentionGraph(sdpa);
         },
         13824.0},
        // Diagonals 4-6: rows keep keys [2, 5], [3, 6], [4, 6].
        {"bottom-right band",
         [] {
             auto sdpa = attention();
             sdpa.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
             sdpa.left_bound = 3;
             sdpa.right_bound = 1;
             return attentionGraph(sdpa);
         },
         16896.0},
        // The deprecated flag composes with a stated right bound (SEM-002): rows 0-2 keep keys
        // [0, 2], [0, 3], [0, 4].
        {"causal flag with a right bound",
         [] {
             auto sdpa = attention();
             sdpa.causal_mask = true;
             sdpa.right_bound = 2;
             return attentionGraph(sdpa);
         },
         18432.0},
        // Diagonals 4-6, one key past each: rows keep 6, 7 and 7 keys.
        {"bottom-right flag with a right bound",
         [] {
             auto sdpa = attention();
             sdpa.causal_mask_bottom_right = true;
             sdpa.right_bound = 1;
             return attentionGraph(sdpa);
         },
         30720.0},
        // causal_mask keeps the stated alignment.
        {"causal flag with bottom-right alignment",
         [] {
             auto sdpa = attention();
             sdpa.causal_mask = true;
             sdpa.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
             return attentionGraph(sdpa);
         },
         27648.0},
    };
    for(const auto& [name, make, flops] : cases)
    {
        SCOPED_TRACE(name);
        const auto published = features(make());
        EXPECT_DOUBLE_EQ(published.at("graph.nodes[0].flops").get<double>(), flops);
        EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), flops);
    }

    // Each SDPA operand or flag on its own: elementwise work and operand contents leave the
    // dense count, a bound narrows the band.
    using Variation = void (*)(SdpaAttributesT&);
    const std::vector<std::tuple<const char*, Variation, double>> variations = {
        {"seq_len_q", [](SdpaAttributesT& op) { op.seq_len_q_tensor_uid = 5; }, 32256.0},
        {"seq_len_kv", [](SdpaAttributesT& op) { op.seq_len_kv_tensor_uid = 5; }, 32256.0},
        {"cu_seq_len_q", [](SdpaAttributesT& op) { op.cu_seq_len_q_tensor_uid = 5; }, 32256.0},
        {"cu_seq_len_kv", [](SdpaAttributesT& op) { op.cu_seq_len_kv_tensor_uid = 5; }, 32256.0},
        {"page_table_k", [](SdpaAttributesT& op) { op.page_table_k_tensor_uid = 5; }, 32256.0},
        {"page_table_v", [](SdpaAttributesT& op) { op.page_table_v_tensor_uid = 5; }, 32256.0},
        {"block_mask", [](SdpaAttributesT& op) { op.block_mask_tensor_uid = 5; }, 32256.0},
        {"sink_token", [](SdpaAttributesT& op) { op.sink_token_tensor_uid = 5; }, 32256.0},
        {"attn_mask", [](SdpaAttributesT& op) { op.attn_mask_tensor_uid = 5; }, 32256.0},
        {"padding_mask", [](SdpaAttributesT& op) { op.padding_mask = true; }, 32256.0},
        {"alibi_mask", [](SdpaAttributesT& op) { op.alibi_mask = true; }, 32256.0},
        {"descale_q", [](SdpaAttributesT& op) { op.descale_q_tensor_uid = 5; }, 32256.0},
        {"descale_k", [](SdpaAttributesT& op) { op.descale_k_tensor_uid = 5; }, 32256.0},
        {"descale_v", [](SdpaAttributesT& op) { op.descale_v_tensor_uid = 5; }, 32256.0},
        {"descale_s", [](SdpaAttributesT& op) { op.descale_s_tensor_uid = 5; }, 32256.0},
        {"scale_s", [](SdpaAttributesT& op) { op.scale_s_tensor_uid = 5; }, 32256.0},
        {"scale_o", [](SdpaAttributesT& op) { op.scale_o_tensor_uid = 5; }, 32256.0},
        {"dropout_mask", [](SdpaAttributesT& op) { op.dropout_mask_tensor_uid = 5; }, 32256.0},
        {"dropout_scale", [](SdpaAttributesT& op) { op.dropout_scale_tensor_uid = 5; }, 32256.0},
        {"dropout_probability",
         [](SdpaAttributesT& op) { op.dropout_probability = 0.1F; },
         32256.0},
        // Row i keeps keys [i, 6]: 7 + 6 + 5 pairs.
        {"left_bound", [](SdpaAttributesT& op) { op.left_bound = 1; }, 27648.0},
        // Row i keeps keys [0, i + 1]: 2 + 3 + 4 pairs.
        {"right_bound", [](SdpaAttributesT& op) { op.right_bound = 1; }, 13824.0},
        // A flag with an unbounded right bound is the flag alone (SEM-002).
        {"causal flag with an unbounded right",
         [](SdpaAttributesT& op) {
             op.causal_mask = true;
             op.right_bound = -1;
         },
         9216.0},
    };
    for(const auto& [name, vary, flops] : variations)
    {
        SCOPED_TRACE(name);
        auto sdpa = attention();
        vary(sdpa);
        const auto published = features(attentionGraph(sdpa));
        EXPECT_DOUBLE_EQ(published.at("graph.nodes[0].flops").get<double>(), flops);
        EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), flops);
        EXPECT_DOUBLE_EQ(published.at("graph.flops_by_type.SdpaAttributes").get<double>(), flops);
    }
}

// Every NodeAttributes member is either counted, with an entry here whose representative
// graph publishes its declared work, or listed in NO_COUNTING_CONVENTION. A new member with
// neither fails, so a new op cannot silently leave every graph holding it without FLOPs.
TEST(TestEngineFeatures, EveryNodeTypeHasDeclaredWorkOrNoCountingConvention)
{
    struct Case
    {
        GraphT (*make)();
        double flops;
    };
    const std::vector<Case> cases = {
        {matmulBroadcastGraph, 840.0},
        {convFwdGraph, 86400.0},
        {[] { return attentionGraph(attention()); }, 32256.0},
        // 2 * dy.numel (384) * w.numel (864) / K (12).
        {[] {
             ConvolutionBwdAttributesT conv;
             conv.dy_tensor_uid = 1;
             conv.w_tensor_uid = 2;
             conv.dx_tensor_uid = 3;
             return single({{2, 12, 4, 4}, {12, 8, 3, 3}, {2, 8, 7, 7}}, conv);
         },
         55296.0},
        // Two groups: 2 * dy.numel (600) * dw.numel (432) / K (12).
        {[] {
             ConvolutionWrwAttributesT conv;
             conv.x_tensor_uid = 1;
             conv.dy_tensor_uid = 2;
             conv.dw_tensor_uid = 3;
             return single({{2, 8, 7, 7}, {2, 12, 5, 5}, {12, 4, 3, 3}}, conv);
         },
         43200.0},
        // 2.5 * the causal top-left forward's 9216.
        {[] {
             SdpaBackwardAttributesT sdpa;
             sdpa.q_tensor_uid = 1;
             sdpa.k_tensor_uid = 2;
             sdpa.v_tensor_uid = 3;
             sdpa.o_tensor_uid = 4;
             sdpa.do_tensor_uid = 5;
             sdpa.stats_tensor_uid = 6;
             sdpa.dq_tensor_uid = 7;
             sdpa.dk_tensor_uid = 8;
             sdpa.dv_tensor_uid = 9;
             sdpa.causal_mask = true;
             return single({{2, 8, 3, 16},
                            {2, 2, 7, 16},
                            {2, 2, 7, 32},
                            {2, 8, 3, 32},
                            {2, 8, 3, 32},
                            {2, 8, 3, 1},
                            {2, 8, 3, 16},
                            {2, 2, 7, 16},
                            {2, 2, 7, 32}},
                           sdpa);
         },
         23040.0},
        // One per output element, boolean or not.
        {[] {
             PointwiseAttributesT compare;
             compare.operation = PointwiseMode::CMP_GT;
             compare.in_0_tensor_uid = 1;
             compare.in_1_tensor_uid = 2;
             compare.out_0_tensor_uid = 3;
             auto graph = single({{2, 3}, {2, 3}, {2, 3}}, compare);
             graph.tensors[2]->data_type = DataType::BOOLEAN;
             return graph;
         },
         6.0},
        // One per input element.
        {[] {
             ReductionAttributesT reduction;
             reduction.mode = ReductionMode::ADD;
             reduction.in_tensor_uid = 1;
             reduction.out_tensor_uid = 2;
             return single({{4, 16}, {4, 1}}, reduction);
         },
         64.0},
        // 2 * x.numel (72).
        {[] {
             BatchnormInferenceAttributesT norm;
             norm.x_tensor_uid = 1;
             norm.mean_tensor_uid = 2;
             norm.inv_variance_tensor_uid = 3;
             norm.scale_tensor_uid = 4;
             norm.bias_tensor_uid = 5;
             norm.y_tensor_uid = 6;
             return single({{2, 4, 3, 3},
                            {1, 4, 1, 1},
                            {1, 4, 1, 1},
                            {1, 4, 1, 1},
                            {1, 4, 1, 1},
                            {2, 4, 3, 3}},
                           norm);
         },
         144.0},
        {[] {
             BatchnormInferenceAttributesVarianceExtT norm;
             norm.x_tensor_uid = 1;
             norm.mean_tensor_uid = 2;
             norm.variance_tensor_uid = 3;
             norm.scale_tensor_uid = 4;
             norm.bias_tensor_uid = 5;
             norm.y_tensor_uid = 6;
             norm.epsilon_tensor_uid = 7;
             return single({{2, 4, 3, 3},
                            {1, 4, 1, 1},
                            {1, 4, 1, 1},
                            {1, 4, 1, 1},
                            {1, 4, 1, 1},
                            {2, 4, 3, 3},
                            {1}},
                           norm);
         },
         144.0},
        // 5 * x.numel (72).
        {[] {
             BatchnormAttributesT norm;
             norm.x_tensor_uid = 1;
             norm.scale_tensor_uid = 2;
             norm.bias_tensor_uid = 3;
             norm.epsilon_tensor_uid = 4;
             norm.y_tensor_uid = 5;
             return single({{2, 4, 3, 3}, {1, 4, 1, 1}, {1, 4, 1, 1}, {1}, {2, 4, 3, 3}}, norm);
         },
         360.0},
        // 8 * x.numel (72).
        {[] {
             BatchnormBackwardAttributesT norm;
             norm.dy_tensor_uid = 1;
             norm.x_tensor_uid = 2;
             norm.scale_tensor_uid = 3;
             norm.dx_tensor_uid = 4;
             norm.dscale_tensor_uid = 5;
             norm.dbias_tensor_uid = 6;
             return single({{2, 4, 3, 3},
                            {2, 4, 3, 3},
                            {1, 4, 1, 1},
                            {2, 4, 3, 3},
                            {1, 4, 1, 1},
                            {1, 4, 1, 1}},
                           norm);
         },
         576.0},
        // 5 * x.numel (512).
        {[] {
             LayernormAttributesT norm;
             norm.x_tensor_uid = 1;
             norm.scale_tensor_uid = 2;
             norm.bias_tensor_uid = 3;
             norm.epsilon_tensor_uid = 4;
             norm.y_tensor_uid = 5;
             norm.normalized_dim_count = 1;
             return single({{8, 64}, {1, 64}, {1, 64}, {1}, {8, 64}}, norm);
         },
         2560.0},
        // 8 * x.numel (512).
        {[] {
             LayernormBackwardAttributesT norm;
             norm.dy_tensor_uid = 1;
             norm.x_tensor_uid = 2;
             norm.scale_tensor_uid = 3;
             norm.dx_tensor_uid = 4;
             norm.dscale_tensor_uid = 5;
             norm.dbias_tensor_uid = 6;
             norm.normalized_dim_count = 1;
             return single({{8, 64}, {8, 64}, {1, 64}, {8, 64}, {1, 64}, {1, 64}}, norm);
         },
         4096.0},
        // 3 * x.numel (512).
        {[] {
             RMSNormAttributesT norm;
             norm.x_tensor_uid = 1;
             norm.scale_tensor_uid = 2;
             norm.epsilon_tensor_uid = 3;
             norm.y_tensor_uid = 4;
             return single({{8, 64}, {1, 64}, {1}, {8, 64}}, norm);
         },
         1536.0},
        // 6 * x.numel (512).
        {[] {
             RMSNormBackwardAttributesT norm;
             norm.dy_tensor_uid = 1;
             norm.x_tensor_uid = 2;
             norm.scale_tensor_uid = 3;
             norm.inv_rms_tensor_uid = 4;
             norm.dx_tensor_uid = 5;
             norm.dscale_tensor_uid = 6;
             return single({{8, 64}, {8, 64}, {1, 64}, {8, 1}, {8, 64}, {1, 64}}, norm);
         },
         3072.0},
        // y.numel (256) * the 3x2 window.
        {[] {
             ResampleFwdAttributesT pool;
             pool.x_tensor_uid = 1;
             pool.y_tensor_uid = 2;
             pool.window = {3, 2};
             pool.stride = {2, 2};
             pool.resample_mode = ResampleMode::MAXPOOL;
             return single({{2, 8, 8, 8}, {2, 8, 4, 4}}, pool);
         },
         1536.0},
        // dy.numel (256) * the 2x2 window.
        {[] {
             ResampleBwdAttributesT pool;
             pool.dy_tensor_uid = 1;
             pool.dx_tensor_uid = 2;
             pool.window = {2, 2};
             pool.stride = {2, 2};
             pool.resample_mode = ResampleMode::AVGPOOL_INCLUDE_PADDING;
             return single({{2, 8, 4, 4}, {2, 8, 8, 8}}, pool);
         },
         1024.0},
        // 2 * x.numel (128).
        {[] {
             BlockScaleQuantizeAttributesT quantize;
             quantize.x_tensor_uid = 1;
             quantize.y_tensor_uid = 2;
             quantize.scale_tensor_uid = 3;
             quantize.block_size = 32;
             return single({{4, 32}, {4, 32}, {4, 1}}, quantize);
         },
         256.0},
        {[] {
             BlockScaleDequantizeAttributesT dequantize;
             dequantize.x_tensor_uid = 1;
             dequantize.scale_tensor_uid = 2;
             dequantize.y_tensor_uid = 3;
             dequantize.block_size = {1, 32};
             auto graph = single({{4, 32}, {4, 1}, {4, 32}}, dequantize);
             graph.tensors[0]->data_type = DataType::FP8_E4M3;
             return graph;
         },
         256.0},
        // The token's 32 rows (not gathering): 2 * 32 * K (64) * N (128).
        {[] {
             MoeGroupedMatmulAttributesT moe;
             moe.token_tensor_uid = 1;
             moe.weight_tensor_uid = 2;
             moe.first_token_offset_tensor_uid = 3;
             moe.output_tensor_uid = 4;
             return single({{1, 32, 64}, {4, 64, 128}, {4}, {1, 32, 128}}, moe);
         },
         524288.0},
        // The weight gradient over the declared 32 rows: 2 * 32 * K (64) * N (128).
        {[] {
             MoeGroupedMatmulBwdAttributesT moe;
             moe.doutput_tensor_uid = 1;
             moe.token_tensor_uid = 2;
             moe.first_token_offset_tensor_uid = 3;
             moe.dweight_tensor_uid = 4;
             return single({{1, 32, 128}, {1, 32, 64}, {4}, {4, 64, 128}}, moe);
         },
         524288.0},
        // 6 * T (16) * HO (4) * V (6) * K (8): S k, the rank-one update and S q.
        {[] {
             GdnAttributesT gdn;
             gdn.q_tensor_uid = 1;
             gdn.k_tensor_uid = 2;
             gdn.v_tensor_uid = 3;
             gdn.g_tensor_uid = 4;
             gdn.beta_tensor_uid = 5;
             gdn.cu_seqlens_tensor_uid = 6;
             gdn.o_tensor_uid = 7;
             return single({{16, 2, 8}, {16, 2, 8}, {16, 4, 6}, {16, 4}, {16, 4}, {3}, {16, 4, 6}},
                           gdn);
         },
         18432.0},
        {[] {
             KdaAttributesT kda;
             kda.q_tensor_uid = 1;
             kda.k_tensor_uid = 2;
             kda.v_tensor_uid = 3;
             kda.g_tensor_uid = 4;
             kda.beta_tensor_uid = 5;
             kda.cu_seqlens_tensor_uid = 6;
             kda.o_tensor_uid = 7;
             return single(
                 {{16, 2, 8}, {16, 2, 8}, {16, 4, 6}, {16, 4, 8}, {16, 4}, {3}, {16, 4, 6}}, kda);
         },
         18432.0},
    };
    std::set<NodeAttributes> counted;
    for(const auto& [make, flops] : cases)
    {
        const auto graph = make();
        counted.insert(graph.nodes[0]->attributes.type);
        const std::string type = EnumNameNodeAttributes(graph.nodes[0]->attributes.type);
        SCOPED_TRACE(type);
        const auto published = features(graph);
        EXPECT_DOUBLE_EQ(published.at("graph.nodes[0].flops").get<double>(), flops);
        EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), flops);
        EXPECT_DOUBLE_EQ(published.at("graph.flops_by_type." + type).get<double>(), flops);
    }
    for(const auto type : EnumValuesNodeAttributes())
    {
        SCOPED_TRACE(EnumNameNodeAttributes(type));
        const bool listed = !hipdnn_plugin_sdk::heuristics::detail::hasCountingConvention(type);
        EXPECT_NE(counted.count(type) == 1, listed)
            << "needs a nodeFlops overload, a logicalFlops case and an entry above, or an "
               "entry in NO_COUNTING_CONVENTION, but not both";
    }
}

TEST(TestEngineFeatures, UnknownWorkIsAbsentNeverZero)
{
    const auto layernorm = [](const std::vector<int64_t>& dims) {
        LayernormAttributesT norm;
        norm.x_tensor_uid = 1;
        norm.y_tensor_uid = 2;
        return single({dims, dims}, norm);
    };
    ResampleFwdAttributesT windowless;
    windowless.x_tensor_uid = 1;
    windowless.y_tensor_uid = 2;
    MoeGroupedMatmulAttributesT moe;
    moe.token_tensor_uid = 1;
    moe.weight_tensor_uid = 2;
    moe.first_token_offset_tensor_uid = 3;
    moe.output_tensor_uid = 4;
    MoeGroupedMatmulBwdAttributesT moeBackward;
    moeBackward.doutput_tensor_uid = 1;
    moeBackward.token_tensor_uid = 2;
    moeBackward.first_token_offset_tensor_uid = 3;
    moeBackward.dweight_tensor_uid = 4;
    GdnAttributesT gdn;
    gdn.q_tensor_uid = 1;
    gdn.o_tensor_uid = 2;
    // Bands outside hipDNN's domain: the frontend rejects left_bound 0 and a bound below -1
    // (SEM-005), the references both deprecated flags together.
    const auto band = [](void (*vary)(SdpaAttributesT&)) {
        auto sdpa = attention();
        vary(sdpa);
        return attentionGraph(sdpa);
    };
    std::vector<std::pair<const char*, GraphT>> cases;
    cases.emplace_back("empty extent", layernorm({0, 64}));
    cases.emplace_back("negative extent", layernorm({-1, 64}));
    cases.emplace_back("no window", single({{2, 8, 8, 8}, {2, 8, 4, 4}}, windowless));
    cases.emplace_back("MoE token K against weight K",
                       single({{1, 32, 60}, {4, 64, 128}, {4}, {1, 32, 128}}, moe));
    cases.emplace_back("MoE backward rows",
                       single({{1, 31, 128}, {1, 32, 64}, {4}, {4, 64, 128}}, moeBackward));
    cases.emplace_back("GDN output tokens", single({{16, 2, 8}, {15, 4, 6}}, gdn));
    cases.emplace_back("left_bound 0", band([](SdpaAttributesT& op) { op.left_bound = 0; }));
    cases.emplace_back("left_bound -2", band([](SdpaAttributesT& op) { op.left_bound = -2; }));
    cases.emplace_back("right_bound -2", band([](SdpaAttributesT& op) { op.right_bound = -2; }));
    cases.emplace_back("both causal flags", band([](SdpaAttributesT& op) {
                           op.causal_mask = true;
                           op.causal_mask_bottom_right = true;
                       }));
    auto shortV = attentionGraph(attention());
    shortV.tensors[2]->dims = {2, 2, 6, 32};
    cases.emplace_back("an unpaged V shorter than K", std::move(shortV));
    auto otherBatch = attentionGraph(attention());
    otherBatch.tensors[2]->dims = {3, 2, 7, 32};
    cases.emplace_back("an unpaged V of another batch", std::move(otherBatch));
    cases.emplace_back("MoE output N against weight N",
                       single({{1, 32, 64}, {4, 64, 128}, {4}, {1, 32, 100}}, moe));
    for(const auto& [name, graph] : cases)
    {
        SCOPED_TRACE(name);
        const auto published = features(graph);
        const std::string type = EnumNameNodeAttributes(graph.nodes[0]->attributes.type);
        EXPECT_FALSE(published.contains("graph.nodes[0].flops"));
        EXPECT_FALSE(published.contains("graph.flops"));
        EXPECT_FALSE(published.contains("graph.flops_by_type." + type));
        // A type the graph does not contain has no work: 0, not unknown.
        EXPECT_DOUBLE_EQ(published.at("graph.flops_by_type.MatmulAttributes").get<double>(), 0.0);
    }
}

TEST(TestEngineFeatures, FusedGraphWorkIsTheSumOfItsNodes)
{
    const auto published = features(convBiasReluGraph());
    // The bias add and the ReLU: one operation per output element (600) each.
    EXPECT_DOUBLE_EQ(published.at("graph.nodes[1].flops").get<double>(), 600.0);
    EXPECT_DOUBLE_EQ(published.at("graph.nodes[2].flops").get<double>(), 600.0);
    EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), 86400.0 + 600.0 + 600.0);
    EXPECT_DOUBLE_EQ(published.at("graph.flops_by_type.ConvolutionFwdAttributes").get<double>(),
                     86400.0);
    EXPECT_DOUBLE_EQ(published.at("graph.flops_by_type.PointwiseAttributes").get<double>(), 1200.0);
    // Only x (784), w (864), bias (12) and the output (600) are external, in half precision;
    // the two virtual intermediates never leave the fused kernel.
    const double bytes = (784.0 + 864.0 + 12.0 + 600.0) * 2.0;
    EXPECT_DOUBLE_EQ(published.at("graph.logical_bytes").get<double>(), bytes);
    EXPECT_DOUBLE_EQ(published.at("graph.arithmetic_intensity").get<double>(), 87600.0 / bytes);
}

TEST(TestEngineFeatures, LogicalBytesCountSubByteElementsFractionally)
{
    PointwiseAttributesT relu;
    relu.operation = PointwiseMode::RELU_FWD;
    relu.in_0_tensor_uid = 1;
    relu.out_0_tensor_uid = 2;
    auto graph = single({{3, 2}, {3, 2}}, relu);
    graph.tensors[0]->data_type = DataType::FP4_E2M1;
    graph.tensors[1]->data_type = DataType::FP6_E2M3;
    auto published = features(graph);
    // Six 4-bit and six 6-bit elements: 3 + 4.5 bytes, not the 12 an allocation rounds to.
    EXPECT_DOUBLE_EQ(published.at("graph.logical_bytes").get<double>(), 7.5);
    EXPECT_DOUBLE_EQ(published.at("graph.arithmetic_intensity").get<double>(), 6.0 / 7.5);

    graph.tensors[1]->data_type = DataType::UNSET;
    published = features(graph);
    EXPECT_FALSE(published.contains("graph.logical_bytes"));
    EXPECT_FALSE(published.contains("graph.arithmetic_intensity"));
    EXPECT_DOUBLE_EQ(published.at("graph.flops").get<double>(), 6.0);
}

// Override shapes may replace the declared ones at execute time; the work is the declared
// problem's, as for any graph (RFC 0019 section 6.8).
TEST(TestEngineFeatures, OverrideShapesCountTheDeclaredShapes)
{
    auto graph = convBiasReluGraph();
    graph.is_override_shape_enabled = true;
    // The shared cache keys on graph content, which ignores the override flag: each form is
    // computed on a cleared cache, so neither answers for the other.
    hipdnn_plugin_sdk::heuristics::problemFeatureCache().clear();
    const auto overridden = features(graph);
    hipdnn_plugin_sdk::heuristics::problemFeatureCache().clear();
    const auto fixed = features(convBiasReluGraph());
    EXPECT_EQ(overridden, fixed);
    EXPECT_DOUBLE_EQ(overridden.at("graph.flops").get<double>(), 87600.0);
    EXPECT_DOUBLE_EQ(overridden.at("graph.flops_by_type.ConvolutionFwdAttributes").get<double>(),
                     86400.0);
    EXPECT_TRUE(overridden.contains("graph.logical_bytes"));
}

/// Owns a packed graph and the wrapper over it, as a plugin receives one.
class PackedGraph
{
public:
    explicit PackedGraph(const GraphT& graph)
    {
        _buffer.Finish(Graph::Pack(_buffer, &graph));
        _wrapper = std::make_unique<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper>(
            _buffer.GetBufferPointer(), _buffer.GetSize());
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& get() const
    {
        return *_wrapper;
    }

private:
    flatbuffers::FlatBufferBuilder _buffer;
    std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper> _wrapper;
};

/// An engine configuration pinning integer knob @p name to @p value.
class PinnedConfig
{
public:
    PinnedConfig(const std::string& name, int64_t value)
    {
        EngineConfigT request;
        auto knob = std::make_unique<KnobSettingT>();
        knob->knob_id = name;
        IntValueT setting;
        setting.value = value;
        knob->value.Set(setting);
        request.knobs.push_back(std::move(knob));
        _buffer.Finish(EngineConfig::Pack(_buffer, &request));
        _wrapper
            = std::make_unique<hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper>(
                _buffer.GetBufferPointer(), _buffer.GetSize());
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& get() const
    {
        return *_wrapper;
    }

private:
    flatbuffers::FlatBufferBuilder _buffer;
    std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper> _wrapper;
};

using hipdnn_plugin_sdk::heuristics::problemFeatureCache;
using hipdnn_plugin_sdk::heuristics::ProblemFeatureCache;
using hipdnn_plugin_sdk::heuristics::sharedProblemFeatures;

/// Engines asked about one graph build its problem features once and each see them, plus
/// only their own constraints.
TEST(TestProblemFeatureCache, EnginesOfOneGraphShareItsProblemFeatures)
{
    problemFeatureCache().clear();
    const PackedGraph graph(convFwdGraph());
    const PinnedConfig first("tile", 4);
    const PinnedConfig second("waves", 2);

    const auto a
        = hipdnn_plugin_sdk::heuristics::engineFeatures(graph.get(), first.get(), Device{});
    const auto b
        = hipdnn_plugin_sdk::heuristics::engineFeatures(graph.get(), second.get(), Device{});
    EXPECT_EQ(problemFeatureCache().size(), 1U);

    auto expectedA = hipdnn_plugin_sdk::heuristics::problemFeatures(graph.get(), Device{}).toJson();
    auto expectedB = expectedA;
    expectedA["constraint.knobs.tile"] = 4;
    expectedB["constraint.knobs.waves"] = 2;
    EXPECT_EQ(a.toJson(), expectedA);
    // One engine's constraint never reaches another's context through the shared base.
    EXPECT_EQ(b.toJson(), expectedB);
    EXPECT_EQ(sharedProblemFeatures(graph.get(), Device{})->toJson().count("constraint.knobs.tile"),
              0U);
}

/// The key is the graph's content and the device: an equal graph in another buffer hits, a
/// different shape or device misses.
TEST(TestProblemFeatureCache, KeysOnGraphContentAndDevice)
{
    problemFeatureCache().clear();
    const PackedGraph graph(convFwdGraph());
    const PackedGraph copy(convFwdGraph());
    auto otherShape = convFwdGraph();
    otherShape.tensors[0]->dims[0] = 4;
    otherShape.tensors[2]->dims[0] = 4;
    const PackedGraph other(otherShape);
    Device bigger;
    bigger.multiProcessorCount = 128;

    const auto shared = sharedProblemFeatures(graph.get(), Device{});
    EXPECT_EQ(sharedProblemFeatures(copy.get(), Device{}), shared);
    EXPECT_NE(sharedProblemFeatures(other.get(), Device{}), shared);
    const auto onBigger = sharedProblemFeatures(graph.get(), bigger);
    EXPECT_NE(onBigger, shared);
    EXPECT_EQ(onBigger->toJson(),
              hipdnn_plugin_sdk::heuristics::problemFeatures(graph.get(), bigger).toJson());
    EXPECT_EQ(problemFeatureCache().size(), 3U);
}

/// Past CAPACITY graphs the least recently used one is rebuilt; a recently used one is not.
TEST(TestProblemFeatureCache, EvictsTheLeastRecentlyUsedGraph)
{
    problemFeatureCache().clear();
    std::vector<std::unique_ptr<PackedGraph>> graphs;
    for(size_t i = 0; i <= ProblemFeatureCache::CAPACITY; ++i)
    {
        auto graph = convFwdGraph();
        graph.tensors[0]->dims[0] = static_cast<int64_t>(i + 1);
        graph.tensors[2]->dims[0] = static_cast<int64_t>(i + 1);
        graphs.push_back(std::make_unique<PackedGraph>(graph));
    }
    const auto oldest = sharedProblemFeatures(graphs[0]->get(), Device{});
    const auto touched = sharedProblemFeatures(graphs[1]->get(), Device{});
    for(size_t i = 2; i < ProblemFeatureCache::CAPACITY; ++i)
    {
        sharedProblemFeatures(graphs[i]->get(), Device{});
    }
    EXPECT_EQ(sharedProblemFeatures(graphs[0]->get(), Device{}), oldest); // Now most recent.
    sharedProblemFeatures(graphs[ProblemFeatureCache::CAPACITY]->get(), Device{});

    EXPECT_EQ(problemFeatureCache().size(), ProblemFeatureCache::CAPACITY);
    EXPECT_EQ(sharedProblemFeatures(graphs[0]->get(), Device{}), oldest);
    EXPECT_NE(sharedProblemFeatures(graphs[1]->get(), Device{}), touched);
}

/// A graph whose content bytes are withheld (IGraph::bytes() empty), as an implementation that
/// cannot identify its graph reports itself. Its tensors and nodes still read.
class ByteslessGraph final : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper
{
public:
    using GraphWrapper::GraphWrapper;

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::SerializedBlobView bytes() const override
    {
        return {};
    }
};

/// A graph without usable content has no key a later call could match, so each caller gets
/// its own computation and the cache holds nothing for it.
TEST(TestProblemFeatureCache, AGraphWithoutContentBytesIsNeverCached)
{
    problemFeatureCache().clear();
    const auto raw = convFwdGraph();
    flatbuffers::FlatBufferBuilder buffer;
    buffer.Finish(Graph::Pack(buffer, &raw));
    const ByteslessGraph graph(buffer.GetBufferPointer(), buffer.GetSize());
    ASSERT_TRUE(graph.isValid());
    ASSERT_EQ(graph.bytes().size, 0U) << "the fixture must actually carry no content";

    const auto first = sharedProblemFeatures(graph, Device{});
    const auto second = sharedProblemFeatures(graph, Device{});

    EXPECT_NE(first, second);
    EXPECT_EQ(first->toJson(), second->toJson());
    EXPECT_EQ(first->toJson(),
              hipdnn_plugin_sdk::heuristics::problemFeatures(graph, Device{}).toJson());
    EXPECT_EQ(problemFeatureCache().size(), 0U);
}

/// Engines of several plugin threads select at once; whatever the interleaving of hits, misses
/// and evictions, each gets exactly the features a lone caller would. Most calls hit a few hot
/// graphs, which is where an unguarded list edit races; more graphs than CAPACITY keep a
/// stream of misses evicting under the hits. Each thread holds its own copy of every graph and
/// its own configuration, as separate selections do: the wrappers are not thread-safe objects.
TEST(TestProblemFeatureCache, ConcurrentCallersEachGetTheirGraphsFeatures)
{
    constexpr size_t GRAPHS = ProblemFeatureCache::CAPACITY + 8;
    constexpr size_t HOT = 6;
    constexpr size_t THREADS = 8;
    constexpr size_t ROUNDS = 4000;

    const auto graphOf = [](size_t i) {
        auto graph = convFwdGraph();
        graph.tensors[0]->dims[0] = static_cast<int64_t>(i + 1);
        graph.tensors[2]->dims[0] = static_cast<int64_t>(i + 1);
        return graph;
    };
    problemFeatureCache().clear();
    std::vector<nlohmann::json> expected;
    expected.reserve(GRAPHS);
    for(size_t i = 0; i < GRAPHS; ++i)
    {
        expected.push_back(
            hipdnn_plugin_sdk::heuristics::problemFeatures(PackedGraph(graphOf(i)).get(), Device{})
                .toJson());
    }

    std::atomic<bool> go{false};
    std::atomic<size_t> wrong{0};
    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for(size_t t = 0; t < THREADS; ++t)
    {
        threads.emplace_back([&, t] {
            std::vector<std::unique_ptr<PackedGraph>> graphs;
            graphs.reserve(GRAPHS);
            for(size_t i = 0; i < GRAPHS; ++i)
            {
                graphs.push_back(std::make_unique<PackedGraph>(graphOf(i)));
            }
            const PinnedConfig config("tile", 4);
            while(!go.load())
            {
                std::this_thread::yield();
            }
            for(size_t round = 0; round < ROUNDS; ++round)
            {
                const bool cold = round % 8 == 0;
                const size_t index = cold ? (round / 8 * 5 + t * 3) % GRAPHS : (round + t) % HOT;
                if(!cold && round % 16 != 1)
                {
                    sharedProblemFeatures(graphs[index]->get(), Device{});
                    continue;
                }
                auto want = expected[index];
                nlohmann::json got;
                if(round % 16 < 8)
                {
                    got = sharedProblemFeatures(graphs[index]->get(), Device{})->toJson();
                }
                else
                {
                    got = hipdnn_plugin_sdk::heuristics::engineFeatures(
                              graphs[index]->get(), config.get(), Device{})
                              .toJson();
                    want["constraint.knobs.tile"] = 4;
                }
                if(got != want)
                {
                    ++wrong;
                }
            }
        });
    }
    go = true;
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(wrong.load(), 0U);
    EXPECT_LE(problemFeatureCache().size(), ProblemFeatureCache::CAPACITY);
}

} // namespace
