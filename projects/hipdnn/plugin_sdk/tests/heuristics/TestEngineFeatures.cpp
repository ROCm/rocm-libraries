// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

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
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        graphBuffer.GetBufferPointer(), graphBuffer.GetSize());
    flatbuffers::FlatBufferBuilder configBuffer;
    configBuffer.Finish(CreateEngineConfig(configBuffer, 1));
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
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

ConvolutionFwdAttributesT convolution(std::vector<int64_t> padding, std::vector<int64_t> stride)
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

// Every feature the hand-written Matmul/ConvolutionFwd/SDPA binders published for these
// graphs, captured from them before the generic binder replaced them. Shipped models
// read these names, so each must keep its name, value and JSON kind.
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

TEST(TestEngineFeatures, GenericOperandsKeepEveryPreviouslyPublishedFeature)
{
    const std::pair<const char*, GraphT (*)()> cases[] = {
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
    EXPECT_EQ(visited, 23U);
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
    EXPECT_FALSE(published.contains("graph.flops"));
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

TEST(TestEngineFeatures, MoeRoutingMakesWorkDataDependent)
{
    GraphT graph;
    addTensor(graph, 1, {1, 32, 64});
    addTensor(graph, 2, {4, 64, 128});
    addTensor(graph, 3, {4, 1, 1});
    addTensor(graph, 4, {1, 32, 1});
    addTensor(graph, 5, {1, 32, 128});
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
    EXPECT_EQ(published.at("graph.nodes[0].token_index.numel"), 32);
    EXPECT_EQ(published.at("graph.nodes[0].mode"),
              static_cast<int64_t>(MoeGroupedMatmulMode::GATHER));
    EXPECT_EQ(published.at("graph.nodes[0].top_k"), 2);
    EXPECT_FALSE(published.contains("graph.flops"));
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
    EXPECT_FALSE(published.contains("graph.flops"));

    // An additive bias is applied to every score whatever its values.
    sdpa = attention();
    sdpa.attn_mask_tensor_uid = 5;
    EXPECT_EQ(features(attentionGraph(sdpa)).at("graph.nodes[0].data_dependent"), false);
}

TEST(TestEngineFeatures, RaggedOperandMakesWorkDataDependent)
{
    auto graph = matmulBroadcastGraph();
    EXPECT_EQ(features(graph).at("graph.nodes[0].data_dependent"), false);
    addTensor(graph, 4, {6, 1, 1});
    graph.tensors[0]->ragged_offset_tensor_uid = 4;
    EXPECT_EQ(features(graph).at("graph.nodes[0].data_dependent"), true);
}
} // namespace
