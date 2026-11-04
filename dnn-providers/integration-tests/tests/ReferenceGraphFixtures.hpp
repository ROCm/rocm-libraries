// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Minimal serialized graphs for tests that decide things from a graph's node types
// and shapes -- reference op coverage, the CPU cost gate, the golden-data planner --
// without depending on the checked-in bundles.

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <nlohmann/json.hpp>

namespace hipdnn_integration_tests::bundle::fixtures
{

// A minimal single-node batchnorm-inference graph, serialized the same way the
// bundle loader does it.
inline constexpr const char* BATCHNORM_GRAPH_JSON = R"({"nodes": [{"inputs": {"x_tensor_uid": 0,
    "mean_tensor_uid": 1, "inv_variance_tensor_uid": 2, "scale_tensor_uid": 3,
    "bias_tensor_uid": 4}, "outputs": {"y_tensor_uid": 5},
    "type": "BatchnormInferenceAttributes", "compute_data_type": "float", "name": ""}],
    "tensors": [
    {"name": "", "uid": 0, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], "data_type": "float", "virtual": false},
    {"name": "", "uid": 1, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 2, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 3, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 4, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 5, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], "data_type": "float", "virtual": false}],
    "io_data_type": "float", "compute_data_type": "float",
    "intermediate_data_type": "float", "name": ""})";

inline flatbuffers::DetachedBuffer buildBatchnormGraph()
{
    flatbuffers::FlatBufferBuilder builder;
    const auto json = nlohmann::json::parse(BATCHNORM_GRAPH_JSON);
    auto offset = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
        builder, json);
    builder.Finish(offset);
    return builder.Release();
}

// The graph JSON converter requires every declared key to be present, nullable
// ones included, so these spell out the full Sdpa input/output sets with only the
// tensors this graph actually uses populated. Trimming them to the fields under
// test fails conversion rather than producing a smaller graph.
inline nlohmann::json sdpaInputs()
{
    nlohmann::json inputs = {{"q_tensor_uid", 0}, {"k_tensor_uid", 1}, {"v_tensor_uid", 2}};
    for(const char* key : {"attn_mask_tensor_uid",
                           "scale_tensor_uid",
                           "seq_len_q_tensor_uid",
                           "seq_len_kv_tensor_uid",
                           "seed_tensor_uid",
                           "offset_tensor_uid",
                           "dropout_mask_tensor_uid",
                           "dropout_scale_tensor_uid",
                           "page_table_k_tensor_uid",
                           "page_table_v_tensor_uid",
                           "block_mask_tensor_uid",
                           "sink_token_tensor_uid",
                           "descale_q_tensor_uid",
                           "descale_k_tensor_uid",
                           "descale_v_tensor_uid",
                           "descale_s_tensor_uid",
                           "scale_s_tensor_uid",
                           "scale_o_tensor_uid"})
    {
        inputs[key] = nullptr;
    }
    return inputs;
}

inline nlohmann::json sdpaOutputs()
{
    nlohmann::json outputs = {{"o_tensor_uid", 3}};
    for(const char* key : {"stats_tensor_uid",
                           "max_tensor_uid",
                           "sum_exp_tensor_uid",
                           "rng_dump_tensor_uid",
                           "amax_s_tensor_uid",
                           "amax_o_tensor_uid"})
    {
        outputs[key] = nullptr;
    }
    return outputs;
}

// A single-node Sdpa graph whose Q/K sequence length is a parameter, so the
// affordability gate can be exercised on either side of its working-set cap
// without depending on the checked-in bundles.
inline flatbuffers::DetachedBuffer buildSdpaGraph(int64_t seq)
{
    const auto dims = nlohmann::json::array({2, 4, seq, 128});
    const auto strides = nlohmann::json::array({4 * seq * 128, seq * 128, 128, 1});
    auto tensor = [&](int64_t uid) {
        return nlohmann::json{{"name", ""},
                              {"uid", uid},
                              {"strides", strides},
                              {"dims", dims},
                              {"data_type", "half"},
                              {"virtual", false}};
    };

    const nlohmann::json graph = {
        {"nodes",
         nlohmann::json::array({{{"inputs", sdpaInputs()},
                                 {"outputs", sdpaOutputs()},
                                 {"type", "SdpaAttributes"},
                                 {"compute_data_type", "float"},
                                 {"name", ""},
                                 // The converter requires every key, nullable ones included, so
                                 // this mirrors a real SdpaFwd bundle's block rather than trimming
                                 // it to the fields the gate reads.
                                 {"attributes",
                                  {{"generate_stats", nullptr},
                                   {"alibi_mask", false},
                                   {"padding_mask", false},
                                   {"causal_mask", false},
                                   {"causal_mask_bottom_right", false},
                                   {"dropout_probability", nullptr},
                                   {"attn_scale_value", 0.08838834764831843},
                                   {"left_bound", -1},
                                   {"right_bound", 0},
                                   {"max_seq_len_kv", nullptr},
                                   {"diagonal_alignment", "BOTTOM_RIGHT"},
                                   {"mma_core_mode", "unset"},
                                   {"implementation", "AUTO"}}}}})},
        {"tensors", nlohmann::json::array({tensor(0), tensor(1), tensor(2), tensor(3)})},
        {"io_data_type", "half"},
        {"compute_data_type", "float"},
        {"intermediate_data_type", "float"},
        {"name", ""}};

    flatbuffers::FlatBufferBuilder builder;
    auto offset = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
        builder, graph);
    builder.Finish(offset);
    return builder.Release();
}

// A single-node SdpaBackward graph. Only Q and K carry the dimensions the cost gate reads
// (Q is [batch, heads, seqQ, headDim], K is [batch, heads, seqKv, headDim]); the other
// operands reuse Q's shape because the gate never looks at them.
inline flatbuffers::DetachedBuffer buildSdpaBackwardGraph(
    int64_t batch, int64_t heads, int64_t seqQ, int64_t seqKv, int64_t headDim)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    const auto strides = [headDim](int64_t seq, int64_t headCount) {
        return std::vector<int64_t>{headCount * seq * headDim, seq * headDim, headDim, 1};
    };
    const std::vector<int64_t> qDims = {batch, heads, seqQ, headDim};
    const std::vector<int64_t> kDims = {batch, heads, seqKv, headDim};
    const auto qStrides = strides(seqQ, heads);
    const auto kStrides = strides(seqKv, heads);

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 1, "q", DataType::HALF, &qStrides, &qDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 2, "k", DataType::HALF, &kStrides, &kDims));
    for(int64_t uid = 3; uid <= 9; ++uid)
    {
        tensors.push_back(
            CreateTensorAttributesDirect(builder, uid, "", DataType::HALF, &qStrides, &qDims));
    }

    auto attributes = CreateSdpaBackwardAttributes(builder, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_bwd",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaBackwardAttributes,
                                     attributes.Union()));

    builder.Finish(CreateGraphDirect(
        builder, "test", DataType::HALF, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes));
    return builder.Release();
}

} // namespace hipdnn_integration_tests::bundle::fixtures
