// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

// Small one-node ragged SDPA graphs (and their dense padded twins) for testing the ragged-as-dense
// rewrite (RaggedSdpaDensify.hpp) in the CPU and GPU executors.
namespace hipdnn_test_sdk::utilities::ragged_densify_test
{

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::Tensor;

// B = 2, H = 2, D = 4, Sq_max = 6, Skv_max = 8; live lengths q = {4, 6}, kv = {8, 5}. Packed
// token-major ([B, H, S, D] dims, strides [S*H*D, D, H*D, 1]) with token-unit offsets
// (multiplier H*D), so batch b starts at token offsets[b].
constexpr int64_t B = 2;
constexpr int64_t H = 2;
constexpr int64_t D = 4;
constexpr int64_t SQ = 6;
constexpr int64_t SKV = 8;
inline const std::vector<int32_t> Q_TOKENS = {0, 4, 10};
inline const std::vector<int32_t> KV_TOKENS = {0, 8, 13};
inline const std::vector<int64_t> Q_LENS = {4, 6};
inline const std::vector<int64_t> KV_LENS = {8, 5};

enum Uid : int64_t
{
    Q = 1,
    K,
    V,
    O,
    STATS,
    DO,
    DQ,
    DK,
    DV,
    Q_OFFSETS,
    KV_OFFSETS,
    BIAS,
    SEQ_Q,
    SEQ_KV,
};

inline std::unique_ptr<TensorAttributesT> tensor(int64_t uid,
                                                 DataType type,
                                                 std::vector<int64_t> dims,
                                                 std::vector<int64_t> strides,
                                                 std::optional<int64_t> offsetUid = std::nullopt)
{
    auto t = std::make_unique<TensorAttributesT>();
    t->uid = uid;
    t->name = "t" + std::to_string(uid);
    t->data_type = type;
    t->dims = std::move(dims);
    t->strides = std::move(strides);
    if(offsetUid.has_value())
    {
        t->ragged_offset_tensor_uid = offsetUid;
        t->ragged_offset_multiplier = t->strides[2];
    }
    return t;
}

inline std::vector<int64_t> packedStrides(int64_t s)
{
    return {s * H * D, D, H * D, 1};
}

inline std::vector<int64_t> denseStrides(int64_t s)
{
    return {H * s * D, s * D, D, 1};
}

// One-node ragged SDPA graph: forward (Q, K, V -> O, packed LSE) or backward (Q, K, V, O, dO,
// packed LSE -> dQ, dK, dV), optionally with a FLOAT bias [1, 1, SQ, SKV].
inline std::vector<uint8_t> raggedGraph(bool backward, bool bias, bool ragged = true)
{
    GraphT graph;
    graph.name = "ragged";
    graph.io_data_type = DataType::FLOAT;
    graph.compute_data_type = DataType::FLOAT;
    graph.intermediate_data_type = DataType::FLOAT;
    const auto qOff = ragged ? std::optional<int64_t>(Q_OFFSETS) : std::nullopt;
    const auto kvOff = ragged ? std::optional<int64_t>(KV_OFFSETS) : std::nullopt;
    const auto qs = ragged ? packedStrides(SQ) : denseStrides(SQ);
    const auto ks = ragged ? packedStrides(SKV) : denseStrides(SKV);
    inline const std::vector<int64_t> lseStrides
        = ragged ? std::vector<int64_t>{SQ * H, 1, H, 1} : std::vector<int64_t>{H * SQ, SQ, 1, 1};
    auto& t = graph.tensors;
    t.push_back(tensor(Q, DataType::FLOAT, {B, H, SQ, D}, qs, qOff));
    t.push_back(tensor(K, DataType::FLOAT, {B, H, SKV, D}, ks, kvOff));
    t.push_back(tensor(V, DataType::FLOAT, {B, H, SKV, D}, ks, kvOff));
    t.push_back(tensor(O, DataType::FLOAT, {B, H, SQ, D}, qs, qOff));
    t.push_back(tensor(STATS, DataType::FLOAT, {B, H, SQ, 1}, lseStrides, qOff));
    if(ragged)
    {
        t.push_back(tensor(Q_OFFSETS, DataType::INT32, {B + 1, 1, 1, 1}, {1, 1, 1, 1}));
        t.push_back(tensor(KV_OFFSETS, DataType::INT32, {B + 1, 1, 1, 1}, {1, 1, 1, 1}));
    }
    else
    {
        t.push_back(tensor(SEQ_Q, DataType::INT32, {B, 1, 1, 1}, {1, 1, 1, 1}));
        t.push_back(tensor(SEQ_KV, DataType::INT32, {B, 1, 1, 1}, {1, 1, 1, 1}));
    }
    if(bias)
    {
        t.push_back(tensor(BIAS, DataType::FLOAT, {1, 1, SQ, SKV}, {SQ * SKV, SQ * SKV, SKV, 1}));
    }
    auto node = std::make_unique<NodeT>();
    node->name = "sdpa";
    node->compute_data_type = DataType::FLOAT;
    if(backward)
    {
        t.push_back(tensor(DO, DataType::FLOAT, {B, H, SQ, D}, qs, qOff));
        t.push_back(tensor(DQ, DataType::FLOAT, {B, H, SQ, D}, qs, qOff));
        t.push_back(tensor(DK, DataType::FLOAT, {B, H, SKV, D}, ks, kvOff));
        t.push_back(tensor(DV, DataType::FLOAT, {B, H, SKV, D}, ks, kvOff));
        SdpaBackwardAttributesT attrs;
        attrs.q_tensor_uid = Q;
        attrs.k_tensor_uid = K;
        attrs.v_tensor_uid = V;
        attrs.o_tensor_uid = O;
        attrs.do_tensor_uid = DO;
        attrs.stats_tensor_uid = STATS;
        attrs.dq_tensor_uid = DQ;
        attrs.dk_tensor_uid = DK;
        attrs.dv_tensor_uid = DV;
        attrs.attn_scale_value = 0.5F;
        attrs.right_bound = 0;
        attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
        if(bias)
        {
            attrs.attn_mask_tensor_uid = BIAS;
        }
        if(!ragged)
        {
            attrs.seq_len_q_tensor_uid = SEQ_Q;
            attrs.seq_len_kv_tensor_uid = SEQ_KV;
            attrs.padding_mask = true;
        }
        node->attributes.Set(std::move(attrs));
    }
    else
    {
        SdpaAttributesT attrs;
        attrs.q_tensor_uid = Q;
        attrs.k_tensor_uid = K;
        attrs.v_tensor_uid = V;
        attrs.o_tensor_uid = O;
        attrs.stats_tensor_uid = STATS;
        attrs.attn_scale_value = 0.5F;
        attrs.right_bound = 0;
        attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
        if(bias)
        {
            attrs.attn_mask_tensor_uid = BIAS;
        }
        if(!ragged)
        {
            attrs.seq_len_q_tensor_uid = SEQ_Q;
            attrs.seq_len_kv_tensor_uid = SEQ_KV;
            attrs.padding_mask = true;
        }
        node->attributes.Set(std::move(attrs));
    }
    graph.nodes.push_back(std::move(node));
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, &graph));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

// Random values for every packed token, NaN everywhere else (a read outside the live tokens
// shows up as NaN in the outputs).
inline std::vector<float> packedValues(int64_t tokens, int64_t width, unsigned seed)
{
    Tensor<float> t({tokens * width});
    t.fillWithRandomValues(-1.0F, 1.0F, seed);
    return {t.memory().hostData(), t.memory().hostData() + tokens * width};
}

// Packed [token, H, D] -> dense contiguous [B, H, S, D] with zeros past each batch's length.
inline std::vector<float> unpack(inline const std::vector<float>& packed,
                                 inline const std::vector<int32_t>& tokens,
                                 int64_t s,
                                 int64_t width)
{
    std::vector<float> dense(static_cast<size_t>(B * H * s * width), 0.0F);
    for(int64_t b = 0; b < B; ++b)
    {
        for(int64_t tok = 0; tok < tokens[b + 1] - tokens[b]; ++tok)
        {
            for(int64_t h = 0; h < H; ++h)
            {
                for(int64_t d = 0; d < width; ++d)
                {
                    dense[static_cast<size_t>(((b * H + h) * s + tok) * width + d)]
                        = packed[static_cast<size_t>(((tokens[b] + tok) * H + h) * width + d)];
                }
            }
        }
    }
    return dense;
}

} // namespace hipdnn_test_sdk::utilities::ragged_densify_test
