// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/RaggedLengthUids.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_test_sdk::utilities::raggedLengthUids;

namespace
{

using UidMap = std::unordered_map<int64_t, int64_t>;

// Uids 1-9 are the node's tensors, 20 the ragged_offset aux, 30 / 31 seq_len_q / seq_len_kv.
constexpr int64_t OFFSET_UID = 20;
constexpr int64_t SEQ_LEN_Q_UID = 30;
constexpr int64_t SEQ_LEN_KV_UID = 31;

std::unique_ptr<TensorAttributesT> makeTensor(int64_t uid, bool ragged)
{
    auto tensor = std::make_unique<TensorAttributesT>();
    tensor->uid = uid;
    tensor->data_type = DataType::FLOAT;
    tensor->dims = {2, 1, 4, 2};
    tensor->strides = {8, 2, 2, 1};
    if(ragged)
    {
        tensor->ragged_offset_tensor_uid = OFFSET_UID;
    }
    return tensor;
}

// A one-node graph. Every tensor in `raggedUids` is ragged, every one in `denseUids` is not.
flatbuffers::DetachedBuffer makeGraph(NodeAttributesUnion attributes,
                                      const std::vector<int64_t>& raggedUids,
                                      const std::vector<int64_t>& denseUids = {})
{
    GraphT graph;
    for(const auto uid : raggedUids)
    {
        graph.tensors.push_back(makeTensor(uid, true));
    }
    for(const auto uid : denseUids)
    {
        graph.tensors.push_back(makeTensor(uid, false));
    }
    for(const auto uid : {OFFSET_UID, SEQ_LEN_Q_UID, SEQ_LEN_KV_UID})
    {
        graph.tensors.push_back(makeTensor(uid, false));
    }
    auto node = std::make_unique<NodeT>();
    node->attributes = std::move(attributes);
    graph.nodes.push_back(std::move(node));

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, &graph));
    return builder.Release();
}

UidMap lengthUids(const flatbuffers::DetachedBuffer& buffer)
{
    return raggedLengthUids(*GetGraph(buffer.data()));
}

// Forward: Q=1, K=2, V=3, O=4, Stats=5, Max=6, Sum_exp=7.
SdpaAttributesT forwardNode(bool seqLenQ = true, bool seqLenKv = true)
{
    SdpaAttributesT attrs;
    attrs.q_tensor_uid = 1;
    attrs.k_tensor_uid = 2;
    attrs.v_tensor_uid = 3;
    attrs.o_tensor_uid = 4;
    attrs.stats_tensor_uid = 5;
    attrs.max_tensor_uid = 6;
    attrs.sum_exp_tensor_uid = 7;
    if(seqLenQ)
    {
        attrs.seq_len_q_tensor_uid = SEQ_LEN_Q_UID;
    }
    if(seqLenKv)
    {
        attrs.seq_len_kv_tensor_uid = SEQ_LEN_KV_UID;
    }
    return attrs;
}

NodeAttributesUnion asUnion(SdpaAttributesT attrs)
{
    NodeAttributesUnion u;
    u.Set(attrs);
    return u;
}

// Backward: Q=1, K=2, V=3, O=4, dO=5, Stats=6, dQ=7, dK=8, dV=9.
NodeAttributesUnion backwardNode()
{
    SdpaBackwardAttributesT attrs;
    attrs.q_tensor_uid = 1;
    attrs.k_tensor_uid = 2;
    attrs.v_tensor_uid = 3;
    attrs.o_tensor_uid = 4;
    attrs.do_tensor_uid = 5;
    attrs.stats_tensor_uid = 6;
    attrs.dq_tensor_uid = 7;
    attrs.dk_tensor_uid = 8;
    attrs.dv_tensor_uid = 9;
    attrs.seq_len_q_tensor_uid = SEQ_LEN_Q_UID;
    attrs.seq_len_kv_tensor_uid = SEQ_LEN_KV_UID;
    NodeAttributesUnion u;
    u.Set(attrs);
    return u;
}

} // namespace

// Q, O, Stats, Max and Sum_exp take seq_len_q; K and V take seq_len_kv.
TEST(TestRaggedLengthUids, ForwardRolesIncludeMaxAndSumExp)
{
    const auto graph = makeGraph(asUnion(forwardNode()), {1, 2, 3, 4, 5, 6, 7});
    EXPECT_EQ(lengthUids(graph),
              (UidMap{{1, SEQ_LEN_Q_UID},
                      {4, SEQ_LEN_Q_UID},
                      {5, SEQ_LEN_Q_UID},
                      {6, SEQ_LEN_Q_UID},
                      {7, SEQ_LEN_Q_UID},
                      {2, SEQ_LEN_KV_UID},
                      {3, SEQ_LEN_KV_UID}}));
}

// Q, O, dO, dQ and Stats take seq_len_q; K, V, dK and dV take seq_len_kv.
TEST(TestRaggedLengthUids, BackwardRoles)
{
    const auto graph = makeGraph(backwardNode(), {1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_EQ(lengthUids(graph),
              (UidMap{{1, SEQ_LEN_Q_UID},
                      {4, SEQ_LEN_Q_UID},
                      {5, SEQ_LEN_Q_UID},
                      {6, SEQ_LEN_Q_UID},
                      {7, SEQ_LEN_Q_UID},
                      {2, SEQ_LEN_KV_UID},
                      {3, SEQ_LEN_KV_UID},
                      {8, SEQ_LEN_KV_UID},
                      {9, SEQ_LEN_KV_UID}}));
}

// A role has an entry only when its node names that seq_len.
TEST(TestRaggedLengthUids, NoEntryWithoutTheNodesSeqLen)
{
    const auto qOnly = makeGraph(asUnion(forwardNode(true, false)), {1, 2, 3, 4, 5, 6, 7});
    const auto lengthsOfQOnly = lengthUids(qOnly);
    EXPECT_EQ(lengthsOfQOnly.count(2), 0u);
    EXPECT_EQ(lengthsOfQOnly.count(3), 0u);
    EXPECT_EQ(lengthsOfQOnly.at(1), SEQ_LEN_Q_UID);

    const auto kvOnly = makeGraph(asUnion(forwardNode(false, true)), {1, 2, 3, 4, 5, 6, 7});
    const auto lengthsOfKvOnly = lengthUids(kvOnly);
    EXPECT_EQ(lengthsOfKvOnly.count(1), 0u);
    EXPECT_EQ(lengthsOfKvOnly.at(2), SEQ_LEN_KV_UID);

    EXPECT_TRUE(
        lengthUids(makeGraph(asUnion(forwardNode(false, false)), {1, 2, 3, 4, 5, 6, 7})).empty());
}

// Only ragged tensors are mapped; the ragged_offset and seq_len tensors themselves never are.
TEST(TestRaggedLengthUids, NonRaggedTensorsHaveNoEntry)
{
    const auto graph = makeGraph(asUnion(forwardNode()), {1, 4}, {2, 3, 5, 6, 7});
    EXPECT_EQ(lengthUids(graph), (UidMap{{1, SEQ_LEN_Q_UID}, {4, SEQ_LEN_Q_UID}}));
}

TEST(TestRaggedLengthUids, GraphWithoutSdpaNodeIsEmpty)
{
    GraphT graph;
    graph.tensors.push_back(makeTensor(1, true));
    auto node = std::make_unique<NodeT>();
    graph.nodes.push_back(std::move(node));
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, &graph));
    EXPECT_TRUE(raggedLengthUids(*GetGraph(builder.GetBufferPointer())).empty());
}

// Q and K are one tensor; seq_len_q and seq_len_kv are different tensors.
TEST(TestRaggedLengthUids, ThrowsWhenOneTensorTakesTwoSeqLens)
{
    auto attrs = forwardNode();
    attrs.k_tensor_uid = attrs.q_tensor_uid;
    const auto graph = makeGraph(asUnion(attrs), {1, 3, 4, 5, 6, 7});
    EXPECT_THROW(lengthUids(graph), std::invalid_argument);
}

// Q and K are one tensor and both roles name the same seq_len tensor.
TEST(TestRaggedLengthUids, OneTensorInBothRolesWithOneSeqLen)
{
    auto attrs = forwardNode();
    attrs.k_tensor_uid = attrs.q_tensor_uid;
    attrs.seq_len_kv_tensor_uid = SEQ_LEN_Q_UID;
    const auto graph = makeGraph(asUnion(attrs), {1, 3, 4});
    EXPECT_EQ(lengthUids(graph),
              (UidMap{{1, SEQ_LEN_Q_UID}, {3, SEQ_LEN_Q_UID}, {4, SEQ_LEN_Q_UID}}));
}

// Two forward nodes share K; the second node's seq_len_kv is a different tensor.
TEST(TestRaggedLengthUids, ThrowsWhenNodesBindSharedTensorToDifferentSeqLens)
{
    constexpr int64_t OTHER_SEQ_LEN_KV_UID = 32;
    GraphT graph;
    for(const auto uid : {1, 2, 3, 4, 11, 13, 14})
    {
        graph.tensors.push_back(makeTensor(uid, true));
    }
    for(const auto uid : {OFFSET_UID, SEQ_LEN_Q_UID, SEQ_LEN_KV_UID, OTHER_SEQ_LEN_KV_UID})
    {
        graph.tensors.push_back(makeTensor(uid, false));
    }
    auto second = forwardNode();
    second.q_tensor_uid = 11;
    second.v_tensor_uid = 13;
    second.o_tensor_uid = 14;
    second.seq_len_kv_tensor_uid = OTHER_SEQ_LEN_KV_UID;
    for(auto attrs : {forwardNode(), second})
    {
        attrs.stats_tensor_uid.reset();
        attrs.max_tensor_uid.reset();
        attrs.sum_exp_tensor_uid.reset();
        auto node = std::make_unique<NodeT>();
        node->attributes = asUnion(attrs);
        graph.nodes.push_back(std::move(node));
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, &graph));
    EXPECT_THROW(raggedLengthUids(*GetGraph(builder.GetBufferPointer())), std::invalid_argument);
}
