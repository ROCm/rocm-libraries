// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Unit tests for GraphView producer/consumer adjacency: linear-chain producer
// and consumer endpoints, leaf/output boundaries, per-node flattened operand
// and result UID lists, the load-bearing use-count vs consumer-count
// distinction, variadic op resolution, tensor lookup, and graceful handling of
// unknown/absent schemas. Graphs are built host-only with the test-SDK builders,
// plus two minimal inline graphs for the fan-in / fan-out cases.

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <vector>

namespace {

namespace gm = hipdnn::graph_matcher;
namespace data = hipdnn_flatbuffers_sdk::data_objects;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace util = hipdnn_test_sdk::utilities;

// A single-consumer pointwise where in_0 == in_1 == the same UID. The tensor is
// referenced by two operand slots of ONE node: use-count 2, consumer-count 1.
flatbuffers::FlatBufferBuilder createSelfConsumingPointwiseGraph() {
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {4, 8};
    const std::vector<int64_t> strides = {8, 1};

    std::vector<flatbuffers::Offset<data::TensorAttributes>> tensors;
    tensors.push_back(data::CreateTensorAttributesDirect(builder, 1, "h", data::DataType::FLOAT,
                                                         &strides, &dims));
    tensors.push_back(data::CreateTensorAttributesDirect(builder, 2, "out", data::DataType::FLOAT,
                                                         &strides, &dims));

    // mul(h, h): in_0 = in_1 = 1, out_0 = 2.
    auto pw =
        data::CreatePointwiseAttributes(builder, data::PointwiseMode::MUL, flatbuffers::nullopt,
                                        flatbuffers::nullopt, flatbuffers::nullopt,
                                        flatbuffers::nullopt,               // axis_tensor_uid
                                        1,                                  // in_0
                                        flatbuffers::Optional<int64_t>(1),  // in_1
                                        flatbuffers::nullopt,               // in_2
                                        2);                                 // out_0

    std::vector<flatbuffers::Offset<data::Node>> nodes;
    nodes.push_back(data::CreateNodeDirect(builder, "mul", data::DataType::FLOAT,
                                           data::NodeAttributes::PointwiseAttributes, pw.Union()));

    auto graph =
        data::CreateGraphDirect(builder, "self_consume", data::DataType::FLOAT,
                                data::DataType::FLOAT, data::DataType::FLOAT, &tensors, &nodes);
    builder.Finish(graph);
    return builder;
}

// One input tensor feeding two distinct pointwise nodes: use-count 2,
// consumer-count 2.
flatbuffers::FlatBufferBuilder createFanoutPointwiseGraph() {
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {4, 8};
    const std::vector<int64_t> strides = {8, 1};

    std::vector<flatbuffers::Offset<data::TensorAttributes>> tensors;
    tensors.push_back(data::CreateTensorAttributesDirect(builder, 1, "x", data::DataType::FLOAT,
                                                         &strides, &dims));
    tensors.push_back(data::CreateTensorAttributesDirect(builder, 2, "out_a", data::DataType::FLOAT,
                                                         &strides, &dims));
    tensors.push_back(data::CreateTensorAttributesDirect(builder, 3, "out_b", data::DataType::FLOAT,
                                                         &strides, &dims));

    auto relu = [&](int64_t out) {
        return data::CreatePointwiseAttributes(builder, data::PointwiseMode::RELU_FWD,
                                               flatbuffers::nullopt, flatbuffers::nullopt,
                                               flatbuffers::nullopt,
                                               flatbuffers::nullopt,  // axis
                                               1,                     // in_0 = x
                                               flatbuffers::nullopt,  // in_1
                                               flatbuffers::nullopt,  // in_2
                                               out);
    };

    std::vector<flatbuffers::Offset<data::Node>> nodes;
    nodes.push_back(data::CreateNodeDirect(builder, "relu_a", data::DataType::FLOAT,
                                           data::NodeAttributes::PointwiseAttributes,
                                           relu(2).Union()));
    nodes.push_back(data::CreateNodeDirect(builder, "relu_b", data::DataType::FLOAT,
                                           data::NodeAttributes::PointwiseAttributes,
                                           relu(3).Union()));

    auto graph =
        data::CreateGraphDirect(builder, "fanout", data::DataType::FLOAT, data::DataType::FLOAT,
                                data::DataType::FLOAT, &tensors, &nodes);
    builder.Finish(graph);
    return builder;
}

}  // namespace

// --- Linear chain: producer/consumer endpoints ---------------------------

// createValidMatmulBiasActivGraph (doBias=true, RELU) wires:
//   node0 matmul:      a=1, b=2      -> c=3 (C_matmul)
//   node1 bias  (pw):  in_0=3, in_1=4 -> out_0=5 (C_bias)
//   node2 activ (pw):  in_0=5         -> out_0=6 (C)
// So C_matmul(3) is produced by node0's result role and consumed by node1's
// first operand role: the merge point that motivates the whole view.
TEST(GraphView, LinearChainProducerAndConsumerEndpoints) {
    auto builder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    const gm::Endpoint* producer = view.producerOf(3);
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer->nodeIndex, 0u);  // matmul
    EXPECT_EQ(producer->roleIndex, 0u);  // results[0] == c
    EXPECT_EQ(producer->slot, 0u);

    const auto& consumers = view.consumersOf(3);
    ASSERT_EQ(consumers.size(), 1u);
    EXPECT_EQ(consumers[0].nodeIndex, 1u);  // bias pointwise
    EXPECT_EQ(consumers[0].roleIndex, 0u);  // operands[0] == in_0
    EXPECT_EQ(consumers[0].slot, 0u);
}

TEST(GraphView, LeafInputHasNoProducer) {
    auto builder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // A (uid 1) and B (uid 2) are graph inputs: no producing node.
    EXPECT_EQ(view.producerOf(1), nullptr);
    EXPECT_EQ(view.producerOf(2), nullptr);
    // ...but they ARE consumed by the matmul.
    EXPECT_EQ(view.useCount(1), 1u);
    EXPECT_EQ(view.useCount(2), 1u);
}

TEST(GraphView, FinalOutputHasNoConsumers) {
    auto builder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // C (uid 6) is the graph output: produced but never consumed.
    ASSERT_NE(view.producerOf(6), nullptr);
    EXPECT_TRUE(view.consumersOf(6).empty());
    EXPECT_EQ(view.useCount(6), 0u);
    EXPECT_EQ(view.consumerNodeCount(6), 0u);
}

TEST(GraphView, OperandAndResultUidsFlattenPerNode) {
    auto builder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_EQ(view.operandUids(0), (std::vector<int64_t>{1, 2}));  // matmul a,b
    EXPECT_EQ(view.resultUids(0), (std::vector<int64_t>{3}));      // matmul c

    // bias pointwise: in_0=3, optional in_1=4 present -> both flattened.
    EXPECT_EQ(view.operandUids(1), (std::vector<int64_t>{3, 4}));
    EXPECT_EQ(view.resultUids(1), (std::vector<int64_t>{5}));

    // activation pointwise: only in_0=5 set (optional in_1/in_2 absent).
    EXPECT_EQ(view.operandUids(2), (std::vector<int64_t>{5}));
    EXPECT_EQ(view.resultUids(2), (std::vector<int64_t>{6}));
}

// --- The load-bearing distinction: use-count vs consumer-count -----------

TEST(GraphView, UseCountCountsSlotsConsumerCountCountsNodes_SelfConsume) {
    auto builder = createSelfConsumingPointwiseGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // h (uid 1) feeds in_0 and in_1 of the SAME node.
    EXPECT_EQ(view.useCount(1), 2u);
    EXPECT_EQ(view.consumerNodeCount(1), 1u);

    const auto& consumers = view.consumersOf(1);
    ASSERT_EQ(consumers.size(), 2u);
    EXPECT_EQ(consumers[0].nodeIndex, 0u);
    EXPECT_EQ(consumers[1].nodeIndex, 0u);
    // Distinct operand roles: in_0 (role 0) and in_1 (role 1).
    EXPECT_EQ(consumers[0].roleIndex, 0u);
    EXPECT_EQ(consumers[1].roleIndex, 1u);
}

TEST(GraphView, UseCountCountsSlotsConsumerCountCountsNodes_Fanout) {
    auto builder = createFanoutPointwiseGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // x (uid 1) feeds in_0 of TWO different nodes.
    EXPECT_EQ(view.useCount(1), 2u);
    EXPECT_EQ(view.consumerNodeCount(1), 2u);

    const auto& consumers = view.consumersOf(1);
    ASSERT_EQ(consumers.size(), 2u);
    EXPECT_NE(consumers[0].nodeIndex, consumers[1].nodeIndex);
}

// --- Variadic op resolution ----------------------------------------------

TEST(GraphView, VariadicCustomOpResolvesInputAndOutputVectors) {
    // createValidCustomOpGraph: inputs=[1], outputs=[2].
    auto builder = util::createValidCustomOpGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_EQ(view.operandUids(0), (std::vector<int64_t>{1}));
    EXPECT_EQ(view.resultUids(0), (std::vector<int64_t>{2}));

    EXPECT_EQ(view.producerOf(2)->nodeIndex, 0u);  // output produced by custom op
    EXPECT_EQ(view.producerOf(1), nullptr);        // input is a leaf
    ASSERT_EQ(view.consumersOf(1).size(), 1u);
    EXPECT_EQ(view.consumersOf(1)[0].nodeIndex, 0u);
}

// --- tensor() lookup ------------------------------------------------------

TEST(GraphView, TensorReturnsAttributesByUidAndNullForAbsent) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    const data::TensorAttributes* a = view.tensor(1);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->uid(), 1);

    const data::TensorAttributes* c = view.tensor(3);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->uid(), 3);

    EXPECT_EQ(view.tensor(999), nullptr);  // no such UID in graph
}

// --- Empty / no-schema graphs skipped without crashing -------------------

TEST(GraphView, EmptyGraphYieldsNoProducersOrConsumers) {
    auto builder = util::createEmptyValidGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);  // must not crash on zero nodes

    EXPECT_EQ(view.producerOf(1), nullptr);
    EXPECT_TRUE(view.consumersOf(1).empty());
    EXPECT_EQ(view.useCount(1), 0u);
    EXPECT_EQ(view.consumerNodeCount(1), 0u);
    EXPECT_EQ(view.tensor(1), nullptr);
}
