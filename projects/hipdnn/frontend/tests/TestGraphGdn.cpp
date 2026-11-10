// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/GdnAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

std::shared_ptr<TensorAttributes> makeTensor(const std::vector<int64_t>& dims,
                                             DataType dataType = DataType::FLOAT)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(auto i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i)
    {
        const auto idx = static_cast<size_t>(i);
        strides[idx] = strides[idx + 1] * dims[idx + 1];
    }
    return Graph::tensor(
        TensorAttributes().set_dim(dims).set_stride(strides).set_data_type(dataType));
}

struct GdnInputs
{
    std::shared_ptr<TensorAttributes> q;
    std::shared_ptr<TensorAttributes> k;
    std::shared_ptr<TensorAttributes> v;
    std::shared_ptr<TensorAttributes> g;
    std::shared_ptr<TensorAttributes> beta;
    std::shared_ptr<TensorAttributes> cuSeqlens;
};

// T = 4 tokens in N = 2 sequences, H = HK = 1, HV = HO = 2, K = 4, V = 2.
GdnInputs makeInputs()
{
    return {makeTensor({4, 1, 4}),
            makeTensor({4, 1, 4}),
            makeTensor({4, 2, 2}),
            makeTensor({4, 2}),
            makeTensor({4, 2}),
            makeTensor({3}, DataType::INT32)};
}

void configureGraph(Graph& graph)
{
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);
}

} // namespace

TEST(TestGraphGdn, OptionalOutputsAreNullUnlessRequested)
{
    Graph graph;
    configureGraph(graph);
    const auto in = makeInputs();

    const auto [o, finalState, stateCheckpoints]
        = graph.gdn(in.q, in.k, in.v, in.g, in.beta, in.cuSeqlens, GdnAttributes());
    ASSERT_NE(o, nullptr);
    EXPECT_EQ(finalState, nullptr);
    EXPECT_EQ(stateCheckpoints, nullptr);
    o->set_output(true);

    const auto result = graph.validate();
    ASSERT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(o->get_dim(), (std::vector<int64_t>{4, 2, 2}));
}

TEST(TestGraphGdn, BuildsAllOutputs)
{
    Graph graph;
    configureGraph(graph);
    const auto in = makeInputs();

    GdnAttributes attributes;
    attributes.set_name("gdn")
        .set_initial_state(makeTensor({2, 2, 2, 4}))
        .set_output_final_state(true)
        .set_checkpoint_every_n_tokens(2)
        .set_use_qk_l2norm(true)
        .set_scale(0.5f);

    const auto [o, finalState, stateCheckpoints]
        = graph.gdn(in.q, in.k, in.v, in.g, in.beta, in.cuSeqlens, attributes);
    ASSERT_NE(o, nullptr);
    ASSERT_NE(finalState, nullptr);
    ASSERT_NE(stateCheckpoints, nullptr);
    o->set_output(true);
    finalState->set_output(true);
    stateCheckpoints->set_output(true);

    const auto result = graph.validate();
    ASSERT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(finalState->get_dim(), (std::vector<int64_t>{2, 2, 2, 4}));
    // C = max(T / n + N, 1) = 4 / 2 + 2 = 4.
    EXPECT_EQ(stateCheckpoints->get_dim(), (std::vector<int64_t>{4, 2, 2, 4}));
    EXPECT_EQ(o->get_name(), "gdn::O");
    EXPECT_EQ(in.q->get_name(), "gdn::Q");
}

TEST(TestGraphGdn, ValidateFailsOnCrossFieldViolation)
{
    Graph graph;
    configureGraph(graph);
    const auto in = makeInputs();

    GdnAttributes attributes;
    attributes.set_safe_gate(true).set_gate_domain(GateDomain::LINEAR);

    const auto outputs = graph.gdn(in.q, in.k, in.v, in.g, in.beta, in.cuSeqlens, attributes);
    outputs[0]->set_output(true);

    const auto result = graph.validate();
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}
