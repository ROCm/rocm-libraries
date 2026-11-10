// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/CausalConv1dAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

std::shared_ptr<TensorAttributes> makeInput(const std::vector<int64_t>& dims, DataType dataType)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim(dims).set_stride(hipdnn_data_sdk::utilities::generateStrides(dims));
    tensor->set_data_type(dataType);
    return tensor;
}

void setGraphTypes(Graph& graph)
{
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);
}

} // namespace

TEST(TestGraphCausalConv1d, BuildGraph)
{
    Graph graph;
    setGraphTypes(graph);

    auto x = makeInput({8, 4}, DataType::FLOAT);
    auto weight = makeInput({4, 4}, DataType::FLOAT);
    auto cuSeqlens = makeInput({3}, DataType::INT32);

    CausalConv1dAttributes attributes;
    attributes.set_name("CausalConv1dNode");
    attributes.set_activation(CausalConv1dActivation::SILU);

    auto [y, finalState] = graph.causal_conv1d(x, weight, cuSeqlens, attributes);

    ASSERT_NE(y, nullptr);
    EXPECT_EQ(y->get_name(), "CausalConv1dNode::Y");
    EXPECT_TRUE(y->get_is_virtual());
    EXPECT_EQ(finalState, nullptr);
    y->set_output(true);

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();
    EXPECT_EQ(y->get_dim(), (std::vector<int64_t>{8, 4}));
    EXPECT_EQ(y->get_stride(), (std::vector<int64_t>{4, 1}));
}

TEST(TestGraphCausalConv1d, BuildGraphWithStatePool)
{
    Graph graph;
    setGraphTypes(graph);

    auto x = makeInput({2, 4}, DataType::FLOAT);
    auto weight = makeInput({4, 3}, DataType::FLOAT);
    auto cuSeqlens = makeInput({3}, DataType::INT64);
    auto bias = makeInput({4}, DataType::FLOAT);
    auto initialState = makeInput({6, 4, 2}, DataType::FLOAT);
    auto stateIndices = makeInput({2}, DataType::INT32);

    CausalConv1dAttributes attributes;
    attributes.set_bias(bias)
        .set_initial_state(initialState)
        .set_state_indices(stateIndices)
        .set_output_final_state(true);

    auto [y, finalState] = graph.causal_conv1d(x, weight, cuSeqlens, attributes);

    ASSERT_NE(y, nullptr);
    ASSERT_NE(finalState, nullptr);
    EXPECT_EQ(finalState->get_name(), "CausalConv1d_0::FINAL_STATE");
    EXPECT_EQ(bias->get_name(), "CausalConv1d_0::BIAS");
    EXPECT_EQ(initialState->get_name(), "CausalConv1d_0::INITIAL_STATE");
    EXPECT_EQ(stateIndices->get_name(), "CausalConv1d_0::STATE_INDICES");
    y->set_output(true);
    finalState->set_output(true);

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();
    EXPECT_EQ(finalState->get_dim(), (std::vector<int64_t>{6, 4, 2}));
    EXPECT_EQ(finalState->get_stride(), (std::vector<int64_t>{8, 2, 1}));
}

TEST(TestGraphCausalConv1d, ValidateFailsStateIndicesWithoutInitialState)
{
    Graph graph;
    setGraphTypes(graph);

    auto x = makeInput({8, 4}, DataType::FLOAT);
    auto weight = makeInput({4, 4}, DataType::FLOAT);
    auto cuSeqlens = makeInput({3}, DataType::INT32);

    CausalConv1dAttributes attributes;
    attributes.set_state_indices(makeInput({2}, DataType::INT32));

    auto [y, finalState] = graph.causal_conv1d(x, weight, cuSeqlens, attributes);
    y->set_output(true);

    EXPECT_TRUE(graph.validate().is_bad());
}
