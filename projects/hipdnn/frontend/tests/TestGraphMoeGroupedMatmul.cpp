// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestGraphMoeGroupedMatmul, BuildGraph)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create input tensors
    auto token = std::make_shared<TensorAttributes>();
    token->set_dim({1, 8, 16}).set_stride({128, 16, 1}).set_data_type(DataType::FLOAT);

    auto weight = std::make_shared<TensorAttributes>();
    weight->set_dim({2, 16, 32}).set_stride({512, 32, 1}).set_data_type(DataType::FLOAT);

    auto firstTokenOffset = std::make_shared<TensorAttributes>();
    firstTokenOffset->set_dim({2, 1, 1}).set_stride({1, 1, 1}).set_data_type(DataType::INT32);

    auto tokenIndex = std::make_shared<TensorAttributes>();
    tokenIndex->set_dim({1, 8, 1}).set_stride({8, 1, 1}).set_data_type(DataType::INT32);

    auto tokenKs = std::make_shared<TensorAttributes>();
    tokenKs->set_dim({1, 8, 1}).set_stride({8, 1, 1}).set_data_type(DataType::INT32);

    // Create attributes
    MoeGroupedMatmulAttributes attributes;
    attributes.set_name("MoeGroupedMatmulNode");
    attributes.set_mode(MoeGroupedMatmulMode::SCATTER);
    attributes.set_top_k(2);

    // Call graph method
    auto output = graph.moe_grouped_matmul(
        token, weight, firstTokenOffset, tokenIndex, tokenKs, attributes);

    // Verify returned tensor is non-null
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->get_name(), "MoeGroupedMatmulNode::OUTPUT");
    EXPECT_TRUE(output->get_is_virtual());

    // Verify graph validates successfully
    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}
