// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulBwdAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestGraphMoeGroupedMatmulBwd, BuildGraph)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create input tensors
    auto dOutput = std::make_shared<TensorAttributes>();
    dOutput->set_dim({1, 8, 32}).set_stride({256, 32, 1}).set_data_type(DataType::FLOAT);

    auto token = std::make_shared<TensorAttributes>();
    token->set_dim({1, 8, 16}).set_stride({128, 16, 1}).set_data_type(DataType::FLOAT);

    auto firstTokenOffset = std::make_shared<TensorAttributes>();
    firstTokenOffset->set_dim({2, 1, 1}).set_stride({1, 1, 1}).set_data_type(DataType::INT32);

    // Create attributes
    MoeGroupedMatmulBwdAttributes attributes;
    attributes.set_name("MoeGroupedMatmulBwdNode");

    // Call graph method
    auto dweight = graph.moe_grouped_matmul_bwd(dOutput, token, firstTokenOffset, attributes);

    // dweight's expert count cannot be derived from any input tensor, so it must be
    // supplied by the caller (see MoeGroupedMatmulBwdNode::infer_properties_node).
    dweight->set_dim({2, 16, 32});

    // Verify returned tensor is non-null
    ASSERT_NE(dweight, nullptr);
    EXPECT_EQ(dweight->get_name(), "MoeGroupedMatmulBwdNode::DWEIGHT");
    EXPECT_TRUE(dweight->get_is_virtual());

    // Verify graph validates successfully
    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}
