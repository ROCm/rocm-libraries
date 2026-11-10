// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/RopeAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestGraphRope, BuildGraph)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create input tensors
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 4, 16, 8}).set_stride({512, 128, 8, 1}).set_data_type(DataType::FLOAT);

    auto freqs = std::make_shared<TensorAttributes>();
    freqs->set_dim({1, 1, 16, 4}).set_stride({64, 64, 4, 1}).set_data_type(DataType::FLOAT);

    // Create attributes
    RopeAttributes attributes;
    attributes.set_name("RopeNode");
    attributes.set_output_scale(0.5);
    attributes.set_rope_dim(4);
    attributes.set_rope_offset(0);

    // Call graph method
    auto y = graph.rope(x, freqs, attributes);

    // Verify returned tensor is non-null
    ASSERT_NE(y, nullptr);
    EXPECT_EQ(y->get_name(), "RopeNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    // Verify graph validates successfully
    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST(TestGraphRope, InfersOutputDimsFromX)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::HALF)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 8}).set_stride({48, 24, 8, 1}).set_data_type(DataType::HALF);
    auto freqs = std::make_shared<TensorAttributes>();
    freqs->set_dim({1, 1, 3, 8}).set_stride({24, 24, 8, 1}).set_data_type(DataType::FLOAT);

    auto y = graph.rope(x, freqs, RopeAttributes{});
    y->set_output(true);

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();
    EXPECT_EQ(y->get_dim(), x->get_dim());
    EXPECT_EQ(y->get_stride(), x->get_stride());
    EXPECT_EQ(y->get_name(), "Rope_0::Y");
    EXPECT_EQ(x->get_name(), "Rope_0::X");
    EXPECT_EQ(freqs->get_name(), "Rope_0::FREQS");
}

TEST(TestGraphRope, ValidateFailsForInvalidRopeDim)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 8}).set_stride({48, 24, 8, 1}).set_data_type(DataType::FLOAT);
    auto freqs = std::make_shared<TensorAttributes>();
    freqs->set_dim({1, 1, 3, 6}).set_stride({18, 18, 6, 1}).set_data_type(DataType::FLOAT);

    RopeAttributes attributes;
    attributes.set_rope_dim(6).set_rope_offset(4);
    auto y = graph.rope(x, freqs, attributes);
    y->set_output(true);

    EXPECT_FALSE(graph.validate().is_good());
}
