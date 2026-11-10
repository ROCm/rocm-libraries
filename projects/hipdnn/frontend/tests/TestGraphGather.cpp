// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/GatherAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestGraphGather, BuildGraph)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create input tensors
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({16, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    auto indices = std::make_shared<TensorAttributes>();
    indices->set_dim({2, 3}).set_stride({3, 1}).set_data_type(DataType::INT32);

    // Create attributes
    GatherAttributes attributes;
    attributes.set_name("GatherNode");
    attributes.set_axis(0);

    // Call graph method
    auto y = graph.gather(x, indices, attributes);

    // Verify returned tensor is non-null
    ASSERT_NE(y, nullptr);
    EXPECT_EQ(y->get_name(), "GatherNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    // Verify graph validates successfully
    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST(TestGraphGather, EmbeddingLookupInfersShapeAndDataType)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::BFLOAT16)
        .set_intermediate_data_type(DataType::FLOAT);

    auto table = std::make_shared<TensorAttributes>();
    table->set_dim({32, 4}).set_stride({4, 1}).set_data_type(DataType::BFLOAT16);
    auto ids = std::make_shared<TensorAttributes>();
    ids->set_dim({2, 5}).set_stride({5, 1}).set_data_type(DataType::INT64);

    auto y = graph.gather(table, ids, GatherAttributes{});
    y->set_output(true);

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();
    EXPECT_EQ(y->get_dim(), (std::vector<int64_t>{2, 5, 4}));
    EXPECT_EQ(y->get_stride(), (std::vector<int64_t>{20, 4, 1}));
    EXPECT_EQ(y->get_data_type(), DataType::BFLOAT16);
    EXPECT_EQ(ids->get_name(), "Gather_0::INDICES");
}

TEST(TestGraphGather, ValidateRejectsOutOfRangeAxis)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT).set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({16, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
    auto indices = std::make_shared<TensorAttributes>();
    indices->set_dim({3}).set_stride({1}).set_data_type(DataType::INT32);

    GatherAttributes attributes;
    attributes.set_axis(2);
    graph.gather(x, indices, attributes);

    EXPECT_FALSE(graph.validate().is_good());
}

TEST(TestGraphGather, ValidateRejectsMissingIndices)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT).set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({16, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    graph.gather(x, nullptr, GatherAttributes{});

    EXPECT_FALSE(graph.validate().is_good());
}
