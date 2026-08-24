// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestGraphLayernormBackward, BuildGraph)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create attributes
    LayernormBackwardAttributes attributes;
    attributes.set_name("LayernormBackwardNode");

    // Create input tensors
    auto dy = std::make_shared<TensorAttributes>();
    dy->set_dim({16, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({16, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({16, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(DataType::FLOAT);
    attributes.set_mean(std::move(mean));

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_dim({16, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(DataType::FLOAT);
    attributes.set_inv_variance(std::move(invVariance));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1}).set_stride({1}).set_data_type(DataType::FLOAT);
    attributes.set_epsilon(std::move(epsilon));

    // Call graph method
    auto results = graph.layernorm_backward(dy, x, scale, attributes);

    // Verify returned dx tensor is non-null
    ASSERT_NE(results[0], nullptr);
    EXPECT_EQ(results[0]->get_name(), "LayernormBackwardNode::DX");
    EXPECT_TRUE(results[0]->get_is_virtual());

    // Verify returned dscale tensor is non-null
    ASSERT_NE(results[1], nullptr);
    EXPECT_EQ(results[1]->get_name(), "LayernormBackwardNode::DSCALE");
    EXPECT_TRUE(results[1]->get_is_virtual());

    // Verify returned dbias tensor is non-null
    ASSERT_NE(results[2], nullptr);
    EXPECT_EQ(results[2]->get_name(), "LayernormBackwardNode::DBIAS");
    EXPECT_TRUE(results[2]->get_is_virtual());

    // Verify graph validates successfully
    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}
