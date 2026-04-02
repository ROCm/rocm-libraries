// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestGraphRMSNormBackward, BuildGraph)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create input tensors
    auto dy = std::make_shared<TensorAttributes>();
    dy->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 1, 1}).set_stride({64, 1, 1, 1}).set_data_type(DataType::FLOAT);

    // Create attributes
    RMSNormBackwardAttributes attributes;
    attributes.set_name("RMSNormBackwardNode");

    // Call graph method
    auto [dx, dscale, dbias] = graph.rmsnorm_backward(dy, x, scale, attributes);

    EXPECT_EQ(dx->get_name(), "RMSNormBackwardNode::DX");
    EXPECT_TRUE(dx->get_is_virtual());

    EXPECT_EQ(dscale->get_name(), "RMSNormBackwardNode::DSCALE");
    EXPECT_TRUE(dscale->get_is_virtual());

    // Verify graph validates successfully
    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}
