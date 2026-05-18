// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ResampleBwdAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestGraphResampleBwd, BuildGraph)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create input tensors
    auto dy = std::make_shared<TensorAttributes>();
    dy->set_dim({1, 3, 16, 16}).set_stride({768, 256, 16, 1}).set_data_type(DataType::FLOAT);

    // Create attributes
    ResampleBwdAttributes attributes;
    attributes.set_name("ResampleBwdNode");
    attributes.set_pre_padding({1, 1});
    attributes.set_post_padding({1, 1});
    attributes.set_stride({2, 2});
    attributes.set_window({3, 3});
    attributes.set_resample_mode(ResampleMode::MAXPOOL);
    attributes.set_padding_mode(PaddingMode::ZERO_PAD);

    // Call graph method
    auto dx = graph.resample_bwd(dy, attributes);

    // Verify returned tensor is non-null
    ASSERT_NE(dx, nullptr);
    EXPECT_EQ(dx->get_name(), "ResampleBwdNode::DX");
    EXPECT_TRUE(dx->get_is_virtual());

    // Verify graph validates successfully
    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}
