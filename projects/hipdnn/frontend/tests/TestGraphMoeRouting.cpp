// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/MoeRoutingAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

std::shared_ptr<TensorAttributes> makeTopkIds()
{
    auto topkIds = std::make_shared<TensorAttributes>();
    topkIds->set_dim({4, 2}).set_stride({2, 1}).set_data_type(DataType::INT32);
    return topkIds;
}

} // namespace

TEST(TestGraphMoeRouting, BuildGraphInfersOutputs)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    MoeRoutingAttributes attributes;
    attributes.set_name("MoeRoutingNode");
    attributes.set_num_experts(4);

    auto [firstTokenOffset, tokenIndex, tokenKs] = graph.moe_routing(makeTopkIds(), attributes);
    ASSERT_NE(firstTokenOffset, nullptr);
    ASSERT_NE(tokenIndex, nullptr);
    ASSERT_NE(tokenKs, nullptr);
    EXPECT_EQ(firstTokenOffset->get_name(), "MoeRoutingNode::FIRST_TOKEN_OFFSET");
    EXPECT_EQ(tokenIndex->get_name(), "MoeRoutingNode::TOKEN_INDEX");
    EXPECT_EQ(tokenKs->get_name(), "MoeRoutingNode::TOKEN_KS");

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();

    EXPECT_EQ(firstTokenOffset->get_dim(), (std::vector<int64_t>{5, 1, 1}));
    EXPECT_EQ(firstTokenOffset->get_data_type(), DataType::INT32);
    EXPECT_EQ(tokenIndex->get_dim(), (std::vector<int64_t>{1, 8, 1}));
    EXPECT_EQ(tokenIndex->get_data_type(), DataType::INT32);
    EXPECT_EQ(tokenKs->get_dim(), (std::vector<int64_t>{1, 8, 1}));
    EXPECT_EQ(tokenKs->get_data_type(), DataType::INT32);
}

TEST(TestGraphMoeRouting, DefaultNameAssigned)
{
    Graph graph;
    auto topkIds = makeTopkIds();
    MoeRoutingAttributes attributes;
    attributes.set_num_experts(2);

    auto outputs = graph.moe_routing(topkIds, attributes);
    EXPECT_EQ(topkIds->get_name(), "MoeRouting_0::TOPK_IDS");
    EXPECT_EQ(outputs[0]->get_name(), "MoeRouting_0::FIRST_TOKEN_OFFSET");
}

TEST(TestGraphMoeRouting, ValidateFailsForSingleExpert)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT).set_compute_data_type(DataType::FLOAT);
    MoeRoutingAttributes attributes;
    attributes.set_num_experts(1);
    graph.moe_routing(makeTopkIds(), attributes);
    EXPECT_FALSE(graph.validate().is_good());
}
