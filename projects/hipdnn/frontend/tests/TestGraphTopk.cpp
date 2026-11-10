// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/attributes/TopkAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

std::shared_ptr<TensorAttributes> makeX()
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 8}).set_stride({24, 8, 1}).set_data_type(DataType::HALF);
    return x;
}

Graph makeGraph()
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);
    return graph;
}

} // namespace

TEST(TestGraphTopk, BuildGraphInfersOutputs)
{
    auto graph = makeGraph();

    TopkAttributes attributes;
    attributes.set_name("TopkNode").set_k(4).set_largest(false);

    auto [values, indices] = graph.topk(makeX(), attributes);

    ASSERT_NE(values, nullptr);
    ASSERT_NE(indices, nullptr);
    EXPECT_EQ(values->get_name(), "TopkNode::VALUES");
    EXPECT_EQ(indices->get_name(), "TopkNode::INDICES");
    EXPECT_TRUE(values->get_is_virtual());
    EXPECT_TRUE(indices->get_is_virtual());
    values->set_output(true);
    indices->set_output(true);

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();

    EXPECT_EQ(values->get_dim(), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(values->get_stride(), (std::vector<int64_t>{12, 4, 1}));
    EXPECT_EQ(values->get_data_type(), DataType::HALF);
    EXPECT_EQ(indices->get_dim(), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(indices->get_stride(), (std::vector<int64_t>{12, 4, 1}));
    EXPECT_EQ(indices->get_data_type(), DataType::INT64);
}

TEST(TestGraphTopk, ExplicitInt32IndicesKept)
{
    auto graph = makeGraph();

    TopkAttributes attributes;
    attributes.set_k(1);

    auto [values, indices] = graph.topk(makeX(), attributes);
    indices->set_data_type(DataType::INT32).set_output(true);
    values->set_output(true);

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();
    EXPECT_EQ(indices->get_data_type(), DataType::INT32);
    EXPECT_EQ(indices->get_dim(), (std::vector<int64_t>{2, 3, 1}));
}

TEST(TestGraphTopk, AutoNamesNodeAndInput)
{
    auto graph = makeGraph();
    auto x = makeX();

    TopkAttributes attributes;
    attributes.set_k(2);

    auto [values, indices] = graph.topk(x, attributes);

    EXPECT_EQ(x->get_name(), "Topk_0::X");
    EXPECT_EQ(values->get_name(), "Topk_0::VALUES");
    EXPECT_EQ(indices->get_name(), "Topk_0::INDICES");
}

TEST(TestGraphTopk, KOutOfRangeFailsValidation)
{
    auto graph = makeGraph();

    TopkAttributes attributes;
    attributes.set_k(9);

    auto [values, indices] = graph.topk(makeX(), attributes);
    values->set_output(true);
    indices->set_output(true);

    EXPECT_FALSE(graph.validate().is_good());
}
