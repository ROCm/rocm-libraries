// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SoftmaxAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

Graph makeGraph()
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);
    return graph;
}

std::shared_ptr<TensorAttributes> makeP()
{
    auto p = std::make_shared<TensorAttributes>();
    p->set_dim({2, 4, 8, 16}).set_stride({512, 128, 16, 1}).set_data_type(DataType::FLOAT);
    return p;
}

} // namespace

TEST(TestGraphSoftmax, BuildGraphWithoutStats)
{
    auto graph = makeGraph();

    SoftmaxAttributes attributes;
    attributes.set_name("SoftmaxNode");

    auto [s, stats] = graph.softmax(makeP(), attributes);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(stats, nullptr);

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();

    EXPECT_EQ(s->get_dim(), (std::vector<int64_t>{2, 4, 8, 16}));
    EXPECT_EQ(s->get_stride(), (std::vector<int64_t>{512, 128, 16, 1}));
}

TEST(TestGraphSoftmax, BuildGraphWithStatsInfersStatsShape)
{
    auto graph = makeGraph();

    SoftmaxAttributes attributes;
    attributes.set_generate_stats(true).set_stats_use_log2(true);

    auto [s, stats] = graph.softmax(makeP(), attributes);
    ASSERT_NE(s, nullptr);
    ASSERT_NE(stats, nullptr);
    s->set_output(true);
    stats->set_output(true);

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();

    EXPECT_EQ(stats->get_dim(), (std::vector<int64_t>{2, 4, 8, 1}));
    EXPECT_EQ(stats->get_stride(), (std::vector<int64_t>{32, 8, 1, 1}));
}

TEST(TestGraphSoftmax, AutoNamesNodeAndTensors)
{
    auto graph = makeGraph();
    auto p = makeP();

    SoftmaxAttributes attributes;
    attributes.set_generate_stats(true);
    auto [s, stats] = graph.softmax(p, attributes);

    EXPECT_EQ(p->get_name(), "Softmax_0::P");
    EXPECT_EQ(s->get_name(), "Softmax_0::S");
    ASSERT_NE(stats, nullptr);
    EXPECT_EQ(stats->get_name(), "Softmax_0::STATS");
}

TEST(TestGraphSoftmax, ValidateRejectsMismatchedSDims)
{
    auto graph = makeGraph();

    auto [s, stats] = graph.softmax(makeP(), SoftmaxAttributes{});
    s->set_dim({2, 4, 8, 8});

    EXPECT_TRUE(graph.validate().is_bad());
}

TEST(TestGraphSoftmax, ValidateRejectsMismatchedStatsDims)
{
    auto graph = makeGraph();

    SoftmaxAttributes attributes;
    attributes.set_generate_stats(true);
    auto [s, stats] = graph.softmax(makeP(), attributes);
    ASSERT_NE(stats, nullptr);
    stats->set_dim({2, 4, 8, 16});

    EXPECT_TRUE(graph.validate().is_bad());
}
