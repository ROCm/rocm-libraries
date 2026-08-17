// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>

using namespace hipdnn_frontend;
using hipdnn_tests::buildConvFpropGraph;

namespace
{

// ============================================================================
// buildConvFpropGraph tests
// ============================================================================

TEST(TestBuildConvFpropGraph, ReturnsNonNullGraph)
{
    auto graph = buildConvFpropGraph();
    ASSERT_NE(graph, nullptr);
}

TEST(TestBuildConvFpropGraph, DefaultGraphName)
{
    auto graph = buildConvFpropGraph();
    EXPECT_EQ(graph->get_name(), "ConvFpropTestGraph");
}

TEST(TestBuildConvFpropGraph, CustomGraphName)
{
    auto graph = buildConvFpropGraph("MyCustomGraph");
    EXPECT_EQ(graph->get_name(), "MyCustomGraph");
}

TEST(TestBuildConvFpropGraph, DefaultDataTypes)
{
    auto graph = buildConvFpropGraph();
    EXPECT_EQ(graph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(graph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(graph->get_io_data_type(), DataType::FLOAT);
}

TEST(TestBuildConvFpropGraph, CustomDataTypes)
{
    auto graph = buildConvFpropGraph("Test", DataType::HALF, DataType::BFLOAT16, DataType::FLOAT);
    EXPECT_EQ(graph->get_compute_data_type(), DataType::HALF);
    EXPECT_EQ(graph->get_intermediate_data_type(), DataType::BFLOAT16);
    EXPECT_EQ(graph->get_io_data_type(), DataType::FLOAT);
}

TEST(TestBuildConvFpropGraph, GraphValidates)
{
    auto graph = buildConvFpropGraph();
    auto result = graph->validate();
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

TEST(TestBuildConvFpropGraph, GraphHasOneNode)
{
    auto graph = buildConvFpropGraph();
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK);

    const auto& nodes = graph->getSubNodes();
    EXPECT_EQ(nodes.size(), 1u);
}

// ============================================================================
// buildTensorMap tests
// ============================================================================

TEST(TestBuildTensorMap, EmptyGraph)
{
    const hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;
    auto tensorMap = hipdnn_tests::buildTensorMap(graphT);
    EXPECT_TRUE(tensorMap.empty());
}

TEST(TestBuildTensorMap, MapsUidsToTensors)
{
    hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;

    auto t1 = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>();
    t1->uid = 10;
    auto t2 = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>();
    t2->uid = 20;

    graphT.tensors.push_back(std::move(t1));
    graphT.tensors.push_back(std::move(t2));

    auto tensorMap = hipdnn_tests::buildTensorMap(graphT);
    ASSERT_EQ(tensorMap.size(), 2u);
    EXPECT_NE(tensorMap.count(10), 0u);
    EXPECT_NE(tensorMap.count(20), 0u);
    EXPECT_EQ(tensorMap.at(10)->uid, 10);
    EXPECT_EQ(tensorMap.at(20)->uid, 20);
}

} // namespace
