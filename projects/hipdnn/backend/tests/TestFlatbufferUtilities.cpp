// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FlatbufferUtilities.hpp"
#include "HipdnnException.hpp"
#include "descriptors/FlatbufferTestUtils.hpp"
#include "descriptors/TestMacros.hpp"
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_backend
{
namespace testing
{

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_backend::test_utilities;

class TestFlatbufferUtilities : public ::testing::Test
{
public:
    static void verifyGraph(const hipdnn_flatbuffers_sdk::data_objects::GraphT& graph)
    {
        EXPECT_EQ(graph.name, "test");
        EXPECT_EQ(graph.compute_data_type, DataType::FLOAT);
        EXPECT_EQ(graph.intermediate_data_type, DataType::HALF);
        EXPECT_EQ(graph.io_data_type, DataType::BFLOAT16);
        EXPECT_EQ(graph.tensors.size(), 3);
        EXPECT_EQ(graph.nodes.size(), 1);
    }
};

TEST_F(TestFlatbufferUtilities, WillCorrectlyUnpackValidGraphBuffer)
{
    auto builder = createValidGraph();

    auto serializedGraph = builder.Release();
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT> graph;
    ASSERT_NO_THROW(flatbuffer_utilities::convertSerializedGraphToGraph(
        serializedGraph.data(), serializedGraph.size(), graph));

    verifyGraph(*graph);
}

TEST_F(TestFlatbufferUtilities, WillStillHaveValidGraphAfterBuilderDestructs)
{
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT> graph;
    {
        auto builder = createValidGraph();

        auto serializedGraph = builder.Release();
        ASSERT_NO_THROW(flatbuffer_utilities::convertSerializedGraphToGraph(
            serializedGraph.data(), serializedGraph.size(), graph));
    }

    verifyGraph(*graph);
}

TEST_F(TestFlatbufferUtilities, WillNotUnpackNullBuffer)
{
    auto [buffer, size] = std::make_pair(static_cast<const uint8_t*>(nullptr), size_t(10));

    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT> graph;
    ASSERT_THROW_HIPDNN_STATUS(
        flatbuffer_utilities::convertSerializedGraphToGraph(buffer, size, graph),
        HIPDNN_STATUS_BAD_PARAM);
    ASSERT_EQ(graph, nullptr);
}

TEST_F(TestFlatbufferUtilities, WillNotUnpackInvalidBuffer)
{
    auto arr = std::array<uint8_t, 10>{0};
    auto [buffer, size] = std::make_pair(arr.data(), size_t(10));

    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT> graph;
    ASSERT_THROW_HIPDNN_STATUS(
        flatbuffer_utilities::convertSerializedGraphToGraph(buffer, size, graph),
        HIPDNN_STATUS_BAD_PARAM);
    ASSERT_EQ(graph, nullptr);
}

TEST_F(TestFlatbufferUtilities, WillNotUnpackWrongSizeBuffer)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();
    auto [buffer, size] = std::make_pair(serializedGraph.data(), serializedGraph.size() - 20);

    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT> graph;
    ASSERT_THROW_HIPDNN_STATUS(
        flatbuffer_utilities::convertSerializedGraphToGraph(buffer, size, graph),
        HIPDNN_STATUS_BAD_PARAM);
    ASSERT_EQ(graph, nullptr);
}

} // namespace testing
} // namespace hipdnn_backend
