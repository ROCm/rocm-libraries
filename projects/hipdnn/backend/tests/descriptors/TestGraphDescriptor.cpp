// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FlatbufferTestUtils.hpp"
#include "FlatbufferUtilities.hpp"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "hipdnn_backend.h"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

using namespace hipdnn_backend;

class TestGraphDescriptor : public ::testing::Test
{
public:
    static flatbuffers::FlatBufferBuilder createValidGraph()
    {
        return hipdnn_sdk::test_utilities::createValidGraph();
    }

    static void verifyGraph(const hipdnn_sdk::data_objects::GraphT& graph)
    {
        EXPECT_EQ(graph.name, "test");
        EXPECT_EQ(graph.compute_type, hipdnn_sdk::data_objects::DataType::FLOAT);
        EXPECT_EQ(graph.intermediate_type, hipdnn_sdk::data_objects::DataType::HALF);
        EXPECT_EQ(graph.io_type, hipdnn_sdk::data_objects::DataType::BFLOAT16);
        EXPECT_EQ(graph.tensors.size(), 0);
        EXPECT_EQ(graph.nodes.size(), 0);
    }
};

TEST_F(TestGraphDescriptor, SerializeDeserializeGraph)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    auto output = descriptor.getSerializedGraph();
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(output.ptr), output.size);
    ASSERT_TRUE(verifier.VerifyBuffer<hipdnn_sdk::data_objects::Graph>());
}

TEST_F(TestGraphDescriptor, WillCorrectlySetGraph)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    ASSERT_NO_THROW(descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size()));

    ASSERT_THROW_HIPDNN_STATUS(descriptor.finalize(), HIPDNN_STATUS_BAD_PARAM);

    auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    ASSERT_NO_THROW(
        descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle));
    ASSERT_NO_THROW(descriptor.finalize());
}

TEST_F(TestGraphDescriptor, WillCorrectlySetGraphReverseOrder)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    ASSERT_NO_THROW(
        descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle));

    ASSERT_THROW_HIPDNN_STATUS(descriptor.finalize(), HIPDNN_STATUS_BAD_PARAM);

    ASSERT_NO_THROW(descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size()));
    ASSERT_NO_THROW(descriptor.finalize());
}

TEST_F(TestGraphDescriptor, WillFailToSetInvalidGraph)
{
    GraphDescriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(descriptor.deserializeGraph(nullptr, 0),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphDescriptor, FinalizeFailInvalidGraph)
{
    GraphDescriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(descriptor.finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptor, GetAttributeReturnsNotSupported)
{
    GraphDescriptor descriptor;
    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        descriptor.getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestGraphDescriptor, SetAttributeReturnsNotSupported)
{
    GraphDescriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(
        descriptor.setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_DATA_TYPE, 0, nullptr),
        HIPDNN_STATUS_NOT_SUPPORTED);
}
