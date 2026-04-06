// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FlatbufferTestUtils.hpp"
#include "FlatbufferUtilities.hpp"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "hipdnn_backend.h"

#include <cstring>
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <nlohmann/json.hpp>
#include <vector>

using namespace hipdnn_backend;

class TestGraphDescriptor : public ::testing::Test
{
public:
    static flatbuffers::FlatBufferBuilder createValidGraph()
    {
        return test_utilities::createValidGraph();
    }

    static void verifyGraph(const hipdnn_data_sdk::data_objects::GraphT& graph)
    {
        EXPECT_EQ(graph.name, "test");
        EXPECT_EQ(graph.compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
        EXPECT_EQ(graph.intermediate_data_type, hipdnn_data_sdk::data_objects::DataType::HALF);
        EXPECT_EQ(graph.io_data_type, hipdnn_data_sdk::data_objects::DataType::BFLOAT16);
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

    auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                            HIPDNN_TYPE_HANDLE,
                            1,
                            static_cast<const void*>(&handle));
    descriptor.finalize();

    auto output = descriptor.getSerializedGraph();
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(output.ptr), output.size);
    ASSERT_TRUE(verifier.VerifyBuffer<hipdnn_data_sdk::data_objects::Graph>());
}

TEST_F(TestGraphDescriptor, WillCorrectlySetGraph)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    ASSERT_NO_THROW(descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size()));

    ASSERT_THROW_HIPDNN_STATUS(descriptor.finalize(), HIPDNN_STATUS_BAD_PARAM);

    auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    ASSERT_NO_THROW(descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                            HIPDNN_TYPE_HANDLE,
                                            1,
                                            static_cast<const void*>(&handle)));
    ASSERT_NO_THROW(descriptor.finalize());
}

TEST_F(TestGraphDescriptor, WillCorrectlySetGraphReverseOrder)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    ASSERT_NO_THROW(descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                            HIPDNN_TYPE_HANDLE,
                                            1,
                                            static_cast<const void*>(&handle)));

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

TEST_F(TestGraphDescriptor, GetAttributeWorksOnDeserializedUnfinalizedGraph)
{
    auto builder = createValidGraph();
    const auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    // Querying OPS on a FlatBuffer-flow graph with no nodes returns count 0
    // (lazy unpack finds zero nodes in _graph and produces an empty _operations)
    int64_t elementCount = -1;
    ASSERT_NO_THROW(descriptor.getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, 0);

    // Query compute data type without finalization - should succeed
    int64_t computeCount = 0;
    hipdnnDataType_t computeDt = HIPDNN_DATA_HALF;
    ASSERT_NO_THROW(descriptor.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT,
                                            HIPDNN_TYPE_DATA_TYPE,
                                            1,
                                            &computeCount,
                                            &computeDt));
    EXPECT_EQ(computeDt, HIPDNN_DATA_FLOAT);
}

TEST_F(TestGraphDescriptor, GetAttributeUnsupportedReturnsNotSupported)
{
    auto builder = createValidGraph();
    const auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());
    const auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                            HIPDNN_TYPE_HANDLE,
                            1,
                            static_cast<const void*>(&handle));
    descriptor.finalize();

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

TEST_F(TestGraphDescriptor, GetSerializedGraphWithoutFinalization)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    // getSerializedGraph should work without finalization
    auto output = descriptor.getSerializedGraph();
    EXPECT_GT(output.size, 0u);
    EXPECT_NE(output.ptr, nullptr);
}

TEST_F(TestGraphDescriptor, GetSerializedGraphEmptyOperationsThrows)
{
    const GraphDescriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(descriptor.getSerializedGraph(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptor, JsonRoundTrip)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    // Deserialize from binary into first descriptor
    GraphDescriptor descriptor1;
    descriptor1.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    // Serialize to JSON
    auto jsonStr = descriptor1.getSerializedJsonGraph();
    ASSERT_FALSE(jsonStr.empty());

    // Verify JSON is valid
    auto parsed = nlohmann::json::parse(jsonStr);
    EXPECT_TRUE(parsed.contains("name"));
    EXPECT_TRUE(parsed.contains("compute_data_type"));
    EXPECT_TRUE(parsed.contains("nodes"));
    EXPECT_TRUE(parsed.contains("tensors"));

    // Round-trip: create a new descriptor from the JSON
    GraphDescriptor descriptor2;
    ASSERT_NO_THROW(
        GraphDescriptor::createFromJsonGraph(descriptor2, jsonStr.c_str(), jsonStr.size()));

    // Both descriptors should produce equivalent serialized graphs
    auto binary1 = descriptor1.getSerializedGraph();
    auto binary2 = descriptor2.getSerializedGraph();

    auto graph1
        = hipdnn_data_sdk::data_objects::UnPackGraph(static_cast<const uint8_t*>(binary1.ptr));
    auto graph2
        = hipdnn_data_sdk::data_objects::UnPackGraph(static_cast<const uint8_t*>(binary2.ptr));

    EXPECT_EQ(graph1->name, graph2->name);
    EXPECT_EQ(graph1->compute_data_type, graph2->compute_data_type);
    EXPECT_EQ(graph1->intermediate_data_type, graph2->intermediate_data_type);
    EXPECT_EQ(graph1->io_data_type, graph2->io_data_type);
    EXPECT_EQ(graph1->tensors.size(), graph2->tensors.size());
    EXPECT_EQ(graph1->nodes.size(), graph2->nodes.size());
}

// ============================================================================
// JSON C API error-path tests
// ============================================================================

TEST_F(TestGraphDescriptor, JsonSerializeNullDescriptor)
{
    size_t size = 0;
    auto status = hipdnnBackendGetSerializedJsonGraph_ext(nullptr, 0, &size, nullptr);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphDescriptor, JsonSerializeNullSize)
{
    // Create a valid descriptor through the binary C API
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t desc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &desc, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(desc, nullptr);

    status = hipdnnBackendGetSerializedJsonGraph_ext(desc, 0, nullptr, nullptr);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(TestGraphDescriptor, JsonSerializeInsufficientBuffer)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t desc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &desc, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(desc, nullptr);

    // Query the required size
    size_t requiredSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(desc, 0, &requiredSize, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(requiredSize, 0u);

    // Provide a buffer that is 1 byte too small
    std::vector<char> buffer(requiredSize - 1);
    size_t returnedSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(
        desc, buffer.size(), &returnedSize, buffer.data());
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(TestGraphDescriptor, JsonSerializeOversizedBuffer)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t desc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &desc, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(desc, nullptr);

    // Query the required size
    size_t requiredSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(desc, 0, &requiredSize, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(requiredSize, 0u);

    // Provide an oversized buffer (larger than required)
    std::vector<char> buffer(requiredSize + 64);
    size_t returnedSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(
        desc, buffer.size(), &returnedSize, buffer.data());
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(returnedSize, requiredSize);

    // Verify the JSON content is valid
    auto parsed = nlohmann::json::parse(buffer.data());
    EXPECT_TRUE(parsed.contains("name"));

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(TestGraphDescriptor, JsonDeserializeNullInput)
{
    hipdnnBackendDescriptor_t desc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeJsonGraph_ext(&desc, nullptr, 0);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphDescriptor, JsonDeserializeEmptySize)
{
    hipdnnBackendDescriptor_t desc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeJsonGraph_ext(&desc, "{}", 0);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptor, JsonRoundTripViaApi)
{
    // Create a valid graph descriptor via the binary C API
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t desc1 = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &desc1, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(desc1, nullptr);

    // Serialize to JSON via C API: first query the size
    size_t jsonSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(desc1, 0, &jsonSize, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(jsonSize, 0u);

    // Then retrieve the JSON data
    std::vector<char> jsonBuffer(jsonSize);
    size_t returnedSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(
        desc1, jsonBuffer.size(), &returnedSize, jsonBuffer.data());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Deserialize from JSON via C API
    hipdnnBackendDescriptor_t desc2 = nullptr;
    status = hipdnnBackendCreateAndDeserializeJsonGraph_ext(
        &desc2, jsonBuffer.data(), std::strlen(jsonBuffer.data()));
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(desc2, nullptr);

    // Verify the round-tripped graph matches by extracting binary from both
    auto graphDesc1 = desc1->asDescriptor<GraphDescriptor>();
    auto graphDesc2 = desc2->asDescriptor<GraphDescriptor>();

    auto binary1 = graphDesc1->getSerializedGraph();
    auto binary2 = graphDesc2->getSerializedGraph();

    auto graph1
        = hipdnn_data_sdk::data_objects::UnPackGraph(static_cast<const uint8_t*>(binary1.ptr));
    auto graph2
        = hipdnn_data_sdk::data_objects::UnPackGraph(static_cast<const uint8_t*>(binary2.ptr));

    EXPECT_EQ(graph1->name, graph2->name);
    EXPECT_EQ(graph1->compute_data_type, graph2->compute_data_type);
    EXPECT_EQ(graph1->intermediate_data_type, graph2->intermediate_data_type);
    EXPECT_EQ(graph1->io_data_type, graph2->io_data_type);
    EXPECT_EQ(graph1->tensors.size(), graph2->tensors.size());
    EXPECT_EQ(graph1->nodes.size(), graph2->nodes.size());

    hipdnnBackendDestroyDescriptor(desc1);
    hipdnnBackendDestroyDescriptor(desc2);
}

TEST_F(TestGraphDescriptor, JsonCacheInvalidatedOnDeserialize)
{
    // Build graph A with name "graphA"
    flatbuffers::FlatBufferBuilder builderA;
    const std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        emptyTensorsA;
    const std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> emptyNodesA;
    auto graphA = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builderA,
        "graphA",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &emptyTensorsA,
        &emptyNodesA);
    builderA.Finish(graphA);
    auto serializedA = builderA.Release();

    // Deserialize graph A and populate the JSON cache
    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedA.data(), serializedA.size());
    auto jsonA = descriptor.getSerializedJsonGraph();
    ASSERT_FALSE(jsonA.empty());

    // Build graph B with name "graphB"
    flatbuffers::FlatBufferBuilder builderB;
    const std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        emptyTensorsB;
    const std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> emptyNodesB;
    auto graphB = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builderB,
        "graphB",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &emptyTensorsB,
        &emptyNodesB);
    builderB.Finish(graphB);
    auto serializedB = builderB.Release();

    // Deserialize graph B (should invalidate the JSON cache)
    descriptor.deserializeGraph(serializedB.data(), serializedB.size());
    auto jsonB = descriptor.getSerializedJsonGraph();

    // Verify the JSON reflects graph B, not the stale graph A
    auto parsed = nlohmann::json::parse(jsonB);
    EXPECT_EQ(parsed["name"], "graphB");
}
