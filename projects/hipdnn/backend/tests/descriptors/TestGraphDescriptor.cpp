// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FlatbufferTestUtils.hpp"
#include "FlatbufferUtilities.hpp"
#include "HipdnnException.hpp"
#include "ScopedBackendDescriptor.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "hipdnn_backend.h"

#include <array>
#include <cstring>
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <nlohmann/json.hpp>
#include <vector>

using namespace hipdnn_backend;
using hipdnn_backend::test_utilities::ScopedBackendDescriptor;

class TestGraphDescriptor : public ::testing::Test
{
public:
    static flatbuffers::FlatBufferBuilder createValidGraph()
    {
        return test_utilities::createValidGraph();
    }

    static void verifyGraph(const hipdnn_flatbuffers_sdk::data_objects::GraphT& graph)
    {
        EXPECT_EQ(graph.name, "test");
        EXPECT_EQ(graph.compute_data_type, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
        EXPECT_EQ(graph.intermediate_data_type,
                  hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
        EXPECT_EQ(graph.io_data_type, hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
        EXPECT_EQ(graph.tensors.size(), 3);
        EXPECT_EQ(graph.nodes.size(), 1);
    }

    static void verifyGraphsEquivalent(const hipdnn_flatbuffers_sdk::data_objects::GraphT& graph1,
                                       const hipdnn_flatbuffers_sdk::data_objects::GraphT& graph2)
    {
        EXPECT_EQ(graph1.name, graph2.name);
        EXPECT_EQ(graph1.compute_data_type, graph2.compute_data_type);
        EXPECT_EQ(graph1.intermediate_data_type, graph2.intermediate_data_type);
        EXPECT_EQ(graph1.io_data_type, graph2.io_data_type);
        EXPECT_EQ(graph1.preferred_engine_id, graph2.preferred_engine_id);

        ASSERT_EQ(graph1.tensors.size(), graph2.tensors.size());
        for(size_t i = 0; i < graph1.tensors.size(); ++i)
        {
            SCOPED_TRACE("tensor[" + std::to_string(i) + "]");
            ASSERT_NE(graph1.tensors[i], nullptr);
            ASSERT_NE(graph2.tensors[i], nullptr);
            EXPECT_EQ(*graph1.tensors[i], *graph2.tensors[i]);
        }

        ASSERT_EQ(graph1.nodes.size(), graph2.nodes.size());
        for(size_t i = 0; i < graph1.nodes.size(); ++i)
        {
            SCOPED_TRACE("node[" + std::to_string(i) + "]");
            ASSERT_NE(graph1.nodes[i], nullptr);
            ASSERT_NE(graph2.nodes[i], nullptr);
            EXPECT_EQ(*graph1.nodes[i], *graph2.nodes[i]);
        }
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
    ASSERT_TRUE(verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::Graph>());

    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(
        static_cast<const uint8_t*>(output.ptr));
    ASSERT_NE(graph, nullptr);
    verifyGraph(*graph);
}

TEST_F(TestGraphDescriptor, DeserializeGraphExtractsAttributes)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    // Verify graph-level attributes were correctly extracted from the FlatBuffer
    hipdnnDataType_t computeType{};
    int64_t count = 0;
    ASSERT_NO_THROW(descriptor.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT,
                                            HIPDNN_TYPE_DATA_TYPE,
                                            1,
                                            &count,
                                            &computeType));
    EXPECT_EQ(computeType, HIPDNN_DATA_FLOAT);

    hipdnnDataType_t intermediateType{};
    count = 0;
    ASSERT_NO_THROW(descriptor.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                                            HIPDNN_TYPE_DATA_TYPE,
                                            1,
                                            &count,
                                            &intermediateType));
    EXPECT_EQ(intermediateType, HIPDNN_DATA_HALF);

    hipdnnDataType_t ioType{};
    count = 0;
    ASSERT_NO_THROW(descriptor.getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &count, &ioType));
    EXPECT_EQ(ioType, HIPDNN_DATA_BFLOAT16);

    std::array<char, 64> name{};
    count = 0;
    ASSERT_NO_THROW(descriptor.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                                            HIPDNN_TYPE_CHAR,
                                            static_cast<int64_t>(name.size()),
                                            &count,
                                            name.data()));
    EXPECT_STREQ(name.data(), "test");

    int64_t opsCount = 0;
    ASSERT_NO_THROW(descriptor.getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &opsCount, nullptr));
    EXPECT_EQ(opsCount, 1);
}

TEST_F(TestGraphDescriptor, WillCorrectlySetGraph)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    ASSERT_NO_THROW(descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size()));

    // Finalize requires a handle
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

    // Querying OPS on a deserialized graph returns the number of unpacked nodes
    int64_t elementCount = -1;
    ASSERT_NO_THROW(descriptor.getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, 1);

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

    // getAttribute with an unsupported attribute name does not require finalization
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

TEST_F(TestGraphDescriptor, EmptyGraphDeserializesButFailsToFinalize)
{
    auto builder = test_utilities::createEmptyGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    ASSERT_NO_THROW(descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size()));

    auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    ASSERT_NO_THROW(descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                            HIPDNN_TYPE_HANDLE,
                                            1,
                                            static_cast<const void*>(&handle)));

    // Finalize fails because the empty graph has no operations
    ASSERT_THROW_HIPDNN_STATUS(descriptor.finalize(), HIPDNN_STATUS_BAD_PARAM);

    // Serialization still works (not gated by finalize)
    descriptor.buildSerializedGraph();
    auto data = descriptor.getSerializedGraph();
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(data.ptr), data.size);
    ASSERT_TRUE(verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::Graph>());
}

TEST_F(TestGraphDescriptor, GetSerializedGraphWithoutPopulationThrows)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    // getSerializedGraph requires the buffer to be populated by finalize() or
    // buildSerializedGraph(). Without either call, the getter throws.
    ASSERT_THROW_HIPDNN_STATUS(descriptor.getSerializedGraph(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptor, GetSerializedGraphEmptyOperationsThrows)
{
    const GraphDescriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(descriptor.getSerializedGraph(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptor, JsonRoundTripViaDescriptorApi)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    // Deserialize from binary into first descriptor
    GraphDescriptor descriptor1;
    descriptor1.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    // Build the serialized buffer and get JSON
    descriptor1.buildSerializedGraph();
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
    descriptor2.buildSerializedGraph();
    auto binary1 = descriptor1.getSerializedGraph();
    auto binary2 = descriptor2.getSerializedGraph();

    auto graph1 = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(
        static_cast<const uint8_t*>(binary1.ptr));
    auto graph2 = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(
        static_cast<const uint8_t*>(binary2.ptr));

    verifyGraphsEquivalent(*graph1, *graph2);
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

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &rawDesc, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc, nullptr);
    const ScopedBackendDescriptor desc(rawDesc);

    status = hipdnnBackendGetSerializedJsonGraph_ext(desc.get(), 0, nullptr, nullptr);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphDescriptor, JsonSerializeInsufficientBuffer)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &rawDesc, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc, nullptr);
    const ScopedBackendDescriptor desc(rawDesc);

    // Query the required size
    size_t requiredSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(desc.get(), 0, &requiredSize, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(requiredSize, 0u);

    // Provide a buffer that is 1 byte too small
    std::vector<char> buffer(requiredSize - 1);
    size_t returnedSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(
        desc.get(), buffer.size(), &returnedSize, buffer.data());
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT);
}

TEST_F(TestGraphDescriptor, JsonSerializeOversizedBuffer)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &rawDesc, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc, nullptr);
    const ScopedBackendDescriptor desc(rawDesc);

    // Query the required size
    size_t requiredSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(desc.get(), 0, &requiredSize, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(requiredSize, 0u);

    // Provide an oversized buffer (larger than required)
    std::vector<char> buffer(requiredSize + 64);
    size_t returnedSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(
        desc.get(), buffer.size(), &returnedSize, buffer.data());
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(returnedSize, requiredSize);

    // Verify the JSON content is valid
    auto parsed = nlohmann::json::parse(buffer.data());
    EXPECT_TRUE(parsed.contains("name"));
}

TEST_F(TestGraphDescriptor, JsonDeserializeNullInput)
{
    hipdnnBackendDescriptor_t desc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeJsonGraph_ext(&desc, nullptr, 1);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphDescriptor, JsonDeserializeNullDescriptor)
{
    auto status = hipdnnBackendCreateAndDeserializeJsonGraph_ext(nullptr, "{}", 2);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphDescriptor, JsonDeserializeEmptySize)
{
    hipdnnBackendDescriptor_t desc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeJsonGraph_ext(&desc, "{}", 0);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptor, JsonDeserializeNonGraphJson)
{
    hipdnnBackendDescriptor_t desc = nullptr;
    const std::string json = R"({"foo": "bar", "baz": 42})";
    auto status
        = hipdnnBackendCreateAndDeserializeJsonGraph_ext(&desc, json.c_str(), json.size() + 1);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TestGraphDescriptor, JsonRoundTripViaApi)
{
    // Create a valid graph descriptor via the binary C API
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t rawDesc1 = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &rawDesc1, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc1, nullptr);
    const ScopedBackendDescriptor scopedDesc1(rawDesc1);

    // Serialize to JSON via C API: first query the size
    size_t jsonSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(scopedDesc1.get(), 0, &jsonSize, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(jsonSize, 0u);

    // Then retrieve the JSON data
    std::vector<char> jsonBuffer(jsonSize);
    size_t returnedSize = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(
        scopedDesc1.get(), jsonBuffer.size(), &returnedSize, jsonBuffer.data());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Deserialize from JSON via C API
    hipdnnBackendDescriptor_t rawDesc2 = nullptr;
    status = hipdnnBackendCreateAndDeserializeJsonGraph_ext(
        &rawDesc2, jsonBuffer.data(), std::strlen(jsonBuffer.data()));
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc2, nullptr);
    const ScopedBackendDescriptor scopedDesc2(rawDesc2);

    // Verify the round-tripped graph matches by extracting binary from both
    auto graphDesc1 = scopedDesc1.get()->asDescriptor<GraphDescriptor>();
    auto graphDesc2 = scopedDesc2.get()->asDescriptor<GraphDescriptor>();

    // desc1 buffer was populated by the GetSerializedJsonGraph call above.
    // desc2 needs explicit buildSerializedGraph() since it was only deserialized.
    graphDesc2->buildSerializedGraph();

    auto binary1 = graphDesc1->getSerializedGraph();
    auto binary2 = graphDesc2->getSerializedGraph();

    auto graph1 = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(
        static_cast<const uint8_t*>(binary1.ptr));
    auto graph2 = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(
        static_cast<const uint8_t*>(binary2.ptr));

    verifyGraphsEquivalent(*graph1, *graph2);
}

TEST_F(TestGraphDescriptor, DeserializeInvalidatesSerializedBuffer)
{
    // Build graph A with name "graphA"
    flatbuffers::FlatBufferBuilder builderA;
    const std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        emptyTensorsA;
    const std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> emptyNodesA;
    auto graphA = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builderA,
        "graphA",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &emptyTensorsA,
        &emptyNodesA);
    builderA.Finish(graphA);
    auto serializedA = builderA.Release();

    // Deserialize graph A
    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedA.data(), serializedA.size());

    // Verify graph A name via getAttribute
    std::array<char, 64> nameBuffer{};
    int64_t count = 0;
    ASSERT_NO_THROW(descriptor.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                                            HIPDNN_TYPE_CHAR,
                                            static_cast<int64_t>(nameBuffer.size()),
                                            &count,
                                            nameBuffer.data()));
    EXPECT_EQ(std::string(nameBuffer.data()), "graphA");

    // Build graph B with name "graphB"
    flatbuffers::FlatBufferBuilder builderB;
    const std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        emptyTensorsB;
    const std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> emptyNodesB;
    auto graphB = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builderB,
        "graphB",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &emptyTensorsB,
        &emptyNodesB);
    builderB.Finish(graphB);
    auto serializedB = builderB.Release();

    // Deserialize graph B (should replace graph A's data)
    descriptor.deserializeGraph(serializedB.data(), serializedB.size());

    // Verify the name reflects graph B, not stale graph A
    nameBuffer.fill(0);
    count = 0;
    ASSERT_NO_THROW(descriptor.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                                            HIPDNN_TYPE_CHAR,
                                            static_cast<int64_t>(nameBuffer.size()),
                                            &count,
                                            nameBuffer.data()));
    EXPECT_EQ(std::string(nameBuffer.data()), "graphB");
}

// ============================================================================
// Binary/JSON serialization error-path tests
// ============================================================================

TEST_F(TestGraphDescriptor, BinarySerializeNullDescriptor)
{
    size_t size = 0;
    auto status = hipdnnBackendGetSerializedBinaryGraph_ext(nullptr, 0, &size, nullptr);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphDescriptor, BinarySerializeNullSize)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &rawDesc, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc, nullptr);
    const ScopedBackendDescriptor desc(rawDesc);

    status = hipdnnBackendGetSerializedBinaryGraph_ext(desc.get(), 0, nullptr, nullptr);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphDescriptor, BinarySerializeEmptyGraph)
{
    // Create a graph descriptor with no operations via the C API
    hipdnnBackendDescriptor_t rawDesc = nullptr;
    auto status = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &rawDesc);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc, nullptr);
    const ScopedBackendDescriptor desc(rawDesc);

    // Serializing a graph with no operations produces a valid (empty) FlatBuffer.
    // finalize() is the gate that enforces non-empty operations for execution.
    size_t size = 0;
    status = hipdnnBackendGetSerializedBinaryGraph_ext(desc.get(), 0, &size, nullptr);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_GT(size, 0u);
}

TEST_F(TestGraphDescriptor, JsonSerializeEmptyGraph)
{
    // Create a graph descriptor with no operations via the C API
    hipdnnBackendDescriptor_t rawDesc = nullptr;
    auto status = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &rawDesc);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc, nullptr);
    const ScopedBackendDescriptor desc(rawDesc);

    // Serializing a graph with no operations produces valid JSON
    size_t size = 0;
    status = hipdnnBackendGetSerializedJsonGraph_ext(desc.get(), 0, &size, nullptr);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_GT(size, 0u);
}

// ============================================================================
// Malformed JSON tests
// ============================================================================

TEST_F(TestGraphDescriptor, MalformedJsonReturnsBadParam)
{
    GraphDescriptor descriptor;
    const std::string badJson = "{ not valid json !!!";
    ASSERT_THROW_HIPDNN_STATUS(
        GraphDescriptor::createFromJsonGraph(descriptor, badJson.c_str(), badJson.size()),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptor, MalformedJsonViaCApiReturnsBadParam)
{
    hipdnnBackendDescriptor_t desc = nullptr;
    const std::string badJson = "{ not valid json !!!";
    auto status
        = hipdnnBackendCreateAndDeserializeJsonGraph_ext(&desc, badJson.c_str(), badJson.size());
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptor, BinaryDeserializeCorruptedData)
{
    const std::vector<uint8_t> garbageData = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04};
    hipdnnBackendDescriptor_t rawDesc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &rawDesc, garbageData.data(), garbageData.size());
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(rawDesc, nullptr);
}

// ============================================================================
// Binary C API error-path tests
// ============================================================================

TEST_F(TestGraphDescriptor, BinarySerializeInsufficientBuffer)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &rawDesc, serializedGraph.data(), serializedGraph.size());
    const ScopedBackendDescriptor desc(rawDesc);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc, nullptr);

    // Query the required size
    size_t requiredSize = 0;
    status = hipdnnBackendGetSerializedBinaryGraph_ext(desc.get(), 0, &requiredSize, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(requiredSize, 0u);

    // Provide a buffer that is 1 byte too small
    std::vector<uint8_t> buffer(requiredSize - 1);
    size_t returnedSize = 0;
    status = hipdnnBackendGetSerializedBinaryGraph_ext(
        desc.get(), buffer.size(), &returnedSize, buffer.data());
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT);
}

TEST_F(TestGraphDescriptor, BinarySerializeOversizedBuffer)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &rawDesc, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc, nullptr);
    const ScopedBackendDescriptor desc(rawDesc);

    // Query the required size
    size_t requiredSize = 0;
    status = hipdnnBackendGetSerializedBinaryGraph_ext(desc.get(), 0, &requiredSize, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(requiredSize, 0u);

    // Provide an oversized buffer
    std::vector<uint8_t> buffer(requiredSize + 64);
    size_t returnedSize = 0;
    status = hipdnnBackendGetSerializedBinaryGraph_ext(
        desc.get(), buffer.size(), &returnedSize, buffer.data());
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(returnedSize, requiredSize);

    // Verify the binary content is valid
    flatbuffers::Verifier verifier(buffer.data(), returnedSize);
    ASSERT_TRUE(verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::Graph>());
}

TEST_F(TestGraphDescriptor, BinaryRoundTripViaApi)
{
    // Create a valid graph descriptor via the binary C API
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t rawDesc1 = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &rawDesc1, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc1, nullptr);
    const ScopedBackendDescriptor scopedDesc1(rawDesc1);

    // Serialize to binary via C API: first query the size
    size_t binarySize = 0;
    status = hipdnnBackendGetSerializedBinaryGraph_ext(scopedDesc1.get(), 0, &binarySize, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(binarySize, 0u);

    // Then retrieve the binary data
    std::vector<uint8_t> binaryBuffer(binarySize);
    size_t returnedSize = 0;
    status = hipdnnBackendGetSerializedBinaryGraph_ext(
        scopedDesc1.get(), binaryBuffer.size(), &returnedSize, binaryBuffer.data());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Deserialize from binary via C API
    hipdnnBackendDescriptor_t rawDesc2 = nullptr;
    status
        = hipdnnBackendCreateAndDeserializeGraph_ext(&rawDesc2, binaryBuffer.data(), returnedSize);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(rawDesc2, nullptr);
    const ScopedBackendDescriptor scopedDesc2(rawDesc2);

    // Verify the round-tripped graph matches
    auto graphDesc1 = scopedDesc1.get()->asDescriptor<GraphDescriptor>();
    auto graphDesc2 = scopedDesc2.get()->asDescriptor<GraphDescriptor>();

    // desc1 buffer was populated by the GetSerializedBinaryGraph call above.
    // desc2 needs explicit buildSerializedGraph() since it was only deserialized.
    graphDesc2->buildSerializedGraph();

    auto binary1 = graphDesc1->getSerializedGraph();
    auto binary2 = graphDesc2->getSerializedGraph();

    auto graph1 = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(
        static_cast<const uint8_t*>(binary1.ptr));
    auto graph2 = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(
        static_cast<const uint8_t*>(binary2.ptr));

    verifyGraphsEquivalent(*graph1, *graph2);
}
