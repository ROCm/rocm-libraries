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
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <nlohmann/json.hpp>
#include <unordered_map>
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

    static std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT>
        unpack(const GraphDescriptor& descriptor)
    {
        const auto serialized = descriptor.getSerializedGraph();
        return hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(
            static_cast<const uint8_t*>(serialized.ptr));
    }

    static void setHandle(GraphDescriptor& descriptor)
    {
        auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
        descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                HIPDNN_TYPE_HANDLE,
                                1,
                                static_cast<const void*>(&handle));
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

    EXPECT_EQ(*graph1, *graph2);
}

TEST_F(TestGraphDescriptor, JsonRoundTripPreservesMinimumApiVersionAndRuntimePassByValue)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createValidGraph();
    const auto* graph = GetGraph(builder.GetBufferPointer());
    auto graphT = std::unique_ptr<GraphT>(graph->UnPack());
    ASSERT_NE(graphT, nullptr);
    ASSERT_FALSE(graphT->tensors.empty());

    graphT->tensors.front()->is_runtime_pass_by_value = true;
    graphT->min_required_engine_api_version = std::make_unique<EngineApiVersion>(1, 2, 0);

    flatbuffers::FlatBufferBuilder versionedBuilder;
    versionedBuilder.Finish(Graph::Pack(versionedBuilder, graphT.get()));

    GraphDescriptor original;
    original.deserializeGraph(versionedBuilder.GetBufferPointer(), versionedBuilder.GetSize());
    original.buildSerializedGraph();

    const auto jsonStr = original.getSerializedJsonGraph();
    const auto parsed = nlohmann::json::parse(jsonStr);
    EXPECT_EQ(parsed.at("min_required_engine_api_version"),
              nlohmann::json({{"major", 1}, {"minor", 2}, {"patch", 0}}));

    GraphDescriptor roundTripped;
    ASSERT_NO_THROW(
        GraphDescriptor::createFromJsonGraph(roundTripped, jsonStr.c_str(), jsonStr.size()));
    roundTripped.buildSerializedGraph();

    const auto binary = roundTripped.getSerializedGraph();
    const auto restored = UnPackGraph(static_cast<const uint8_t*>(binary.ptr));
    ASSERT_NE(restored, nullptr);
    ASSERT_NE(restored->min_required_engine_api_version, nullptr);
    EXPECT_EQ(restored->min_required_engine_api_version->major(), 1);
    EXPECT_EQ(restored->min_required_engine_api_version->minor(), 2);
    EXPECT_EQ(restored->min_required_engine_api_version->patch(), 0);
    ASSERT_FALSE(restored->tensors.empty());
    EXPECT_TRUE(restored->tensors.front()->is_runtime_pass_by_value);
}

TEST_F(TestGraphDescriptor, JsonSerializationEmitsOverrideShapeTrue)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    bool overrideShapeEnabled = true;
    ASSERT_NO_THROW(
        descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                HIPDNN_TYPE_BOOLEAN,
                                1,
                                &overrideShapeEnabled));

    descriptor.buildSerializedGraph();
    const auto jsonStr = descriptor.getSerializedJsonGraph();
    const auto parsed = nlohmann::json::parse(jsonStr);

    ASSERT_TRUE(parsed.contains("is_override_shape_enabled"));
    EXPECT_TRUE(parsed.at("is_override_shape_enabled").get<bool>());
}

TEST_F(TestGraphDescriptor, JsonRoundTripPreservesOverrideShapeTrue)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor original;
    original.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    bool overrideShapeEnabled = true;
    ASSERT_NO_THROW(original.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                          HIPDNN_TYPE_BOOLEAN,
                                          1,
                                          &overrideShapeEnabled));

    original.buildSerializedGraph();
    const auto jsonStr = original.getSerializedJsonGraph();

    GraphDescriptor roundTripped;
    ASSERT_NO_THROW(
        GraphDescriptor::createFromJsonGraph(roundTripped, jsonStr.c_str(), jsonStr.size()));

    bool output = false;
    int64_t count = 0;
    ASSERT_NO_THROW(
        roundTripped.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                  HIPDNN_TYPE_BOOLEAN,
                                  1,
                                  &count,
                                  &output));
    EXPECT_EQ(count, 1);
    EXPECT_TRUE(output);
}

TEST_F(TestGraphDescriptor, JsonMissingOverrideShapeFieldDefaultsFalse)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());
    descriptor.buildSerializedGraph();

    auto parsed = nlohmann::json::parse(descriptor.getSerializedJsonGraph());
    parsed.erase("is_override_shape_enabled");
    const auto jsonStr = parsed.dump();

    GraphDescriptor fromJson;
    ASSERT_NO_THROW(
        GraphDescriptor::createFromJsonGraph(fromJson, jsonStr.c_str(), jsonStr.size()));

    bool output = true;
    int64_t count = 0;
    ASSERT_NO_THROW(fromJson.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                          HIPDNN_TYPE_BOOLEAN,
                                          1,
                                          &count,
                                          &output));
    EXPECT_EQ(count, 1);
    EXPECT_FALSE(output);
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

    EXPECT_EQ(*graph1, *graph2);
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

TEST_F(TestGraphDescriptor, DeserializeFailsAfterFinalizeAndPreservesIdentity)
{
    auto builder = createValidGraph();
    auto serialized = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serialized.data(), serialized.size());
    setHandle(descriptor);
    descriptor.finalize();
    const auto original = unpack(descriptor);
    ASSERT_NE(original->id, nullptr);
    const auto originalId = hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*original->id);

    ASSERT_THROW_HIPDNN_STATUS(descriptor.deserializeGraph(serialized.data(), serialized.size()),
                               HIPDNN_STATUS_NOT_INITIALIZED);
    EXPECT_TRUE(descriptor.isFinalized());
    const auto preserved = unpack(descriptor);
    ASSERT_NE(preserved->id, nullptr);
    EXPECT_EQ(hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*preserved->id), originalId);
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

    EXPECT_EQ(*graph1, *graph2);
}

TEST_F(TestGraphDescriptor, FinalizingLegacyGraphGeneratesValidStableUuid)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();
    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());
    descriptor.buildSerializedGraph();
    ASSERT_EQ(unpack(descriptor)->id, nullptr);

    setHandle(descriptor);
    descriptor.finalize();
    const auto first = unpack(descriptor);
    ASSERT_NE(first->id, nullptr);
    const auto firstId = hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*first->id);
    EXPECT_TRUE(hipdnn_flatbuffers_sdk::utilities::isUuidV4(firstId));

    const auto second = unpack(descriptor);
    ASSERT_NE(second->id, nullptr);
    EXPECT_EQ(firstId, hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*second->id));
}

TEST_F(TestGraphDescriptor, EquivalentGraphsReceiveDifferentUuids)
{
    auto firstBuilder = createValidGraph();
    auto firstSerialized = firstBuilder.Release();
    GraphDescriptor first;
    first.deserializeGraph(firstSerialized.data(), firstSerialized.size());
    setHandle(first);
    first.finalize();

    auto secondBuilder = createValidGraph();
    auto secondSerialized = secondBuilder.Release();
    GraphDescriptor second;
    second.deserializeGraph(secondSerialized.data(), secondSerialized.size());
    setHandle(second);
    second.finalize();

    const auto firstGraph = unpack(first);
    const auto secondGraph = unpack(second);
    ASSERT_NE(firstGraph->id, nullptr);
    ASSERT_NE(secondGraph->id, nullptr);
    EXPECT_NE(*firstGraph->id, *secondGraph->id);
}

TEST_F(TestGraphDescriptor, ExistingUuidSurvivesDeserializeHandleAndFinalize)
{
    auto legacyBuilder = createValidGraph();
    auto legacySerialized = legacyBuilder.Release();
    GraphDescriptor original;
    original.deserializeGraph(legacySerialized.data(), legacySerialized.size());
    setHandle(original);
    original.finalize();
    const auto originalGraph = unpack(original);
    ASSERT_NE(originalGraph->id, nullptr);
    const auto expectedId = hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*originalGraph->id);

    const auto originalData = original.getSerializedGraph();
    GraphDescriptor revived;
    revived.deserializeGraph(static_cast<const uint8_t*>(originalData.ptr), originalData.size);
    setHandle(revived);
    revived.finalize();
    const auto revivedGraph = unpack(revived);
    ASSERT_NE(revivedGraph->id, nullptr);
    EXPECT_EQ(hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*revivedGraph->id), expectedId);
}

TEST_F(TestGraphDescriptor, ContentMutationReplacesInheritedUuid)
{
    auto legacyBuilder = createValidGraph();
    auto legacySerialized = legacyBuilder.Release();
    GraphDescriptor original;
    original.deserializeGraph(legacySerialized.data(), legacySerialized.size());
    setHandle(original);
    original.finalize();
    const auto originalGraph = unpack(original);
    ASSERT_NE(originalGraph->id, nullptr);
    const auto originalId = hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*originalGraph->id);

    const auto originalData = original.getSerializedGraph();
    GraphDescriptor mutated;
    mutated.deserializeGraph(static_cast<const uint8_t*>(originalData.ptr), originalData.size);
    const std::array name{'m', 'u', 't', 'a', 't', 'e', 'd', '\0'};
    mutated.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                         HIPDNN_TYPE_CHAR,
                         static_cast<int64_t>(name.size()),
                         name.data());
    setHandle(mutated);
    mutated.finalize();
    const auto mutatedGraph = unpack(mutated);
    ASSERT_NE(mutatedGraph->id, nullptr);
    EXPECT_NE(hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*mutatedGraph->id), originalId);
}

TEST_F(TestGraphDescriptor, PreservesOpaqueSerializedGraphId)
{
    auto builder = createValidGraph();
    auto serialized = builder.Release();
    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(serialized.data());
    std::array<uint8_t, 16> opaqueId{};
    opaqueId[6] = 0x30;
    opaqueId[8] = 0x40;
    graph->id = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::Uuid>(
        flatbuffers::span<const uint8_t, 16>(opaqueId));

    flatbuffers::FlatBufferBuilder identifiedBuilder;
    identifiedBuilder.Finish(
        hipdnn_flatbuffers_sdk::data_objects::Graph::Pack(identifiedBuilder, graph.get()));
    auto identified = identifiedBuilder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(identified.data(), identified.size());
    descriptor.buildSerializedGraph();
    const auto roundTripped = unpack(descriptor);
    ASSERT_NE(roundTripped->id, nullptr);
    EXPECT_EQ(hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*roundTripped->id), opaqueId);
}

TEST_F(TestGraphDescriptor, JsonRoundTripPreservesUuid)
{
    auto legacyBuilder = createValidGraph();
    auto legacySerialized = legacyBuilder.Release();
    GraphDescriptor original;
    original.deserializeGraph(legacySerialized.data(), legacySerialized.size());
    setHandle(original);
    original.finalize();
    const auto originalGraph = unpack(original);
    ASSERT_NE(originalGraph->id, nullptr);
    const auto expectedId = hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*originalGraph->id);

    const auto json = original.getSerializedJsonGraph();
    EXPECT_EQ(nlohmann::json::parse(json).at("id"),
              hipdnn_flatbuffers_sdk::utilities::formatUuid(expectedId));

    GraphDescriptor revived;
    GraphDescriptor::createFromJsonGraph(revived, json.c_str(), json.size());
    revived.buildSerializedGraph();
    const auto revivedGraph = unpack(revived);
    ASSERT_NE(revivedGraph->id, nullptr);
    EXPECT_EQ(hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*revivedGraph->id), expectedId);
}

TEST_F(TestGraphDescriptor, ToStringReportsGraphId)
{
    auto builder = createValidGraph();
    auto serialized = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serialized.data(), serialized.size());
    EXPECT_NE(descriptor.toString().find("id=(none)"), std::string::npos);

    setHandle(descriptor);
    descriptor.finalize();
    const auto id = hipdnn_flatbuffers_sdk::utilities::formatUuid(
        hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*unpack(descriptor)->id));
    EXPECT_NE(descriptor.toString().find("id=" + id), std::string::npos);
}

// ============================================================================
// HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT (RFC 0008)
// ============================================================================

TEST_F(TestGraphDescriptor, IsOverrideShapeEnabledDefaultsToFalseWhenUnset)
{
    // A freshly-created descriptor that never had IS_OVERRIDE_SHAPE_ENABLED set
    // should report false (the wire default for an absent optional bool).
    const GraphDescriptor descriptor;

    bool value = true;
    int64_t count = 0;
    ASSERT_NO_THROW(
        descriptor.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                HIPDNN_TYPE_BOOLEAN,
                                1,
                                &count,
                                &value));
    EXPECT_FALSE(value);
}

TEST_F(TestGraphDescriptor, IsOverrideShapeEnabledSetGetTrueRoundTrip)
{
    GraphDescriptor descriptor;

    bool input = true;
    ASSERT_NO_THROW(descriptor.setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT, HIPDNN_TYPE_BOOLEAN, 1, &input));

    bool output = false;
    int64_t count = 0;
    ASSERT_NO_THROW(
        descriptor.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                HIPDNN_TYPE_BOOLEAN,
                                1,
                                &count,
                                &output));
    EXPECT_TRUE(output);
}

TEST_F(TestGraphDescriptor, IsOverrideShapeEnabledTrueSurvivesSerializationRoundTrip)
{
    // Build a valid graph, set the opt-in flag, serialize, deserialize, verify
    // the flag is preserved as true through the flatbuffer round-trip.
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor original;
    original.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    bool input = true;
    ASSERT_NO_THROW(original.setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT, HIPDNN_TYPE_BOOLEAN, 1, &input));

    auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    ASSERT_NO_THROW(original.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                          HIPDNN_TYPE_HANDLE,
                                          1,
                                          static_cast<const void*>(&handle)));
    ASSERT_NO_THROW(original.finalize());

    auto serialized = original.getSerializedGraph();

    GraphDescriptor revived;
    revived.deserializeGraph(static_cast<const uint8_t*>(serialized.ptr), serialized.size);

    bool output = false;
    int64_t count = 0;
    ASSERT_NO_THROW(revived.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                         HIPDNN_TYPE_BOOLEAN,
                                         1,
                                         &count,
                                         &output));
    EXPECT_TRUE(output);
}

TEST_F(TestGraphDescriptor, LegacyGraphWithoutOverrideShapeFieldRoundTripsToFalse)
{
    // createValidGraph() does NOT set is_override_shape_enabled — it produces a
    // wire image equivalent to a legacy graph that predates this field. Verify
    // deserialize+get reports the wire default (false) without throwing.
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    bool value = true;
    int64_t count = 0;
    ASSERT_NO_THROW(
        descriptor.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                HIPDNN_TYPE_BOOLEAN,
                                1,
                                &count,
                                &value));
    EXPECT_FALSE(value);
}

// ============================================================================
// GraphDescriptor::hasRaggedTensors() (RFC 0014)
// ============================================================================

TEST_F(TestGraphDescriptor, HasRaggedTensorsFalseForNonRaggedGraph)
{
    auto builder = createValidGraph();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    EXPECT_FALSE(descriptor.hasRaggedTensors());
}

TEST_F(TestGraphDescriptor, HasRaggedTensorsFalseForEmptyDescriptor)
{
    // No operations => no tensors => not ragged.
    const GraphDescriptor descriptor;
    EXPECT_FALSE(descriptor.hasRaggedTensors());
}

TEST_F(TestGraphDescriptor, HasRaggedTensorsTrueWhenTensorCarriesRaggedOffset)
{
    auto builder = test_utilities::createValidGraphWithRaggedTensor();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    EXPECT_TRUE(descriptor.hasRaggedTensors());
}

TEST_F(TestGraphDescriptor, HasRaggedTensorsSurvivesSerializationRoundTrip)
{
    auto builder = test_utilities::createValidGraphWithRaggedTensor();
    auto serializedGraph = builder.Release();

    GraphDescriptor original;
    original.deserializeGraph(serializedGraph.data(), serializedGraph.size());

    auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    ASSERT_NO_THROW(original.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                          HIPDNN_TYPE_HANDLE,
                                          1,
                                          static_cast<const void*>(&handle)));
    ASSERT_NO_THROW(original.finalize());

    auto serialized = original.getSerializedGraph();

    GraphDescriptor revived;
    revived.deserializeGraph(static_cast<const uint8_t*>(serialized.ptr), serialized.size);

    EXPECT_TRUE(revived.hasRaggedTensors());
}

TEST_F(TestGraphDescriptor, HasRaggedTensorsSurvivesJsonRoundTrip)
{
    auto builder = test_utilities::createValidGraphWithRaggedTensor();
    auto serializedGraph = builder.Release();

    GraphDescriptor original;
    original.deserializeGraph(serializedGraph.data(), serializedGraph.size());
    original.buildSerializedGraph();
    const auto jsonStr = original.getSerializedJsonGraph();

    GraphDescriptor fromJson;
    ASSERT_NO_THROW(
        GraphDescriptor::createFromJsonGraph(fromJson, jsonStr.c_str(), jsonStr.size()));

    EXPECT_TRUE(fromJson.hasRaggedTensors());
}

// The ragged-offset aux must be re-emitted into the tensor list when the graph
// is rebuilt from its operations, carrying its dims/strides/dtype, with X still
// resolving to it.
TEST_F(TestGraphDescriptor, RaggedOffsetAuxEmittedIntoTensorsOnReserialize)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    using hipdnn_tests::constants::K_FPROP_TENSOR_RAGGED_OFFSET_DIMS;
    using hipdnn_tests::constants::K_FPROP_TENSOR_RAGGED_OFFSET_STRIDES;
    using hipdnn_tests::constants::K_FPROP_TENSOR_RAGGED_OFFSET_UID;
    using hipdnn_tests::constants::K_FPROP_TENSOR_X_UID;

    auto builder = test_utilities::createValidGraphWithRaggedTensor();
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());
    descriptor.buildSerializedGraph();

    const auto serialized = descriptor.getSerializedGraph();
    auto graph = UnPackGraph(static_cast<const uint8_t*>(serialized.ptr));
    ASSERT_NE(graph, nullptr);

    std::unordered_map<int64_t, const TensorAttributesT*> tensorMap;
    for(const auto& t : graph->tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_RAGGED_OFFSET_UID), 0u)
        << "ragged-offset aux must be emitted into the re-serialized tensor list.";
    const auto* aux = tensorMap.at(K_FPROP_TENSOR_RAGGED_OFFSET_UID);
    EXPECT_EQ(aux->data_type, DataType::INT64);
    EXPECT_EQ(aux->dims, hipdnn_tests::toVec(K_FPROP_TENSOR_RAGGED_OFFSET_DIMS));
    EXPECT_EQ(aux->strides, hipdnn_tests::toVec(K_FPROP_TENSOR_RAGGED_OFFSET_STRIDES));

    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_X_UID), 0u);
    const auto* xTensor = tensorMap.at(K_FPROP_TENSOR_X_UID);
    ASSERT_TRUE(xTensor->ragged_offset_tensor_uid.has_value());
    EXPECT_EQ(xTensor->ragged_offset_tensor_uid.value(), K_FPROP_TENSOR_RAGGED_OFFSET_UID);
    EXPECT_NE(tensorMap.count(xTensor->ragged_offset_tensor_uid.value()), 0u)
        << "X's ragged_offset_tensor_uid must resolve within the tensor list.";
}

// The ragged-offset aux must survive a JSON round-trip: deserialize -> JSON ->
// createFromJsonGraph -> re-serialize must still emit the aux with its
// dims/strides/dtype and keep X resolving to it. Guards the JSON path (Path C),
// which HasRaggedTensorsSurvivesJsonRoundTrip cannot (it only checks the link).
TEST_F(TestGraphDescriptor, RaggedOffsetAuxEmittedIntoTensorsOnJsonRoundTrip)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    using hipdnn_tests::constants::K_FPROP_TENSOR_RAGGED_OFFSET_DIMS;
    using hipdnn_tests::constants::K_FPROP_TENSOR_RAGGED_OFFSET_STRIDES;
    using hipdnn_tests::constants::K_FPROP_TENSOR_RAGGED_OFFSET_UID;
    using hipdnn_tests::constants::K_FPROP_TENSOR_X_UID;

    auto builder = test_utilities::createValidGraphWithRaggedTensor();
    auto serializedGraph = builder.Release();

    GraphDescriptor original;
    original.deserializeGraph(serializedGraph.data(), serializedGraph.size());
    original.buildSerializedGraph();
    const auto jsonStr = original.getSerializedJsonGraph();

    GraphDescriptor fromJson;
    ASSERT_NO_THROW(
        GraphDescriptor::createFromJsonGraph(fromJson, jsonStr.c_str(), jsonStr.size()));
    fromJson.buildSerializedGraph();

    const auto serialized = fromJson.getSerializedGraph();
    auto graph = UnPackGraph(static_cast<const uint8_t*>(serialized.ptr));
    ASSERT_NE(graph, nullptr);

    std::unordered_map<int64_t, const TensorAttributesT*> tensorMap;
    for(const auto& t : graph->tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_RAGGED_OFFSET_UID), 0u)
        << "ragged-offset aux must survive the JSON round-trip.";
    const auto* aux = tensorMap.at(K_FPROP_TENSOR_RAGGED_OFFSET_UID);
    EXPECT_EQ(aux->data_type, DataType::INT64);
    EXPECT_EQ(aux->dims, hipdnn_tests::toVec(K_FPROP_TENSOR_RAGGED_OFFSET_DIMS));
    EXPECT_EQ(aux->strides, hipdnn_tests::toVec(K_FPROP_TENSOR_RAGGED_OFFSET_STRIDES));

    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_X_UID), 0u);
    const auto* xTensor = tensorMap.at(K_FPROP_TENSOR_X_UID);
    ASSERT_TRUE(xTensor->ragged_offset_tensor_uid.has_value());
    EXPECT_EQ(xTensor->ragged_offset_tensor_uid.value(), K_FPROP_TENSOR_RAGGED_OFFSET_UID);
    EXPECT_NE(tensorMap.count(xTensor->ragged_offset_tensor_uid.value()), 0u)
        << "X's ragged_offset_tensor_uid must resolve within the tensor list.";
}

// A tensor whose ragged_offset_tensor_uid names an aux that is absent from the
// tensor list must be rejected on deserialize.
TEST_F(TestGraphDescriptor, DeserializeDanglingRaggedOffsetUidThrows)
{
    auto builder = test_utilities::createValidGraphWithRaggedTensor(/*includeAux=*/false);
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(
        descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size()),
        HIPDNN_STATUS_BAD_PARAM);
}

// A virtual ragged-offset aux has no backing storage and must be rejected.
TEST_F(TestGraphDescriptor, DeserializeVirtualRaggedAuxThrows)
{
    auto builder = test_utilities::createValidGraphWithRaggedTensor(/*includeAux=*/true,
                                                                    /*auxVirtual=*/true);
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(
        descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size()),
        HIPDNN_STATUS_BAD_PARAM);
}

// A ragged-offset aux may not itself carry a ragged offset; nesting is rejected.
TEST_F(TestGraphDescriptor, DeserializeNestedRaggedAuxThrows)
{
    auto builder = test_utilities::createValidGraphWithRaggedTensor(/*includeAux=*/true,
                                                                    /*auxVirtual=*/false,
                                                                    /*auxNested=*/true);
    auto serializedGraph = builder.Release();

    GraphDescriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(
        descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size()),
        HIPDNN_STATUS_BAD_PARAM);
}
