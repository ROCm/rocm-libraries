// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/DescriptorTestUtils.hpp"
#include "descriptors/FlatbufferTestUtils.hpp"
#include "descriptors/ScopedBackendDescriptor.hpp"
#include "descriptors/mocks/MockDescriptor.hpp"
#include "descriptors/mocks/MockEnginePluginResourceManager.hpp"
#include "descriptors/mocks/MockHandle.hpp"

#include "descriptors/ExecutionPlanDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "hipdnn_backend.h"

#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/flatbuffers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/execution_plan_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/serialized_graph_and_plan_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/SerializedGraphContainer.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using namespace hipdnn_backend::test_utilities;
using namespace ::testing;

using ::testing::Return;

namespace fbs = hipdnn_flatbuffers_sdk::data_objects;

// Tests for the container-aware graph-and-plan serialization C API surface:
// combo serialize, query contents, and the container-aware graph/plan
// deserialize entry points. makeContainer() hand-assembles an HDGP container;
// makeBareGraphBlob()/makeBarePlanBlob() produce the unwrapped legacy blobs.
class TestGraphAndPlanSerializationApi : public ::testing::Test
{
public:
    std::shared_ptr<ExecutionPlanDescriptor> getExecutionPlanDescriptor() const
    {
        return _planWrapper->asDescriptor<ExecutionPlanDescriptor>();
    }

    std::shared_ptr<MockGraphDescriptor> getMockGraph() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(
            _mockGraphWrapper.get());
    }

    std::shared_ptr<MockEngineDescriptor> getMockEngine() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(
            _mockEngineWrapper.get());
    }

    std::shared_ptr<MockEngineConfigDescriptor> getMockEngineConfig() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineConfigDescriptor>(
            _mockEngineConfigWrapper.get());
    }

    static hipdnnEnginePluginExecutionContext_t getExecutionContext()
    {
        return reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(0xFFFFFFFF);
    }

    void setHandle()
    {
        EXPECT_CALL(*_mockEnginePluginResourceManager, createExecutionContext(_, _, _))
            .WillOnce(Return(getExecutionContext()));
        EXPECT_CALL(*_mockEnginePluginResourceManager, destroyExecutionContext(_, _));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getWorkspaceSize(_, _))
            .WillOnce(Return(1024));

        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillOnce(Return(_mockEnginePluginResourceManager));
        getExecutionPlanDescriptor()->setAttribute(
            HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_mockHandle);
    }

    void setEngineConfig()
    {
        EXPECT_CALL(*getMockGraph(), getHandle()).WillRepeatedly(Return(_mockHandle.get()));
        EXPECT_CALL(*getMockGraph(), getSerializedGraph()).WillRepeatedly(Invoke([this]() {
            return hipdnnPluginConstData_t{_serializedGraph.data(), _serializedGraph.size()};
        }));
        EXPECT_CALL(*getMockGraph(), isOverrideShapeEnabled()).WillRepeatedly(Return(false));
        EXPECT_CALL(*getMockEngine(), getEngineId()).WillOnce(Return(ENGINE_ID));
        EXPECT_CALL(*getMockEngine(), getGraph()).WillOnce(Return(getMockGraph()));

        EXPECT_CALL(*getMockEngineConfig(), isFinalized()).WillOnce(Return(true));
        EXPECT_CALL(*getMockEngineConfig(), getEngine()).WillOnce(Return(getMockEngine()));
        EXPECT_CALL(*getMockEngineConfig(), getSerializedEngineConfig()).WillOnce(Invoke([]() {
            return hipdnnPluginConstData_t{nullptr, 0};
        }));

        getExecutionPlanDescriptor()->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_mockEngineConfigWrapper);
    }

    // Drives the ExecutionPlanDescriptor into a finalized state so its
    // hipdnnBackendDescriptor_t can be handed to the combo serialize C API.
    void makeExecutionPlanFinalized()
    {
        setHandle();
        setEngineConfig();
        ASSERT_NO_THROW(getExecutionPlanDescriptor()->finalize());
    }

    // Combo serialize may serialize the plugin context across the size-query and
    // fill calls, so program the expectation with WillRepeatedly.
    void expectPlanSerialization()
    {
        EXPECT_CALL(*_mockEnginePluginResourceManager,
                    serializeExecutionContext(ENGINE_ID, getExecutionContext(), _))
            .WillRepeatedly([](int64_t,
                               hipdnnEnginePluginExecutionContext_t,
                               std::vector<uint8_t>& serializedContext) {
                serializedContext = std::vector<uint8_t>{7, 8, 9};
            });
    }

    // A real Graph FlatBuffer (conv-fprop) usable as a bare-graph blob.
    static flatbuffers::DetachedBuffer makeBareGraphBlob()
    {
        return createValidGraph().Release();
    }

    // A bare SerializedExecutionPlan FlatBuffer, i.e. NOT wrapped in an HDGP
    // container.
    flatbuffers::DetachedBuffer makeBarePlanBlob(uint32_t version = 1,
                                                 int64_t workspaceSize = 1024) const
    {
        flatbuffers::FlatBufferBuilder builder;
        auto tensorUids = builder.CreateVector(_tensorUids);
        auto pluginPayload = builder.CreateVector(std::vector<uint8_t>{4, 5, 6});

        auto plan = fbs::CreateSerializedExecutionPlan(builder,
                                                       version,
                                                       ENGINE_ID,
                                                       workspaceSize,
                                                       tensorUids,
                                                       pluginPayload,
                                                       /*is_override_shape_enabled=*/false);
        builder.Finish(plan);
        return builder.Release();
    }

    // Hand-assembles an HDGP container. An empty planBytes vector produces a
    // graph-only container (no plan_blob). Deliberately does NOT go through the
    // production buildGraphAndPlanContainer() writer (and so does not force-align
    // the embedded blobs): this feeds read-path/extraction tests that must stay
    // independent of the writer. Alignment is asserted separately against real
    // C-API output, not against this helper.
    static std::vector<uint8_t> makeContainer(const std::vector<uint8_t>& graphBytes,
                                              const std::vector<uint8_t>& planBytes)
    {
        flatbuffers::FlatBufferBuilder builder;

        flatbuffers::Offset<flatbuffers::Vector<uint8_t>> graphVec = 0;
        if(!graphBytes.empty())
        {
            graphVec = builder.CreateVector(graphBytes);
        }
        flatbuffers::Offset<flatbuffers::Vector<uint8_t>> planVec = 0;
        if(!planBytes.empty())
        {
            planVec = builder.CreateVector(planBytes);
        }

        auto root = fbs::CreateSerializedGraphAndPlan(builder, graphVec, planVec);
        fbs::FinishSerializedGraphAndPlanBuffer(builder, root);

        return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
    }

    // Produces a buffer that carries the HDGP file identifier yet fails the
    // FlatBuffers verifier. Truncating a valid container keeps the identifier in
    // bytes [4, 8) (so it is still recognized as a graph-and-plan container)
    // while leaving the root offset pointing past the end of the buffer.
    std::vector<uint8_t> makeIdentifiedButCorruptContainer() const
    {
        auto graphBlob = makeBareGraphBlob();
        auto planBlob = makeBarePlanBlob();
        auto valid = makeContainer({graphBlob.data(), graphBlob.data() + graphBlob.size()},
                                   {planBlob.data(), planBlob.data() + planBlob.size()});

        // 16 bytes retains the 8-byte header (root offset + identifier) and a
        // few more bytes, while dropping the graph and plan vectors the offsets
        // point at; the container embeds both blobs so it is always far larger.
        std::vector<uint8_t> corrupt(valid.begin(), valid.begin() + 16);
        EXPECT_GE(corrupt.size(), 8u);
        EXPECT_TRUE(fbs::SerializedGraphAndPlanBufferHasIdentifier(corrupt.data()));
        return corrupt;
    }

    // Verifies a deserialized graph descriptor reconstructs the conv-fprop
    // op-graph emitted by createValidGraph(): same tensors, nodes, attributes.
    static void verifyDeserializedGraphMatchesSource(hipdnnBackendDescriptor_t desc)
    {
        ASSERT_NE(desc, nullptr);
        EXPECT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);

        auto graphDesc = desc->asDescriptor<GraphDescriptor>();
        // A C-API-deserialized graph is unfinalized; populate its serialized
        // buffer before reading it back.
        graphDesc->buildSerializedGraph();
        auto roundTripped = graphDesc->getSerializedGraph();
        auto reconstructed = fbs::UnPackGraph(static_cast<const uint8_t*>(roundTripped.ptr));
        ASSERT_NE(reconstructed, nullptr);

        auto sourceBlob = makeBareGraphBlob();
        auto source = fbs::UnPackGraph(sourceBlob.data());
        ASSERT_NE(source, nullptr);

        EXPECT_EQ(reconstructed->name, source->name);
        EXPECT_EQ(reconstructed->compute_data_type, source->compute_data_type);
        EXPECT_EQ(reconstructed->intermediate_data_type, source->intermediate_data_type);
        EXPECT_EQ(reconstructed->io_data_type, source->io_data_type);

        ASSERT_EQ(reconstructed->tensors.size(), source->tensors.size());
        for(size_t i = 0; i < source->tensors.size(); ++i)
        {
            SCOPED_TRACE("tensor[" + std::to_string(i) + "]");
            ASSERT_NE(reconstructed->tensors[i], nullptr);
            ASSERT_NE(source->tensors[i], nullptr);
            EXPECT_EQ(reconstructed->tensors[i]->uid, source->tensors[i]->uid);
            EXPECT_EQ(*reconstructed->tensors[i], *source->tensors[i]);
        }

        ASSERT_EQ(reconstructed->nodes.size(), source->nodes.size());
        for(size_t i = 0; i < source->nodes.size(); ++i)
        {
            SCOPED_TRACE("node[" + std::to_string(i) + "]");
            ASSERT_NE(reconstructed->nodes[i], nullptr);
            ASSERT_NE(source->nodes[i], nullptr);
            EXPECT_EQ(*reconstructed->nodes[i], *source->nodes[i]);
        }
    }

protected:
    static constexpr int64_t ENGINE_ID = 0;

    std::unique_ptr<HipdnnBackendDescriptor> _planWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineConfigWrapper = nullptr;
    std::unique_ptr<MockHandle> _mockHandle = nullptr;
    std::shared_ptr<MockEnginePluginResourceManager> _mockEnginePluginResourceManager = nullptr;
    flatbuffers::DetachedBuffer _serializedGraph;
    std::vector<int64_t> _tensorUids{11, 22, 33};

    void SetUp() override
    {
        flatbuffers::FlatBufferBuilder builder;
        fbs::GraphT graph;
        for(auto uid : _tensorUids)
        {
            auto tensor = std::make_unique<fbs::TensorAttributesT>();
            tensor->uid = uid;
            graph.tensors.push_back(std::move(tensor));
        }
        builder.Finish(fbs::Graph::Pack(builder, &graph));
        _serializedGraph = builder.Release();

        _planWrapper = createDescriptor<ExecutionPlanDescriptor>();
        _mockGraphWrapper = createDescriptor<MockGraphDescriptor>();
        _mockEngineWrapper = createDescriptor<MockEngineDescriptor>();
        _mockEngineConfigWrapper = createDescriptor<MockEngineConfigDescriptor>();
        _mockHandle = std::make_unique<MockHandle>();
        _mockEnginePluginResourceManager = std::make_shared<MockEnginePluginResourceManager>();
    }
};

// ============================================================================
// Combo serialize (hipdnnBackendGetSerializedBinaryGraphAndPlan_ext)
// ============================================================================

TEST_F(TestGraphAndPlanSerializationApi, ComboSerializeProducesContainerWithGraphAndPlan)
{
    auto graphBlob = makeBareGraphBlob();
    hipdnnBackendDescriptor_t rawGraph = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateAndDeserializeGraph_ext(&rawGraph, graphBlob.data(), graphBlob.size()),
        HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor graphDesc(rawGraph);

    makeExecutionPlanFinalized();
    expectPlanSerialization();

    size_t size = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  graphDesc.get(), _planWrapper.get(), 0, &size, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(size, 0u);

    std::vector<uint8_t> blob(size);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  graphDesc.get(), _planWrapper.get(), size, &size, blob.data()),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_FALSE(blob.empty());

    EXPECT_TRUE(fbs::SerializedGraphAndPlanBufferHasIdentifier(blob.data()));

    flatbuffers::Verifier verifier(blob.data(), blob.size());
    ASSERT_TRUE(fbs::VerifySerializedGraphAndPlanBuffer(verifier));

    const auto* container = fbs::GetSerializedGraphAndPlan(blob.data());
    ASSERT_NE(container->graph_blob(), nullptr);
    EXPECT_GT(container->graph_blob()->size(), 0u);
    ASSERT_NE(container->plan_blob(), nullptr);
    EXPECT_GT(container->plan_blob()->size(), 0u);

    // The embedded graph/plan are parsed in place as FlatBuffers roots with int64
    // fields, so the C-API container must place them on 8-byte-aligned offsets.
    {
        const auto graphView = hipdnn_flatbuffers_sdk::flatbuffer_utilities::extractGraphBlob(
            blob.data(), blob.size());
        const auto planView = hipdnn_flatbuffers_sdk::flatbuffer_utilities::extractPlanBlob(
            blob.data(), blob.size());
        EXPECT_EQ(reinterpret_cast<uintptr_t>(graphView.data) % 8, 0u);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(planView.data) % 8, 0u);
    }
}

TEST_F(TestGraphAndPlanSerializationApi, ComboSerializeGraphOnlyWhenPlanDescIsNull)
{
    auto graphBlob = makeBareGraphBlob();
    hipdnnBackendDescriptor_t rawGraph = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateAndDeserializeGraph_ext(&rawGraph, graphBlob.data(), graphBlob.size()),
        HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor graphDesc(rawGraph);

    size_t size = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  graphDesc.get(), nullptr, 0, &size, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(size, 0u);

    std::vector<uint8_t> blob(size);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  graphDesc.get(), nullptr, size, &size, blob.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_TRUE(fbs::SerializedGraphAndPlanBufferHasIdentifier(blob.data()));

    const auto* container = fbs::GetSerializedGraphAndPlan(blob.data());
    ASSERT_NE(container->graph_blob(), nullptr);
    EXPECT_GT(container->graph_blob()->size(), 0u);
    EXPECT_TRUE(container->plan_blob() == nullptr || container->plan_blob()->empty());
}

TEST_F(TestGraphAndPlanSerializationApi, ComboSerializeTwoCallSizeQuery)
{
    auto graphBlob = makeBareGraphBlob();
    hipdnnBackendDescriptor_t rawGraph = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateAndDeserializeGraph_ext(&rawGraph, graphBlob.data(), graphBlob.size()),
        HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor graphDesc(rawGraph);

    makeExecutionPlanFinalized();
    expectPlanSerialization();

    // A null serializedBlob is a size query and still writes *blobByteSize.
    size_t querySize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  graphDesc.get(), _planWrapper.get(), 0, &querySize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(querySize, 0u);

    std::vector<uint8_t> blob(querySize);
    size_t fillSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  graphDesc.get(), _planWrapper.get(), querySize, &fillSize, blob.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(fillSize, querySize);
}

TEST_F(TestGraphAndPlanSerializationApi, ComboSerializeRejectsInsufficientBuffer)
{
    auto graphBlob = makeBareGraphBlob();
    hipdnnBackendDescriptor_t rawGraph = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateAndDeserializeGraph_ext(&rawGraph, graphBlob.data(), graphBlob.size()),
        HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor graphDesc(rawGraph);

    makeExecutionPlanFinalized();
    expectPlanSerialization();

    size_t size = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  graphDesc.get(), _planWrapper.get(), 0, &size, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(size, 1u);

    std::vector<uint8_t> tooSmall(size - 1);
    size_t returnedSize = 0;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  graphDesc.get(), _planWrapper.get(), size - 1, &returnedSize, tooSmall.data()),
              HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT);
}

TEST_F(TestGraphAndPlanSerializationApi, ComboSerializeRejectsNullSizePointer)
{
    auto graphBlob = makeBareGraphBlob();
    hipdnnBackendDescriptor_t rawGraph = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateAndDeserializeGraph_ext(&rawGraph, graphBlob.data(), graphBlob.size()),
        HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor graphDesc(rawGraph);

    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  graphDesc.get(), nullptr, 0, nullptr, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphAndPlanSerializationApi, ComboSerializeRejectsNullGraphDescriptor)
{
    size_t size = 0;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(nullptr, nullptr, 0, &size, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// ============================================================================
// Query contents (hipdnnBackendGetSerializedBinaryContents_ext)
// ============================================================================

TEST_F(TestGraphAndPlanSerializationApi, QueryContentsContainerWithPlanReturnsGraphAndPlanFlags)
{
    auto graphBlob = makeBareGraphBlob();
    auto planBlob = makeBarePlanBlob();
    auto container = makeContainer({graphBlob.data(), graphBlob.data() + graphBlob.size()},
                                   {planBlob.data(), planBlob.data() + planBlob.size()});

    int flags = -1;
    EXPECT_EQ(
        hipdnnBackendGetSerializedBinaryContents_ext(container.data(), container.size(), &flags),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(flags, HIPDNN_SERIALIZED_CONTENT_GRAPH | HIPDNN_SERIALIZED_CONTENT_EXECUTION_PLAN);
}

TEST_F(TestGraphAndPlanSerializationApi, QueryContentsContainerWithoutPlanReturnsGraphFlagOnly)
{
    auto graphBlob = makeBareGraphBlob();
    auto container = makeContainer({graphBlob.data(), graphBlob.data() + graphBlob.size()}, {});

    int flags = -1;
    EXPECT_EQ(
        hipdnnBackendGetSerializedBinaryContents_ext(container.data(), container.size(), &flags),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(flags, HIPDNN_SERIALIZED_CONTENT_GRAPH);
    EXPECT_EQ(flags & HIPDNN_SERIALIZED_CONTENT_EXECUTION_PLAN, 0);
}

TEST_F(TestGraphAndPlanSerializationApi, QueryContentsLegacyBareGraphReturnsGraphFlagOnly)
{
    auto graphBlob = makeBareGraphBlob();
    hipdnnBackendDescriptor_t rawGraph = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateAndDeserializeGraph_ext(&rawGraph, graphBlob.data(), graphBlob.size()),
        HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor graphDesc(rawGraph);

    // Produce a legacy bare-graph blob via the existing graph serializer.
    size_t size = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(graphDesc.get(), 0, &size, nullptr),
              HIPDNN_STATUS_SUCCESS);
    std::vector<uint8_t> legacyBlob(size);
    ASSERT_EQ(
        hipdnnBackendGetSerializedBinaryGraph_ext(graphDesc.get(), size, &size, legacyBlob.data()),
        HIPDNN_STATUS_SUCCESS);

    int flags = -1;
    EXPECT_EQ(
        hipdnnBackendGetSerializedBinaryContents_ext(legacyBlob.data(), legacyBlob.size(), &flags),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(flags, HIPDNN_SERIALIZED_CONTENT_GRAPH);
}

TEST_F(TestGraphAndPlanSerializationApi, QueryContentsShortBlobTreatedAsLegacyGraph)
{
    std::array<uint8_t, 4> shortBlob{1, 2, 3, 4};

    int flags = -1;
    EXPECT_EQ(
        hipdnnBackendGetSerializedBinaryContents_ext(shortBlob.data(), shortBlob.size(), &flags),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(flags, HIPDNN_SERIALIZED_CONTENT_GRAPH);
}

TEST_F(TestGraphAndPlanSerializationApi, QueryContentsGarbageWithoutIdentifierTreatedAsLegacyGraph)
{
    std::array<uint8_t, 8> garbage{0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03};

    int flags = -1;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryContents_ext(garbage.data(), garbage.size(), &flags),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(flags, HIPDNN_SERIALIZED_CONTENT_GRAPH);
}

TEST_F(TestGraphAndPlanSerializationApi, QueryContentsIdentifiedButCorruptContainerRejected)
{
    auto corrupt = makeIdentifiedButCorruptContainer();

    int flags = -1;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryContents_ext(corrupt.data(), corrupt.size(), &flags),
              HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphAndPlanSerializationApi, QueryContentsBarePlanBlobTreatedAsLegacyGraph)
{
    // A bare SerializedExecutionPlan blob declares no file_identifier, so the
    // unified query path classifies it as a legacy GRAPH. Bare plan blobs must
    // NOT be routed through this path; they stay on deserialize_compiled_plan.
    auto planBlob = makeBarePlanBlob();

    int flags = -1;
    EXPECT_EQ(
        hipdnnBackendGetSerializedBinaryContents_ext(planBlob.data(), planBlob.size(), &flags),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(flags, HIPDNN_SERIALIZED_CONTENT_GRAPH);
}

// ============================================================================
// Query contents error paths
// ============================================================================

TEST_F(TestGraphAndPlanSerializationApi, QueryContentsRejectsNullArgs)
{
    int flags = 0;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryContents_ext(nullptr, 0, &flags),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    const uint8_t someByte = 0;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryContents_ext(&someByte, 1, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// ============================================================================
// Graph deserialize (hipdnnBackendCreateAndDeserializeGraph_ext)
// ============================================================================

TEST_F(TestGraphAndPlanSerializationApi, DeserializeGraphFromContainerExtractsGraph)
{
    auto graphBlob = makeBareGraphBlob();
    auto planBlob = makeBarePlanBlob();
    auto container = makeContainer({graphBlob.data(), graphBlob.data() + graphBlob.size()},
                                   {planBlob.data(), planBlob.data() + planBlob.size()});

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateAndDeserializeGraph_ext(&rawDesc, container.data(), container.size()),
        HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor desc(rawDesc);

    verifyDeserializedGraphMatchesSource(desc.get());
}

TEST_F(TestGraphAndPlanSerializationApi, DeserializeGraphFromLegacyBareBlobUnchanged)
{
    auto graphBlob = makeBareGraphBlob();

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateAndDeserializeGraph_ext(&rawDesc, graphBlob.data(), graphBlob.size()),
        HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor desc(rawDesc);

    verifyDeserializedGraphMatchesSource(desc.get());
}

TEST_F(TestGraphAndPlanSerializationApi, DeserializeGraphRejectsContainerWithEmptyGraphBlob)
{
    auto planBlob = makeBarePlanBlob();
    auto container = makeContainer({}, {planBlob.data(), planBlob.data() + planBlob.size()});

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    EXPECT_EQ(
        hipdnnBackendCreateAndDeserializeGraph_ext(&rawDesc, container.data(), container.size()),
        HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(rawDesc, nullptr);
}

TEST_F(TestGraphAndPlanSerializationApi, DeserializeGraphIdentifiedButCorruptContainerRejected)
{
    auto corrupt = makeIdentifiedButCorruptContainer();

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    EXPECT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(&rawDesc, corrupt.data(), corrupt.size()),
              HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(rawDesc, nullptr);
}

// ============================================================================
// Plan deserialize (hipdnnBackendCreateAndDeserializeExecutionPlan_ext)
//
// The C API consults handle->getPluginResourceManager(), so MockHandle is
// bridged in via reinterpret_cast<hipdnnHandle_t>(&mockHandle). Cases that fail
// before the resource manager is reached use a NiceMock + Times(AtMost(1)).
// ============================================================================

TEST_F(TestGraphAndPlanSerializationApi, DeserializeExecutionPlanFromContainerExtractsPlan)
{
    auto graphBlob = makeBareGraphBlob();
    auto planBlob = makeBarePlanBlob();
    auto container = makeContainer({graphBlob.data(), graphBlob.data() + graphBlob.size()},
                                   {planBlob.data(), planBlob.data() + planBlob.size()});

    EXPECT_CALL(*_mockEnginePluginResourceManager,
                createExecutionContextFromSerialized(ENGINE_ID, _))
        .WillOnce(Return(getExecutionContext()));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyExecutionContext(_, _));

    MockHandle mockHandle;
    EXPECT_CALL(mockHandle, getPluginResourceManager())
        .WillRepeatedly(Return(_mockEnginePluginResourceManager));

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateAndDeserializeExecutionPlan_ext(
                  reinterpret_cast<hipdnnHandle_t>(&mockHandle),
                  &rawDesc,
                  container.data(),
                  container.size()),
              HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor desc(rawDesc);

    ASSERT_NE(desc.get(), nullptr);
    EXPECT_EQ(desc.get()->getType(), HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);

    // A finalized plan exposes a queryable workspace size.
    int64_t workspaceSize = 0;
    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc.get(),
                                        HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        &count,
                                        &workspaceSize),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(workspaceSize, 1024);
}

TEST_F(TestGraphAndPlanSerializationApi, DeserializeExecutionPlanFromBareBlobBackCompat)
{
    auto planBlob = makeBarePlanBlob();

    EXPECT_CALL(*_mockEnginePluginResourceManager,
                createExecutionContextFromSerialized(ENGINE_ID, _))
        .WillOnce(Return(getExecutionContext()));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyExecutionContext(_, _));

    MockHandle mockHandle;
    EXPECT_CALL(mockHandle, getPluginResourceManager())
        .WillRepeatedly(Return(_mockEnginePluginResourceManager));

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateAndDeserializeExecutionPlan_ext(
                  reinterpret_cast<hipdnnHandle_t>(&mockHandle),
                  &rawDesc,
                  planBlob.data(),
                  planBlob.size()),
              HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor desc(rawDesc);

    ASSERT_NE(desc.get(), nullptr);
    EXPECT_EQ(desc.get()->getType(), HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);

    int64_t workspaceSize = 0;
    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc.get(),
                                        HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        &count,
                                        &workspaceSize),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(workspaceSize, 1024);
}

TEST_F(TestGraphAndPlanSerializationApi,
       DeserializeExecutionPlanFromContainerWithoutPlanReturnsError)
{
    auto graphBlob = makeBareGraphBlob();
    auto container = makeContainer({graphBlob.data(), graphBlob.data() + graphBlob.size()}, {});

    // The plan-less container is rejected before the resource manager is
    // consulted, so the mock is permissive.
    NiceMock<MockHandle> mockHandle;
    EXPECT_CALL(mockHandle, getPluginResourceManager())
        .Times(AtMost(1))
        .WillRepeatedly(Return(_mockEnginePluginResourceManager));

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    EXPECT_EQ(hipdnnBackendCreateAndDeserializeExecutionPlan_ext(
                  reinterpret_cast<hipdnnHandle_t>(&mockHandle),
                  &rawDesc,
                  container.data(),
                  container.size()),
              HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(rawDesc, nullptr);
}

TEST_F(TestGraphAndPlanSerializationApi,
       DeserializeExecutionPlanIdentifiedButCorruptContainerRejected)
{
    auto corrupt = makeIdentifiedButCorruptContainer();

    // The corrupt container is rejected during verification, before the resource
    // manager is consulted, so the mock is permissive.
    NiceMock<MockHandle> mockHandle;
    EXPECT_CALL(mockHandle, getPluginResourceManager())
        .Times(AtMost(1))
        .WillRepeatedly(Return(_mockEnginePluginResourceManager));

    hipdnnBackendDescriptor_t rawDesc = nullptr;
    EXPECT_EQ(hipdnnBackendCreateAndDeserializeExecutionPlan_ext(
                  reinterpret_cast<hipdnnHandle_t>(&mockHandle),
                  &rawDesc,
                  corrupt.data(),
                  corrupt.size()),
              HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(rawDesc, nullptr);
}
