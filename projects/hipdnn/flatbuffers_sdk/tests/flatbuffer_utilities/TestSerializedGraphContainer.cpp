// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/serialized_graph_and_plan_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/SerializedGraphContainer.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

// Wraps an inner graph buffer (and optional plan bytes) into an "HDGP"
// SerializedGraphAndPlan container via the production write helper, returning the
// released buffer. Using buildGraphAndPlanContainer (not CreateVector directly)
// keeps these tests on the same force-aligned path the backend ships.
flatbuffers::DetachedBuffer buildContainer(const std::vector<uint8_t>* graphBytes,
                                           const std::vector<uint8_t>* planBytes)
{
    const uint8_t* graphData
        = (graphBytes != nullptr && !graphBytes->empty()) ? graphBytes->data() : nullptr;
    const size_t graphSize = (graphBytes != nullptr) ? graphBytes->size() : 0;
    const uint8_t* planData
        = (planBytes != nullptr && !planBytes->empty()) ? planBytes->data() : nullptr;
    const size_t planSize = (planBytes != nullptr) ? planBytes->size() : 0;
    return buildGraphAndPlanContainer(graphData, graphSize, planData, planSize);
}

std::vector<uint8_t> toVector(const uint8_t* data, size_t size)
{
    return {data, data + size};
}

} // namespace

TEST(TestSerializedGraphContainer, BareGraphIsNotContainer)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto bareGraph = builder.Release();

    EXPECT_FALSE(isGraphAndPlanContainer(bareGraph.data(), bareGraph.size()));
}

TEST(TestSerializedGraphContainer, BareGraphExtractGraphPassthrough)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto bareGraph = builder.Release();

    const auto view = extractGraphBlob(bareGraph.data(), bareGraph.size());
    EXPECT_EQ(view.data, bareGraph.data());
    EXPECT_EQ(view.size, bareGraph.size());
}

TEST(TestSerializedGraphContainer, BareGraphExtractPlanPassthrough)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto bareGraph = builder.Release();

    const auto view = extractPlanBlob(bareGraph.data(), bareGraph.size());
    EXPECT_EQ(view.data, bareGraph.data());
    EXPECT_EQ(view.size, bareGraph.size());
}

TEST(TestSerializedGraphContainer, BareGraphFromSerializedBlobValid)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto bareGraph = builder.Release();

    const auto wrapper = GraphWrapper::fromSerializedBlob(bareGraph.data(), bareGraph.size());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.nodeCount(), 1);
}

TEST(TestSerializedGraphContainer, ContainerIsDetected)
{
    flatbuffers::FlatBufferBuilder graphBuilder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto innerGraph = graphBuilder.Release();
    const auto graphBytes = toVector(innerGraph.data(), innerGraph.size());
    const std::vector<uint8_t> planBytes = {0x01, 0x02, 0x03, 0x04};

    auto container = buildContainer(&graphBytes, &planBytes);

    EXPECT_TRUE(isGraphAndPlanContainer(container.data(), container.size()));
}

TEST(TestSerializedGraphContainer, ContainerExtractGraphPeelsInnerGraph)
{
    flatbuffers::FlatBufferBuilder graphBuilder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto innerGraph = graphBuilder.Release();
    const auto graphBytes = toVector(innerGraph.data(), innerGraph.size());
    const std::vector<uint8_t> planBytes = {0x01, 0x02, 0x03, 0x04};

    auto container = buildContainer(&graphBytes, &planBytes);

    const auto view = extractGraphBlob(container.data(), container.size());
    ASSERT_NE(view.data, nullptr);
    EXPECT_EQ(view.size, graphBytes.size());
    EXPECT_EQ(toVector(view.data, view.size), graphBytes);

    const GraphWrapper wrapper(view.data, view.size);
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.nodeCount(), 1);
}

TEST(TestSerializedGraphContainer, ContainerExtractPlanReturnsPlanBytes)
{
    flatbuffers::FlatBufferBuilder graphBuilder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto innerGraph = graphBuilder.Release();
    const auto graphBytes = toVector(innerGraph.data(), innerGraph.size());
    const std::vector<uint8_t> planBytes = {0x01, 0x02, 0x03, 0x04, 0x05};

    auto container = buildContainer(&graphBytes, &planBytes);

    const auto view = extractPlanBlob(container.data(), container.size());
    ASSERT_NE(view.data, nullptr);
    ASSERT_EQ(view.size, planBytes.size());
    EXPECT_EQ(toVector(view.data, view.size), planBytes);
}

TEST(TestSerializedGraphContainer, ContainerFromSerializedBlobValid)
{
    flatbuffers::FlatBufferBuilder graphBuilder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto innerGraph = graphBuilder.Release();
    const auto graphBytes = toVector(innerGraph.data(), innerGraph.size());
    const std::vector<uint8_t> planBytes = {0x01, 0x02, 0x03, 0x04};

    auto container = buildContainer(&graphBytes, &planBytes);

    const auto wrapper = GraphWrapper::fromSerializedBlob(container.data(), container.size());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.nodeCount(), 1);
}

TEST(TestSerializedGraphContainer, ContainerGraphOnlyExtractPlanReturnsEmpty)
{
    flatbuffers::FlatBufferBuilder graphBuilder
        = hipdnn_test_sdk::utilities::createEmptyValidGraph();
    auto innerGraph = graphBuilder.Release();
    const auto graphBytes = toVector(innerGraph.data(), innerGraph.size());

    auto container = buildContainer(&graphBytes, nullptr);

    // The graph blob is mandatory and present, so extraction succeeds.
    EXPECT_NO_THROW(extractGraphBlob(container.data(), container.size()));

    // The plan blob is optional: a graph-only container yields an empty view
    // rather than throwing.
    const auto planView = extractPlanBlob(container.data(), container.size());
    EXPECT_EQ(planView.data, nullptr);
    EXPECT_EQ(planView.size, 0u);
}

TEST(TestSerializedGraphContainer, ContainerPlanOnlyExtractGraphThrows)
{
    const std::vector<uint8_t> planBytes = {0x01, 0x02, 0x03, 0x04};

    auto container = buildContainer(nullptr, &planBytes);

    EXPECT_THROW(extractGraphBlob(container.data(), container.size()), std::invalid_argument);
    EXPECT_NO_THROW(extractPlanBlob(container.data(), container.size()));
}

TEST(TestSerializedGraphContainer, SmallBufferIsNotContainer)
{
    const std::vector<uint8_t> tiny = {0x01, 0x02, 0x03};

    EXPECT_FALSE(isGraphAndPlanContainer(tiny.data(), tiny.size()));

    const auto graphView = extractGraphBlob(tiny.data(), tiny.size());
    EXPECT_EQ(graphView.data, tiny.data());
    EXPECT_EQ(graphView.size, tiny.size());

    const auto planView = extractPlanBlob(tiny.data(), tiny.size());
    EXPECT_EQ(planView.data, tiny.data());
    EXPECT_EQ(planView.size, tiny.size());
}

TEST(TestSerializedGraphContainer, NullBufferIsNotContainer)
{
    EXPECT_FALSE(isGraphAndPlanContainer(nullptr, 0));
}

TEST(TestSerializedGraphContainer, NonContainerBufferPassthrough)
{
    // An 8+ byte buffer that does not carry the "HDGP" identifier is treated as
    // a bare blob and passed through without throwing.
    const std::vector<uint8_t> bytes(16, 0x00);

    EXPECT_FALSE(isGraphAndPlanContainer(bytes.data(), bytes.size()));

    const auto graphView = extractGraphBlob(bytes.data(), bytes.size());
    EXPECT_EQ(graphView.data, bytes.data());
    EXPECT_EQ(graphView.size, bytes.size());

    const auto planView = extractPlanBlob(bytes.data(), bytes.size());
    EXPECT_EQ(planView.data, bytes.data());
    EXPECT_EQ(planView.size, bytes.size());
}

TEST(TestSerializedGraphContainer, TruncatedContainerWithIdentifierThrows)
{
    flatbuffers::FlatBufferBuilder graphBuilder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto innerGraph = graphBuilder.Release();
    const auto graphBytes = toVector(innerGraph.data(), innerGraph.size());
    const std::vector<uint8_t> planBytes = {0x01, 0x02, 0x03, 0x04};

    auto container = buildContainer(&graphBytes, &planBytes);

    // Truncate the container while preserving the leading bytes that carry the
    // "HDGP" file identifier, so detection still fires but verification fails.
    const size_t truncatedSize = container.size() / 2;
    ASSERT_GE(truncatedSize, static_cast<size_t>(8));
    const std::vector<uint8_t> truncated(container.data(), container.data() + truncatedSize);

    EXPECT_TRUE(isGraphAndPlanContainer(truncated.data(), truncated.size()));
    EXPECT_THROW(verifyGraphAndPlanContainer(truncated.data(), truncated.size()),
                 std::invalid_argument);
    EXPECT_THROW(extractGraphBlob(truncated.data(), truncated.size()), std::invalid_argument);
}

TEST(TestSerializedGraphContainer, ContainerWithMalformedGraphBlobPeelsButWrapperInvalid)
{
    // A non-empty graph blob is peeled out without throwing, but the inner bytes
    // are garbage and fail the Graph verifier, so the wrapper is invalid.
    const std::vector<uint8_t> garbage = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04};
    const std::vector<uint8_t> planBytes = {0x01, 0x02, 0x03, 0x04};

    auto container = buildContainer(&garbage, &planBytes);

    EXPECT_NO_THROW(extractGraphBlob(container.data(), container.size()));

    const auto wrapper = GraphWrapper::fromSerializedBlob(container.data(), container.size());
    EXPECT_FALSE(wrapper.isValid());
}

TEST(TestSerializedGraphContainer, ContainerWithBothBlobsEmpty)
{
    auto container = buildContainer(nullptr, nullptr);

    EXPECT_TRUE(isGraphAndPlanContainer(container.data(), container.size()));

    // The graph blob is mandatory, so its absence throws.
    EXPECT_THROW(extractGraphBlob(container.data(), container.size()), std::invalid_argument);

    // The plan blob is optional, so its absence yields an empty view.
    const auto planView = extractPlanBlob(container.data(), container.size());
    EXPECT_EQ(planView.data, nullptr);
    EXPECT_EQ(planView.size, 0u);
}

TEST(TestSerializedGraphContainer, DifferentIdentifierIsNotContainer)
{
    // A buffer carrying a different 4-char flatbuffer identifier ("HDDP",
    // device-properties) is not an HDGP container, so it is passed through.
    std::vector<uint8_t> bytes(16, 0x00);
    bytes[4] = 'H';
    bytes[5] = 'D';
    bytes[6] = 'D';
    bytes[7] = 'P';

    EXPECT_FALSE(isGraphAndPlanContainer(bytes.data(), bytes.size()));

    const auto graphView = extractGraphBlob(bytes.data(), bytes.size());
    EXPECT_EQ(graphView.data, bytes.data());
    EXPECT_EQ(graphView.size, bytes.size());

    const auto planView = extractPlanBlob(bytes.data(), bytes.size());
    EXPECT_EQ(planView.data, bytes.data());
    EXPECT_EQ(planView.size, bytes.size());
}

TEST(TestSerializedGraphContainer, EightByteBoundary)
{
    // An 8-byte buffer is the minimum size detection inspects. Without the
    // "HDGP" identifier it is not a container and is passed through.
    const std::vector<uint8_t> nonContainer(8, 0x00);

    EXPECT_FALSE(isGraphAndPlanContainer(nonContainer.data(), nonContainer.size()));

    const auto graphView = extractGraphBlob(nonContainer.data(), nonContainer.size());
    EXPECT_EQ(graphView.data, nonContainer.data());
    EXPECT_EQ(graphView.size, nonContainer.size());

    const auto planView = extractPlanBlob(nonContainer.data(), nonContainer.size());
    EXPECT_EQ(planView.data, nonContainer.data());
    EXPECT_EQ(planView.size, nonContainer.size());

    // An 8-byte buffer that carries the "HDGP" identifier is detected as a
    // container, but it is not a real container so verification (and graph
    // extraction) throws.
    std::vector<uint8_t> withIdentifier(8, 0x00);
    withIdentifier[4] = 'H';
    withIdentifier[5] = 'D';
    withIdentifier[6] = 'G';
    withIdentifier[7] = 'P';

    EXPECT_TRUE(isGraphAndPlanContainer(withIdentifier.data(), withIdentifier.size()));
    EXPECT_THROW(verifyGraphAndPlanContainer(withIdentifier.data(), withIdentifier.size()),
                 std::invalid_argument);
    EXPECT_THROW(extractGraphBlob(withIdentifier.data(), withIdentifier.size()),
                 std::invalid_argument);
}

TEST(TestSerializedGraphContainer, InnerBlobsAreEightByteAligned)
{
    // Regression guard for the alignment fix: the embedded graph and plan blobs
    // are parsed in place as FlatBuffers roots that carry int64 fields, so their
    // peeled base addresses must be 8-byte aligned. A plain [ubyte] vector is
    // only 4-byte aligned; buildGraphAndPlanContainer force-aligns both to 8.
    flatbuffers::FlatBufferBuilder graphBuilder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto innerGraph = graphBuilder.Release();
    const auto graphBytes = toVector(innerGraph.data(), innerGraph.size());
    const std::vector<uint8_t> planBytes = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};

    // Graph + plan container: both peeled blobs must be 8-aligned.
    auto bothContainer = buildContainer(&graphBytes, &planBytes);
    const auto graphView = extractGraphBlob(bothContainer.data(), bothContainer.size());
    const auto planView = extractPlanBlob(bothContainer.data(), bothContainer.size());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(graphView.data) % 8, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(planView.data) % 8, 0u);

    // Graph-only container: the graph blob must still be 8-aligned.
    auto graphOnly = buildContainer(&graphBytes, nullptr);
    const auto graphOnlyView = extractGraphBlob(graphOnly.data(), graphOnly.size());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(graphOnlyView.data) % 8, 0u);
}
