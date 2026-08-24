// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <set>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;

TEST(TestGraphWrapper, NullBufferIsInvalid)
{
    const GraphWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.getGraph(), std::invalid_argument);
}

TEST(TestGraphWrapper, NonGraphBufferIsInvalid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidEngineDetails(123);
    auto serializedGraph = builder.Release();

    const GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());

    EXPECT_FALSE(wrapper.isValid());
}

TEST(TestGraphWrapper, ValidGraphReturnsCorrectNodeCountForEmptyGraph)
{
    flatbuffers::FlatBufferBuilder builder = hipdnn_test_sdk::utilities::createEmptyValidGraph();
    auto serializedGraph = builder.Release();

    const GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.nodeCount(), 0);
}

TEST(TestGraphWrapper, ValidGraphReturnsCorrectNodeCount)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();

    const GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.nodeCount(), 1);
}

TEST(TestGraphWrapper, HasSupportedTypesReturnsTrueIfAllSupported)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();

    const GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());

    std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes> supported
        = {hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes};
    EXPECT_TRUE(wrapper.hasOnlySupportedAttributes(supported));

    supported.insert(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes);
    EXPECT_TRUE(wrapper.hasOnlySupportedAttributes(supported));
}

TEST(TestGraphWrapper, HasSupportedTypesReturnsFalseIfAnyUnsupported)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();

    const GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());

    std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes> supported
        = {hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes};
    EXPECT_FALSE(wrapper.hasOnlySupportedAttributes(supported));

    supported.insert(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes);
    EXPECT_FALSE(wrapper.hasOnlySupportedAttributes(supported));
}

TEST(TestGraphWrapper, GetTensorMapEmptyGraph)
{
    flatbuffers::FlatBufferBuilder builder = hipdnn_test_sdk::utilities::createEmptyValidGraph();
    auto serializedGraph = builder.Release();

    const GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());
    ASSERT_TRUE(wrapper.isValid());

    const auto& tensorMap = wrapper.getTensorMap();
    EXPECT_TRUE(tensorMap.empty());
}

TEST(TestGraphWrapper, GetTensorMapReturnsCorrectTensors)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    const std::vector<int64_t> strides = {1, 1, 1, 1};
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    auto graph = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(builder,
                                                                         "test",
                                                                         DataType::FLOAT,
                                                                         DataType::HALF,
                                                                         DataType::BFLOAT16,
                                                                         &tensorAttributes,
                                                                         &nodes);
    builder.Finish(graph);

    auto serializedGraph = builder.Release();
    const GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());
    ASSERT_TRUE(wrapper.isValid());

    const auto& tensorMap = wrapper.getTensorMap();
    EXPECT_EQ(tensorMap.size(), 2);
    EXPECT_NE(tensorMap.find(1), tensorMap.end());
    EXPECT_NE(tensorMap.find(2), tensorMap.end());
    EXPECT_EQ(tensorMap.at(1)->uid(), 1);
    EXPECT_EQ(tensorMap.at(2)->uid(), 2);
}

TEST(TestGraphWrapper, GetNodeWrapper)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();

    const GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());
    ASSERT_TRUE(wrapper.isValid());
    ASSERT_EQ(wrapper.nodeCount(), 1);

    const auto& nodeWrapper = wrapper.getNodeWrapper(0);

    EXPECT_EQ(nodeWrapper.attributesType(),
              hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes);
    EXPECT_THROW(wrapper.getNodeWrapper(1), std::out_of_range);
}

/// `bytes()` is the cache-key seam: `GraphContentKey` reads the buffer back through it
/// and calls `GetRoot` without re-verifying, so a buffer is retained only once the
/// verifier has accepted it.
TEST(TestGraphWrapper, BytesReturnsTheVerifiedBuffer)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();

    const GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());
    ASSERT_TRUE(wrapper.isValid());

    const auto bytes = wrapper.bytes();
    EXPECT_EQ(bytes.data, serializedGraph.data());
    EXPECT_EQ(bytes.size, serializedGraph.size());
}

TEST(TestGraphWrapper, BytesIsEmptyForANullBuffer)
{
    const GraphWrapper wrapper(nullptr, 0);
    ASSERT_FALSE(wrapper.isValid());

    const auto bytes = wrapper.bytes();
    EXPECT_EQ(bytes.data, nullptr);
    EXPECT_EQ(bytes.size, 0U);
}

/// A buffer the verifier rejects must not reach bytes(): handing it out would let a
/// caller GetRoot() a buffer getGraph() itself refuses to read.
TEST(TestGraphWrapper, BytesIsEmptyWhenVerificationFails)
{
    auto builder = hipdnn_test_sdk::utilities::createValidEngineDetails(123);
    auto serializedGraph = builder.Release();

    const GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());
    ASSERT_FALSE(wrapper.isValid());

    const auto bytes = wrapper.bytes();
    EXPECT_EQ(bytes.data, nullptr);
    EXPECT_EQ(bytes.size, 0U);
}

/// Every `IGraph` implementation in this codebase overrides `bytes()` (`GraphWrapper` here,
/// plus the test doubles under plugin_sdk/tests/ingestor), so without this test the
/// base-class default body never runs. It stands in for an implementation -- in-tree or
/// out-of-tree -- that predates `bytes()` or simply forgets to override it.
class GraphWithNoBytesOverride : public IGraph
{
public:
    const hipdnn_flatbuffers_sdk::data_objects::Graph& getGraph() const override
    {
        throw std::logic_error("GraphWithNoBytesOverride carries no graph");
    }
    bool isValid() const override
    {
        return false;
    }
    uint32_t nodeCount() const override
    {
        return 0;
    }
    bool hasOnlySupportedAttributes(
        std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes> /*supportedAttributes*/)
        const override
    {
        return true;
    }
    const hipdnn_flatbuffers_sdk::data_objects::Node& getNode(uint32_t /*index*/) const override
    {
        throw std::logic_error("GraphWithNoBytesOverride carries no nodes");
    }
    const INodeWrapper& getNodeWrapper(uint32_t /*index*/) const override
    {
        throw std::logic_error("GraphWithNoBytesOverride carries no nodes");
    }
    const std::vector<std::unique_ptr<INodeWrapper>>& nodeWrappers() const override
    {
        throw std::logic_error("GraphWithNoBytesOverride carries no nodes");
    }
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        getTensorMap() const override
    {
        return _tensors;
    }

    // Deliberately does not override bytes(): this class exists to exercise IGraph's default.

private:
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
        _tensors;
};

/// An implementation that does not supply bytes() must report empty rather than garbage,
/// so `GraphContentKey` declines to cache or match under it instead of treating it as an
/// empty graph that matches every other unkeyable one.
TEST(TestGraphWrapper, ImplementationWithNoBytesOverrideReportsEmpty)
{
    const GraphWithNoBytesOverride graph;

    const auto bytes = graph.bytes();
    EXPECT_EQ(bytes.data, nullptr);
    EXPECT_EQ(bytes.size, 0U);
}
