// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/GraphContentKey.hpp>

#include "ContentCarryingTestGraph.hpp"
#include "KernelIngestorTestFixtures.hpp"

namespace hipdnn_plugin_sdk::ingestor::testing
{
namespace
{

using Spec = ContentCarryingTestGraph::Spec;

/// The relation under test: two graphs are equal when a kernel measurement taken on one
/// is a valid measurement for the other.
///
/// MAINTENANCE: this file is the field-set pin for the graph half of the key. The
/// traversal is generated from `graph.fbs`, so a new field is hashed automatically --
/// what it cannot decide is whether that participation is *correct*. A field that
/// changes kernel behaviour needs a discriminates-on case here; one that does not needs
/// `(cache_ignore)` in the schema and an equals-case here. Pinned: every `Graph` field
/// the fixture can express, the tensor fields the traversal reads (uid, dtype, dims,
/// strides), and the node fields it reads (name, compute dtype, attribute discriminant,
/// two payload fields). Not pinned: the interiors of `TensorAttributesT` and each
/// `NodeAttributes` union member -- walked by the generated traversal and covered
/// automatically when the schema regenerates.
GraphContentKey keyFor(const ContentCarryingTestGraph& graph)
{
    return GraphContentKey{graph};
}

TEST(TestIngestorGraphContentKey, IdenticalContentComparesEqual)
{
    const ContentCarryingTestGraph first{Spec{}};
    const ContentCarryingTestGraph second{Spec{}};

    EXPECT_EQ(keyFor(first), keyFor(second));
    EXPECT_EQ(keyFor(first).hash(), keyFor(second).hash());
}

// Graph.id is minted per finalize, so two runs of the same computation produce
// different ids. If the key saw it, every lookup would miss.
TEST(TestIngestorGraphContentKey, ADifferentGraphIdStillComparesEqual)
{
    Spec first;
    first.graphId = makeGraphId(0xA1);
    Spec second;
    second.graphId = makeGraphId(0xB2);

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{first}), keyFor(ContentCarryingTestGraph{second}));
}

// The preferred engine selects who runs the computation, never what is computed, so a
// measurement transfers across it.
TEST(TestIngestorGraphContentKey, ADifferentPreferredEngineIdStillComparesEqual)
{
    Spec first;
    first.preferredEngineId = 7;
    Spec second;
    second.preferredEngineId = 99;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{first}), keyFor(ContentCarryingTestGraph{second}));
}

// The flag permits shapes to be overridden; it does not change them. The dims and
// strides that will actually run are compared in full either way.
TEST(TestIngestorGraphContentKey, ADifferentOverrideShapeFlagStillComparesEqual)
{
    Spec enabled;
    enabled.isOverrideShapeEnabled = true;
    const Spec disabled;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{enabled}),
              keyFor(ContentCarryingTestGraph{disabled}));
}

/// The production shape of the case above: the backend derives the stamped version from
/// `is_override_shape_enabled` (PluginVersionConstants.hpp:58-93), so two real graphs
/// differing only in that flag carry *different* versions, and the generated operator==
/// compares it. Without clearing it, the exclusion above is defeated in production while
/// its own test still passes.
TEST(TestIngestorGraphContentKey, TheDerivedApiVersionDoesNotDefeatTheOverrideShapeExclusion)
{
    Spec withOverride;
    withOverride.isOverrideShapeEnabled = true;
    withOverride.minRequiredApiVersion = hipdnn_data_sdk::utilities::Version{1, 1, 0};

    Spec without;
    without.isOverrideShapeEnabled = false;
    without.minRequiredApiVersion = hipdnn_data_sdk::utilities::Version{1, 0, 0};

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{withOverride}),
              keyFor(ContentCarryingTestGraph{without}))
        << "a measurement transfers across the override-shape flag, so the version it "
           "stamps must not split the key";
}

/// The version is excluded because it is *derived*: its content-bearing inputs are each
/// compared directly on the tensors that carry them.
TEST(TestIngestorGraphContentKey, ADifferentApiVersionAloneDoesNotSplitTheKey)
{
    Spec early;
    early.minRequiredApiVersion = hipdnn_data_sdk::utilities::Version{1, 0, 0};
    Spec later;
    later.minRequiredApiVersion = hipdnn_data_sdk::utilities::Version{9, 9, 9};

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{early}), keyFor(ContentCarryingTestGraph{later}))
        << "the version is a derived summary; the facts it summarises are compared on "
           "their own";
}

TEST(TestIngestorGraphContentKey, ADifferentTensorShapeComparesUnequal)
{
    Spec narrow;
    narrow.tensors[0].dims = {4, 8};
    Spec wide;
    wide.tensors[0].dims = {4, 16};

    EXPECT_NE(keyFor(ContentCarryingTestGraph{narrow}), keyFor(ContentCarryingTestGraph{wide}));
}

TEST(TestIngestorGraphContentKey, ADifferentTensorStrideComparesUnequal)
{
    Spec packed;
    packed.tensors[0].strides = {8, 1};
    Spec strided;
    strided.tensors[0].strides = {16, 1};

    EXPECT_NE(keyFor(ContentCarryingTestGraph{packed}), keyFor(ContentCarryingTestGraph{strided}));
}

TEST(TestIngestorGraphContentKey, ADifferentTensorDataTypeComparesUnequal)
{
    Spec asFloat;
    asFloat.tensors[0].dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    Spec asHalf;
    asHalf.tensors[0].dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

TEST(TestIngestorGraphContentKey, ADifferentTensorUidComparesUnequal)
{
    Spec first;
    first.tensors[0].uid = 1;
    Spec second;
    second.tensors[0].uid = 42;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{first}), keyFor(ContentCarryingTestGraph{second}));
}

TEST(TestIngestorGraphContentKey, ADifferentNodeCountComparesUnequal)
{
    const Spec single;
    Spec doubled;
    doubled.nodes = {ContentCarryingTestGraph::NodeSpec{}, ContentCarryingTestGraph::NodeSpec{}};

    EXPECT_NE(keyFor(ContentCarryingTestGraph{single}), keyFor(ContentCarryingTestGraph{doubled}));
}

TEST(TestIngestorGraphContentKey, ADifferentNodeComputeDataTypeComparesUnequal)
{
    const Spec asFloat;
    Spec asHalf;
    asHalf.nodes[0].computeDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

// Inside the union payload, not merely its discriminant: an ADD and a MUL are the same
// node type and would collide on any key that stopped at attributes_type.
TEST(TestIngestorGraphContentKey, ADifferentPointwiseOperationComparesUnequal)
{
    const Spec addition;
    Spec multiplication;
    multiplication.nodes[0].operation = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{addition}),
              keyFor(ContentCarryingTestGraph{multiplication}));
}

TEST(TestIngestorGraphContentKey, ADifferentGraphComputeDataTypeComparesUnequal)
{
    const Spec asFloat;
    Spec asHalf;
    asHalf.computeDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

// fnv1aHash collapses null and empty input to sentinel 0, so a key of 0 would alias
// every contentless graph onto one bucket -- the version tag and node count emitted
// ahead of any content are what prevent it.
TEST(TestIngestorGraphContentKey, AnEmptyGraphKeysToANonZeroHash)
{
    const TestGraph empty(makeGraphId(0xC3));

    EXPECT_NE(GraphContentKey{empty}.hash(), 0U);
}

TEST(TestIngestorGraphContentKey, AnEmptyGraphStillComparesEqualToAnotherEmptyGraph)
{
    const TestGraph first(makeGraphId(0xC4));
    const TestGraph second(makeGraphId(0xC5));

    EXPECT_EQ(GraphContentKey{first}, GraphContentKey{second});
}

/// Names are excluded, marked `(cache_ignore)` in graph.fbs, which is what this pins:
/// two identically-shaped graphs run the same kernels whatever they are called, so a
/// measurement transfers regardless of name.
TEST(TestIngestorGraphContentKey, ADifferentGraphNameComparesEqual)
{
    const Spec named;
    Spec renamed;
    renamed.name = "a_different_graph";

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{named}), keyFor(ContentCarryingTestGraph{renamed}));
}

TEST(TestIngestorGraphContentKey, ADifferentGraphIoDataTypeComparesUnequal)
{
    const Spec asFloat;
    Spec asHalf;
    asHalf.ioDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

TEST(TestIngestorGraphContentKey, ADifferentGraphIntermediateDataTypeComparesUnequal)
{
    const Spec asFloat;
    Spec asHalf;
    asHalf.intermediateDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

/// Same reasoning as the graph name above, one level down: the node's *attributes* are
/// compared in full, so excluding the label does not weaken discrimination.
TEST(TestIngestorGraphContentKey, ADifferentNodeNameComparesEqual)
{
    const Spec named;
    Spec renamed;
    renamed.nodes[0].name = "a_different_node";

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{named}), keyFor(ContentCarryingTestGraph{renamed}));
}

/// Inside the union payload again, on a field the discriminant cannot distinguish: two
/// adds wired to different tensors are different computations.
TEST(TestIngestorGraphContentKey, ADifferentPointwiseOperandUidComparesUnequal)
{
    const Spec wired;
    Spec rewired;
    rewired.nodes[0].in0TensorUid = 99;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{wired}), keyFor(ContentCarryingTestGraph{rewired}));
}

TEST(TestIngestorGraphContentKey, AnEmptyGraphDoesNotMatchAContentCarryingOne)
{
    const TestGraph empty(makeGraphId(0xC6));
    const ContentCarryingTestGraph populated{Spec{}};

    EXPECT_NE(GraphContentKey{empty}, keyFor(populated));
}

/// An IGraph that does **not** override `bytes()`: stands in for an out-of-tree
/// implementor written before the method existed. It must compile (non-breaking) and
/// its key must be unusable and match nothing, including another unkeyable graph.
class UnkeyableGraph : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph
{
public:
    const hipdnn_flatbuffers_sdk::data_objects::Graph& getGraph() const override
    {
        throw std::logic_error("UnkeyableGraph carries no graph");
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
        std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes> /*supported*/) const override
    {
        return true;
    }
    const hipdnn_flatbuffers_sdk::data_objects::Node& getNode(uint32_t /*index*/) const override
    {
        throw std::logic_error("UnkeyableGraph carries no nodes");
    }
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper&
        getNodeWrapper(uint32_t /*index*/) const override
    {
        throw std::logic_error("UnkeyableGraph carries no nodes");
    }
    const std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper>>&
        nodeWrappers() const override
    {
        throw std::logic_error("UnkeyableGraph carries no nodes");
    }
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        getTensorMap() const override
    {
        return _tensors;
    }

private:
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
        _tensors;
};

TEST(TestIngestorGraphContentKey, AGraphWithNoBytesYieldsAnUnusableKey)
{
    const UnkeyableGraph graph;

    EXPECT_FALSE(GraphContentKey{graph}.isUsable());
}

TEST(TestIngestorGraphContentKey, TwoUnkeyableGraphsDoNotMatchEachOther)
{
    const UnkeyableGraph first;
    const UnkeyableGraph second;

    EXPECT_NE(GraphContentKey{first}, GraphContentKey{second})
        << "absence of content is a permanent miss, never a wildcard that matches "
           "every other unkeyable graph";
}

TEST(TestIngestorGraphContentKey, AnUnkeyableGraphDoesNotMatchARealOne)
{
    const UnkeyableGraph unkeyable;
    const ContentCarryingTestGraph real{Spec{}};

    EXPECT_NE(GraphContentKey{unkeyable}, keyFor(real));
}

/// The hash narrows; the content decides. Hashes are forced to agree so the structural
/// comparison is actually reached -- `operator==` short-circuits on a hash mismatch.
TEST(TestIngestorGraphContentKey, EqualHashesWithDifferentContentStillCompareUnequal)
{
    struct CollidingKey : GraphContentKey
    {
        using GraphContentKey::GraphContentKey;

        static CollidingKey with(const ContentCarryingTestGraph& graph, uint64_t forcedHash)
        {
            CollidingKey key{graph};
            key.forceHash(forcedHash);
            return key;
        }
    };

    Spec narrow;
    narrow.tensors[0].dims = {4, 8};
    Spec wide;
    wide.tensors[0].dims = {4, 16};

    const ContentCarryingTestGraph first{narrow};
    const ContentCarryingTestGraph second{wide};

    const auto firstKey = CollidingKey::with(first, 0xC0FFEE);
    const auto secondKey = CollidingKey::with(second, 0xC0FFEE);

    ASSERT_EQ(firstKey.hash(), secondKey.hash()) << "the collision must actually be forced";
    EXPECT_NE(static_cast<const GraphContentKey&>(firstKey),
              static_cast<const GraphContentKey&>(secondKey))
        << "a hash collision with differing content must resolve to a miss, never a hit";
}

/// The same forcing, in the direction that must still match: equal content plus equal
/// hashes is a genuine hit.
TEST(TestIngestorGraphContentKey, EqualHashesWithEqualContentStillCompareEqual)
{
    struct CollidingKey : GraphContentKey
    {
        using GraphContentKey::GraphContentKey;

        static CollidingKey with(const ContentCarryingTestGraph& graph, uint64_t forcedHash)
        {
            CollidingKey key{graph};
            key.forceHash(forcedHash);
            return key;
        }
    };

    const ContentCarryingTestGraph first{Spec{}};
    const ContentCarryingTestGraph second{Spec{}};

    const auto firstKey = CollidingKey::with(first, 0xC0FFEE);
    const auto secondKey = CollidingKey::with(second, 0xC0FFEE);

    EXPECT_EQ(static_cast<const GraphContentKey&>(firstKey),
              static_cast<const GraphContentKey&>(secondKey));
}

TEST(TestIngestorGraphContentKey, StdHashAgreesWithTheKeysOwnHash)
{
    const ContentCarryingTestGraph graph{Spec{}};
    const auto key = keyFor(graph);

    EXPECT_EQ(std::hash<GraphContentKey>{}(key), static_cast<size_t>(key.hash()));
}

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
