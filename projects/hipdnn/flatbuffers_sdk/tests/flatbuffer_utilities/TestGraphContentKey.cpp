// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>

#include <optional>

#include "ContentCarryingTestGraph.hpp"

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing
{
namespace
{

using Spec = ContentCarryingTestGraph::Spec;

/// Two graphs are equal when a kernel measurement taken on one is valid for the other.
///
/// This file exercises every field the graph half of the key compares, one test per
/// field, following `graph.fbs`'s traversal: a field that changes kernel behaviour gets
/// a discriminates-on case; one that does not gets `(cache_ignore)` and an equals-case;
/// a tensor reference gets `(cache_uid)` and both -- equal under renumbering, unequal
/// under rewiring. Unlike `TestDeviceKey.cpp`'s structured-binding pin, nothing here
/// forces a new schema field to get a case: this coverage is maintained by hand.
GraphContentKey keyFor(const ContentCarryingTestGraph& graph)
{
    return GraphContentKey{graph};
}

TEST(TestGraphContentKey, IdenticalContentComparesEqual)
{
    const ContentCarryingTestGraph first{Spec{}};
    const ContentCarryingTestGraph second{Spec{}};

    EXPECT_EQ(keyFor(first), keyFor(second));
    EXPECT_EQ(keyFor(first).hash(), keyFor(second).hash());
}

// Graph.id is minted per finalize, so two runs of the same computation produce
// different ids. If the key saw it, every lookup would miss.
TEST(TestGraphContentKey, ADifferentGraphIdStillComparesEqual)
{
    Spec first;
    first.graphId = makeGraphId(0xA1);
    Spec second;
    second.graphId = makeGraphId(0xB2);

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{first}), keyFor(ContentCarryingTestGraph{second}));
}

// The preferred engine selects who runs the computation, never what is computed, so a
// measurement transfers across it.
TEST(TestGraphContentKey, ADifferentPreferredEngineIdStillComparesEqual)
{
    Spec first;
    first.preferredEngineId = 7;
    Spec second;
    second.preferredEngineId = 99;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{first}), keyFor(ContentCarryingTestGraph{second}));
}

// The flag permits shapes to be overridden; it does not change them. The dims and
// strides that will actually run are compared in full either way.
TEST(TestGraphContentKey, ADifferentOverrideShapeFlagStillComparesEqual)
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
TEST(TestGraphContentKey, TheDerivedApiVersionDoesNotDefeatTheOverrideShapeExclusion)
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
TEST(TestGraphContentKey, ADifferentApiVersionAloneDoesNotSplitTheKey)
{
    Spec early;
    early.minRequiredApiVersion = hipdnn_data_sdk::utilities::Version{1, 0, 0};
    Spec later;
    later.minRequiredApiVersion = hipdnn_data_sdk::utilities::Version{9, 9, 9};

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{early}), keyFor(ContentCarryingTestGraph{later}))
        << "the version is a derived summary; the facts it summarises are compared on "
           "their own";
}

TEST(TestGraphContentKey, ADifferentTensorShapeComparesUnequal)
{
    Spec narrow;
    narrow.tensors[0].dims = {4, 8};
    Spec wide;
    wide.tensors[0].dims = {4, 16};

    EXPECT_NE(keyFor(ContentCarryingTestGraph{narrow}), keyFor(ContentCarryingTestGraph{wide}));
}

TEST(TestGraphContentKey, ADifferentTensorStrideComparesUnequal)
{
    Spec packed;
    packed.tensors[0].strides = {8, 1};
    Spec strided;
    strided.tensors[0].strides = {16, 1};

    EXPECT_NE(keyFor(ContentCarryingTestGraph{packed}), keyFor(ContentCarryingTestGraph{strided}));
}

TEST(TestGraphContentKey, ADifferentTensorDataTypeComparesUnequal)
{
    Spec asFloat;
    asFloat.tensors[0].dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    Spec asHalf;
    asHalf.tensors[0].dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

/// Uids are caller-assigned labels, auto-filled when the caller leaves them unset
/// (GraphTensorIds.hpp:17-58), so one computation can carry different numbers.
TEST(TestGraphContentKey, ARenumberedGraphComparesEqual)
{
    Spec low;
    low.tensors
        = {ContentCarryingTestGraph::TensorSpec{1}, ContentCarryingTestGraph::TensorSpec{2}};
    low.nodes[0].in0TensorUid = 1;
    low.nodes[0].out0TensorUid = 2;

    Spec high;
    high.tensors
        = {ContentCarryingTestGraph::TensorSpec{1000}, ContentCarryingTestGraph::TensorSpec{2000}};
    high.nodes[0].in0TensorUid = 1000;
    high.nodes[0].out0TensorUid = 2000;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{low}), keyFor(ContentCarryingTestGraph{high}))
        << "uids are labels; the same wiring under different numbers is the same graph";
    EXPECT_EQ(keyFor(ContentCarryingTestGraph{low}).hash(),
              keyFor(ContentCarryingTestGraph{high}).hash());
}

/// Both graphs carry the same tensors in the same order and differ only in which operand
/// slot the second one feeds, so only the canonicalized reference distinguishes them.
TEST(TestGraphContentKey, MovingATensorToADifferentOperandSlotComparesUnequal)
{
    Spec asSecondInput;
    asSecondInput.nodes[0].in1TensorUid = 2;

    Spec asOutput;
    asOutput.nodes[0].out0TensorUid = 2;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{asSecondInput}),
              keyFor(ContentCarryingTestGraph{asOutput}))
        << "which slot a tensor feeds is content, and survives canonicalization";
}

/// Aliasing is invisible to tensor-list position: both graphs walk the same tensors in
/// the same order, and only the operand references record that one add reads one tensor
/// twice.
TEST(TestGraphContentKey, AliasingTwoOperandsOntoOneTensorComparesUnequal)
{
    Spec distinct;
    distinct.nodes[0].in0TensorUid = 1;
    distinct.nodes[0].in1TensorUid = 2;

    Spec aliased;
    aliased.nodes[0].in0TensorUid = 1;
    aliased.nodes[0].in1TensorUid = 1;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{distinct}), keyFor(ContentCarryingTestGraph{aliased}))
        << "x + y and x + x are different computations";
}

/// Renumbering does not rescue a rewiring, so the equality above is about labels alone.
TEST(TestGraphContentKey, ARewiredGraphStaysUnequalAfterRenumbering)
{
    Spec aliased;
    aliased.nodes[0].in1TensorUid = 1;

    Spec renumberedDistinct;
    renumberedDistinct.tensors
        = {ContentCarryingTestGraph::TensorSpec{1000}, ContentCarryingTestGraph::TensorSpec{2000}};
    renumberedDistinct.nodes[0].in0TensorUid = 1000;
    renumberedDistinct.nodes[0].in1TensorUid = 2000;
    renumberedDistinct.nodes[0].out0TensorUid = 2000;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{aliased}),
              keyFor(ContentCarryingTestGraph{renumberedDistinct}));
}

/// Tensors agreeing on every compared attribute are interchangeable, so swapping which
/// feeds which operand keys the same: the described content is the same work.
///
/// Both graphs are built as `buildGraphFromOperations` (GraphDescriptor.cpp:95-123)
/// would, emitting tensors in first-encounter order over each operation's operands, so
/// swapping two operands also swaps their positions in the tensor list.
TEST(TestGraphContentKey, SwappingTwoIdenticallyDescribedOperandsComparesEqual)
{
    Spec straight;
    straight.tensors = {ContentCarryingTestGraph::TensorSpec{1},
                        ContentCarryingTestGraph::TensorSpec{2},
                        ContentCarryingTestGraph::TensorSpec{3}};
    straight.nodes[0].in0TensorUid = 1;
    straight.nodes[0].in1TensorUid = 2;
    straight.nodes[0].out0TensorUid = 3;

    Spec swapped;
    swapped.tensors = {ContentCarryingTestGraph::TensorSpec{2},
                       ContentCarryingTestGraph::TensorSpec{1},
                       ContentCarryingTestGraph::TensorSpec{3}};
    swapped.nodes[0].in0TensorUid = 2;
    swapped.nodes[0].in1TensorUid = 1;
    swapped.nodes[0].out0TensorUid = 3;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{straight}), keyFor(ContentCarryingTestGraph{swapped}))
        << "identically described tensors are interchangeable; the work is the same";
}

/// The companion: once the swapped tensors differ in shape, the same reordering permutes
/// two unlike entries and the key must split.
TEST(TestGraphContentKey, SwappingTwoDifferentlyDescribedOperandsComparesUnequal)
{
    ContentCarryingTestGraph::TensorSpec narrow{1};
    narrow.dims = {4, 8};
    narrow.strides = {8, 1};
    ContentCarryingTestGraph::TensorSpec wide{2};
    wide.dims = {16, 32};
    wide.strides = {32, 1};
    const ContentCarryingTestGraph::TensorSpec output{3};

    Spec straight;
    straight.tensors = {narrow, wide, output};
    straight.nodes[0].in0TensorUid = 1;
    straight.nodes[0].in1TensorUid = 2;
    straight.nodes[0].out0TensorUid = 3;

    Spec swapped;
    swapped.tensors = {wide, narrow, output};
    swapped.nodes[0].in0TensorUid = 2;
    swapped.nodes[0].in1TensorUid = 1;
    swapped.nodes[0].out0TensorUid = 3;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{straight}), keyFor(ContentCarryingTestGraph{swapped}))
        << "once the operands differ in shape, which one feeds which slot is content";
}

/// `(cache_uid)` canonicalizes a nullable reference's value but leaves presence alone:
/// a ragged tensor is not a dense one.
TEST(TestGraphContentKey, ARaggedTensorComparesUnequalToADenseOne)
{
    const Spec dense;
    Spec ragged;
    ragged.tensors[0].raggedOffsetTensorUid = 2;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{dense}), keyFor(ContentCarryingTestGraph{ragged}))
        << "presence of a ragged offset is content, not a label";
}

/// The value behind that presence is a label, so renumbering it holds equal.
TEST(TestGraphContentKey, ARenumberedRaggedReferenceComparesEqual)
{
    Spec low;
    low.tensors
        = {ContentCarryingTestGraph::TensorSpec{1}, ContentCarryingTestGraph::TensorSpec{2}};
    low.tensors[0].raggedOffsetTensorUid = 2;
    low.nodes[0].in0TensorUid = 1;
    low.nodes[0].out0TensorUid = 2;

    Spec high;
    high.tensors
        = {ContentCarryingTestGraph::TensorSpec{1000}, ContentCarryingTestGraph::TensorSpec{2000}};
    high.tensors[0].raggedOffsetTensorUid = 2000;
    high.nodes[0].in0TensorUid = 1000;
    high.nodes[0].out0TensorUid = 2000;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{low}), keyFor(ContentCarryingTestGraph{high}));
}

/// A ragged offset is an ordinary tensor reference, so which offset tensor a primary
/// points at is content: sharing one offset across two inputs is different addressing
/// from two separate ones.
TEST(TestGraphContentKey, SharedAndSeparateRaggedOffsetsCompareUnequal)
{
    Spec separate;
    separate.tensors = {ContentCarryingTestGraph::TensorSpec{1},
                        ContentCarryingTestGraph::TensorSpec{2},
                        ContentCarryingTestGraph::TensorSpec{3},
                        ContentCarryingTestGraph::TensorSpec{90},
                        ContentCarryingTestGraph::TensorSpec{91}};
    separate.tensors[0].raggedOffsetTensorUid = 90;
    separate.tensors[1].raggedOffsetTensorUid = 91;
    separate.nodes[0].in0TensorUid = 1;
    separate.nodes[0].in1TensorUid = 2;
    separate.nodes[0].out0TensorUid = 3;

    Spec shared = separate;
    shared.tensors = {ContentCarryingTestGraph::TensorSpec{1},
                      ContentCarryingTestGraph::TensorSpec{2},
                      ContentCarryingTestGraph::TensorSpec{3},
                      ContentCarryingTestGraph::TensorSpec{90}};
    shared.tensors[0].raggedOffsetTensorUid = 90;
    shared.tensors[1].raggedOffsetTensorUid = 90;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{separate}), keyFor(ContentCarryingTestGraph{shared}));
}

TEST(TestGraphContentKey, ADifferentNodeCountComparesUnequal)
{
    const Spec single;
    Spec doubled;
    doubled.nodes = {ContentCarryingTestGraph::NodeSpec{}, ContentCarryingTestGraph::NodeSpec{}};

    EXPECT_NE(keyFor(ContentCarryingTestGraph{single}), keyFor(ContentCarryingTestGraph{doubled}));
}

TEST(TestGraphContentKey, ADifferentNodeComputeDataTypeComparesUnequal)
{
    const Spec asFloat;
    Spec asHalf;
    asHalf.nodes[0].computeDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

// Inside the union payload rather than only its discriminant: an ADD and a MUL are the
// same node type and would collide on any key that stopped at attributes_type.
TEST(TestGraphContentKey, ADifferentPointwiseOperationComparesUnequal)
{
    const Spec addition;
    Spec multiplication;
    multiplication.nodes[0].operation = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{addition}),
              keyFor(ContentCarryingTestGraph{multiplication}));
}

/// The graph-level data types are defaults: the frontend stamps them onto each node and
/// tensor left unset (Attributes::fill_from_context), and those are compared. Folding
/// them would split the key between an explicit type and the same type defaulted.
TEST(TestGraphContentKey, ADifferentGraphComputeDataTypeComparesEqual)
{
    const Spec asFloat;
    Spec asHalf;
    asHalf.computeDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

/// The exclusion above holds only while the types it populates discriminate.
TEST(TestGraphContentKey, TheStampedNodeDataTypeStillDiscriminates)
{
    const Spec asFloat;
    Spec asHalf;
    asHalf.nodes[0].computeDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}))
        << "precision reaches the key through the per-node type the graph default stamps";
}

// fnv1aHash collapses null and empty input to sentinel 0, so a key of 0 would alias
// every contentless graph onto one bucket -- the version tag and node count emitted
// ahead of any content are what prevent it.
TEST(TestGraphContentKey, AnEmptyGraphKeysToANonZeroHash)
{
    const EmptyTestGraph empty(makeGraphId(0xC3));

    EXPECT_NE(GraphContentKey{empty}.hash(), 0U);
}

TEST(TestGraphContentKey, AnEmptyGraphStillComparesEqualToAnotherEmptyGraph)
{
    const EmptyTestGraph first(makeGraphId(0xC4));
    const EmptyTestGraph second(makeGraphId(0xC5));

    EXPECT_EQ(GraphContentKey{first}, GraphContentKey{second});
}

/// Names carry `(cache_ignore)`: two identically shaped graphs run the same kernels
/// whatever they are called.
TEST(TestGraphContentKey, ADifferentGraphNameComparesEqual)
{
    const Spec named;
    Spec renamed;
    renamed.name = "a_different_graph";

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{named}), keyFor(ContentCarryingTestGraph{renamed}));
}

TEST(TestGraphContentKey, ADifferentGraphIoDataTypeComparesEqual)
{
    const Spec asFloat;
    Spec asHalf;
    asHalf.ioDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

TEST(TestGraphContentKey, ADifferentGraphIntermediateDataTypeComparesEqual)
{
    const Spec asFloat;
    Spec asHalf;
    asHalf.intermediateDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

/// The io/intermediate counterpart of the node-dtype case above.
TEST(TestGraphContentKey, ThePerTensorDataTypeStillDiscriminates)
{
    const Spec asFloat;
    Spec asHalf;
    asHalf.tensors[0].dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{asFloat}), keyFor(ContentCarryingTestGraph{asHalf}));
}

/// One level down from the graph name: the node's attributes are compared in full.
TEST(TestGraphContentKey, ADifferentNodeNameComparesEqual)
{
    const Spec named;
    Spec renamed;
    renamed.nodes[0].name = "a_different_node";

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{named}), keyFor(ContentCarryingTestGraph{renamed}));
}

/// Inside the union payload again, on a field the discriminant cannot distinguish: two
/// adds wired to different tensors are different computations.
TEST(TestGraphContentKey, ADifferentPointwiseOperandUidComparesUnequal)
{
    const Spec wired;
    Spec rewired;
    rewired.nodes[0].in0TensorUid = 99;

    EXPECT_NE(keyFor(ContentCarryingTestGraph{wired}), keyFor(ContentCarryingTestGraph{rewired}));
}

TEST(TestGraphContentKey, AnEmptyGraphDoesNotMatchAContentCarryingOne)
{
    const EmptyTestGraph empty(makeGraphId(0xC6));
    const ContentCarryingTestGraph populated{Spec{}};

    EXPECT_NE(GraphContentKey{empty}, keyFor(populated));
}

/// An `IGraph` that does not override `bytes()`, as an out-of-tree implementor predating
/// the method would not. It must compile, and its key must match nothing.
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

TEST(TestGraphContentKey, AGraphWithNoBytesYieldsAnUnusableKey)
{
    const UnkeyableGraph graph;

    EXPECT_FALSE(GraphContentKey{graph}.isUsable());
}

/// Hash 0, unusable, and matching nothing are one fact: a key with a live hash but no
/// content would be bucketable yet not equal to itself.
TEST(TestGraphContentKey, AnUnusableKeyHashesToZero)
{
    const UnkeyableGraph graph;
    const GraphContentKey key{graph};

    ASSERT_FALSE(key.isUsable());
    EXPECT_EQ(key.hash(), 0U) << "an unkeyable graph must not be filed under a live hash";
}

/// `isValid()` and `bytes()` are independent predicates on `IGraph`, so a valid graph
/// supplying no bytes is legal. Only the retained bytes decide keyability.
class ValidButByteless : public UnkeyableGraph
{
public:
    bool isValid() const override
    {
        return true;
    }
};

TEST(TestGraphContentKey, AValidGraphWithNoBytesIsStillUnkeyableAndSelfConsistent)
{
    const ValidButByteless graph;
    const GraphContentKey key{graph};

    EXPECT_FALSE(key.isUsable());
    EXPECT_EQ(key.hash(), 0U)
        << "validity must not mint a hash the content cannot back; the two gates are "
           "independent on IGraph and only the retained bytes decide keyability";
}

TEST(TestGraphContentKey, TwoUnkeyableGraphsDoNotMatchEachOther)
{
    const UnkeyableGraph first;
    const UnkeyableGraph second;

    EXPECT_NE(GraphContentKey{first}, GraphContentKey{second})
        << "absence of content is a permanent miss, never a wildcard that matches "
           "every other unkeyable graph";
}

TEST(TestGraphContentKey, AnUnkeyableGraphDoesNotMatchARealOne)
{
    const UnkeyableGraph unkeyable;
    const ContentCarryingTestGraph real{Spec{}};

    EXPECT_NE(GraphContentKey{unkeyable}, keyFor(real));
}

/// The hash narrows; the content decides. Hashes are forced to agree so the structural
/// comparison is actually reached -- `operator==` short-circuits on a hash mismatch.
TEST(TestGraphContentKey, EqualHashesWithDifferentContentStillCompareUnequal)
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
TEST(TestGraphContentKey, EqualHashesWithEqualContentStillCompareEqual)
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

TEST(TestGraphContentKey, StdHashAgreesWithTheKeysOwnHash)
{
    const ContentCarryingTestGraph graph{Spec{}};
    const auto key = keyFor(graph);

    EXPECT_EQ(std::hash<GraphContentKey>{}(key), static_cast<size_t>(key.hash()));
}

/// A uid the domain cannot resolve folds its own value, tagged as unresolved. Folding
/// every dangling reference to one shared sentinel instead would make these two graphs
/// -- which point at different absent tensors -- compare equal and share a kernel.
/// The generator test pins the emitted source; this pins the compiled behaviour.
TEST(TestGraphContentKey, DanglingReferencesToDifferentUidsCompareUnequal)
{
    Spec first;
    first.tensors
        = {ContentCarryingTestGraph::TensorSpec{1}, ContentCarryingTestGraph::TensorSpec{2}};
    first.tensors[0].raggedOffsetTensorUid = 900;

    Spec second = first;
    second.tensors[0].raggedOffsetTensorUid = 901;

    const ContentCarryingTestGraph firstGraph{first};
    const ContentCarryingTestGraph secondGraph{second};

    EXPECT_NE(keyFor(firstGraph), keyFor(secondGraph));
}

/// An unresolved fold carries the raw uid, a resolved one carries an ordinal, and the
/// two spaces must not overlap: uid 1 resolves to ordinal 0 here, so a dangling uid 0
/// would collide with it if the resolved tag were dropped.
TEST(TestGraphContentKey, ADanglingReferenceDoesNotAliasAResolvedOrdinal)
{
    Spec resolved;
    resolved.tensors
        = {ContentCarryingTestGraph::TensorSpec{1}, ContentCarryingTestGraph::TensorSpec{2}};
    resolved.tensors[1].raggedOffsetTensorUid = 1;

    Spec dangling = resolved;
    dangling.tensors[1].raggedOffsetTensorUid = 0;

    const ContentCarryingTestGraph resolvedGraph{resolved};
    const ContentCarryingTestGraph danglingGraph{dangling};

    EXPECT_NE(keyFor(resolvedGraph), keyFor(danglingGraph));
}

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
TEST(TestGraphContentKey, JsonRoundTripPreservesEquality)
{
    const ContentCarryingTestGraph graph{Spec{}};
    const auto original = keyFor(graph);

    const auto restored = GraphContentKey::fromJson(original.toJson());

    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(original, *restored);
}

TEST(TestGraphContentKey, JsonRoundTripPreservesDiscriminatingContent)
{
    Spec narrow;
    narrow.tensors[0].dims = {4, 8};
    Spec wide;
    wide.tensors[0].dims = {4, 16};

    const auto narrowRestored
        = GraphContentKey::fromJson(keyFor(ContentCarryingTestGraph{narrow}).toJson());
    const auto wideRestored
        = GraphContentKey::fromJson(keyFor(ContentCarryingTestGraph{wide}).toJson());

    ASSERT_TRUE(narrowRestored.has_value());
    ASSERT_TRUE(wideRestored.has_value());
    EXPECT_NE(*narrowRestored, *wideRestored)
        << "a codec that lost discriminating content would make these collide";
}

TEST(TestGraphContentKey, ANonBase64FieldDeclinesWithoutThrowing)
{
    auto json = keyFor(ContentCarryingTestGraph{Spec{}}).toJson();
    json["content_base64"] = std::string("not valid base64!!!");

    std::optional<GraphContentKey> restored;
    EXPECT_NO_THROW(restored = GraphContentKey::fromJson(json));
    EXPECT_FALSE(restored.has_value());
}

TEST(TestGraphContentKey, ATruncatedBase64FieldDeclinesWithoutThrowing)
{
    auto json = keyFor(ContentCarryingTestGraph{Spec{}}).toJson();
    const auto content = json["content_base64"].get<std::string>();
    ASSERT_GT(content.size(), 4U);
    json["content_base64"] = content.substr(0, content.size() - 1);

    std::optional<GraphContentKey> restored;
    EXPECT_NO_THROW(restored = GraphContentKey::fromJson(json));
    EXPECT_FALSE(restored.has_value());
}

TEST(TestGraphContentKey, AMissingContentFieldDeclinesWithoutThrowing)
{
    const auto json = nlohmann::json::object();

    std::optional<GraphContentKey> restored;
    EXPECT_NO_THROW(restored = GraphContentKey::fromJson(json));
    EXPECT_FALSE(restored.has_value());
}

TEST(TestGraphContentKey, NonObjectJsonDeclinesWithoutThrowing)
{
    const nlohmann::json json = "not an object";

    std::optional<GraphContentKey> restored;
    EXPECT_NO_THROW(restored = GraphContentKey::fromJson(json));
    EXPECT_FALSE(restored.has_value());
}

/// Base64-decodable bytes that do not reverify as a `Graph` flatbuffer must also
/// decline, not hand back a key that looks usable but reads nonsense.
TEST(TestGraphContentKey, ValidBase64OfNonGraphBytesDeclinesWithoutThrowing)
{
    nlohmann::json json;
    json["content_base64"] = std::string("AAAAAAAAAAAAAAAAAAAAAAAA");

    std::optional<GraphContentKey> restored;
    EXPECT_NO_THROW(restored = GraphContentKey::fromJson(json));
    EXPECT_FALSE(restored.has_value());
}

/// `EmptyTestGraph` still retains real bytes and a non-zero hash, so it round-trips
/// like any other usable key.
TEST(TestGraphContentKey, AnEmptyGraphRoundTripsAndStillMatchesAnotherEmptyGraph)
{
    const EmptyTestGraph first(makeGraphId(0xD1));
    const EmptyTestGraph second(makeGraphId(0xD2));

    const auto restoredFirst = GraphContentKey::fromJson(GraphContentKey{first}.toJson());
    const auto restoredSecond = GraphContentKey::fromJson(GraphContentKey{second}.toJson());

    ASSERT_TRUE(restoredFirst.has_value());
    ASSERT_TRUE(restoredSecond.has_value());
    EXPECT_NE(restoredFirst->hash(), 0U);
    EXPECT_EQ(*restoredFirst, *restoredSecond);
}

/// No retained bytes at all: `toJson()` still succeeds (an empty base64 field is
/// well-formed, not corruption) but the restored key must stay unusable.
TEST(TestGraphContentKey, AnUnusableKeyRoundTripsToAnUnusableKeyThatMatchesNothing)
{
    const UnkeyableGraph graph;
    const GraphContentKey original{graph};
    ASSERT_FALSE(original.isUsable());

    const auto restored = GraphContentKey::fromJson(original.toJson());

    ASSERT_TRUE(restored.has_value())
        << "an empty content field is well-formed JSON, not corruption";
    EXPECT_FALSE(restored->isUsable());
    EXPECT_EQ(restored->hash(), 0U);
    EXPECT_NE(original, *restored)
        << "absence of content is a permanent miss; round-tripping an unusable key must "
           "not make it start matching, including matching the key it came from";
}
#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

} // namespace
} // namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing
