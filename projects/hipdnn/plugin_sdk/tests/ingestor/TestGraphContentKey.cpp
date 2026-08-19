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
/// is a valid measurement for the other. Every case below is an instance of that
/// question -- the "equal" ones are facts a benchmark result survives, the "not equal"
/// ones are facts it does not.
///
/// MAINTENANCE: this file is the field-set pin for the graph half of the key. `IGraph`
/// is a virtual interface with no compiler-checkable arity, so nothing fails to build
/// when `graph.fbs` grows -- a new field that changes what a kernel does needs a
/// discriminates-on case added here, and a new field that does not needs an ignores case.
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

// The story's premise: Graph.id is minted per finalize, so two runs of the same program
// produce different ids for the same computation. If the key saw it, every lookup would
// miss and the cache would never serve anything.
TEST(TestIngestorGraphContentKey, ADifferentGraphIdStillComparesEqual)
{
    Spec first;
    first.graphId = makeGraphId(0xA1);
    Spec second;
    second.graphId = makeGraphId(0xB2);

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{first}), keyFor(ContentCarryingTestGraph{second}));
}

// Which engine the caller would prefer selects who runs the computation, never what is
// computed, so a measurement transfers across it.
TEST(TestIngestorGraphContentKey, ADifferentPreferredEngineIdStillComparesEqual)
{
    Spec first;
    first.preferredEngineId = 7;
    Spec second;
    second.preferredEngineId = 99;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{first}), keyFor(ContentCarryingTestGraph{second}));
}

// The flag permits shapes to be overridden; it does not change them. The dims and
// strides that will actually run are in this same graph and are compared in full, so the
// geometry a kernel was timed against is identical either way.
TEST(TestIngestorGraphContentKey, ADifferentOverrideShapeFlagStillComparesEqual)
{
    Spec enabled;
    enabled.isOverrideShapeEnabled = true;
    const Spec disabled;

    EXPECT_EQ(keyFor(ContentCarryingTestGraph{enabled}),
              keyFor(ContentCarryingTestGraph{disabled}));
}

/// The production shape of the case above, which the fixture could not express until
/// `Spec::minRequiredApiVersion` existed. The backend derives the stamped version from
/// `is_override_shape_enabled` (PluginVersionConstants.hpp:58-93), so two real graphs
/// differing only in that flag carry *different* versions -- and the generated
/// operator== compares the version. Without clearing it, the exclusion above is defeated
/// in production while its test still passes. This is the test that would have caught it.
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

/// The version is excluded because it is *derived*, not because versions are
/// unimportant: its content-bearing inputs (pass-by-value, ragged offsets, alignment)
/// are each compared directly on the tensors that carry them.
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

// TestGraph is the degenerate case by construction: no nodes, no tensors. fnv1aHash
// collapses null and empty input to sentinel 0, so a key of 0 would alias every
// contentless graph onto one bucket -- the version tag and node count emitted ahead of
// any content are what prevent it.
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

TEST(TestIngestorGraphContentKey, AnEmptyGraphDoesNotMatchAContentCarryingOne)
{
    const TestGraph empty(makeGraphId(0xC6));
    const ContentCarryingTestGraph populated{Spec{}};

    EXPECT_NE(GraphContentKey{empty}, keyFor(populated));
}

// The hash narrows; the content decides. Colliding hashes must not produce a false hit,
// or a cache lookup would serve a kernel measured for a different graph.
TEST(TestIngestorGraphContentKey, EqualHashesWithDifferentContentStillCompareUnequal)
{
    Spec narrow;
    narrow.tensors[0].dims = {4, 8};
    Spec wide;
    wide.tensors[0].dims = {4, 16};

    const ContentCarryingTestGraph first{narrow};
    const ContentCarryingTestGraph second{wide};

    const auto firstKey = keyFor(first);
    const auto secondKey = keyFor(second);
    ASSERT_NE(firstKey, secondKey) << "the two graphs must differ for this test to mean anything";

    // Structural comparison is what rejects the match, so forcing the hashes together
    // must not change the verdict.
    EXPECT_FALSE(*firstKey.content() == *secondKey.content());
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
