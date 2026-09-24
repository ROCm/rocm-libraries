// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The reference supported-op sets are a commitment: a bundle inside a set gets a
// validation test with no skip path, and one outside it is silently absent from the
// suite. Both halves of that need pinning, and so does the second axis the gate
// grew -- graph features such as ragged offsets, which live on the tensors and are
// invisible to a node-type set.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <nlohmann/json.hpp>

#include "RaggedGraphTestUtils.hpp"
#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/bundle/ReferenceOpCoverage.hpp"

using hipdnn_integration_tests::ReferenceExecutorType;
using hipdnn_integration_tests::bundle::exclusionReasons;
using hipdnn_integration_tests::bundle::formatExclusionReasons;
using hipdnn_integration_tests::bundle::graphNodeTypes;
using hipdnn_integration_tests::bundle::graphUsesRaggedTensors;
using hipdnn_integration_tests::bundle::K_FP8_TENSORS;
using hipdnn_integration_tests::bundle::K_NO_NODES;
using hipdnn_integration_tests::bundle::K_RAGGED_TENSORS;
using hipdnn_integration_tests::bundle::K_UNREADABLE_GRAPH;
using hipdnn_integration_tests::bundle::NodeAttributes;
using hipdnn_integration_tests::bundle::referenceCoversGraph;
using hipdnn_integration_tests::bundle::referenceSupportedOps;
using hipdnn_integration_tests::test_utils::markFirstTensorRagged;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

// A minimal single-node batchnorm-inference graph, serialized the same way the
// bundle loader does it.
const char* BATCHNORM_GRAPH_JSON = R"({"nodes": [{"inputs": {"x_tensor_uid": 0,
    "mean_tensor_uid": 1, "inv_variance_tensor_uid": 2, "scale_tensor_uid": 3,
    "bias_tensor_uid": 4}, "outputs": {"y_tensor_uid": 5},
    "type": "BatchnormInferenceAttributes", "compute_data_type": "float", "name": ""}],
    "tensors": [
    {"name": "", "uid": 0, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], "data_type": "float", "virtual": false},
    {"name": "", "uid": 1, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 2, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 3, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 4, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 5, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], "data_type": "float", "virtual": false}],
    "io_data_type": "float", "compute_data_type": "float",
    "intermediate_data_type": "float", "name": ""})";

const char* NO_NODE_GRAPH_JSON = R"({"nodes": [], "tensors": [
    {"name": "", "uid": 0, "strides": [1], "dims": [1], "data_type": "float", "virtual": false}],
    "io_data_type": "float", "compute_data_type": "float",
    "intermediate_data_type": "float", "name": ""})";

flatbuffers::DetachedBuffer buildGraph(const std::string& graphJson)
{
    flatbuffers::FlatBufferBuilder builder;
    const auto json = nlohmann::json::parse(graphJson);
    auto offset = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
        builder, json);
    builder.Finish(offset);
    return builder.Release();
}

flatbuffers::DetachedBuffer buildBatchnormGraph()
{
    return buildGraph(BATCHNORM_GRAPH_JSON);
}

// A minimal single-node SDPA graph. SdpaAttributes is in the GPU reference's op set,
// so this is the control the ragged and FP8 cases are varied from: each differs in
// exactly one tensor field, which is what makes their verdicts attributable.
flatbuffers::FlatBufferBuilder
    buildSdpaGraph(hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                   = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    const std::vector<int64_t> dims = {1, 2, 64, 16};
    return hipdnn_integration_tests::test_utils::createSdpaFwdGraph(
        0, 1, 2, 3, dims, dims, dims, dims, dataType);
}

} // namespace

// ---------------------------------------------------------------------------
// The sets themselves
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, BothReferenceSetsAreNonEmpty)
{
    EXPECT_FALSE(referenceSupportedOps(ReferenceExecutorType::CPU).empty());
    EXPECT_FALSE(referenceSupportedOps(ReferenceExecutorType::GPU).empty());
}

// The two references cover different ops on purpose — the GPU one dispatches
// through a signature-keyed plan registry and grows only as builders are written.
// If these ever became identical the split would be pointless, so it is worth
// noticing.
TEST(TestReferenceOpCoverage, SetsAreIndependent)
{
    EXPECT_NE(referenceSupportedOps(ReferenceExecutorType::CPU),
              referenceSupportedOps(ReferenceExecutorType::GPU));
}

// ---------------------------------------------------------------------------
// Graph inspection
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, NodeTypesAreReadFromTheGraph)
{
    const auto graph = buildBatchnormGraph();
    const auto types = graphNodeTypes(graph.data(), graph.size());

    ASSERT_TRUE(types.has_value());
    ASSERT_EQ(types->size(), 1u);
    EXPECT_EQ(*types->begin(), NodeAttributes::BatchnormInferenceAttributes);
}

// An unreadable buffer must not be treated as "covered by everything" — that would
// register a validation test for a bundle nobody can run.
TEST(TestReferenceOpCoverage, UnreadableGraphIsNotCovered)
{
    const std::vector<uint8_t> garbage(64, 0xAB);

    EXPECT_FALSE(graphNodeTypes(garbage.data(), garbage.size()).has_value());
    EXPECT_FALSE(referenceCoversGraph(ReferenceExecutorType::CPU, garbage.data(), garbage.size()));
    EXPECT_FALSE(referenceCoversGraph(ReferenceExecutorType::GPU, garbage.data(), garbage.size()));
}

// "Not covered" and "nothing is uncovered" must not both be true of one graph: the
// registration log prints an exclusion count next to the ops responsible for it, so
// an unreadable graph that named no ops would report a gap with no reason attached.
TEST(TestReferenceOpCoverage, UnreadableGraphNamesItselfAsTheReason)
{
    const std::vector<uint8_t> garbage(64, 0xAB);

    const auto reasons
        = exclusionReasons(ReferenceExecutorType::CPU, garbage.data(), garbage.size());
    ASSERT_EQ(reasons.size(), 1u);
    EXPECT_EQ(reasons.front(), K_UNREADABLE_GRAPH);
}

// A zero-node graph was where the verdict and the reason list had already drifted
// apart: referenceCoversGraph() called it "not covered" while uncoveredNodeTypes()
// returned {}, so the summary reported an exclusion with nothing attached. The
// verdict is unchanged; the reason is now named.
TEST(TestReferenceOpCoverage, GraphWithNoNodesNamesItselfAsTheReason)
{
    const auto graph = buildGraph(NO_NODE_GRAPH_JSON);

    EXPECT_FALSE(referenceCoversGraph(ReferenceExecutorType::CPU, graph.data(), graph.size()));

    const auto reasons = exclusionReasons(ReferenceExecutorType::CPU, graph.data(), graph.size());
    ASSERT_EQ(reasons.size(), 1u);
    EXPECT_EQ(reasons.front(), K_NO_NODES);
}

// ---------------------------------------------------------------------------
// Coverage decision
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, CpuCoversBatchnormInference)
{
    const auto graph = buildBatchnormGraph();
    EXPECT_TRUE(referenceCoversGraph(ReferenceExecutorType::CPU, graph.data(), graph.size()));
    EXPECT_TRUE(exclusionReasons(ReferenceExecutorType::CPU, graph.data(), graph.size()).empty());
}

// The GPU reference has no batchnorm plan builder, so bundles using it are absent
// from the GPU validation suite rather than skipped inside it.
TEST(TestReferenceOpCoverage, DeviceReferenceDoesNotCoverBatchnormInference)
{
    const auto graph = buildBatchnormGraph();
    EXPECT_FALSE(referenceCoversGraph(ReferenceExecutorType::GPU, graph.data(), graph.size()));

    const auto reasons = exclusionReasons(ReferenceExecutorType::GPU, graph.data(), graph.size());
    ASSERT_EQ(reasons.size(), 1u);
    EXPECT_EQ(reasons.front(), "BatchnormInferenceAttributes");
}

// ---------------------------------------------------------------------------
// Graph features
//
// Ragged and dtype live on TensorAttributes, so a node-type set cannot see them:
// the ragged and dense SDPA graphs below are both SdpaAttributes. That is the whole
// reason the gate grew a second axis.
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, GraphFeaturesAreReadFromTensorsNotNodes)
{
    auto dense = buildSdpaGraph();
    auto ragged = markFirstTensorRagged(dense.GetBufferPointer());
    const std::vector<uint8_t> garbage(64, 0xAB);

    const auto denseIsRagged = graphUsesRaggedTensors(dense.GetBufferPointer(), dense.GetSize());
    ASSERT_TRUE(denseIsRagged.has_value());
    EXPECT_FALSE(*denseIsRagged);

    const auto raggedIsRagged = graphUsesRaggedTensors(ragged.data(), ragged.size());
    ASSERT_TRUE(raggedIsRagged.has_value());
    EXPECT_TRUE(*raggedIsRagged);

    EXPECT_FALSE(graphUsesRaggedTensors(garbage.data(), garbage.size()).has_value());
}

// The regression: four SDPA `_ragged` bundles were registered for GpuRef validation
// and failed with a ~98% mismatch, because the reference read their padded BSHD dims
// as BHSD. SdpaAttributes being in gpuSupportedOps() is not enough.
TEST(TestReferenceOpCoverage, DeviceReferenceDoesNotCoverRaggedGraphs)
{
    auto dense = buildSdpaGraph();
    auto ragged = markFirstTensorRagged(dense.GetBufferPointer());

    EXPECT_FALSE(referenceCoversGraph(ReferenceExecutorType::GPU, ragged.data(), ragged.size()));

    const auto reasons = exclusionReasons(ReferenceExecutorType::GPU, ragged.data(), ragged.size());
    EXPECT_EQ(reasons, std::vector<std::string>{std::string(K_RAGGED_TENSORS)});
}

// The risk in gating on features is over-rejecting: a gate that excluded all SDPA
// would silently drop every dense SDPA bundle from the GpuRef suite and still look
// green, since absence is how exclusion is expressed. This pins that it does not.
TEST(TestReferenceOpCoverage, NonRaggedSdpaIsStillCoveredByTheDeviceReference)
{
    auto graph = buildSdpaGraph();
    auto* buffer = graph.GetBufferPointer();
    const auto size = graph.GetSize();

    EXPECT_TRUE(referenceCoversGraph(ReferenceExecutorType::GPU, buffer, size));
    EXPECT_TRUE(exclusionReasons(ReferenceExecutorType::GPU, buffer, size).empty());
}

// Reasons compose rather than short-circuit: the CPU reference has no SDPA builder
// *and* cannot read ragged offsets, and the summary should say both.
TEST(TestReferenceOpCoverage, RaggedIsNamedAlongsideTheUncoveredOp)
{
    auto dense = buildSdpaGraph();
    auto ragged = markFirstTensorRagged(dense.GetBufferPointer());

    const auto reasons = exclusionReasons(ReferenceExecutorType::CPU, ragged.data(), ragged.size());
    EXPECT_EQ(std::set<std::string>(reasons.begin(), reasons.end()),
              (std::set<std::string>{"SdpaAttributes", std::string(K_RAGGED_TENSORS)}));
}

// FP8 is excluded for the GPU reference only: no GPU plan builder registers an FP8
// signature and the reference SDPA kernel takes no descale parameters, whereas the
// CPU reference may legitimately handle FP8 for ops in its own set.
TEST(TestReferenceOpCoverage, Fp8IsExcludedOnlyForTheDeviceReference)
{
    auto graph = buildSdpaGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3);
    auto* buffer = graph.GetBufferPointer();
    const auto size = graph.GetSize();

    EXPECT_EQ(exclusionReasons(ReferenceExecutorType::GPU, buffer, size),
              std::vector<std::string>{std::string(K_FP8_TENSORS)});

    const auto cpuReasons = exclusionReasons(ReferenceExecutorType::CPU, buffer, size);
    EXPECT_EQ(std::count(cpuReasons.begin(), cpuReasons.end(), std::string(K_FP8_TENSORS)), 0);
}

// ---------------------------------------------------------------------------
// Registration diagnostic
//
// The exclusion tally alone says a gap exists without saying what to implement to
// close it.
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, NoExclusionsAddsNothingToTheSummary)
{
    EXPECT_EQ(formatExclusionReasons({}), "");
}

TEST(TestReferenceOpCoverage, ExcludedOpsAreNamedAndSeparated)
{
    EXPECT_EQ(formatExclusionReasons({"BatchnormInferenceAttributes"}),
              " (BatchnormInferenceAttributes)");
    EXPECT_EQ(formatExclusionReasons({"ReductionAttributes", "ConvolutionBwdDataAttributes"}),
              " (ConvolutionBwdDataAttributes, ReductionAttributes)");
}

// The sentinels sort ahead of op names, so a reason list reads feature-first.
TEST(TestReferenceOpCoverage, SentinelReasonsSortAheadOfOpNames)
{
    EXPECT_EQ(formatExclusionReasons({"SdpaAttributes", std::string(K_RAGGED_TENSORS)}),
              " (<ragged tensors>, SdpaAttributes)");
}

// NOLINTEND(readability-identifier-naming)
