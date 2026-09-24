// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The reference supported-op sets are a commitment: a bundle inside a set gets a
// validation test with no skip path, and one outside it is silently absent from the
// suite. Both halves of that need pinning, as do graph features such as ragged.

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
using hipdnn_integration_tests::bundle::graphUsesSeqLenTensors;
using hipdnn_integration_tests::bundle::K_FP8_TENSORS;
using hipdnn_integration_tests::bundle::K_NO_NODES;
using hipdnn_integration_tests::bundle::K_RAGGED_TENSORS;
using hipdnn_integration_tests::bundle::K_SEQ_LEN_TENSORS;
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

// Single-node SDPA graph covered by the GPU reference; the ragged and FP8 cases vary
// one tensor field from it.
flatbuffers::FlatBufferBuilder
    buildSdpaGraph(hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                   = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                   hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT attrs = {})
{
    const std::vector<int64_t> dims = {1, 2, 64, 16};
    return hipdnn_integration_tests::test_utils::createSdpaFwdGraph(
        0, 1, 2, 3, dims, dims, dims, dims, dataType, attrs);
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
// Graph features: gated independently of the node type.
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

TEST(TestReferenceOpCoverage, DeviceReferenceDoesNotCoverRaggedGraphs)
{
    auto dense = buildSdpaGraph();
    auto ragged = markFirstTensorRagged(dense.GetBufferPointer());

    EXPECT_FALSE(referenceCoversGraph(ReferenceExecutorType::GPU, ragged.data(), ragged.size()));

    const auto reasons = exclusionReasons(ReferenceExecutorType::GPU, ragged.data(), ragged.size());
    EXPECT_EQ(reasons, std::vector<std::string>{std::string(K_RAGGED_TENSORS)});
}

// Guards against over-rejecting: excluded bundles are silently absent from the suite.
TEST(TestReferenceOpCoverage, NonRaggedSdpaIsStillCoveredByTheDeviceReference)
{
    auto graph = buildSdpaGraph();
    auto* buffer = graph.GetBufferPointer();
    const auto size = graph.GetSize();

    EXPECT_TRUE(referenceCoversGraph(ReferenceExecutorType::GPU, buffer, size));
    EXPECT_TRUE(exclusionReasons(ReferenceExecutorType::GPU, buffer, size).empty());
}

TEST(TestReferenceOpCoverage, RaggedIsNamedAlongsideTheUncoveredOp)
{
    auto dense = buildSdpaGraph();
    auto ragged = markFirstTensorRagged(dense.GetBufferPointer());

    const auto reasons = exclusionReasons(ReferenceExecutorType::CPU, ragged.data(), ragged.size());
    EXPECT_EQ(std::set<std::string>(reasons.begin(), reasons.end()),
              (std::set<std::string>{"SdpaAttributes", std::string(K_RAGGED_TENSORS)}));
}

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

TEST(TestReferenceOpCoverage, SeqLenIsReadFromSdpaAttributes)
{
    const auto dense = buildSdpaGraph();
    const auto denseUsesSeqLen = graphUsesSeqLenTensors(dense.GetBufferPointer(), dense.GetSize());
    ASSERT_TRUE(denseUsesSeqLen.has_value());
    EXPECT_FALSE(*denseUsesSeqLen);

    hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT attrs;
    attrs.seq_len_q_tensor_uid = 4;
    const auto seqLenQ
        = buildSdpaGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, attrs);
    const auto seqLenQUsesSeqLen
        = graphUsesSeqLenTensors(seqLenQ.GetBufferPointer(), seqLenQ.GetSize());
    ASSERT_TRUE(seqLenQUsesSeqLen.has_value());
    EXPECT_TRUE(*seqLenQUsesSeqLen);

    const std::vector<uint8_t> garbage(64, 0xAB);
    EXPECT_FALSE(graphUsesSeqLenTensors(garbage.data(), garbage.size()).has_value());
}

TEST(TestReferenceOpCoverage, SeqLenIsExcludedForBothReferences)
{

    hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT attrs;
    attrs.seq_len_kv_tensor_uid = 4;
    auto graph = buildSdpaGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, attrs);
    auto* buffer = graph.GetBufferPointer();
    const auto size = graph.GetSize();

    EXPECT_EQ(exclusionReasons(ReferenceExecutorType::GPU, buffer, size),
              std::vector<std::string>{std::string(K_SEQ_LEN_TENSORS)});

    const auto cpuReasons = exclusionReasons(ReferenceExecutorType::CPU, buffer, size);
    EXPECT_EQ(std::count(cpuReasons.begin(), cpuReasons.end(), std::string(K_SEQ_LEN_TENSORS)), 1);
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
