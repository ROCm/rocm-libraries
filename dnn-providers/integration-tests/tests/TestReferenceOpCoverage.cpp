// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The reference supported-op sets are a commitment: a bundle inside a set gets a
// validation test with no skip path, and one outside it is silently absent from the
// suite. Both halves of that need pinning.

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <nlohmann/json.hpp>

#include "harness/bundle/ReferenceOpCoverage.hpp"

using hipdnn_integration_tests::ReferenceExecutorType;
using hipdnn_integration_tests::bundle::findKnownReferenceGap;
using hipdnn_integration_tests::bundle::formatUncoveredOps;
using hipdnn_integration_tests::bundle::graphNodeTypes;
using hipdnn_integration_tests::bundle::K_UNREADABLE_GRAPH;
using hipdnn_integration_tests::bundle::knownReferenceGaps;
using hipdnn_integration_tests::bundle::NodeAttributes;
using hipdnn_integration_tests::bundle::referenceCoversGraph;
using hipdnn_integration_tests::bundle::referenceShapeIsAffordable;
using hipdnn_integration_tests::bundle::referenceSupportedOps;
using hipdnn_integration_tests::bundle::uncoveredNodeTypes;

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

flatbuffers::DetachedBuffer buildBatchnormGraph()
{
    flatbuffers::FlatBufferBuilder builder;
    const auto json = nlohmann::json::parse(BATCHNORM_GRAPH_JSON);
    auto offset = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
        builder, json);
    builder.Finish(offset);
    return builder.Release();
}

// The graph JSON converter requires every declared key to be present, nullable
// ones included, so these spell out the full Sdpa input/output sets with only the
// tensors this graph actually uses populated. Trimming them to the fields under
// test fails conversion rather than producing a smaller graph.
nlohmann::json sdpaInputs()
{
    nlohmann::json inputs = {{"q_tensor_uid", 0}, {"k_tensor_uid", 1}, {"v_tensor_uid", 2}};
    for(const char* key : {"attn_mask_tensor_uid",
                           "scale_tensor_uid",
                           "seq_len_q_tensor_uid",
                           "seq_len_kv_tensor_uid",
                           "seed_tensor_uid",
                           "offset_tensor_uid",
                           "dropout_mask_tensor_uid",
                           "dropout_scale_tensor_uid",
                           "page_table_k_tensor_uid",
                           "page_table_v_tensor_uid",
                           "block_mask_tensor_uid",
                           "sink_token_tensor_uid",
                           "descale_q_tensor_uid",
                           "descale_k_tensor_uid",
                           "descale_v_tensor_uid",
                           "descale_s_tensor_uid",
                           "scale_s_tensor_uid",
                           "scale_o_tensor_uid"})
    {
        inputs[key] = nullptr;
    }
    return inputs;
}

nlohmann::json sdpaOutputs()
{
    nlohmann::json outputs = {{"o_tensor_uid", 3}};
    for(const char* key : {"stats_tensor_uid",
                           "max_tensor_uid",
                           "sum_exp_tensor_uid",
                           "rng_dump_tensor_uid",
                           "amax_s_tensor_uid",
                           "amax_o_tensor_uid"})
    {
        outputs[key] = nullptr;
    }
    return outputs;
}

// A single-node Sdpa graph whose Q/K sequence length is a parameter, so the
// affordability gate can be exercised on either side of its working-set cap
// without depending on the checked-in bundles.
flatbuffers::DetachedBuffer buildSdpaGraph(int64_t seq)
{
    const auto dims = nlohmann::json::array({2, 4, seq, 128});
    const auto strides = nlohmann::json::array({4 * seq * 128, seq * 128, 128, 1});
    auto tensor = [&](int64_t uid) {
        return nlohmann::json{{"name", ""},
                              {"uid", uid},
                              {"strides", strides},
                              {"dims", dims},
                              {"data_type", "half"},
                              {"virtual", false}};
    };

    const nlohmann::json graph = {
        {"nodes",
         nlohmann::json::array({{{"inputs", sdpaInputs()},
                                 {"outputs", sdpaOutputs()},
                                 {"type", "SdpaAttributes"},
                                 {"compute_data_type", "float"},
                                 {"name", ""},
                                 // The converter requires every key, nullable ones included, so
                                 // this mirrors a real SdpaFwd bundle's block rather than trimming
                                 // it to the fields the gate reads.
                                 {"attributes",
                                  {{"generate_stats", nullptr},
                                   {"alibi_mask", false},
                                   {"padding_mask", false},
                                   {"causal_mask", false},
                                   {"causal_mask_bottom_right", false},
                                   {"dropout_probability", nullptr},
                                   {"attn_scale_value", 0.08838834764831843},
                                   {"left_bound", -1},
                                   {"right_bound", 0},
                                   {"max_seq_len_kv", nullptr},
                                   {"diagonal_alignment", "BOTTOM_RIGHT"},
                                   {"mma_core_mode", "float"},
                                   {"implementation", "AUTO"}}}}})},
        {"tensors", nlohmann::json::array({tensor(0), tensor(1), tensor(2), tensor(3)})},
        {"io_data_type", "half"},
        {"compute_data_type", "float"},
        {"intermediate_data_type", "float"},
        {"name", ""}};

    flatbuffers::FlatBufferBuilder builder;
    auto offset = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
        builder, graph);
    builder.Finish(offset);
    return builder.Release();
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

    const auto uncovered
        = uncoveredNodeTypes(ReferenceExecutorType::CPU, garbage.data(), garbage.size());
    ASSERT_EQ(uncovered.size(), 1u);
    EXPECT_EQ(uncovered.front(), K_UNREADABLE_GRAPH);
}

// ---------------------------------------------------------------------------
// Coverage decision
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, CpuCoversBatchnormInference)
{
    const auto graph = buildBatchnormGraph();
    EXPECT_TRUE(referenceCoversGraph(ReferenceExecutorType::CPU, graph.data(), graph.size()));
    EXPECT_TRUE(uncoveredNodeTypes(ReferenceExecutorType::CPU, graph.data(), graph.size()).empty());
}

// The GPU reference has no batchnorm plan builder, so bundles using it are absent
// from the GPU validation suite rather than skipped inside it.
TEST(TestReferenceOpCoverage, DeviceReferenceDoesNotCoverBatchnormInference)
{
    const auto graph = buildBatchnormGraph();
    EXPECT_FALSE(referenceCoversGraph(ReferenceExecutorType::GPU, graph.data(), graph.size()));

    const auto uncovered
        = uncoveredNodeTypes(ReferenceExecutorType::GPU, graph.data(), graph.size());
    ASSERT_EQ(uncovered.size(), 1u);
    EXPECT_EQ(uncovered.front(), "BatchnormInferenceAttributes");
}

// ---------------------------------------------------------------------------
// Registration diagnostic
//
// The exclusion tally alone says a gap exists without saying which op to
// implement to close it, which is what made uncoveredNodeTypes() dead code.
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, NoExclusionsAddsNothingToTheSummary)
{
    EXPECT_EQ(formatUncoveredOps({}), "");
}

TEST(TestReferenceOpCoverage, ExcludedOpsAreNamedAndSeparated)
{
    EXPECT_EQ(formatUncoveredOps({"BatchnormInferenceAttributes"}),
              " (BatchnormInferenceAttributes)");
    EXPECT_EQ(formatUncoveredOps({"ReductionAttributes", "ConvolutionBwdDataAttributes"}),
              " (ConvolutionBwdDataAttributes, ReductionAttributes)");
}

// The gap table is load-bearing: an entry inverts a bundle's expectation, so a
// malformed one silently stops validating real data. These pin its shape.
//
// A bundle id may legitimately appear under both references — the varlen bundles
// are declined by each for its own reason — so the invariant is that a lookup
// returns an entry belonging to the reference asked for, not that the other
// reference has none.
TEST(TestReferenceOpCoverage, KnownGapLookupIsScopedToTheReferenceAsked)
{
    for(const auto& gap : knownReferenceGaps())
    {
        const auto* found = findKnownReferenceGap(gap.reference, gap.bundleId);
        ASSERT_NE(found, nullptr) << gap.bundleId << " is not findable under its own reference";
        EXPECT_EQ(found->reference, gap.reference);
        EXPECT_EQ(found->bundleId, gap.bundleId);
    }
}

// The fp8 batch bundles are a GPU-only gap: the CPU reference implements fp8, so
// listing them for CPU would wrongly assert it cannot run them. Pins the asymmetry
// rather than leaving it to a comment.
TEST(TestReferenceOpCoverage, Fp8BatchGapsAreNotListedForCpu)
{
    for(const char* bundleId : {"quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small.Small",
                                "quick_SdpaFwd_bhsd_fp8_hd128_nomask_batch_Small.Small"})
    {
        EXPECT_NE(findKnownReferenceGap(ReferenceExecutorType::GPU, bundleId), nullptr) << bundleId;
        EXPECT_EQ(findKnownReferenceGap(ReferenceExecutorType::CPU, bundleId), nullptr)
            << bundleId << " is listed as a CPU gap, but the CPU reference implements fp8";
    }
}

TEST(TestReferenceOpCoverage, KnownGapLookupMissesAreNull)
{
    EXPECT_EQ(findKnownReferenceGap(ReferenceExecutorType::GPU, "no_such_bundle.Case"), nullptr);
    EXPECT_EQ(findKnownReferenceGap(ReferenceExecutorType::CPU, ""), nullptr);
}

// A gap with no reason is just a silently disabled bundle, which is the thing the
// list exists to avoid. A duplicated one means two entries disagree about why.
TEST(TestReferenceOpCoverage, EveryKnownGapCarriesAReasonAndIsUnique)
{
    std::set<std::pair<int, std::string>> seen;
    for(const auto& gap : knownReferenceGaps())
    {
        EXPECT_FALSE(gap.bundleId.empty());
        EXPECT_FALSE(gap.reason.empty()) << gap.bundleId << " has no reason recorded";
        EXPECT_TRUE(seen.emplace(static_cast<int>(gap.reference), std::string(gap.bundleId)).second)
            << "duplicate gap entry for " << gap.bundleId;
    }
}

// The CPU reference is scalar, so it validates Sdpa only for quick-tier bundles at
// modest shapes. Both caps are needed: tier alone would keep the seq-4096 bundle
// (it lives in quick and takes minutes), and size alone would keep the whole
// standard tier. Neither excluded bundle goes unverified -- the GPU lane has them.
TEST(TestReferenceOpCoverage, CpuSdpaIsLimitedToQuickTier)
{
    const auto graph = buildSdpaGraph(256);

    EXPECT_TRUE(referenceShapeIsAffordable(
        ReferenceExecutorType::CPU, "quick_SdpaFwd_x.Small", graph.data(), graph.size()));
    EXPECT_FALSE(referenceShapeIsAffordable(
        ReferenceExecutorType::CPU, "standard_SdpaFwd_x.Medium", graph.data(), graph.size()));
}

TEST(TestReferenceOpCoverage, CpuSdpaIsLimitedByWorkingSet)
{
    // Not named `small`: <rpcndr.h> defines that as a macro on Windows.
    const auto smallGraph = buildSdpaGraph(256);
    const auto hugeGraph = buildSdpaGraph(8192);

    EXPECT_TRUE(referenceShapeIsAffordable(
        ReferenceExecutorType::CPU, "quick_SdpaFwd_x.Small", smallGraph.data(), smallGraph.size()));
    EXPECT_FALSE(referenceShapeIsAffordable(
        ReferenceExecutorType::CPU, "quick_SdpaFwd_x.Small", hugeGraph.data(), hugeGraph.size()));
}

// The gate is CPU-only: the GPU reference runs every one of these in milliseconds
// and is what keeps the excluded bundles covered.
TEST(TestReferenceOpCoverage, OnlyTheCpuReferenceIsGatedOnCost)
{
    const auto huge = buildSdpaGraph(8192);

    EXPECT_TRUE(referenceShapeIsAffordable(
        ReferenceExecutorType::GPU, "standard_SdpaFwd_x.Medium", huge.data(), huge.size()));
}

// Non-Sdpa ops are cheap at every checked-in shape, so neither cap applies to them
// -- a standard-tier batchnorm bundle must still be validated on CPU.
TEST(TestReferenceOpCoverage, NonSdpaGraphsAreNeverGated)
{
    const auto graph = buildBatchnormGraph();

    EXPECT_TRUE(referenceShapeIsAffordable(ReferenceExecutorType::CPU,
                                           "standard_BatchnormFwdInference_x.Large",
                                           graph.data(),
                                           graph.size()));
}

// NOLINTEND(readability-identifier-naming)
