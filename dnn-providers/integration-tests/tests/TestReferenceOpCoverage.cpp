// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The reference supported-op sets are a commitment: a bundle inside a set gets a
// validation test with no skip path, and one outside it is silently absent from the
// suite. Both halves of that need pinning.

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ReferenceGraphFixtures.hpp"
#include "harness/reference-validation/ReferenceOpCoverage.hpp"

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
using hipdnn_integration_tests::bundle::fixtures::buildBatchnormGraph;
using hipdnn_integration_tests::bundle::fixtures::buildSdpaGraph;

// NOLINTBEGIN(readability-identifier-naming)

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

TEST(TestReferenceOpCoverage, RaggedGapsAreListedForBothReferences)
{
    for(const char* bundleId : {"quick_SdpaFwd_bshd_bf16_hd128_causal_ragged_Small.Small",
                                "quick_SdpaFwd_bshd_bf16_hd128_nomask_ragged_Small.Small",
                                "quick_SdpaFwd_bshd_bf16_hd192_causal_ragged_Small.Small",
                                "quick_SdpaFwd_bshd_bf16_hd192_nomask_ragged_Small.Small"})
    {
        EXPECT_NE(findKnownReferenceGap(ReferenceExecutorType::GPU, bundleId), nullptr) << bundleId;
        EXPECT_NE(findKnownReferenceGap(ReferenceExecutorType::CPU, bundleId), nullptr) << bundleId;
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
