// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "harness/SupportClaimsAutoGen.hpp"
#include "harness/SupportMatrixCollector.hpp"

namespace
{

using hipdnn_integration_tests::CondensedSupportData;
using hipdnn_integration_tests::condenseSupportClaims;
using hipdnn_integration_tests::GraphSupportRecord;
using hipdnn_integration_tests::SupportMatcher;

// Build a record with just the fields the condenser reads: opChain,
// ioDtype, layout, testName, and supportingEngines.
GraphSupportRecord makeRecord(std::string op,
                              std::string io,
                              std::string layout,
                              bool engineSupports,
                              std::string testName = "TestSuite.TestCase",
                              std::string engineName = "TEST_ENGINE")
{
    GraphSupportRecord r;
    r.opChain = std::move(op);
    r.ioDtype = std::move(io);
    r.layout = std::move(layout);
    r.testName = std::move(testName);
    if(engineSupports)
    {
        r.supportingEngines.insert(std::move(engineName));
    }
    return r;
}

// Sugar: a matcher containing exactly these (op, in->out pair, layout)
// cross products. Used for set-equality assertions where insertion
// order shouldn't matter.
struct ExpectedMatcher
{
    std::vector<std::string> opChains;
    std::vector<std::string> ioDtypePairs; // "in->out" form
    std::vector<std::string> layouts;
};

bool matcherEquals(const SupportMatcher& actual, const ExpectedMatcher& expected)
{
    return actual.opChains == expected.opChains && actual.ioDtypePairs == expected.ioDtypePairs
           && actual.layouts == expected.layouts;
}

} // namespace

// -- Empty input --------------------------------------------------------

TEST(TestSupportClaimsCondenser, EmptyRecordsProducesEmptyOutput)
{
    auto result = condenseSupportClaims({}, "TEST_ENGINE");
    EXPECT_TRUE(result.matchers.empty());
    EXPECT_TRUE(result.unsupportedObservations.empty());
}

TEST(TestSupportClaimsCondenser, OnlyUnsupportedRecordsProducesNoMatchers)
{
    std::vector<GraphSupportRecord> records = {
        makeRecord("ConvFprop", "fp16", "NCHW", /*engineSupports=*/false),
        makeRecord("ConvFprop", "fp32", "NHWC", /*engineSupports=*/false),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    EXPECT_TRUE(result.matchers.empty());
    EXPECT_EQ(result.unsupportedObservations.size(), 2u);
}

// -- Single-cell and full-rectangle base cases --------------------------

TEST(TestSupportClaimsCondenser, SingleSupportedRecordEmitsSingleCellMatcher)
{
    std::vector<GraphSupportRecord> records = {
        makeRecord("ConvFprop", "fp32", "NCHW", true),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    ASSERT_EQ(result.matchers.size(), 1u);
    EXPECT_TRUE(matcherEquals(result.matchers[0], {{"ConvFprop"}, {"fp32->fp32"}, {"NCHW"}}));
}

TEST(TestSupportClaimsCondenser, FullRectangleCondensesToSingleMatcher)
{
    // 2 dtypes × 2 layouts, all supported → one rectangle.
    std::vector<GraphSupportRecord> records = {
        makeRecord("ConvFprop", "fp16", "NCHW", true),
        makeRecord("ConvFprop", "fp16", "NHWC", true),
        makeRecord("ConvFprop", "fp32", "NCHW", true),
        makeRecord("ConvFprop", "fp32", "NHWC", true),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    ASSERT_EQ(result.matchers.size(), 1u);
    EXPECT_TRUE(matcherEquals(result.matchers[0],
                              {{"ConvFprop"}, {"fp16->fp16", "fp32->fp32"}, {"NCHW", "NHWC"}}));
}

// -- The reviewer's counterexample (RFC 0012 §7 correctness) -----------
//
// U-tuples form an anti-diagonal; the only safe rectangles are the two
// single-cell ones along the main diagonal. Greedy axis-shrink would
// emit at most one rectangle for the op — silently dropping the second
// cell of S coverage. The corrected algorithm must emit both.

TEST(TestSupportClaimsCondenser, AntiDiagonalForbiddenCellsRequireTwoRectangles)
{
    std::vector<GraphSupportRecord> records = {
        // S: diagonal
        makeRecord("ConvFprop", "fp16", "NHWC", true),
        makeRecord("ConvFprop", "fp32", "NCHW", true),
        // U: anti-diagonal
        makeRecord("ConvFprop", "fp16", "NCHW", false),
        makeRecord("ConvFprop", "fp32", "NHWC", false),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    ASSERT_EQ(result.matchers.size(), 2u);

    // Order in result is sorted by opChains; for a single op_chain that
    // means matchers are sorted by their ios/layouts content. We check
    // set membership rather than position to keep the test resilient to
    // tiebreak ordering changes.
    std::set<std::vector<std::string>> matcherSignatures;
    for(const auto& m : result.matchers)
    {
        EXPECT_EQ(m.opChains, (std::vector<std::string>{"ConvFprop"}));
        std::vector<std::string> sig;
        sig.insert(sig.end(), m.ioDtypePairs.begin(), m.ioDtypePairs.end());
        sig.push_back("|");
        sig.insert(sig.end(), m.layouts.begin(), m.layouts.end());
        matcherSignatures.insert(std::move(sig));
    }
    EXPECT_TRUE(matcherSignatures.count({"fp16->fp16", "|", "NHWC"}))
        << "Expected matcher {fp16->fp16}×{NHWC} in cover";
    EXPECT_TRUE(matcherSignatures.count({"fp32->fp32", "|", "NCHW"}))
        << "Expected matcher {fp32->fp32}×{NCHW} in cover";
}

// -- Grouping across op_chains -----------------------------------------

TEST(TestSupportClaimsCondenser, OpChainsSharingRectangleGroupIntoOneMatcher)
{
    std::vector<GraphSupportRecord> records = {
        makeRecord("ConvFprop", "fp16", "NCHW", true),
        makeRecord("ConvFprop", "fp32", "NCHW", true),
        makeRecord("ConvDgrad", "fp16", "NCHW", true),
        makeRecord("ConvDgrad", "fp32", "NCHW", true),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    ASSERT_EQ(result.matchers.size(), 1u);
    EXPECT_TRUE(matcherEquals(
        result.matchers[0], {{"ConvDgrad", "ConvFprop"}, {"fp16->fp16", "fp32->fp32"}, {"NCHW"}}));
}

TEST(TestSupportClaimsCondenser, OpChainsWithDifferentSafeRectanglesGetSeparateMatchers)
{
    // ConvFprop: full {fp16,fp32}×{NCHW,NHWC} coverage
    // ConvDgrad: only fp16/NCHW is in S (U blocks others)
    std::vector<GraphSupportRecord> records = {
        makeRecord("ConvFprop", "fp16", "NCHW", true),
        makeRecord("ConvFprop", "fp16", "NHWC", true),
        makeRecord("ConvFprop", "fp32", "NCHW", true),
        makeRecord("ConvFprop", "fp32", "NHWC", true),
        makeRecord("ConvDgrad", "fp16", "NCHW", true),
        makeRecord("ConvDgrad", "fp16", "NHWC", false),
        makeRecord("ConvDgrad", "fp32", "NCHW", false),
        makeRecord("ConvDgrad", "fp32", "NHWC", false),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    ASSERT_EQ(result.matchers.size(), 2u);

    // Find each by op set.
    const SupportMatcher* fpropMatcher = nullptr;
    const SupportMatcher* dgradMatcher = nullptr;
    for(const auto& m : result.matchers)
    {
        if(m.opChains == std::vector<std::string>{"ConvFprop"})
        {
            fpropMatcher = &m;
        }
        else if(m.opChains == std::vector<std::string>{"ConvDgrad"})
        {
            dgradMatcher = &m;
        }
    }
    ASSERT_NE(fpropMatcher, nullptr);
    ASSERT_NE(dgradMatcher, nullptr);
    EXPECT_EQ(fpropMatcher->ioDtypePairs, (std::vector<std::string>{"fp16->fp16", "fp32->fp32"}));
    EXPECT_EQ(fpropMatcher->layouts, (std::vector<std::string>{"NCHW", "NHWC"}));
    EXPECT_EQ(dgradMatcher->ioDtypePairs, (std::vector<std::string>{"fp16->fp16"}));
    EXPECT_EQ(dgradMatcher->layouts, (std::vector<std::string>{"NCHW"}));
}

// -- Engine-name filtering ---------------------------------------------

TEST(TestSupportClaimsCondenser, RecordsForOtherEnginesAreIgnored)
{
    // Same observation supported by a different engine — counted as
    // "engine doesn't support" from our perspective, so the cell lands
    // in U and the matcher set is empty.
    std::vector<GraphSupportRecord> records = {
        makeRecord("ConvFprop",
                   "fp32",
                   "NCHW",
                   true,
                   /*testName=*/"SomeTest.Correctness/0",
                   /*engineName=*/"OTHER_ENGINE"),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    EXPECT_TRUE(result.matchers.empty());
    EXPECT_EQ(result.unsupportedObservations.size(), 1u);
}

TEST(TestSupportClaimsCondenser, RecordsWithEmptyOpChainAreSkipped)
{
    // Legacy callers of the string-string recordGraphSupport overload
    // produce records with empty opChain. The condenser must skip them
    // rather than emitting a matcher with an empty op_chain string.
    GraphSupportRecord legacy;
    legacy.opChain = "";
    legacy.ioDtype = "fp32";
    legacy.layout = "NCHW";
    legacy.supportingEngines.insert("TEST_ENGINE");
    auto result = condenseSupportClaims({legacy}, "TEST_ENGINE");
    EXPECT_TRUE(result.matchers.empty());
}

// -- Determinism --------------------------------------------------------

TEST(TestSupportClaimsCondenser, OutputIsDeterministicAcrossInputOrder)
{
    // Build two record vectors with identical content but reversed
    // order. The condenser's output must be byte-identical.
    std::vector<GraphSupportRecord> ordered = {
        makeRecord("ConvFprop", "fp16", "NCHW", true),
        makeRecord("ConvFprop", "fp16", "NHWC", true),
        makeRecord("ConvFprop", "fp32", "NCHW", true),
        makeRecord("ConvDgrad", "fp32", "NHWC", false),
    };
    std::vector<GraphSupportRecord> reversed(ordered.rbegin(), ordered.rend());

    auto a = condenseSupportClaims(ordered, "TEST_ENGINE");
    auto b = condenseSupportClaims(reversed, "TEST_ENGINE");

    ASSERT_EQ(a.matchers.size(), b.matchers.size());
    for(size_t i = 0; i < a.matchers.size(); ++i)
    {
        EXPECT_EQ(a.matchers[i].opChains, b.matchers[i].opChains);
        EXPECT_EQ(a.matchers[i].ioDtypePairs, b.matchers[i].ioDtypePairs);
        EXPECT_EQ(a.matchers[i].layouts, b.matchers[i].layouts);
    }
}

// -- Unobserved cells ---------------------------------------------------
//
// When an axis value appears in some op_chains but not others, the
// algorithm shouldn't synthesize coverage for unobserved combinations.

// -- S∩U conflict detection --------------------------------------------
//
// RFC 0012 §7's Safety invariant says a tuple can be in S xor U, never
// both. When it lands in both, the op_chain string is too coarse for
// the engine's actual dispatch granularity and the condenser must
// refuse rather than silently exclude.

TEST(TestSupportClaimsCondenser, TupleInBothSAndUProducesConflict)
{
    // Same (op, io, layout) tuple reported as supported by one test and
    // unsupported by another. Real-world cause: BatchnormFwdTrainingActiv
    // runs both FULL_TRAINING and WITH_BATCH_STATS scenarios — same
    // node type but different optional inputs wired up.
    std::vector<GraphSupportRecord> records = {
        makeRecord("Batchnorm + Pointwise:RELU_FWD",
                   "fp32",
                   "NCHW",
                   /*supports=*/true,
                   "BNTrainingActiv2dFp32.Correctness/0"),
        makeRecord("Batchnorm + Pointwise:RELU_FWD",
                   "fp32",
                   "NCHW",
                   /*supports=*/false,
                   "BNTrainingActiv2dFp32.Correctness/1"),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    ASSERT_EQ(result.conflictingObservations.size(), 1u);
    const auto& conflict = result.conflictingObservations[0];
    EXPECT_EQ(conflict.opChain, "Batchnorm + Pointwise:RELU_FWD");
    EXPECT_EQ(conflict.inputDtype, "fp32");
    // Always-pairs schema: even symmetric conflicts carry an explicit
    // outputDtype so the diagnostic always names the full pair.
    EXPECT_EQ(conflict.outputDtype, "fp32");
    EXPECT_EQ(conflict.layout, "NCHW");
    ASSERT_EQ(conflict.supportedBy.size(), 1u);
    EXPECT_EQ(conflict.supportedBy[0], "BNTrainingActiv2dFp32.Correctness/0");
    ASSERT_EQ(conflict.unsupportedBy.size(), 1u);
    EXPECT_EQ(conflict.unsupportedBy[0], "BNTrainingActiv2dFp32.Correctness/1");
    // No matchers emitted when conflict detected — caller (write driver)
    // refuses to write.
    EXPECT_TRUE(result.matchers.empty());
}

TEST(TestSupportClaimsCondenser, AllConflictsAreReportedNotJustFirst)
{
    // The diagnostic is most useful when it lists every conflict in
    // one pass — engineer can address all variant-tag gaps in one PR.
    std::vector<GraphSupportRecord> records = {
        makeRecord("ConvFprop", "fp16", "NCHW", true, "A"),
        makeRecord("ConvFprop", "fp16", "NCHW", false, "B"),
        makeRecord("ConvDgrad", "fp32", "NHWC", true, "C"),
        makeRecord("ConvDgrad", "fp32", "NHWC", false, "D"),
        makeRecord("Batchnorm", "bf16", "NCDHW", true, "E"),
        makeRecord("Batchnorm", "bf16", "NCDHW", false, "F"),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    EXPECT_EQ(result.conflictingObservations.size(), 3u);
}

TEST(TestSupportClaimsCondenser, ConflictListsAllTestCasesOnEachSide)
{
    // The supportedBy / unsupportedBy lists are sorted and complete so
    // an engineer reading the diagnostic can pattern-match across
    // test names to spot the scenario split (e.g. all "Correctness/0"
    // even, all "Correctness/1" odd).
    std::vector<GraphSupportRecord> records = {
        makeRecord("Batchnorm", "fp32", "NCHW", true, "Smoke/Foo.Correctness/0"),
        makeRecord("Batchnorm", "fp32", "NCHW", true, "Smoke/Foo.Correctness/2"),
        makeRecord("Batchnorm", "fp32", "NCHW", true, "Smoke/Foo.Correctness/4"),
        makeRecord("Batchnorm", "fp32", "NCHW", false, "Smoke/Foo.Correctness/1"),
        makeRecord("Batchnorm", "fp32", "NCHW", false, "Smoke/Foo.Correctness/3"),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    ASSERT_EQ(result.conflictingObservations.size(), 1u);
    const auto& conflict = result.conflictingObservations[0];
    EXPECT_EQ(conflict.supportedBy.size(), 3u);
    EXPECT_EQ(conflict.unsupportedBy.size(), 2u);
    // Sort guarantee.
    EXPECT_TRUE(std::is_sorted(conflict.supportedBy.begin(), conflict.supportedBy.end()));
    EXPECT_TRUE(std::is_sorted(conflict.unsupportedBy.begin(), conflict.unsupportedBy.end()));
}

TEST(TestSupportClaimsCondenser, UnobservedCellsArePreferredOutOfRectangle)
{
    // S = {(fp16,NCHW), (fp32,NCHW)}; (fp16,NHWC) and (fp32,NHWC) are
    // unobserved (no record at all). Claiming {fp16,fp32}×{NCHW,NHWC}
    // would be technically safe (NHWC isn't in U) but risks Rule A on
    // first enforce run. The tiebreak prefers smaller rectangles, so
    // {fp16,fp32}×{NCHW} should win.
    std::vector<GraphSupportRecord> records = {
        makeRecord("ConvFprop", "fp16", "NCHW", true),
        makeRecord("ConvFprop", "fp32", "NCHW", true),
    };
    auto result = condenseSupportClaims(records, "TEST_ENGINE");
    ASSERT_EQ(result.matchers.size(), 1u);
    EXPECT_EQ(result.matchers[0].layouts, (std::vector<std::string>{"NCHW"}));
    EXPECT_EQ(result.matchers[0].ioDtypePairs,
              (std::vector<std::string>{"fp16->fp16", "fp32->fp32"}));
}
