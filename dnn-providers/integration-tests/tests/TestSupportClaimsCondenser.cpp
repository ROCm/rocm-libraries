// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

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
// ioDtype, layout, and supportingEngines. graphName / testName don't
// affect grouping.
GraphSupportRecord makeRecord(std::string op,
                              std::string io,
                              std::string layout,
                              bool engineSupports,
                              std::string engineName = "TEST_ENGINE")
{
    GraphSupportRecord r;
    r.opChain = std::move(op);
    r.ioDtype = std::move(io);
    r.layout = std::move(layout);
    if(engineSupports)
    {
        r.supportingEngines.insert(std::move(engineName));
    }
    return r;
}

// Sugar: a matcher containing exactly these (op, io, layout) cross
// products. Used for set-equality assertions where insertion order
// shouldn't matter.
struct ExpectedMatcher
{
    std::vector<std::string> opChains;
    std::vector<std::string> ioDtypes;
    std::vector<std::string> layouts;
};

bool matcherEquals(const SupportMatcher& actual, const ExpectedMatcher& expected)
{
    return actual.opChains == expected.opChains && actual.ioDtypes == expected.ioDtypes
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
    EXPECT_TRUE(matcherEquals(result.matchers[0], {{"ConvFprop"}, {"fp32"}, {"NCHW"}}));
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
    EXPECT_TRUE(
        matcherEquals(result.matchers[0], {{"ConvFprop"}, {"fp16", "fp32"}, {"NCHW", "NHWC"}}));
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
        sig.insert(sig.end(), m.ioDtypes.begin(), m.ioDtypes.end());
        sig.push_back("|");
        sig.insert(sig.end(), m.layouts.begin(), m.layouts.end());
        matcherSignatures.insert(std::move(sig));
    }
    EXPECT_TRUE(matcherSignatures.count({"fp16", "|", "NHWC"}))
        << "Expected matcher {fp16}×{NHWC} in cover";
    EXPECT_TRUE(matcherSignatures.count({"fp32", "|", "NCHW"}))
        << "Expected matcher {fp32}×{NCHW} in cover";
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
    EXPECT_TRUE(matcherEquals(result.matchers[0],
                              {{"ConvDgrad", "ConvFprop"}, {"fp16", "fp32"}, {"NCHW"}}));
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
    EXPECT_EQ(fpropMatcher->ioDtypes, (std::vector<std::string>{"fp16", "fp32"}));
    EXPECT_EQ(fpropMatcher->layouts, (std::vector<std::string>{"NCHW", "NHWC"}));
    EXPECT_EQ(dgradMatcher->ioDtypes, (std::vector<std::string>{"fp16"}));
    EXPECT_EQ(dgradMatcher->layouts, (std::vector<std::string>{"NCHW"}));
}

// -- Engine-name filtering ---------------------------------------------

TEST(TestSupportClaimsCondenser, RecordsForOtherEnginesAreIgnored)
{
    // Same observation supported by a different engine — counted as
    // "engine doesn't support" from our perspective, so the cell lands
    // in U and the matcher set is empty.
    std::vector<GraphSupportRecord> records = {
        makeRecord("ConvFprop", "fp32", "NCHW", true, "OTHER_ENGINE"),
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
        EXPECT_EQ(a.matchers[i].ioDtypes, b.matchers[i].ioDtypes);
        EXPECT_EQ(a.matchers[i].layouts, b.matchers[i].layouts);
    }
}

// -- Unobserved cells ---------------------------------------------------
//
// When an axis value appears in some op_chains but not others, the
// algorithm shouldn't synthesize coverage for unobserved combinations.

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
    EXPECT_EQ(result.matchers[0].ioDtypes, (std::vector<std::string>{"fp16", "fp32"}));
}
