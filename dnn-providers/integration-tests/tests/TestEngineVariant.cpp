// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A catalog sweep case names one candidate of the engine (`metadata.engine_variant`): the
// metadata parse that reads it, and the plan creation that builds exactly that candidate and not
// the engine's heuristic pick. No device and no engine plugin: the plan creation is a template,
// proven here against a graph that only counts which call was made.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <nlohmann/json.hpp>

#include "harness/BundleMetadata.hpp"
#include "harness/bundle/EngineVariant.hpp"
#include "harness/bundle/LoadedEngine.hpp"

using hipdnn_frontend::Error;
using hipdnn_frontend::ErrorCode;
using hipdnn_frontend::KnobSetting;
using hipdnn_integration_tests::BundleMetadataError;
using hipdnn_integration_tests::parseBundleMetadataJson;
using hipdnn_integration_tests::bundle::createCasePlans;
using hipdnn_integration_tests::bundle::EngineVariantSpec;
using hipdnn_integration_tests::bundle::LoadedEngine;

// NOLINTBEGIN(readability-identifier-naming) -- the fake mirrors the frontend Graph's names

namespace
{

nlohmann::json metadataWith(const std::string& engineVariant)
{
    return nlohmann::json::parse(R"({"format_version": 1, "engine_variant": )" + engineVariant
                                 + "}");
}

/// Counts which of the frontend Graph's two plan-creation calls was made, and with what.
struct FakeGraph
{
    int heuristicCalls = 0;
    int extCalls = 0;
    int64_t extEngineId = -1;
    std::vector<KnobSetting> extKnobs;
    Error extResult;

    Error create_execution_plans()
    {
        ++heuristicCalls;
        return {};
    }

    Error create_execution_plan_ext(int64_t engineId, const std::vector<KnobSetting>& settings)
    {
        ++extCalls;
        extEngineId = engineId;
        extKnobs = settings;
        return extResult;
    }
};

constexpr int64_t ENGINE_ID = 7;

std::optional<LoadedEngine> engineUnderTest()
{
    return LoadedEngine{ENGINE_ID, "ENGINE_UNDER_TEST"};
}

EngineVariantSpec chosen()
{
    EngineVariantSpec variant;
    variant.id = "tile_128x64";
    variant.knobs.emplace_back("HIPDNN_KNOB_TILE_M", int64_t{128});
    variant.knobs.emplace_back("HIPDNN_KNOB_SCALE", 0.5);
    variant.knobs.emplace_back("HIPDNN_KNOB_MODE", std::string("fast"));
    return variant;
}

} // namespace

// ---------------------------------------------------------------------------
// The metadata parse
// ---------------------------------------------------------------------------

// Bug caught: a case without an engine_variant acquiring one, which would force every ordinary
// case off the heuristic path.
TEST(TestParseEngineVariant, AbsentLeavesTheHeuristicPick)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(R"({"format_version": 1})"));
    EXPECT_FALSE(meta.engineVariant.has_value());
}

// Bug caught: a knob value read back in another type than the listing printed it in (an integer
// knob as a real, a real as an integer), which selects another candidate, or none.
TEST(TestParseEngineVariant, KnobValuesKeepTheirType)
{
    const auto meta = parseBundleMetadataJson(metadataWith(
        R"({"id": "tile_128x64",
            "knobs": {"HIPDNN_KNOB_TILE_M": 128, "HIPDNN_KNOB_SCALE": 0.5,
                      "HIPDNN_KNOB_WHOLE": 2.0, "HIPDNN_KNOB_MODE": "fast"}})"));

    ASSERT_TRUE(meta.engineVariant.has_value());
    EXPECT_EQ(meta.engineVariant->id, "tile_128x64");
    EXPECT_FALSE(meta.engineVariant->isDefault());

    // The object's keys come back in name order.
    const std::vector<KnobSetting> expected{KnobSetting("HIPDNN_KNOB_MODE", std::string("fast")),
                                            KnobSetting("HIPDNN_KNOB_SCALE", 0.5),
                                            KnobSetting("HIPDNN_KNOB_TILE_M", int64_t{128}),
                                            KnobSetting("HIPDNN_KNOB_WHOLE", 2.0)};
    ASSERT_EQ(meta.engineVariant->knobs.size(), expected.size());
    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_TRUE(meta.engineVariant->knobs[i] == expected[i])
            << meta.engineVariant->knobs[i].toString() << " vs " << expected[i].toString();
    }
}

// An engine whose candidates carry no knobs is a legal catalog: the id alone names the case.
TEST(TestParseEngineVariant, KnobsAreOptional)
{
    const auto meta = parseBundleMetadataJson(metadataWith(R"({"id": "k0"})"));
    ASSERT_TRUE(meta.engineVariant.has_value());
    EXPECT_EQ(meta.engineVariant->id, "k0");
    EXPECT_TRUE(meta.engineVariant->knobs.empty());
}

// Bug caught: the default (an engine that cannot enumerate) carrying knobs that would be
// silently ignored by the heuristic path while the record names a candidate.
TEST(TestParseEngineVariant, TheDefaultIsTheHeuristicPickAndTakesNoKnobs)
{
    const auto meta = parseBundleMetadataJson(metadataWith(R"({"id": "default", "knobs": {}})"));
    ASSERT_TRUE(meta.engineVariant.has_value());
    EXPECT_TRUE(meta.engineVariant->isDefault());

    EXPECT_THROW(parseBundleMetadataJson(
                     metadataWith(R"({"id": "default", "knobs": {"HIPDNN_KNOB_TILE_M": 1}})")),
                 BundleMetadataError);
}

// Bug caught: a malformed engine_variant degrading to the heuristic pick, so a typo silently runs
// a case that no longer exercises the candidate it was drawn for.
TEST(TestParseEngineVariant, MalformedVariantsThrow)
{
    for(const auto* text : {R"("tile_128x64")",
                            R"([])",
                            R"(null)",
                            R"({})",
                            R"({"knobs": {}})",
                            R"({"id": ""})",
                            R"({"id": 3})",
                            R"({"id": null})",
                            R"({"id": "a", "knobs": []})",
                            R"({"id": "a", "knobs": "x"})",
                            R"({"id": "a", "knobs": null})",
                            R"({"id": "a", "knob": {}})",
                            R"({"id": "a", "knobs": {"K": true}})",
                            R"({"id": "a", "knobs": {"K": null}})",
                            R"({"id": "a", "knobs": {"K": [1]}})",
                            R"({"id": "a", "knobs": {"K": {"v": 1}}})",
                            R"({"id": "a", "knobs": {"K": 18446744073709551615}})"})
    {
        EXPECT_THROW(parseBundleMetadataJson(metadataWith(text), "case_0"), BundleMetadataError)
            << text;
    }
}

// The error names where the metadata came from, so an author can find the case.
TEST(TestParseEngineVariant, TheErrorNamesTheCase)
{
    try
    {
        parseBundleMetadataJson(metadataWith(R"({"knobs": {}})"), "sweep.json#case_3");
        FAIL() << "expected BundleMetadataError";
    }
    catch(const BundleMetadataError& error)
    {
        EXPECT_NE(std::string(error.what()).find("sweep.json#case_3"), std::string::npos)
            << error.what();
        EXPECT_NE(std::string(error.what()).find("id"), std::string::npos) << error.what();
    }
}

// ---------------------------------------------------------------------------
// The plan creation
// ---------------------------------------------------------------------------

// Bug caught: a chosen candidate built through the heuristic path, which builds whatever the
// ranking puts first and reports the sweep's candidate as covered; or built with other knobs.
TEST(TestCreateCasePlans, AChosenCandidateIsBuiltWithItsKnobsAndNotByTheHeuristic)
{
    FakeGraph graph;
    const auto variant = chosen();

    const auto result = createCasePlans(graph, engineUnderTest(), variant);

    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(graph.extCalls, 1);
    EXPECT_EQ(graph.heuristicCalls, 0);
    EXPECT_EQ(graph.extEngineId, ENGINE_ID);
    ASSERT_EQ(graph.extKnobs.size(), variant.knobs.size());
    for(size_t i = 0; i < variant.knobs.size(); ++i)
    {
        EXPECT_TRUE(graph.extKnobs[i] == variant.knobs[i]) << graph.extKnobs[i].toString();
    }
}

// Bug caught: the heuristic path lost for ordinary cases, or reached through the ext call.
TEST(TestCreateCasePlans, ACaseWithoutAVariantTakesTheHeuristicPath)
{
    FakeGraph graph;

    const auto result = createCasePlans(graph, engineUnderTest(), std::nullopt);

    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(graph.heuristicCalls, 1);
    EXPECT_EQ(graph.extCalls, 0);
}

// Bug caught: the one-entry catalog of an engine that cannot enumerate being built through the
// ext call with no knobs, which an engine that has no knob tuple for its pick may refuse.
TEST(TestCreateCasePlans, TheDefaultVariantTakesTheHeuristicPath)
{
    FakeGraph graph;
    const EngineVariantSpec variant{"default", {}};

    const auto result = createCasePlans(graph, engineUnderTest(), variant);

    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(graph.heuristicCalls, 1);
    EXPECT_EQ(graph.extCalls, 0);
}

// Bug caught: a failure of the chosen candidate (an engine that refuses the tuple) swallowed, or
// answered by falling back to the heuristic pick, which would pass the case on another kernel.
TEST(TestCreateCasePlans, AFailureOfTheChosenCandidateIsReturnedAndNotRetried)
{
    FakeGraph graph;
    graph.extResult = Error{ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED, "no such tuple"};

    const auto result = createCasePlans(graph, engineUnderTest(), chosen());

    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.get_message(), "no such tuple");
    EXPECT_EQ(graph.extCalls, 1);
    EXPECT_EQ(graph.heuristicCalls, 0);
}

// Bug caught: a candidate built on an engine nobody named, which a knob tuple cannot select.
TEST(TestCreateCasePlans, AChosenCandidateNeedsAnEngine)
{
    FakeGraph graph;

    const auto result = createCasePlans(graph, std::nullopt, chosen());

    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("--test-engine"), std::string::npos)
        << result.get_message();
    EXPECT_EQ(graph.extCalls, 0);
    EXPECT_EQ(graph.heuristicCalls, 0);
}

// NOLINTEND(readability-identifier-naming)
