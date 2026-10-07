// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// How what happened to one catalog entry becomes a status and a report line, and how a
// graph's entries add up.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "harness/catalog-sweep/CatalogSweepTypes.hpp"

using namespace hipdnn_integration_tests::catalog_sweep;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

EntryOutcome stoppedAt(EntryStage stage)
{
    EntryOutcome outcome;
    outcome.failedStage = stage;
    outcome.message = "boom";
    return outcome;
}

SweepRunContext context()
{
    return {"gfx1151", "1.2.3.4"};
}

} // namespace

// ---- classification ---------------------------------------------------------------

TEST(TestCatalogSweepClassification, AnEntryThatRanAndMatchedPasses)
{
    EXPECT_EQ(classifyEntry({}), CatalogStatus::PASS);
}

TEST(TestCatalogSweepClassification, EveryPlanBuildingStageFailsAsABuild)
{
    EXPECT_EQ(classifyEntry(stoppedAt(EntryStage::CREATE_PLAN)), CatalogStatus::FAIL_BUILD);
    EXPECT_EQ(classifyEntry(stoppedAt(EntryStage::CHECK_SUPPORT)), CatalogStatus::FAIL_BUILD);
    EXPECT_EQ(classifyEntry(stoppedAt(EntryStage::BUILD_PLANS)), CatalogStatus::FAIL_BUILD);
}

TEST(TestCatalogSweepClassification, ARunThatBrokeFailsAsAnExecution)
{
    EXPECT_EQ(classifyEntry(stoppedAt(EntryStage::EXECUTE)), CatalogStatus::FAIL_EXECUTE);
}

TEST(TestCatalogSweepClassification, AnOutputOutOfToleranceFailsNumerically)
{
    EntryOutcome outcome;
    outcome.outputMismatch = true;
    EXPECT_EQ(classifyEntry(outcome), CatalogStatus::FAIL_NUMERIC);
}

// A kernel that writes nothing leaves the NaN the harness prefilled. The validator may
// not call that a mismatch (a NaN reference, a loose tolerance), but it is never a pass.
TEST(TestCatalogSweepClassification, ANanInTheOutputFailsEvenWhenTheValidatorAcceptedIt)
{
    EntryOutcome outcome;
    outcome.outputMismatch = false;
    outcome.nanElements = 1;
    EXPECT_EQ(classifyEntry(outcome), CatalogStatus::FAIL_NUMERIC);
}

TEST(TestCatalogSweepClassification, AComparisonThatCouldNotRunFailsNumerically)
{
    EXPECT_EQ(classifyEntry(stoppedAt(EntryStage::COMPARE)), CatalogStatus::FAIL_NUMERIC);
}

TEST(TestCatalogSweepClassification, OnlyPassAndFilteredAreNotFailures)
{
    EXPECT_FALSE(isFailure(CatalogStatus::PASS));
    EXPECT_FALSE(isFailure(CatalogStatus::FILTERED));
    for(const auto status : {CatalogStatus::FAIL_NUMERIC,
                             CatalogStatus::FAIL_BUILD,
                             CatalogStatus::FAIL_EXECUTE,
                             CatalogStatus::NO_CATALOG})
    {
        EXPECT_TRUE(isFailure(status)) << toString(status);
    }
}

// ---- the report line ----------------------------------------------------------------

TEST(TestCatalogSweepRecord, CarriesExactlyTheDocumentedFields)
{
    CatalogRecord record;
    record.graph = "quick_Pointwise.Small";
    record.engineId = 42;
    record.engineName = "ROCKE_POINTWISE";
    record.entryIndex = 3;
    record.status = CatalogStatus::FAIL_NUMERIC;
    record.maxAbsErr = 0.5;
    record.mismatchPct = 12.5;
    record.message = "output y mismatches";
    record.compileTimeMs = 810.25;

    const auto json = toJson(record, context());

    EXPECT_EQ(json.size(), 12U);
    EXPECT_EQ(json.at("graph"), "quick_Pointwise.Small");
    EXPECT_EQ(json.at("engine_id"), 42);
    EXPECT_EQ(json.at("engine_name"), "ROCKE_POINTWISE");
    EXPECT_EQ(json.at("entry_index"), 3);
    EXPECT_EQ(json.at("status"), "FAIL_NUMERIC");
    EXPECT_DOUBLE_EQ(json.at("max_abs_err").get<double>(), 0.5);
    EXPECT_DOUBLE_EQ(json.at("mismatch_pct").get<double>(), 12.5);
    EXPECT_EQ(json.at("message"), "output y mismatches");
    EXPECT_DOUBLE_EQ(json.at("compile_time_ms").get<double>(), 810.25);
    EXPECT_EQ(json.at("device_arch"), "gfx1151");
    EXPECT_EQ(json.at("hipdnn_version"), "1.2.3.4");
    EXPECT_TRUE(json.contains("knobs"));
    EXPECT_FALSE(json.contains("kind"))
        << "an entry record must stay distinguishable from a summary";
}

// The knob set identifies an entry, so the integer 2 and the string "2" must not collapse
// into each other on the way to the report.
TEST(TestCatalogSweepRecord, KnobValuesKeepTheirType)
{
    CatalogRecord record;
    record.knobs = {{"tile.m", std::int64_t{2}}, {"tile.name", std::string("2")}, {"scale", 0.5}};

    const auto json = toJson(record, context());
    const auto& knobs = json.at("knobs");

    ASSERT_EQ(knobs.size(), 3U);
    EXPECT_EQ(knobs[0].at("knob_id"), "tile.m");
    EXPECT_TRUE(knobs[0].at("value").is_number_integer());
    EXPECT_EQ(knobs[0].at("value").get<std::int64_t>(), 2);
    EXPECT_TRUE(knobs[1].at("value").is_string());
    EXPECT_EQ(knobs[1].at("value").get<std::string>(), "2");
    EXPECT_TRUE(knobs[2].at("value").is_number_float());
}

TEST(TestCatalogSweepRecord, AnEngineWithoutKnobsReportsAnEmptyArray)
{
    const auto json = toJson(CatalogRecord{}, context());
    EXPECT_TRUE(json.at("knobs").is_array());
    EXPECT_TRUE(json.at("knobs").empty());
}

TEST(TestCatalogSweepRecord, MeasurementsThatDoNotExistOrAreNotFiniteAreNull)
{
    CatalogRecord record;
    record.maxAbsErr = std::numeric_limits<double>::infinity();
    record.mismatchPct = std::nan("");

    const auto json = toJson(record, context());
    EXPECT_TRUE(json.at("max_abs_err").is_null());
    EXPECT_TRUE(json.at("mismatch_pct").is_null());
    EXPECT_TRUE(json.at("compile_time_ms").is_null());
    // And the line must survive a round trip through a parser: that is how it is read.
    EXPECT_NO_THROW(static_cast<void>(nlohmann::json::parse(json.dump())));
}

// ---- reconciliation -------------------------------------------------------------------

TEST(TestCatalogSweepReconcile, EveryReturnedEntryReportedMeansTheIdentityHolds)
{
    GraphTally tally;
    tally.noteReturned(6);
    for(const auto status : {CatalogStatus::PASS,
                             CatalogStatus::PASS,
                             CatalogStatus::FAIL_NUMERIC,
                             CatalogStatus::FAIL_BUILD,
                             CatalogStatus::FILTERED,
                             CatalogStatus::FILTERED})
    {
        tally.noteRecord(status);
    }

    EXPECT_TRUE(tally.reconciles());
    EXPECT_EQ(tally.failed(), 2U);
    EXPECT_EQ(formatReconcileLine("g.t", tally),
              "reconcile graph=g.t entries=6 = passed=2 + failed=2 + filtered=2 "
              "[fail_numeric=1 fail_build=1 fail_execute=0 no_catalog=0]");
}

// An engine with no catalog is one slot the engine failed to fill: it counts as returned
// and as failed, so the identity covers it.
TEST(TestCatalogSweepReconcile, AnEngineWithoutACatalogCountsAsOneFailedEntry)
{
    GraphTally tally;
    tally.noteReturned(3); // two entries of one engine, plus the catalog-less engine
    tally.noteRecord(CatalogStatus::PASS);
    tally.noteRecord(CatalogStatus::PASS);
    tally.noteRecord(CatalogStatus::NO_CATALOG);

    EXPECT_TRUE(tally.reconciles());
    EXPECT_EQ(tally.failed(), 1U);
    EXPECT_EQ(tally.noCatalog, 1U);
}

TEST(TestCatalogSweepReconcile, AnEntryThatWasNeverReportedBreaksTheIdentityLoudly)
{
    GraphTally tally;
    tally.noteReturned(3);
    tally.noteRecord(CatalogStatus::PASS);
    tally.noteRecord(CatalogStatus::FILTERED);

    EXPECT_FALSE(tally.reconciles());
    EXPECT_NE(formatReconcileLine("g", tally).find("MISMATCH"), std::string::npos);
}

TEST(TestCatalogSweepReconcile, ASummaryRecordIsToldApartFromAnEntryAndCarriesTheCounts)
{
    GraphTally tally;
    tally.noteReturned(2);
    tally.noteRecord(CatalogStatus::PASS);
    tally.noteRecord(CatalogStatus::FAIL_EXECUTE);

    const auto json = reconcileToJson("g.t", tally, GraphOutcome::SWEPT, "", context());

    EXPECT_EQ(json.at("kind"), "reconcile");
    EXPECT_EQ(json.at("graph"), "g.t");
    EXPECT_EQ(json.at("entries"), 2);
    EXPECT_EQ(json.at("passed"), 1);
    EXPECT_EQ(json.at("failed"), 1);
    EXPECT_EQ(json.at("filtered"), 0);
    EXPECT_EQ(json.at("fail_execute"), 1);
    EXPECT_TRUE(json.at("reconciled").get<bool>());
    EXPECT_EQ(json.at("graph_outcome"), "SWEPT");
    EXPECT_EQ(json.at("line").get<std::string>(), formatReconcileLine("g.t", tally));
}

// NOLINTEND(readability-identifier-naming)
