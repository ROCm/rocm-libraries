// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The writer and the enforcer are two halves of one contract. The writer
// records what the hardware said; the enforcer later re-reads that record and
// re-judges the same query. If the two disagree about which cell a claim lives
// in, every unit test on either side can still pass while a real run prints
// UNCLAIMED_SUPPORT forever and no amount of re-authoring clears it.
//
// These tests close that loop -- write a sidecar from observations, then judge
// the same query against the file that came out -- for the three things an
// engineer actually does:
//
//   1. bootstrap -- no sidecar yet; claims appear
//   2. append    -- a later run on another arch or engine adds to the file
//                   without disturbing what the first run wrote
//   3. re-run    -- same hardware, same answers, zero diff
//
// Both for a single-graph bundle and for a template sweep, because those use
// different files, different schemas, and different lookup paths.
//
// The judge helper below calls observeSupport() -- the enforcer's own read path
// -- once per engine, with the arch and platform passed in rather than read from
// TestConfig. Those two reads only answer "what machine am I on", which is not
// what is under test, and depending on them would tie these tests to a GPU being
// present. Everything else is production code: the sidecar is located by the same
// factories the harness uses and parsed by the same loader, so a disagreement
// about which cell a claim lives in cannot hide behind a test-local lookup.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_frontend/Error.hpp>

#include "harness/bundle/GraphSession.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/SupportClaimWriter.hpp"
#include "harness/bundle/SupportClaims.hpp"
#include "harness/bundle/SupportVerdict.hpp"

#include "ScratchDirectory.hpp"
#include "SupportClaimTestUtils.hpp"

using hipdnn_frontend::ErrorCode;
using hipdnn_integration_tests::bundle::enginesAccept;
using hipdnn_integration_tests::bundle::formatVerdictMessage;
using hipdnn_integration_tests::bundle::LoadedEngine;
using hipdnn_integration_tests::bundle::ObservedSupportCell;
using hipdnn_integration_tests::bundle::observeSupport;
using hipdnn_integration_tests::bundle::RankedEngines;
using hipdnn_integration_tests::bundle::SidecarState;
using hipdnn_integration_tests::bundle::singleGraphClaimLocator;
using hipdnn_integration_tests::bundle::SupportClaimLocator;
using hipdnn_integration_tests::bundle::SupportResult;
using hipdnn_integration_tests::bundle::SupportVerdict;
using hipdnn_integration_tests::bundle::sweepCaseClaimLocator;
using hipdnn_integration_tests::bundle::writeObservedSupportClaims;
using hipdnn_integration_tests::bundle::test_utils::readFile;
using hipdnn_integration_tests::bundle::test_utils::singleGraphObservation;
using hipdnn_integration_tests::bundle::test_utils::sweepCaseObservation;
using hipdnn_integration_tests::scratch::makeDir;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t MIOPEN_ID = 1;
constexpr int64_t HIP_KERNEL_ID = 2;

const std::vector<LoadedEngine>& bothEngines()
{
    static const std::vector<LoadedEngine> s_engines
        = {{MIOPEN_ID, "MIOPEN_ENGINE"}, {HIP_KERNEL_ID, "HIP_KERNEL_ENGINE"}};
    return s_engines;
}

/// A query that resolved, ranking `rankedIds`, judged from `engine`'s seat.
/// `accepted` is per-engine, so this cannot be hoisted out of the loop below.
RankedEngines ranked(const std::vector<int64_t>& rankedIds, const LoadedEngine& engine)
{
    RankedEngines engines;
    engines.status = hipdnn_frontend::Error{ErrorCode::OK, ""};
    engines.rankedIds = rankedIds;
    engines.accepted = enginesAccept(ErrorCode::OK, rankedIds, engine);
    return engines;
}

/// Read back what the writer just wrote, through the enforcer's own path.
///
/// The locator is built by the same factory the writer's observations carry, so
/// this asks the file the writer actually touched -- there is no second opinion
/// here about where a claim lives. Sweep and single-graph differ only in the
/// locator; observeSupport() dispatches on locator.isSweep() itself.
std::vector<SupportResult> judge(const SupportClaimLocator& locator,
                                 const std::vector<int64_t>& rankedIds,
                                 const std::string& arch,
                                 const std::string& platform)
{
    std::vector<SupportResult> results;
    for(const auto& engine : bothEngines())
    {
        const auto observation
            = observeSupport(ranked(rankedIds, engine), locator, engine, arch, platform);

        // Every call site below writes the sidecar first. A NONE here would mean
        // the judge is looking somewhere the writer never wrote, which is the
        // exact divergence these tests exist to catch.
        EXPECT_EQ(observation.sidecar, SidecarState::CHECKED)
            << "sidecar not read at " << locator.sidecarPath;

        results.insert(results.end(), observation.results.begin(), observation.results.end());
    }
    return results;
}

/// The round-trip property. Nothing the writer just recorded may come back as a
/// surprise: UNCLAIMED_SUPPORT means the file is missing a line the writer
/// should have added, CLAIM_BROKEN means it added one the query does not back.
void expectNoSurprises(const std::vector<SupportResult>& results)
{
    for(const auto& result : results)
    {
        EXPECT_NE(result.verdict, SupportVerdict::UNCLAIMED_SUPPORT)
            << formatVerdictMessage(result);
        EXPECT_NE(result.verdict, SupportVerdict::CLAIM_BROKEN) << formatVerdictMessage(result);
        EXPECT_NE(result.verdict, SupportVerdict::QUERY_ERRORED) << formatVerdictMessage(result);
    }
}

/// The verdict recorded for `engineName`, or nullopt if none was.
///
/// Absence is an answer, not a lookup failure: chooseVerdict() records nothing
/// for an engine that is neither claimed here nor in the ranked list, which is
/// the case where there is simply nothing to enforce.
std::optional<SupportVerdict> verdictFor(const std::vector<SupportResult>& results,
                                         const std::string& engineName)
{
    for(const auto& result : results)
    {
        if(result.engineName == engineName)
        {
            return result.verdict;
        }
    }
    return std::nullopt;
}

} // namespace

// ---------------------------------------------------------------------------
// Single graph: bootstrap -> append -> re-run
// ---------------------------------------------------------------------------

TEST(TestSupportClaimRoundTrip, SingleGraphBootstrapAppendAndReRun)
{
    const ScopedDirectory dir = makeDir("test_round_trip_");
    const auto bundlePath = dir.path() / "Small.json";
    const auto sidecarPath = dir.path() / "Small.support.json";

    // --- Run 1: gfx942, one engine (mode C). Nothing on disk yet. ---
    const std::vector<ObservedSupportCell> firstRun = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };
    ASSERT_EQ(writeObservedSupportClaims(firstRun).filesWritten, 1u);
    ASSERT_TRUE(std::filesystem::exists(sidecarPath));

    {
        const auto verdicts
            = judge(singleGraphClaimLocator(bundlePath), {MIOPEN_ID}, "gfx942", "linux");
        expectNoSurprises(verdicts);
        EXPECT_EQ(verdictFor(verdicts, "MIOPEN_ENGINE"), SupportVerdict::CLAIM_ACCEPTED);
        // Never queried, never claimed: silence, not a failure. No claim to
        // enforce means no record at all, which is why this asks for absence
        // rather than for a verdict naming it.
        EXPECT_FALSE(verdictFor(verdicts, "HIP_KERNEL_ENGINE").has_value());
    }

    // --- Run 2: a second machine (gfx90a) where both engines answer yes. ---
    const std::vector<ObservedSupportCell> secondRun = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx90a", "linux", true),
        singleGraphObservation(bundlePath, "HIP_KERNEL_ENGINE", "gfx90a", "linux", true),
    };
    EXPECT_EQ(writeObservedSupportClaims(secondRun).filesWritten, 1u);

    {
        const auto verdicts = judge(
            singleGraphClaimLocator(bundlePath), {MIOPEN_ID, HIP_KERNEL_ID}, "gfx90a", "linux");
        expectNoSurprises(verdicts);
        EXPECT_EQ(verdictFor(verdicts, "MIOPEN_ENGINE"), SupportVerdict::CLAIM_ACCEPTED);
        EXPECT_EQ(verdictFor(verdicts, "HIP_KERNEL_ENGINE"), SupportVerdict::CLAIM_ACCEPTED);
    }

    // Run 1's claim is still there — gfx942 was never observed this time, and an
    // unobserved cell must not be rewritten.
    {
        const auto verdicts
            = judge(singleGraphClaimLocator(bundlePath), {MIOPEN_ID}, "gfx942", "linux");
        expectNoSurprises(verdicts);
        EXPECT_EQ(verdictFor(verdicts, "MIOPEN_ENGINE"), SupportVerdict::CLAIM_ACCEPTED);
    }

    // --- Run 3: same machine, same answers. Zero diff (RFC 0015 §9.5). ---
    const auto contentBeforeReRun = readFile(sidecarPath);
    const auto reRunSummary = writeObservedSupportClaims(secondRun);
    EXPECT_EQ(reRunSummary.filesWritten, 0u);
    EXPECT_EQ(reRunSummary.filesUnchanged, 1u);
    EXPECT_EQ(readFile(sidecarPath), contentBeforeReRun);
}

// ---------------------------------------------------------------------------
// Single graph: a cell that loses support leaves no permanent CLAIM_BROKEN
// ---------------------------------------------------------------------------

TEST(TestSupportClaimRoundTrip, SingleGraphWithdrawnSupportJudgesCleanAfterReWrite)
{
    const ScopedDirectory dir = makeDir("test_round_trip_");
    const auto bundlePath = dir.path() / "Small.json";
    const auto sidecarPath = dir.path() / "Small.support.json";

    writeObservedSupportClaims({
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    });
    ASSERT_TRUE(std::filesystem::exists(sidecarPath));

    // The engine stops accepting the graph. Judged against the stale file this
    // is CLAIM_BROKEN, which is exactly the failure re-authoring exists to fix.
    {
        const auto verdicts
            = judge(singleGraphClaimLocator(bundlePath), /*rankedIds=*/{}, "gfx942", "linux");
        EXPECT_EQ(verdictFor(verdicts, "MIOPEN_ENGINE"), SupportVerdict::CLAIM_BROKEN);
    }

    writeObservedSupportClaims({
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", false),
    });

    {
        const auto verdicts
            = judge(singleGraphClaimLocator(bundlePath), /*rankedIds=*/{}, "gfx942", "linux");
        expectNoSurprises(verdicts);
        // The claim is gone and the engine declines: nothing left to enforce, so
        // no verdict is recorded at all.
        EXPECT_FALSE(verdictFor(verdicts, "MIOPEN_ENGINE").has_value());
    }
}

// ---------------------------------------------------------------------------
// Sweep: bootstrap -> append -> re-run, keyed by case id
// ---------------------------------------------------------------------------

TEST(TestSupportClaimRoundTrip, SweepBootstrapAppendAndReRun)
{
    const ScopedDirectory dir = makeDir("test_round_trip_");
    const auto sweepPath = dir.path() / "sweep.json";
    const auto sidecarPath = dir.path() / "support.json";

    // --- Run 1: gfx942, both engines queried (mode B); only MIOPEN answers. ---
    const std::vector<ObservedSupportCell> firstRun = {
        sweepCaseObservation(sweepPath, "case_a", "MIOPEN_ENGINE", "gfx942", "linux", true),
        sweepCaseObservation(sweepPath, "case_a", "HIP_KERNEL_ENGINE", "gfx942", "linux", false),
        sweepCaseObservation(sweepPath, "case_b", "MIOPEN_ENGINE", "gfx942", "linux", true),
        sweepCaseObservation(sweepPath, "case_b", "HIP_KERNEL_ENGINE", "gfx942", "linux", false),
    };
    ASSERT_EQ(writeObservedSupportClaims(firstRun).filesWritten, 1u);
    ASSERT_TRUE(std::filesystem::exists(sidecarPath));

    for(const auto* caseId : {"case_a", "case_b"})
    {
        const auto verdicts
            = judge(sweepCaseClaimLocator(sweepPath, caseId), {MIOPEN_ID}, "gfx942", "linux");
        expectNoSurprises(verdicts);
        EXPECT_EQ(verdictFor(verdicts, "MIOPEN_ENGINE"), SupportVerdict::CLAIM_ACCEPTED) << caseId;
        // Queried and declined, so nothing was claimed for it: silence, not a
        // verdict naming it.
        EXPECT_FALSE(verdictFor(verdicts, "HIP_KERNEL_ENGINE").has_value()) << caseId;
    }

    // --- Run 2: gfx90a, where both engines answer yes for both cases. ---
    const std::vector<ObservedSupportCell> secondRun = {
        sweepCaseObservation(sweepPath, "case_a", "MIOPEN_ENGINE", "gfx90a", "linux", true),
        sweepCaseObservation(sweepPath, "case_a", "HIP_KERNEL_ENGINE", "gfx90a", "linux", true),
        sweepCaseObservation(sweepPath, "case_b", "MIOPEN_ENGINE", "gfx90a", "linux", true),
        sweepCaseObservation(sweepPath, "case_b", "HIP_KERNEL_ENGINE", "gfx90a", "linux", true),
    };
    EXPECT_EQ(writeObservedSupportClaims(secondRun).filesWritten, 1u);

    for(const auto* caseId : {"case_a", "case_b"})
    {
        const auto verdicts = judge(sweepCaseClaimLocator(sweepPath, caseId),
                                    {MIOPEN_ID, HIP_KERNEL_ID},
                                    "gfx90a",
                                    "linux");
        expectNoSurprises(verdicts);
        EXPECT_EQ(verdictFor(verdicts, "MIOPEN_ENGINE"), SupportVerdict::CLAIM_ACCEPTED) << caseId;
        EXPECT_EQ(verdictFor(verdicts, "HIP_KERNEL_ENGINE"), SupportVerdict::CLAIM_ACCEPTED)
            << caseId;
    }

    // Run 1's gfx942 claims survived the append.
    {
        const auto verdicts
            = judge(sweepCaseClaimLocator(sweepPath, "case_a"), {MIOPEN_ID}, "gfx942", "linux");
        expectNoSurprises(verdicts);
        EXPECT_EQ(verdictFor(verdicts, "MIOPEN_ENGINE"), SupportVerdict::CLAIM_ACCEPTED);
    }

    // Both cases still share a support footprint per engine, so each engine
    // holds exactly one group -- the append re-grouped canonically rather than
    // splitting case_a and case_b apart.
    {
        auto json = nlohmann::json::parse(readFile(sidecarPath));
        EXPECT_EQ(json["claims"]["MIOPEN_ENGINE"].size(), 1u);
        EXPECT_EQ(json["claims"]["MIOPEN_ENGINE"][0]["cases"].size(), 2u);
        EXPECT_EQ(json["claims"]["HIP_KERNEL_ENGINE"].size(), 1u);
        EXPECT_EQ(json["claims"]["HIP_KERNEL_ENGINE"][0]["cases"].size(), 2u);
    }

    // --- Run 3: same machine, same answers. Zero diff. ---
    const auto contentBeforeReRun = readFile(sidecarPath);
    const auto reRunSummary = writeObservedSupportClaims(secondRun);
    EXPECT_EQ(reRunSummary.filesWritten, 0u);
    EXPECT_EQ(reRunSummary.filesUnchanged, 1u);
    EXPECT_EQ(readFile(sidecarPath), contentBeforeReRun);
}

// ---------------------------------------------------------------------------
// Sweep: one case diverging from its group is judged on its own support
// ---------------------------------------------------------------------------

TEST(TestSupportClaimRoundTrip, SweepDivergingCaseKeepsItsOwnVerdict)
{
    const ScopedDirectory dir = makeDir("test_round_trip_");
    const auto sweepPath = dir.path() / "sweep.json";
    const auto sidecarPath = dir.path() / "support.json";

    writeObservedSupportClaims({
        sweepCaseObservation(sweepPath, "case_a", "MIOPEN_ENGINE", "gfx942", "linux", true),
        sweepCaseObservation(sweepPath, "case_b", "MIOPEN_ENGINE", "gfx942", "linux", true),
    });
    ASSERT_TRUE(std::filesystem::exists(sidecarPath));

    // case_b regresses; case_a does not. The two must part ways in the file.
    writeObservedSupportClaims({
        sweepCaseObservation(sweepPath, "case_b", "MIOPEN_ENGINE", "gfx942", "linux", false),
    });

    const auto caseAVerdicts
        = judge(sweepCaseClaimLocator(sweepPath, "case_a"), {MIOPEN_ID}, "gfx942", "linux");
    expectNoSurprises(caseAVerdicts);
    EXPECT_EQ(verdictFor(caseAVerdicts, "MIOPEN_ENGINE"), SupportVerdict::CLAIM_ACCEPTED);

    // case_b is unclaimed now, so a query that still says yes reports the file
    // is behind rather than passing silently.
    const auto caseBVerdicts
        = judge(sweepCaseClaimLocator(sweepPath, "case_b"), {MIOPEN_ID}, "gfx942", "linux");
    EXPECT_EQ(verdictFor(caseBVerdicts, "MIOPEN_ENGINE"), SupportVerdict::UNCLAIMED_SUPPORT);

    // ...and a query that agrees with the withdrawal is simply silent: nothing
    // claimed, nothing ranked, so no verdict is recorded at all.
    const auto caseBDeclined
        = judge(sweepCaseClaimLocator(sweepPath, "case_b"), /*rankedIds=*/{}, "gfx942", "linux");
    expectNoSurprises(caseBDeclined);
    EXPECT_FALSE(verdictFor(caseBDeclined, "MIOPEN_ENGINE").has_value());
}

// NOLINTEND(readability-identifier-naming)
