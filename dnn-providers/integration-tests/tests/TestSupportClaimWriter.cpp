// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness/bundle/SupportClaimWriter.hpp"
#include "harness/bundle/SupportClaims.hpp"

#include "ScratchDirectory.hpp"
#include "SupportClaimTestUtils.hpp"

using hipdnn_integration_tests::bundle::dumpCanonical;
using hipdnn_integration_tests::bundle::ObservedGraphSupport;
using hipdnn_integration_tests::bundle::parseSupportClaimsJson;
using hipdnn_integration_tests::bundle::parseSweepSupportClaimsJson;
using hipdnn_integration_tests::bundle::writeObservedSupportClaims;
using hipdnn_integration_tests::bundle::test_utils::readFile;
using hipdnn_integration_tests::bundle::test_utils::singleGraphObservation;
using hipdnn_integration_tests::bundle::test_utils::sweepCaseObservation;
using hipdnn_integration_tests::scratch::makeDir;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

// ---------------------------------------------------------------------------
// Single-graph: basic write
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, SingleGraphWriteCreatesNewSidecar)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "Small.json";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto summary = writeObservedSupportClaims(observations);
    EXPECT_EQ(summary.filesWritten, 1u);
    EXPECT_EQ(summary.filesUnchanged, 0u);
    EXPECT_TRUE(summary.errors.empty());

    const auto sidecarPath = dir.path() / "Small.support.json";
    ASSERT_TRUE(std::filesystem::exists(sidecarPath));

    auto json = nlohmann::json::parse(readFile(sidecarPath));
    const auto claims = parseSupportClaimsJson(json);
    EXPECT_TRUE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
}

// ---------------------------------------------------------------------------
// Idempotency: writing identical observations twice yields unchanged second write
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, IdenticalObservationsWriteThenUnchanged)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "Small.json";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto firstSummary = writeObservedSupportClaims(observations);
    EXPECT_EQ(firstSummary.filesWritten, 1u);

    const auto sidecarPath = dir.path() / "Small.support.json";
    const auto firstContent = readFile(sidecarPath);

    const auto secondSummary = writeObservedSupportClaims(observations);
    EXPECT_EQ(secondSummary.filesUnchanged, 1u);
    EXPECT_EQ(secondSummary.filesWritten, 0u);

    EXPECT_EQ(readFile(sidecarPath), firstContent);
}

// ---------------------------------------------------------------------------
// Surgical write: unobserved engine block is semantically preserved
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, UnobservedEngineBlockIsPreserved)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sidecarPath = dir.path() / "Small.support.json";

    nlohmann::json existingJson;
    existingJson["version"] = 1;
    existingJson["claims"]["OTHER_ENGINE"]["gfx90a"] = nlohmann::json::array({"linux"});
    std::ofstream(sidecarPath) << dumpCanonical(existingJson);

    const auto bundlePath = dir.path() / "Small.json";
    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    writeObservedSupportClaims(observations);

    auto json = nlohmann::json::parse(readFile(sidecarPath));
    const auto claims = parseSupportClaimsJson(json);
    EXPECT_TRUE(claims.isClaimed("OTHER_ENGINE", "gfx90a", "linux"));
    EXPECT_TRUE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
}

// ---------------------------------------------------------------------------
// Empty observations: existing sidecar survives untouched
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, EmptyObservationsLeaveExistingSidecarUntouched)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sidecarPath = dir.path() / "Small.support.json";

    nlohmann::json existingJson;
    existingJson["version"] = 1;
    existingJson["claims"]["MIOPEN_ENGINE"]["gfx942"] = nlohmann::json::array({"linux"});
    std::ofstream(sidecarPath) << dumpCanonical(existingJson);

    const std::vector<ObservedGraphSupport> observations;

    const auto summary = writeObservedSupportClaims(observations);
    EXPECT_EQ(summary.filesWritten, 0u);
    EXPECT_EQ(summary.filesUnchanged, 0u);

    auto json = nlohmann::json::parse(readFile(sidecarPath));
    const auto claims = parseSupportClaimsJson(json);
    EXPECT_TRUE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
}

// ---------------------------------------------------------------------------
// Resolved decline erases platform, collapsing empty arch and engine
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, DeclineErasesPlatformAndCollapsesEmptyKeys)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sidecarPath = dir.path() / "Small.support.json";

    nlohmann::json existingJson;
    existingJson["version"] = 1;
    existingJson["claims"]["MIOPEN_ENGINE"]["gfx942"] = nlohmann::json::array({"linux"});
    std::ofstream(sidecarPath) << dumpCanonical(existingJson);

    const auto bundlePath = dir.path() / "Small.json";
    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", false),
    };

    writeObservedSupportClaims(observations);

    auto json = nlohmann::json::parse(readFile(sidecarPath));
    const auto claims = parseSupportClaimsJson(json);
    EXPECT_FALSE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_TRUE(claims.claims.find("MIOPEN_ENGINE") == claims.claims.end());
}

TEST(TestSupportClaimWriter, DeclineErasesOnlyTargetedPlatform)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sidecarPath = dir.path() / "Small.support.json";

    nlohmann::json existingJson;
    existingJson["version"] = 1;
    existingJson["claims"]["MIOPEN_ENGINE"]["gfx942"] = nlohmann::json::array({"linux", "windows"});
    std::ofstream(sidecarPath) << dumpCanonical(existingJson);

    const auto bundlePath = dir.path() / "Small.json";
    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", false),
    };

    writeObservedSupportClaims(observations);

    auto json = nlohmann::json::parse(readFile(sidecarPath));
    const auto claims = parseSupportClaimsJson(json);
    EXPECT_FALSE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_TRUE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "windows"));
}

// ---------------------------------------------------------------------------
// Multiple engines in one sidecar
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, MultipleEngineObservationsInOneSidecar)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "Small.json";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
        singleGraphObservation(bundlePath, "HIP_KERNEL_ENGINE", "gfx942", "linux", true),
        singleGraphObservation(bundlePath, "HIP_KERNEL_ENGINE", "gfx942", "windows", false),
    };

    writeObservedSupportClaims(observations);

    const auto sidecarPath = dir.path() / "Small.support.json";
    auto json = nlohmann::json::parse(readFile(sidecarPath));
    const auto claims = parseSupportClaimsJson(json);
    EXPECT_TRUE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_TRUE(claims.isClaimed("HIP_KERNEL_ENGINE", "gfx942", "linux"));
    EXPECT_FALSE(claims.isClaimed("HIP_KERNEL_ENGINE", "gfx942", "windows"));
}

// ---------------------------------------------------------------------------
// Sweep: basic write
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, SweepWriteCreatesNewSidecar)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sweepPath = dir.path() / "sweep.json";

    const std::vector<ObservedGraphSupport> observations = {
        sweepCaseObservation(sweepPath, "case_a", "MIOPEN_ENGINE", "gfx942", "linux", true),
        sweepCaseObservation(sweepPath, "case_b", "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto summary = writeObservedSupportClaims(observations);
    EXPECT_EQ(summary.filesWritten, 1u);

    const auto sidecarPath = dir.path() / "support.json";
    ASSERT_TRUE(std::filesystem::exists(sidecarPath));

    auto json = nlohmann::json::parse(readFile(sidecarPath));
    const auto claims = parseSweepSupportClaimsJson(json);
    EXPECT_TRUE(claims.isClaimed("case_a", "MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_TRUE(claims.isClaimed("case_b", "MIOPEN_ENGINE", "gfx942", "linux"));
}

// ---------------------------------------------------------------------------
// Sweep: cases with identical support are grouped together
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, SweepGroupsCasesWithIdenticalSupport)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sweepPath = dir.path() / "sweep.json";

    const std::vector<ObservedGraphSupport> observations = {
        sweepCaseObservation(sweepPath, "case_a", "MIOPEN_ENGINE", "gfx942", "linux", true),
        sweepCaseObservation(sweepPath, "case_b", "MIOPEN_ENGINE", "gfx942", "linux", true),
        sweepCaseObservation(sweepPath, "case_c", "MIOPEN_ENGINE", "gfx942", "linux", false),
    };

    writeObservedSupportClaims(observations);

    const auto sidecarPath = dir.path() / "support.json";
    auto json = nlohmann::json::parse(readFile(sidecarPath));

    // case_a and case_b have identical support → one group
    // case_c has no support → no group (empty support maps are dropped)
    const auto& engineGroups = json["claims"]["MIOPEN_ENGINE"];
    ASSERT_EQ(engineGroups.size(), 1u);
    EXPECT_EQ(engineGroups[0]["cases"].size(), 2u);
}

// ---------------------------------------------------------------------------
// Sweep: changed support moves case to different group
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, SweepChangedSupportMovesCaseToCorrectGroup)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sidecarPath = dir.path() / "support.json";

    // Pre-existing: case_a and case_b in one group, both supported
    nlohmann::json existingJson;
    existingJson["version"] = 1;
    nlohmann::json group;
    group["cases"] = nlohmann::json::array({"case_a", "case_b"});
    group["support"]["gfx942"] = nlohmann::json::array({"linux"});
    existingJson["claims"]["MIOPEN_ENGINE"] = nlohmann::json::array({group});
    std::ofstream(sidecarPath) << dumpCanonical(existingJson);

    const auto sweepPath = dir.path() / "sweep.json";
    const std::vector<ObservedGraphSupport> observations = {
        // case_b loses support on gfx942/linux
        sweepCaseObservation(sweepPath, "case_b", "MIOPEN_ENGINE", "gfx942", "linux", false),
    };

    writeObservedSupportClaims(observations);

    auto json = nlohmann::json::parse(readFile(sidecarPath));
    const auto claims = parseSweepSupportClaimsJson(json);
    EXPECT_TRUE(claims.isClaimed("case_a", "MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_FALSE(claims.isClaimed("case_b", "MIOPEN_ENGINE", "gfx942", "linux"));

    // case_a should be alone in its group now
    const auto& engineGroups = json["claims"]["MIOPEN_ENGINE"];
    ASSERT_EQ(engineGroups.size(), 1u);
    EXPECT_EQ(engineGroups[0]["cases"].size(), 1u);
    EXPECT_EQ(engineGroups[0]["cases"][0], "case_a");
}

// ---------------------------------------------------------------------------
// Sweep: idempotency
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, SweepIdenticalObservationsWriteThenUnchanged)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sweepPath = dir.path() / "sweep.json";

    const std::vector<ObservedGraphSupport> observations = {
        sweepCaseObservation(sweepPath, "case_a", "MIOPEN_ENGINE", "gfx942", "linux", true),
        sweepCaseObservation(sweepPath, "case_b", "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto firstSummary = writeObservedSupportClaims(observations);
    EXPECT_EQ(firstSummary.filesWritten, 1u);

    const auto sidecarPath = dir.path() / "support.json";
    const auto firstContent = readFile(sidecarPath);

    const auto secondSummary = writeObservedSupportClaims(observations);
    EXPECT_EQ(secondSummary.filesUnchanged, 1u);
    EXPECT_EQ(secondSummary.filesWritten, 0u);

    EXPECT_EQ(readFile(sidecarPath), firstContent);
}

// ---------------------------------------------------------------------------
// Sweep: unobserved engine is semantically preserved
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, SweepUnobservedEngineBlockIsPreserved)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sidecarPath = dir.path() / "support.json";

    nlohmann::json group;
    group["cases"] = nlohmann::json::array({"case_x"});
    group["support"]["gfx90a"] = nlohmann::json::array({"linux"});

    nlohmann::json existingJson;
    existingJson["version"] = 1;
    existingJson["claims"]["OTHER_ENGINE"] = nlohmann::json::array({group});
    std::ofstream(sidecarPath) << dumpCanonical(existingJson);

    const auto sweepPath = dir.path() / "sweep.json";
    const std::vector<ObservedGraphSupport> observations = {
        sweepCaseObservation(sweepPath, "case_a", "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    writeObservedSupportClaims(observations);

    auto json = nlohmann::json::parse(readFile(sidecarPath));
    const auto claims = parseSweepSupportClaimsJson(json);
    EXPECT_TRUE(claims.isClaimed("case_x", "OTHER_ENGINE", "gfx90a", "linux"));
    EXPECT_TRUE(claims.isClaimed("case_a", "MIOPEN_ENGINE", "gfx942", "linux"));
}

// ---------------------------------------------------------------------------
// Canonical output format: sorted keys, 2-space indent, trailing newline
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, OutputIsCanonicalJson)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "Small.json";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    writeObservedSupportClaims(observations);

    const auto sidecarPath = dir.path() / "Small.support.json";
    const auto content = readFile(sidecarPath);

    // Trailing newline
    EXPECT_FALSE(content.empty());
    EXPECT_EQ(content.back(), '\n');

    // Re-serializing the parsed JSON with the same canonical format yields identical bytes
    auto json = nlohmann::json::parse(content);
    EXPECT_EQ(dumpCanonical(json), content);
}

TEST(TestSupportClaimWriter, UnparseableSingleGraphSidecarReportsErrorAndSurvives)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "Corrupt.json";
    const auto sidecarPath = dir.path() / "Corrupt.support.json";

    std::ofstream(sidecarPath) << "not valid json {{{";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto summary = writeObservedSupportClaims(observations);

    EXPECT_EQ(summary.errors.size(), 1u);
    EXPECT_NE(summary.errors[0].find("unparseable"), std::string::npos);
    EXPECT_EQ(readFile(sidecarPath), "not valid json {{{");
}

TEST(TestSupportClaimWriter, SchemaInvalidSingleGraphSidecarReportsErrorAndSurvives)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "BadSchema.json";
    const auto sidecarPath = dir.path() / "BadSchema.support.json";

    // Valid JSON but unsupported schema version — parseSupportClaimsJson throws
    std::ofstream(sidecarPath) << R"({"version": 999, "claims": {}})";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto summary = writeObservedSupportClaims(observations);

    EXPECT_EQ(summary.errors.size(), 1u);
    EXPECT_NE(summary.errors[0].find("unparseable"), std::string::npos);
    EXPECT_EQ(readFile(sidecarPath), R"({"version": 999, "claims": {}})");
}

TEST(TestSupportClaimWriter, UnparseableSweepSidecarReportsErrorAndSurvives)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sweepDir = dir.path() / "sweep";
    std::filesystem::create_directories(sweepDir);
    const auto sidecarPath = sweepDir / "support.json";

    std::ofstream(sidecarPath) << "corrupt sweep data!!!";

    const std::vector<ObservedGraphSupport> observations = {
        sweepCaseObservation(
            sweepDir / "sweep.json", "case_0", "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto summary = writeObservedSupportClaims(observations);

    EXPECT_EQ(summary.errors.size(), 1u);
    EXPECT_NE(summary.errors[0].find("unparseable"), std::string::npos);
    EXPECT_EQ(readFile(sidecarPath), "corrupt sweep data!!!");
}

// ---------------------------------------------------------------------------
// Malformed observations: observationDefect() refuses and skips the sidecar
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, EmptyArchObservationIsRefusedAndLeavesFileUntouched)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "Small.json";
    const auto sidecarPath = dir.path() / "Small.support.json";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "", "linux", true),
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto summary = writeObservedSupportClaims(observations);

    EXPECT_FALSE(std::filesystem::exists(sidecarPath));
    ASSERT_EQ(summary.errors.size(), 1u);
    EXPECT_NE(summary.errors[0].find("empty arch"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Mixed sweep / single-graph observations for one path are refused
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, MismatchedSweepFlagIsRefusedAndLeavesFileUntouched)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto sidecarPath = dir.path() / "support.json";

    ObservedGraphSupport singleObs;
    singleObs.claimLocator.sidecarPath = sidecarPath;
    singleObs.claimLocator.diagnosticPath = sidecarPath.string();
    singleObs.engineName = "MIOPEN_ENGINE";
    singleObs.arch = "gfx942";
    singleObs.platform = "linux";
    singleObs.engineIsSupported = true;

    ObservedGraphSupport sweepObs;
    sweepObs.claimLocator.sidecarPath = sidecarPath;
    sweepObs.claimLocator.caseId = "case_a";
    sweepObs.claimLocator.diagnosticPath = sidecarPath.string() + "#case_a";
    sweepObs.engineName = "MIOPEN_ENGINE";
    sweepObs.arch = "gfx942";
    sweepObs.platform = "linux";
    sweepObs.engineIsSupported = true;

    const std::vector<ObservedGraphSupport> observations = {singleObs, sweepObs};

    const auto summary = writeObservedSupportClaims(observations);

    EXPECT_FALSE(std::filesystem::exists(sidecarPath));
    ASSERT_EQ(summary.errors.size(), 1u);
    EXPECT_NE(summary.errors[0].find("both single-graph and sweep"), std::string::npos);
}

// ---------------------------------------------------------------------------
// All engines decline: no sidecar should be created
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, AllEnginesDeclinedCreatesNoSidecar)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "Small.json";
    const auto sidecarPath = dir.path() / "Small.support.json";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", false),
        singleGraphObservation(bundlePath, "HIP_KERNEL_ENGINE", "gfx942", "linux", false),
    };

    const auto summary = writeObservedSupportClaims(observations);

    EXPECT_FALSE(std::filesystem::exists(sidecarPath));
    EXPECT_EQ(summary.filesWritten, 0u);
    EXPECT_TRUE(summary.errors.empty());
}

// ---------------------------------------------------------------------------
// observationDefect: remaining branches (empty engine, platform, sidecar path)
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, EmptyEngineNameObservationIsRefusedAndLeavesFileUntouched)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "Small.json";
    const auto sidecarPath = dir.path() / "Small.support.json";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "", "gfx942", "linux", true),
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto summary = writeObservedSupportClaims(observations);

    EXPECT_FALSE(std::filesystem::exists(sidecarPath));
    ASSERT_EQ(summary.errors.size(), 1u);
    EXPECT_NE(summary.errors[0].find("empty engine name"), std::string::npos);
}

TEST(TestSupportClaimWriter, EmptyPlatformObservationIsRefusedAndLeavesFileUntouched)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "Small.json";
    const auto sidecarPath = dir.path() / "Small.support.json";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "", true),
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto summary = writeObservedSupportClaims(observations);

    EXPECT_FALSE(std::filesystem::exists(sidecarPath));
    ASSERT_EQ(summary.errors.size(), 1u);
    EXPECT_NE(summary.errors[0].find("empty platform"), std::string::npos);
}

TEST(TestSupportClaimWriter, EmptySidecarPathObservationIsRefusedAndLeavesFileUntouched)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePath = dir.path() / "Small.json";

    ObservedGraphSupport defective;
    defective.claimLocator.sidecarPath = "";
    defective.claimLocator.diagnosticPath = "<empty>";
    defective.engineName = "MIOPEN_ENGINE";
    defective.arch = "gfx942";
    defective.platform = "linux";
    defective.engineIsSupported = true;

    const std::vector<ObservedGraphSupport> observations = {
        defective,
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    const auto summary = writeObservedSupportClaims(observations);

    ASSERT_EQ(summary.errors.size(), 1u);
    EXPECT_NE(summary.errors[0].find("empty sidecar path"), std::string::npos);
    EXPECT_EQ(summary.filesWritten, 1u);
}

// ---------------------------------------------------------------------------
// Skipped bundle: unobserved sidecar preserved while sibling is written
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, UnobservedBundleSidecarIsPreservedWhenSiblingIsWritten)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto bundlePathA = dir.path() / "A.json";
    const auto sidecarPathA = dir.path() / "A.support.json";
    const auto sidecarPathB = dir.path() / "B.support.json";

    nlohmann::json existingJsonA;
    existingJsonA["version"] = 1;
    existingJsonA["claims"]["OLD_ENGINE"]["gfx90a"] = nlohmann::json::array({"linux"});
    std::ofstream(sidecarPathA, std::ios::binary) << dumpCanonical(existingJsonA);

    nlohmann::json existingJsonB;
    existingJsonB["version"] = 1;
    existingJsonB["claims"]["UNTOUCHED_ENGINE"]["gfx942"] = nlohmann::json::array({"linux"});
    const auto seedB = dumpCanonical(existingJsonB);
    std::ofstream(sidecarPathB, std::ios::binary) << seedB;

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePathA, "NEW_ENGINE", "gfx942", "linux", true),
    };

    const auto summary = writeObservedSupportClaims(observations);

    EXPECT_EQ(summary.filesWritten, 1u);
    EXPECT_TRUE(summary.errors.empty());
    EXPECT_EQ(readFile(sidecarPathB), seedB);
}

// ---------------------------------------------------------------------------
// Write failure: read-only directory triggers OpenFailed
// WriteFailed is not covered — triggering rename failure requires cross-device
// or disk-full conditions that are not reliably testable without filesystem
// mocking.
// ---------------------------------------------------------------------------

TEST(TestSupportClaimWriter, ReadOnlyDirectoryReportsOpenFailedAndSkips)
{
    const ScopedDirectory dir = makeDir("test_writer_");
    const auto subdir = dir.path() / "readonly";
    std::filesystem::create_directories(subdir);

    const auto bundlePath = subdir / "Small.json";

    const std::vector<ObservedGraphSupport> observations = {
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true),
    };

    std::filesystem::permissions(subdir,
                                 std::filesystem::perms::owner_read
                                     | std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace);

    const auto summary = writeObservedSupportClaims(observations);

    std::filesystem::permissions(
        subdir, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);

    ASSERT_EQ(summary.errors.size(), 1u);
    EXPECT_NE(summary.errors[0].find("could not open"), std::string::npos);
    EXPECT_EQ(summary.filesSkipped, 1u);
}

// NOLINTEND(readability-identifier-naming)
