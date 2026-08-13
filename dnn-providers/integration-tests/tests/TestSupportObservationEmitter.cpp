// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The emitter's whole job is to produce lines that
// scripts/harvest_support_observations.py can index, so most of what is
// asserted here is the record's *shape*: the exact keys, and a `bundle` value
// that is a POSIX-spelled directory relative to the bundle root. A record the
// consumer cannot place is silently dropped on that side, which would look like
// missing coverage rather than a bug here.

#include <gtest/gtest.h>

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness/bundle/SupportObservationEmitter.hpp"

#include "SupportClaimTestUtils.hpp"

using hipdnn_integration_tests::bundle::emitSupportObservations;
using hipdnn_integration_tests::bundle::ObservationProvenance;
using hipdnn_integration_tests::bundle::ObservedSupport;
using hipdnn_integration_tests::bundle::SupportObservation;
using hipdnn_integration_tests::bundle::toObservationRecord;
using hipdnn_integration_tests::bundle::test_utils::makeScopedTestDir;
using hipdnn_integration_tests::bundle::test_utils::readFile;
using hipdnn_integration_tests::bundle::test_utils::singleGraphObservation;
using hipdnn_integration_tests::bundle::test_utils::sweepCaseObservation;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

/// A provenance block with every field distinguishable, so a test can tell a
/// passed-through value from a defaulted one.
ObservationProvenance testProvenance()
{
    return {"6.4.0", "abc123", "ci-12345", "2026-08-13T12:00:00Z"};
}

/// Splits emitted text into lines, dropping the trailing empty one.
std::vector<std::string> splitLines(const std::string& text)
{
    std::vector<std::string> lines;
    std::istringstream stream(text);
    for(std::string line; std::getline(stream, line);)
    {
        if(!line.empty())
        {
            lines.push_back(line);
        }
    }
    return lines;
}

} // namespace

// ---------------------------------------------------------------------------
// Record shape
// ---------------------------------------------------------------------------

TEST(TestSupportObservationEmitter, SingleGraphRecordCarriesGraphAndNullCaseId)
{
    const std::filesystem::path root = "/bundles";
    const auto observation
        = singleGraphObservation(root / "quick" / "Batchnorm" / "Default" / "Small.json",
                                 "MIOPEN_ENGINE",
                                 "gfx942",
                                 "linux",
                                 ObservedSupport::SUPPORTED);

    const auto record = toObservationRecord(observation, root, testProvenance());

    EXPECT_EQ(record.at("bundle"), "quick/Batchnorm/Default");
    EXPECT_EQ(record.at("graph"), "Small");
    EXPECT_TRUE(record.at("case_id").is_null());
    EXPECT_EQ(record.at("engine"), "MIOPEN_ENGINE");
    EXPECT_EQ(record.at("arch"), "gfx942");
    EXPECT_EQ(record.at("platform"), "linux");
    EXPECT_EQ(record.at("verdict"), "supported");
    EXPECT_EQ(record.at("enforcement_level"), "full");
}

TEST(TestSupportObservationEmitter, SweepRecordCarriesCaseIdAndNoGraph)
{
    const std::filesystem::path root = "/bundles";
    const auto observation = sweepCaseObservation(root / "standard" / "Conv" / "sweep.json",
                                                  "case_7",
                                                  "MIOPEN_ENGINE",
                                                  "gfx950",
                                                  "linux",
                                                  ObservedSupport::DECLINED);

    const auto record = toObservationRecord(observation, root, testProvenance());

    EXPECT_EQ(record.at("bundle"), "standard/Conv");
    EXPECT_EQ(record.at("case_id"), "case_7");
    EXPECT_EQ(record.at("verdict"), "declined");
    // A sweep's sidecar is a bare support.json shared by every case, so there is
    // no per-record graph name to write and stem-stripping it would yield the
    // meaningless "support".
    EXPECT_FALSE(record.contains("graph"));
}

TEST(TestSupportObservationEmitter, ProvenanceIsPassedThroughVerbatim)
{
    const std::filesystem::path root = "/bundles";
    const auto observation = singleGraphObservation(root / "quick" / "Small.json",
                                                    "MIOPEN_ENGINE",
                                                    "gfx942",
                                                    "linux",
                                                    ObservedSupport::SUPPORTED);

    const auto record = toObservationRecord(observation, root, testProvenance());
    const auto& provenance = record.at("provenance");

    EXPECT_EQ(provenance.at("rocm_version"), "6.4.0");
    EXPECT_EQ(provenance.at("commit"), "abc123");
    EXPECT_EQ(provenance.at("run_id"), "ci-12345");
    EXPECT_EQ(provenance.at("timestamp"), "2026-08-13T12:00:00Z");
}

TEST(TestSupportObservationEmitter, EmptyProvenanceStillProducesTheKeys)
{
    const std::filesystem::path root = "/bundles";
    const auto observation = singleGraphObservation(root / "quick" / "Small.json",
                                                    "MIOPEN_ENGINE",
                                                    "gfx942",
                                                    "linux",
                                                    ObservedSupport::SUPPORTED);

    // A CI job that forgot to export ROCM_VERSION loses traceability on its
    // records, not the records themselves.
    const auto record = toObservationRecord(observation, root, {});

    EXPECT_EQ(record.at("provenance").at("rocm_version"), "");
    EXPECT_EQ(record.at("provenance").at("timestamp"), "");
    EXPECT_EQ(record.at("verdict"), "supported");
}

// ---------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------

TEST(TestSupportObservationEmitter, UnknownIsEmittedRatherThanDropped)
{
    const ScopedDirectory dir = makeScopedTestDir("test_emitter");
    const std::filesystem::path root = "/bundles";

    const std::vector<SupportObservation> observations = {
        singleGraphObservation(root / "quick" / "Small.json",
                               "MIOPEN_ENGINE",
                               "gfx942",
                               "linux",
                               ObservedSupport::UNKNOWN),
    };

    std::ostringstream mirror;
    const auto summary
        = emitSupportObservations(observations, dir.path() / "obs.jsonl", root, {}, mirror);

    // "We asked and the query broke" and "no shard ever reached this cell" are
    // different facts, and only the first one can leave a line.
    ASSERT_EQ(summary.recordsEmitted, 1u);
    EXPECT_TRUE(summary.errors.empty());

    const auto record = nlohmann::json::parse(splitLines(readFile(dir.path() / "obs.jsonl")).at(0));
    EXPECT_EQ(record.at("verdict"), "unknown");
}

TEST(TestSupportObservationEmitter, EnforcementLevelIsSpelledAsOnDisk)
{
    const std::filesystem::path root = "/bundles";
    auto observation = singleGraphObservation(root / "quick" / "Small.json",
                                              "MIOPEN_ENGINE",
                                              "gfx942",
                                              "linux",
                                              ObservedSupport::SUPPORTED);
    observation.enforcementLevel = hipdnn_integration_tests::EnforcementLevel::APPLICABILITY;

    const auto record = toObservationRecord(observation, root, {});

    EXPECT_EQ(record.at("enforcement_level"), "applicability");
}

// ---------------------------------------------------------------------------
// Bundle key derivation
// ---------------------------------------------------------------------------

TEST(TestSupportObservationEmitter, BundleKeyIsPosixEvenForNestedPaths)
{
    const std::filesystem::path root = "/bundles";
    const auto observation
        = singleGraphObservation(root / "full" / "Sdpa" / "Fwd" / "Causal" / "Big.json",
                                 "MIOPEN_ENGINE",
                                 "gfx942",
                                 "linux",
                                 ObservedSupport::SUPPORTED);

    const auto record = toObservationRecord(observation, root, {});

    EXPECT_EQ(record.at("bundle"), "full/Sdpa/Fwd/Causal");
}

TEST(TestSupportObservationEmitter, BundleAtRootGetsDotAsItsKey)
{
    const std::filesystem::path root = "/bundles";
    const auto observation = singleGraphObservation(
        root / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", ObservedSupport::SUPPORTED);

    const auto record = toObservationRecord(observation, root, {});

    // What relative() spells for "the root itself"; the consumer resolves it the
    // same way.
    EXPECT_EQ(record.at("bundle"), ".");
}

TEST(TestSupportObservationEmitter, PathOutsideRootFallsBackToTheAbsolutePath)
{
    const auto observation = singleGraphObservation("/elsewhere/Odd/Small.json",
                                                    "MIOPEN_ENGINE",
                                                    "gfx942",
                                                    "linux",
                                                    ObservedSupport::SUPPORTED);

    const auto record = toObservationRecord(observation, "/bundles", {});

    // relative() would answer "../elsewhere/Odd" here, which the consumer would
    // happily index under a bundle that does not exist. An absolute path is an
    // orphan it warns about instead.
    EXPECT_EQ(record.at("bundle"), "/elsewhere/Odd");
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

TEST(TestSupportObservationEmitter, MirrorsEveryLineBehindTheStdoutTag)
{
    const ScopedDirectory dir = makeScopedTestDir("test_emitter");
    const std::filesystem::path root = "/bundles";

    const std::vector<SupportObservation> observations = {
        singleGraphObservation(root / "a" / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", true),
        singleGraphObservation(
            root / "b" / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", false),
    };

    std::ostringstream mirror;
    const auto summary
        = emitSupportObservations(observations, dir.path() / "obs.jsonl", root, {}, mirror);
    ASSERT_EQ(summary.recordsEmitted, 2u);

    const auto mirrored = splitLines(mirror.str());
    const auto written = splitLines(readFile(dir.path() / "obs.jsonl"));
    ASSERT_EQ(mirrored.size(), 2u);
    ASSERT_EQ(written.size(), 2u);

    constexpr const char* k_tag = "[[HIPDNN_SUPPORT_OBS]] ";
    for(std::size_t i = 0; i < mirrored.size(); ++i)
    {
        ASSERT_EQ(mirrored[i].rfind(k_tag, 0), 0u) << mirrored[i];
        // The tagged console line and the artifact line must be the same record,
        // or a job that harvests from logs disagrees with one that harvests from
        // the file.
        EXPECT_EQ(mirrored[i].substr(std::string(k_tag).size()), written[i]);
    }
}

TEST(TestSupportObservationEmitter, AppendsRatherThanTruncatingAcrossCalls)
{
    const ScopedDirectory dir = makeScopedTestDir("test_emitter");
    const std::filesystem::path root = "/bundles";
    const auto outputPath = dir.path() / "obs.jsonl";

    std::ostringstream mirror;
    emitSupportObservations(
        {singleGraphObservation(
            root / "a" / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", true)},
        outputPath,
        root,
        {},
        mirror);
    emitSupportObservations(
        {singleGraphObservation(
            root / "b" / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", true)},
        outputPath,
        root,
        {},
        mirror);

    // Sharded binaries share one path and the consumer's merge is a union, so a
    // truncating second writer would silently lose a whole shard.
    const auto lines = splitLines(readFile(outputPath));
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(nlohmann::json::parse(lines[0]).at("bundle"), "a");
    EXPECT_EQ(nlohmann::json::parse(lines[1]).at("bundle"), "b");
}

TEST(TestSupportObservationEmitter, CreatesMissingParentDirectories)
{
    const ScopedDirectory dir = makeScopedTestDir("test_emitter");
    const std::filesystem::path root = "/bundles";
    const auto outputPath = dir.path() / "nested" / "deeper" / "obs.jsonl";

    std::ostringstream mirror;
    const auto summary = emitSupportObservations(
        {singleGraphObservation(
            root / "a" / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", true)},
        outputPath,
        root,
        {},
        mirror);

    EXPECT_TRUE(summary.errors.empty());
    EXPECT_EQ(summary.recordsEmitted, 1u);
    EXPECT_TRUE(std::filesystem::exists(outputPath));
}

TEST(TestSupportObservationEmitter, NoObservationsWritesNothingAndReportsNoError)
{
    const ScopedDirectory dir = makeScopedTestDir("test_emitter");
    const auto outputPath = dir.path() / "obs.jsonl";

    std::ostringstream mirror;
    const auto summary = emitSupportObservations({}, outputPath, "/bundles", {}, mirror);

    // An empty run is a coverage problem for the caller to warn about, not an
    // error here — and it must not be able to fail the build.
    EXPECT_EQ(summary.recordsEmitted, 0u);
    EXPECT_TRUE(summary.errors.empty());
    EXPECT_TRUE(mirror.str().empty());
}

TEST(TestSupportObservationEmitter, UnwritablePathIsReportedAsAnError)
{
    const ScopedDirectory dir = makeScopedTestDir("test_emitter");
    const std::filesystem::path root = "/bundles";

    // A directory where the file should be: open-for-append fails, and the
    // caller learns about it through the summary rather than an exception.
    const auto outputPath = dir.path() / "obs.jsonl";
    std::filesystem::create_directories(outputPath);

    std::ostringstream mirror;
    const auto summary = emitSupportObservations(
        {singleGraphObservation(
            root / "a" / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", true)},
        outputPath,
        root,
        {},
        mirror);

    EXPECT_EQ(summary.recordsEmitted, 0u);
    EXPECT_FALSE(summary.errors.empty());
}

// ---------------------------------------------------------------------------
// Timestamp
// ---------------------------------------------------------------------------

TEST(TestSupportObservationEmitter, TimestampIsRfc3339Utc)
{
    const auto timestamp = hipdnn_integration_tests::bundle::currentUtcTimestamp();

    ASSERT_EQ(timestamp.size(), 20u) << timestamp;
    EXPECT_EQ(timestamp[4], '-');
    EXPECT_EQ(timestamp[7], '-');
    EXPECT_EQ(timestamp[10], 'T');
    EXPECT_EQ(timestamp[13], ':');
    EXPECT_EQ(timestamp[16], ':');
    EXPECT_EQ(timestamp[19], 'Z');
}

// NOLINTEND(readability-identifier-naming)
