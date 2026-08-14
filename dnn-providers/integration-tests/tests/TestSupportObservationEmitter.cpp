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
#include <fstream>
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
using hipdnn_integration_tests::bundle::test_utils::singleGraphObservation;
using hipdnn_integration_tests::bundle::test_utils::sweepCaseObservation;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

ObservationProvenance testProvenance()
{
    return {"6.4.0", "abc123", "ci-12345", "2026-08-13T12:00:00Z"};
}

std::vector<std::string> readJsonlLines(const std::filesystem::path& path)
{
    std::vector<std::string> lines;
    std::ifstream file(path);
    for(std::string line; std::getline(file, line);)
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

TEST(TestSupportObservationEmitter, SingleGraphRecordShape)
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
    EXPECT_TRUE(record.at("case_id").is_null());
    EXPECT_EQ(record.at("engine"), "MIOPEN_ENGINE");
    EXPECT_EQ(record.at("arch"), "gfx942");
    EXPECT_EQ(record.at("platform"), "linux");
    EXPECT_EQ(record.at("verdict"), "supported");
    EXPECT_EQ(record.at("enforcement_level"), "full");
}

TEST(TestSupportObservationEmitter, SweepRecordCarriesCaseId)
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

    const auto summary = emitSupportObservations(observations, dir.path() / "obs.jsonl", root, {});

    ASSERT_EQ(summary.recordsEmitted, 1u);
    EXPECT_TRUE(summary.errors.empty());

    const auto record = nlohmann::json::parse(readJsonlLines(dir.path() / "obs.jsonl").at(0));
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

    EXPECT_EQ(record.at("bundle"), "/elsewhere/Odd");
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

TEST(TestSupportObservationEmitter, AppendsRatherThanTruncatingAcrossCalls)
{
    const ScopedDirectory dir = makeScopedTestDir("test_emitter");
    const std::filesystem::path root = "/bundles";
    const auto outputPath = dir.path() / "obs.jsonl";

    emitSupportObservations(
        {singleGraphObservation(
            root / "a" / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", true)},
        outputPath,
        root,
        {});
    emitSupportObservations(
        {singleGraphObservation(
            root / "b" / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", true)},
        outputPath,
        root,
        {});

    const auto lines = readJsonlLines(outputPath);
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(nlohmann::json::parse(lines[0]).at("bundle"), "a");
    EXPECT_EQ(nlohmann::json::parse(lines[1]).at("bundle"), "b");
}

TEST(TestSupportObservationEmitter, CreatesMissingParentDirectories)
{
    const ScopedDirectory dir = makeScopedTestDir("test_emitter");
    const std::filesystem::path root = "/bundles";
    const auto outputPath = dir.path() / "nested" / "deeper" / "obs.jsonl";

    const auto summary = emitSupportObservations(
        {singleGraphObservation(
            root / "a" / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", true)},
        outputPath,
        root,
        {});

    EXPECT_TRUE(summary.errors.empty());
    EXPECT_EQ(summary.recordsEmitted, 1u);
    EXPECT_TRUE(std::filesystem::exists(outputPath));
}

TEST(TestSupportObservationEmitter, NoObservationsWritesNothingAndReportsNoError)
{
    const ScopedDirectory dir = makeScopedTestDir("test_emitter");
    const auto outputPath = dir.path() / "obs.jsonl";

    const auto summary = emitSupportObservations({}, outputPath, "/bundles", {});

    EXPECT_EQ(summary.recordsEmitted, 0u);
    EXPECT_TRUE(summary.errors.empty());
    EXPECT_FALSE(std::filesystem::exists(outputPath));
}

TEST(TestSupportObservationEmitter, UnwritablePathIsReportedAsAnError)
{
    const ScopedDirectory dir = makeScopedTestDir("test_emitter");
    const std::filesystem::path root = "/bundles";

    const auto outputPath = dir.path() / "obs.jsonl";
    std::filesystem::create_directories(outputPath);

    const auto summary = emitSupportObservations(
        {singleGraphObservation(
            root / "a" / "Small.json", "MIOPEN_ENGINE", "gfx942", "linux", true)},
        outputPath,
        root,
        {});

    EXPECT_EQ(summary.recordsEmitted, 0u);
    EXPECT_FALSE(summary.errors.empty());
}

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
