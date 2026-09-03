// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

#include "harness/bundle/SupportObservationLog.hpp"

#include "SupportClaimTestUtils.hpp"

using hipdnn_integration_tests::bundle::ObservedSupport;
using hipdnn_integration_tests::bundle::SupportObservationLog;
using hipdnn_integration_tests::bundle::test_utils::makeScopedTestDir;
using hipdnn_integration_tests::bundle::test_utils::singleGraphObservation;
using hipdnn_integration_tests::bundle::test_utils::sweepCaseObservation;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

class TestSupportObservationLog : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SupportObservationLog::get().reset();
    }
    void TearDown() override
    {
        SupportObservationLog::get().reset();
    }
};

// ---------------------------------------------------------------------------
// toSnapshotJsons: empty log
// ---------------------------------------------------------------------------

TEST_F(TestSupportObservationLog, EmptyLogProducesNoSnapshots)
{
    const ScopedDirectory dir = makeScopedTestDir("test_obs_log");
    const auto snapshots = SupportObservationLog::get().toSnapshotJsons(dir.path());
    EXPECT_TRUE(snapshots.empty());
}

// ---------------------------------------------------------------------------
// toSnapshotJsons: single target
// ---------------------------------------------------------------------------

TEST_F(TestSupportObservationLog, SingleTargetProducesOneSnapshot)
{
    const ScopedDirectory dir = makeScopedTestDir("test_obs_log");
    const auto bundlePath = dir.path() / "quick" / "A" / "Small.json";

    SupportObservationLog::get().record(
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true));
    SupportObservationLog::get().record(
        singleGraphObservation(bundlePath, "CK_ENGINE", "gfx942", "linux", false));

    const auto snapshots = SupportObservationLog::get().toSnapshotJsons(dir.path());
    ASSERT_EQ(snapshots.size(), 1u);

    const auto& snap = snapshots[0];
    EXPECT_EQ(snap["schema_version"], 1);
    EXPECT_EQ(snap["target"]["arch"], "gfx942");
    EXPECT_EQ(snap["target"]["platform"], "linux");
    EXPECT_EQ(snap["observations"].size(), 2u);
}

// ---------------------------------------------------------------------------
// toSnapshotJsons: multi-GPU grouping
// ---------------------------------------------------------------------------

TEST_F(TestSupportObservationLog, MultipleTargetsProduceMultipleSnapshots)
{
    const ScopedDirectory dir = makeScopedTestDir("test_obs_log");
    const auto bundlePath = dir.path() / "quick" / "A" / "Small.json";

    SupportObservationLog::get().record(
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true));
    SupportObservationLog::get().record(
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx90a", "linux", true));
    SupportObservationLog::get().record(
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx1151", "windows", false));

    const auto snapshots = SupportObservationLog::get().toSnapshotJsons(dir.path());
    ASSERT_EQ(snapshots.size(), 3u);

    std::set<std::string> arches;
    for(const auto& snap : snapshots)
    {
        EXPECT_EQ(snap["schema_version"], 1);
        EXPECT_EQ(snap["observations"].size(), 1u);
        arches.insert(snap["target"]["arch"].get<std::string>());
    }
    EXPECT_TRUE(arches.count("gfx942"));
    EXPECT_TRUE(arches.count("gfx90a"));
    EXPECT_TRUE(arches.count("gfx1151"));
}

// ---------------------------------------------------------------------------
// toSnapshotJsons: observations land in the correct target bucket
// ---------------------------------------------------------------------------

TEST_F(TestSupportObservationLog, ObservationsGroupedByTarget)
{
    const ScopedDirectory dir = makeScopedTestDir("test_obs_log");
    const auto bundleA = dir.path() / "quick" / "A" / "Small.json";
    const auto bundleB = dir.path() / "quick" / "B" / "Small.json";

    SupportObservationLog::get().record(
        singleGraphObservation(bundleA, "MIOPEN_ENGINE", "gfx942", "linux", true));
    SupportObservationLog::get().record(
        singleGraphObservation(bundleB, "MIOPEN_ENGINE", "gfx942", "linux", false));
    SupportObservationLog::get().record(
        singleGraphObservation(bundleA, "CK_ENGINE", "gfx90a", "linux", true));

    const auto snapshots = SupportObservationLog::get().toSnapshotJsons(dir.path());
    ASSERT_EQ(snapshots.size(), 2u);

    for(const auto& snap : snapshots)
    {
        const auto arch = snap["target"]["arch"].get<std::string>();
        if(arch == "gfx942")
        {
            EXPECT_EQ(snap["observations"].size(), 2u);
        }
        else if(arch == "gfx90a")
        {
            EXPECT_EQ(snap["observations"].size(), 1u);
            EXPECT_EQ(snap["observations"][0]["engine"], "CK_ENGINE");
        }
        else
        {
            FAIL() << "unexpected arch: " << arch;
        }
    }
}

// ---------------------------------------------------------------------------
// toSnapshotJsons: bundle path relativisation
// ---------------------------------------------------------------------------

TEST_F(TestSupportObservationLog, BundlePathIsRelativeToBundleRoot)
{
    const ScopedDirectory dir = makeScopedTestDir("test_obs_log");
    const auto bundlePath = dir.path() / "quick" / "Conv" / "Default" / "Small.json";

    SupportObservationLog::get().record(
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true));

    const auto snapshots = SupportObservationLog::get().toSnapshotJsons(dir.path());
    ASSERT_EQ(snapshots.size(), 1u);
    EXPECT_EQ(snapshots[0]["observations"][0]["bundle"], "quick/Conv/Default");
}

// ---------------------------------------------------------------------------
// toSnapshotJsons: verdict fields
// ---------------------------------------------------------------------------

TEST_F(TestSupportObservationLog, VerdictStringsAreCorrect)
{
    const ScopedDirectory dir = makeScopedTestDir("test_obs_log");
    const auto bundlePath = dir.path() / "quick" / "A" / "Small.json";

    SupportObservationLog::get().record(
        singleGraphObservation(bundlePath, "E1", "gfx942", "linux", ObservedSupport::SUPPORTED));
    SupportObservationLog::get().record(
        singleGraphObservation(bundlePath, "E2", "gfx942", "linux", ObservedSupport::DECLINED));

    const auto snapshots = SupportObservationLog::get().toSnapshotJsons(dir.path());
    ASSERT_EQ(snapshots.size(), 1u);

    std::set<std::string> verdicts;
    for(const auto& obs : snapshots[0]["observations"])
    {
        verdicts.insert(obs["verdict"].get<std::string>());
    }
    EXPECT_TRUE(verdicts.count("supported"));
    EXPECT_TRUE(verdicts.count("declined"));
}

// ---------------------------------------------------------------------------
// Upsert: higher verdict wins
// ---------------------------------------------------------------------------

TEST_F(TestSupportObservationLog, UpsertKeepsHigherVerdict)
{
    const ScopedDirectory dir = makeScopedTestDir("test_obs_log");
    const auto bundlePath = dir.path() / "quick" / "A" / "Small.json";

    SupportObservationLog::get().record(singleGraphObservation(
        bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", ObservedSupport::DECLINED));
    SupportObservationLog::get().record(singleGraphObservation(
        bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", ObservedSupport::SUPPORTED));

    const auto snapshots = SupportObservationLog::get().toSnapshotJsons(dir.path());
    ASSERT_EQ(snapshots.size(), 1u);
    ASSERT_EQ(snapshots[0]["observations"].size(), 1u);
    EXPECT_EQ(snapshots[0]["observations"][0]["verdict"], "supported");
}

TEST_F(TestSupportObservationLog, UpsertDoesNotDowngrade)
{
    const ScopedDirectory dir = makeScopedTestDir("test_obs_log");
    const auto bundlePath = dir.path() / "quick" / "A" / "Small.json";

    SupportObservationLog::get().record(singleGraphObservation(
        bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", ObservedSupport::SUPPORTED));
    SupportObservationLog::get().record(singleGraphObservation(
        bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", ObservedSupport::DECLINED));

    const auto snapshots = SupportObservationLog::get().toSnapshotJsons(dir.path());
    ASSERT_EQ(snapshots.size(), 1u);
    ASSERT_EQ(snapshots[0]["observations"].size(), 1u);
    EXPECT_EQ(snapshots[0]["observations"][0]["verdict"], "supported");
}

// ---------------------------------------------------------------------------
// toSnapshotJsons: sweep case_id
// ---------------------------------------------------------------------------

TEST_F(TestSupportObservationLog, SweepCaseIdIsCarried)
{
    const ScopedDirectory dir = makeScopedTestDir("test_obs_log");
    const auto sweepPath = dir.path() / "full" / "S" / "graph.template.json";

    SupportObservationLog::get().record(sweepCaseObservation(
        sweepPath, "case_0", "MIOPEN_ENGINE", "gfx942", "linux", ObservedSupport::SUPPORTED));

    const auto snapshots = SupportObservationLog::get().toSnapshotJsons(dir.path());
    ASSERT_EQ(snapshots.size(), 1u);
    EXPECT_EQ(snapshots[0]["observations"][0]["case_id"], "case_0");
}

TEST_F(TestSupportObservationLog, SingleGraphCaseIdIsNull)
{
    const ScopedDirectory dir = makeScopedTestDir("test_obs_log");
    const auto bundlePath = dir.path() / "quick" / "A" / "Small.json";

    SupportObservationLog::get().record(
        singleGraphObservation(bundlePath, "MIOPEN_ENGINE", "gfx942", "linux", true));

    const auto snapshots = SupportObservationLog::get().toSnapshotJsons(dir.path());
    ASSERT_EQ(snapshots.size(), 1u);
    EXPECT_TRUE(snapshots[0]["observations"][0]["case_id"].is_null());
}

// NOLINTEND(readability-identifier-naming)
