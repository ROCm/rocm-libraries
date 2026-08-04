// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for RFC 0015 §9's harness-side write hook:
// IntegrationBundleVerificationHarness::recordSupportObservationForWrite().
// Drives the method directly (it is protected, non-virtual, and calls only
// virtual primitives that this file's TestableWriteHarness stubs), so these
// tests never touch a real handle, plugin, or the TestConfig singleton.

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/SupportClaimWriter.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests::bundle;

namespace
{

class ScopedTempDir
{
public:
    ScopedTempDir()
    {
        static std::atomic<int> counter{0};
        _path = std::filesystem::temp_directory_path()
                / ("hipdnn_write_harness_test_" + std::to_string(counter.fetch_add(1)));
        std::filesystem::create_directories(_path);
    }

    ~ScopedTempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(_path, ec);
    }

    ScopedTempDir(const ScopedTempDir&) = delete;
    ScopedTempDir& operator=(const ScopedTempDir&) = delete;

    const std::filesystem::path& path() const
    {
        return _path;
    }

private:
    std::filesystem::path _path;
};

class TestableWriteHarness : public IntegrationBundleVerificationHarness
{
public:
    TestableWriteHarness()
        : IntegrationBundleVerificationHarness(/*requiresDevice=*/false)
    {
    }

    using IntegrationBundleVerificationHarness::recordSupportObservationForWrite;

    bool writeModeRequested = true;
    std::optional<int64_t> pinnedEngineId;
    std::string pinnedEngineName = "MIOPEN_ENGINE";
    std::string arch = "gfx942";
    std::string platform = "linux";

protected:
    bool writeSupportClaimsRequested() const override
    {
        return writeModeRequested;
    }

    std::optional<int64_t> resolveActivePreferredEngineId() const override
    {
        return pinnedEngineId;
    }

    std::string resolveActivePreferredEngineName() const override
    {
        return pinnedEngineName;
    }

    std::string currentArchToken() const override
    {
        return arch;
    }

    std::string currentPlatform() const override
    {
        return platform;
    }
};

std::shared_ptr<IntegrationTestBundle>
    makeBundleWithWriteTarget(std::optional<ClaimWriteTarget> target)
{
    auto bundle = std::make_shared<IntegrationTestBundle>();
    bundle->claimWriteTarget = std::move(target);
    return bundle;
}

} // namespace

class TestRecordSupportObservationForWrite : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ClaimObservationCollector::get().reset();
    }

    void TearDown() override
    {
        ClaimObservationCollector::get().reset();
    }
};

TEST_F(TestRecordSupportObservationForWrite, WriteModeOffRecordsNothing)
{
    TestableWriteHarness harness;
    harness.writeModeRequested = false;
    harness.pinnedEngineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
    harness.setBundle(makeBundleWithWriteTarget(ClaimWriteTarget{"/tmp/Foo.json", std::nullopt}),
                      "unit-test-bundle");

    harness.recordSupportObservationForWrite(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::OK, ""), {*harness.pinnedEngineId});

    EXPECT_TRUE(ClaimObservationCollector::get().empty());
}

TEST_F(TestRecordSupportObservationForWrite, NoWriteTargetRecordsNothing)
{
    TestableWriteHarness harness;
    harness.pinnedEngineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
    harness.setBundle(makeBundleWithWriteTarget(std::nullopt), "unit-test-bundle");

    harness.recordSupportObservationForWrite(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::OK, ""), {*harness.pinnedEngineId});

    EXPECT_TRUE(ClaimObservationCollector::get().empty());
}

TEST_F(TestRecordSupportObservationForWrite, NoPinnedEngineRecordsNothingModeA)
{
    TestableWriteHarness harness;
    harness.pinnedEngineId = std::nullopt; // mode A: nothing pinned this pass
    harness.setBundle(makeBundleWithWriteTarget(ClaimWriteTarget{"/tmp/Foo.json", std::nullopt}),
                      "unit-test-bundle");

    harness.recordSupportObservationForWrite(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::OK, ""), {42});

    EXPECT_TRUE(ClaimObservationCollector::get().empty());
}

TEST_F(TestRecordSupportObservationForWrite, UnresolvedQueryRecordsNothing)
{
    TestableWriteHarness harness;
    harness.pinnedEngineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
    harness.setBundle(makeBundleWithWriteTarget(ClaimWriteTarget{"/tmp/Foo.json", std::nullopt}),
                      "unit-test-bundle");

    harness.recordSupportObservationForWrite(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR, "device lost"),
        {});

    EXPECT_TRUE(ClaimObservationCollector::get().empty());
}

TEST_F(TestRecordSupportObservationForWrite, SupportedVerdictWritesSupportedClaim)
{
    ScopedTempDir dir;
    const auto graphJsonPath = dir.path() / "Foo.json";
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");

    TestableWriteHarness harness;
    harness.pinnedEngineId = engineId;
    harness.setBundle(makeBundleWithWriteTarget(ClaimWriteTarget{graphJsonPath, std::nullopt}),
                      "unit-test-bundle");

    harness.recordSupportObservationForWrite(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::OK, ""), {engineId});

    ASSERT_FALSE(ClaimObservationCollector::get().empty());
    ClaimObservationCollector::get().writeAll();

    auto loaded = loadSupportClaims(graphJsonPath);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
}

TEST_F(TestRecordSupportObservationForWrite, DeclinedVerdictRemovesExistingClaim)
{
    ScopedTempDir dir;
    const auto graphJsonPath = dir.path() / "Foo.json";
    const auto sidecarPath = dir.path() / "Foo.support.json";
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");

    SupportClaims preexisting;
    preexisting.version = 1;
    preexisting.claims["MIOPEN_ENGINE"]["gfx942"] = {"linux"};
    writeSupportClaimsFile(sidecarPath, preexisting);

    TestableWriteHarness harness;
    harness.pinnedEngineId = engineId;
    harness.setBundle(makeBundleWithWriteTarget(ClaimWriteTarget{graphJsonPath, std::nullopt}),
                      "unit-test-bundle");

    // Resolved query (GRAPH_NOT_SUPPORTED), engine id absent from the ranked
    // list -- a real coverage-loss observation.
    harness.recordSupportObservationForWrite(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED, ""), {});

    ClaimObservationCollector::get().writeAll();

    auto loaded = loadSupportClaims(graphJsonPath);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_FALSE(loaded->isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
}

// NOLINTEND(readability-identifier-naming)
