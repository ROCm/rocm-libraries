// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "harness/golden/GoldenBundleDiscovery.hpp"

using namespace hipdnn_integration_tests::golden;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

class GoldenBundleDiscoveryFixture : public ::testing::Test
{
protected:
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        _tempDir = std::filesystem::temp_directory_path()
                   / ("golden_discovery_test_" + std::to_string(::testing::UnitTest::GetInstance()
                                                                    ->current_test_info()
                                                                    ->line()));
        std::filesystem::remove_all(_tempDir);
        std::filesystem::create_directories(_tempDir);
    }

    void TearDown() override
    {
        if(!_tempDir.empty() && std::filesystem::exists(_tempDir))
        {
            std::filesystem::remove_all(_tempDir);
        }
    }

    // Writes a minimal but schema-valid batchnorm-inference graph (nchw, fp32).
    // The bundle directory name becomes the test name; suite is derived from
    // graph content -> "BatchnormInference_nchw_fp32".
    static void createMinimalBundle(const std::filesystem::path& dir, const std::string& name)
    {
        std::filesystem::create_directories(dir);
        std::ofstream ofs(dir / (name + ".json"));
        ofs << R"({"nodes": [{"inputs": {"x_tensor_uid": 0, "mean_tensor_uid": 1, )"
               R"("inv_variance_tensor_uid": 2, "scale_tensor_uid": 3, "bias_tensor_uid": 4}, )"
               R"("outputs": {"y_tensor_uid": 5}, "type": "BatchnormInferenceAttributes", )"
               R"("compute_data_type": "float", "name": ""}], "tensors": [)"
               R"({"name": "", "uid": 0, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
               R"("data_type": "float", "virtual": false}, )"
               R"({"name": "", "uid": 1, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
               R"("data_type": "float", "virtual": false}, )"
               R"({"name": "", "uid": 2, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
               R"("data_type": "float", "virtual": false}, )"
               R"({"name": "", "uid": 3, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
               R"("data_type": "float", "virtual": false}, )"
               R"({"name": "", "uid": 4, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
               R"("data_type": "float", "virtual": false}, )"
               R"({"name": "", "uid": 5, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
               R"("data_type": "float", "virtual": false}], "io_data_type": "float", )"
               R"("compute_data_type": "float", "intermediate_data_type": "float", "name": ""})";
    }

    // Populates one bundle in every tier so the per-tier "must exist and be
    // non-empty" check passes. Each tier gets a distinct bundle name so test
    // names stay unique.
    void populateAllTiers()
    {
        createMinimalBundle(_tempDir / "quick" / "Bn" / "q", "q");
        createMinimalBundle(_tempDir / "standard" / "Bn" / "s", "s");
        createMinimalBundle(_tempDir / "comprehensive" / "Bn" / "c", "c");
        createMinimalBundle(_tempDir / "full" / "Bn" / "f", "f");
    }
};

} // namespace

TEST_F(GoldenBundleDiscoveryFixture, MissingTierThrows)
{
    // Only quick populated; standard/comprehensive/full are absent -> hard fail.
    createMinimalBundle(_tempDir / "quick" / "Bn" / "q", "q");
    EXPECT_THROW(discoverGoldenBundles(_tempDir), std::runtime_error);
}

TEST_F(GoldenBundleDiscoveryFixture, EmptyTierThrows)
{
    // All four tier folders exist but quick has no bundles -> hard fail.
    for(const char* tier : {"quick", "standard", "comprehensive", "full"})
    {
        std::filesystem::create_directories(_tempDir / tier);
    }
    EXPECT_THROW(discoverGoldenBundles(_tempDir), std::runtime_error);
}

TEST_F(GoldenBundleDiscoveryFixture, StrayTopLevelDirThrows)
{
    populateAllTiers();
    // A typo'd tier directory at the top level must be rejected.
    std::filesystem::create_directories(_tempDir / "quik");
    EXPECT_THROW(discoverGoldenBundles(_tempDir), std::runtime_error);
}

TEST_F(GoldenBundleDiscoveryFixture, UnparseableJsonThrows)
{
    populateAllTiers();
    // Corrupt one bundle in the quick tier (processed first).
    auto badDir = _tempDir / "quick" / "Bn" / "bad";
    std::filesystem::create_directories(badDir);
    std::ofstream(badDir / "bad.json") << "{ this is not valid json";
    EXPECT_THROW(discoverGoldenBundles(_tempDir), std::runtime_error);
}

TEST_F(GoldenBundleDiscoveryFixture, CollisionThrows)
{
    populateAllTiers();
    // Two bundles in the same tier with identical graph content + scenario name
    // generate the same test name -> collision.
    createMinimalBundle(_tempDir / "quick" / "OpA" / "SameName", "SameName");
    createMinimalBundle(_tempDir / "quick" / "OpB" / "SameName", "SameName");
    EXPECT_THROW(discoverGoldenBundles(_tempDir), std::runtime_error);
}

TEST_F(GoldenBundleDiscoveryFixture, DiscoversBundlesAcrossAllTiers)
{
    populateAllTiers();

    auto result = discoverGoldenBundles(_tempDir);
    ASSERT_EQ(result.size(), 4u);

    // Locate the quick-tier and standard-tier bundles to check the RFC naming
    // scheme: quick carries no tier prefix; standard is prefixed "Standard/".
    auto findByTest = [&](const std::string& testName) -> const DiscoveredBundle* {
        for(const auto& b : result)
        {
            if(b.testName == testName)
            {
                return &b;
            }
        }
        return nullptr;
    };

    const auto* quickBundle = findByTest("q");
    ASSERT_NE(quickBundle, nullptr);
    EXPECT_EQ(quickBundle->suiteName, "BatchnormInference_nchw_fp32");

    const auto* stdBundle = findByTest("s");
    ASSERT_NE(stdBundle, nullptr);
    EXPECT_EQ(stdBundle->suiteName, "Standard/BatchnormInference_nchw_fp32");
}

TEST_F(GoldenBundleDiscoveryFixture, SkipsMetaJson)
{
    populateAllTiers();
    // A meta.json companion must not be counted as a bundle.
    auto bundleDir = _tempDir / "quick" / "Bn" / "withmeta";
    createMinimalBundle(bundleDir, "withmeta");
    std::ofstream(bundleDir / "withmeta.meta.json") << "{}";

    auto result = discoverGoldenBundles(_tempDir);
    // 4 from populateAllTiers + 1 "withmeta"; the meta.json adds nothing.
    EXPECT_EQ(result.size(), 5u);
}

TEST(SanitizeForGtest, ReplacesInvalidChars)
{
    EXPECT_EQ(sanitizeForGtest("hello world!"), "hello_world_");
    EXPECT_EQ(sanitizeForGtest("Conv-Fprop.v2"), "Conv_Fprop_v2");
    EXPECT_EQ(sanitizeForGtest("already_valid_123"), "already_valid_123");
}

TEST(TierPrefix, MatchesRfcScheme)
{
    EXPECT_EQ(tierPrefix("quick"), "");
    EXPECT_EQ(tierPrefix("standard"), "Standard/");
    EXPECT_EQ(tierPrefix("comprehensive"), "Comprehensive/");
    EXPECT_EQ(tierPrefix("full"), "Full/");
}

// NOLINTEND(readability-identifier-naming)
