// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>

#include "harness/golden/GoldenBundleDiscovery.hpp"

using namespace hipdnn_integration_tests::golden;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

class TestGoldenBundleDiscoveryFixture : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        auto path
            = std::filesystem::temp_directory_path()
              / ("golden_discovery_test_"
                 + std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
        std::filesystem::remove_all(path);
        _scopedDir.emplace(path);
        _tempDir = _scopedDir->path();
    }

    // Writes a minimal but schema-valid batchnorm-inference graph (nchw, fp32).
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
    // names stay unique. Path convention: {tier}/{op}/{layout}/{dtype}/{bundle}/
    void populateAllTiers()
    {
        createMinimalBundle(_tempDir / "quick" / "BatchnormInference" / "nchw" / "fp32" / "q", "q");
        createMinimalBundle(_tempDir / "standard" / "BatchnormInference" / "nchw" / "fp32" / "s",
                            "s");
        createMinimalBundle(
            _tempDir / "comprehensive" / "BatchnormInference" / "nchw" / "fp32" / "c", "c");
        createMinimalBundle(_tempDir / "full" / "BatchnormInference" / "nchw" / "fp32" / "f", "f");
    }
};

} // namespace

TEST_F(TestGoldenBundleDiscoveryFixture, MissingTierThrows)
{
    // Only quick populated; standard/comprehensive/full are absent -> hard fail.
    createMinimalBundle(_tempDir / "quick" / "BatchnormInference" / "nchw" / "fp32" / "q", "q");
    EXPECT_THROW(discoverGoldenBundles(_tempDir), std::runtime_error);
}

TEST_F(TestGoldenBundleDiscoveryFixture, EmptyTierThrows)
{
    // All four tier folders exist but quick has no bundles -> hard fail.
    for(const char* tier : {"quick", "standard", "comprehensive", "full"})
    {
        std::filesystem::create_directories(_tempDir / tier);
    }
    EXPECT_THROW(discoverGoldenBundles(_tempDir), std::runtime_error);
}

TEST_F(TestGoldenBundleDiscoveryFixture, StrayTopLevelDirThrows)
{
    populateAllTiers();
    // A typo'd tier directory at the top level must be rejected.
    std::filesystem::create_directories(_tempDir / "quik");
    EXPECT_THROW(discoverGoldenBundles(_tempDir), std::runtime_error);
}

TEST_F(TestGoldenBundleDiscoveryFixture, WrongDepthBundleThrows)
{
    populateAllTiers();
    // Bundle placed at wrong depth (missing layout and dtype levels).
    auto badDir = _tempDir / "quick" / "BadOp" / "bad";
    createMinimalBundle(badDir, "bad");
    EXPECT_THROW(discoverGoldenBundles(_tempDir), std::runtime_error);
}

TEST_F(TestGoldenBundleDiscoveryFixture, CollisionThrows)
{
    populateAllTiers();
    // Two bundles whose op names differ only by dash vs underscore
    // both sanitize to the same suite + test name -> collision.
    createMinimalBundle(_tempDir / "quick" / "Op-A" / "nchw" / "fp32" / "SameName", "SameName");
    createMinimalBundle(_tempDir / "quick" / "Op_A" / "nchw" / "fp32" / "SameName", "SameName");
    EXPECT_THROW(discoverGoldenBundles(_tempDir), std::runtime_error);
}

TEST_F(TestGoldenBundleDiscoveryFixture, DiscoversBundlesAcrossAllTiers)
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

TEST_F(TestGoldenBundleDiscoveryFixture, SkipsMetaJson)
{
    populateAllTiers();
    // Both a bare meta.json and a {Name}.meta.json companion must be ignored.
    auto bundleDir = _tempDir / "quick" / "BatchnormInference" / "nchw" / "fp32" / "withmeta";
    createMinimalBundle(bundleDir, "withmeta");
    std::ofstream(bundleDir / "withmeta.meta.json") << "{}";
    std::ofstream(bundleDir / "meta.json") << "{}";

    auto result = discoverGoldenBundles(_tempDir);
    // 4 from populateAllTiers + 1 "withmeta"; neither meta file adds a bundle.
    EXPECT_EQ(result.size(), 5u);
}

TEST_F(TestGoldenBundleDiscoveryFixture, ScanFilesByExtensionIsGenericAndSorted)
{
    // The generic scanner carries no golden-ref knowledge: it returns every
    // matching file (including meta files), recursively, in sorted order.
    auto root = _tempDir / "scan";
    std::filesystem::create_directories(root / "sub");
    std::ofstream(root / "b.json") << "{}";
    std::ofstream(root / "a.json") << "{}";
    std::ofstream(root / "sub" / "c.json") << "{}";
    std::ofstream(root / "sub" / "c.meta.json") << "{}";
    std::ofstream(root / "note.txt") << "ignore me";

    auto json = scanFilesByExtension(root, ".json");
    ASSERT_EQ(json.size(), 4u); // a, b, sub/c, sub/c.meta — .txt excluded
    EXPECT_TRUE(std::is_sorted(json.begin(), json.end()));
    EXPECT_EQ(json.front().filename(), "a.json");
}

TEST(TestGoldenMetaFile, IdentifiesCompanionMetadata)
{
    EXPECT_TRUE(isGoldenMetaFile("dir/meta.json"));
    EXPECT_TRUE(isGoldenMetaFile("dir/resnet50.meta.json"));
    EXPECT_TRUE(isGoldenMetaFile("resnet50.meta.json"));
    EXPECT_FALSE(isGoldenMetaFile("dir/resnet50.json"));
    EXPECT_FALSE(isGoldenMetaFile("dir/metadata.json"));
    EXPECT_FALSE(isGoldenMetaFile("dir/meta.bin"));
}

TEST(TestSanitizeForGtest, ReplacesInvalidChars)
{
    EXPECT_EQ(sanitizeForGtest("hello world!"), "hello_world_");
    EXPECT_EQ(sanitizeForGtest("Conv-Fprop.v2"), "Conv_Fprop_v2");
    EXPECT_EQ(sanitizeForGtest("already_valid_123"), "already_valid_123");
}

TEST_F(TestGoldenBundleDiscoveryFixture, UnparseableJsonIsDiscoveredButLoadThrows)
{
    populateAllTiers();
    auto badDir = _tempDir / "quick" / "BadOp" / "nchw" / "fp32" / "Malformed";
    std::filesystem::create_directories(badDir);
    std::ofstream(badDir / "Malformed.json") << "{{NOT VALID JSON AT ALL";

    auto bundles = discoverGoldenBundles(_tempDir);
    auto it = std::find_if(bundles.begin(), bundles.end(), [](const DiscoveredBundle& b) {
        return b.testName == "Malformed";
    });
    ASSERT_NE(it, bundles.end()) << "Malformed bundle should be discovered (valid .json path)";

    EXPECT_THROW(hipdnn_test_sdk::utilities::loadGraphAndTensors(it->jsonPath), std::exception);
}

TEST(TestTierPrefix, MatchesRfcScheme)
{
    EXPECT_EQ(tierPrefix("quick"), "");
    EXPECT_EQ(tierPrefix("standard"), "Standard/");
    EXPECT_EQ(tierPrefix("comprehensive"), "Comprehensive/");
    EXPECT_EQ(tierPrefix("full"), "Full/");
}

// NOLINTEND(readability-identifier-naming)
