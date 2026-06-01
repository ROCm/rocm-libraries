// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <hipdnn_test_sdk/utilities/BundleMetadata.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

using hipdnn_test_sdk::utilities::BundleMetadata;
using hipdnn_test_sdk::utilities::checkArchCompatibility;
using hipdnn_test_sdk::utilities::checkVramRequirement;
using hipdnn_test_sdk::utilities::isMetaJsonFile;
using hipdnn_test_sdk::utilities::loadBundleMetadata;
using hipdnn_test_sdk::utilities::metaJsonPath;
using hipdnn_test_sdk::utilities::writeBundleMetadata;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

namespace
{

/// Helper: create a temporary directory with a fake bundle JSON and optional
/// .meta.json companion. Auto-cleans on destruction.
class TempBundle
{
public:
    explicit TempBundle(const std::string& metaJsonContent = "")
        : _dir(std::filesystem::temp_directory_path()
               / ("test_bundle_" + std::to_string(std::rand())))
    {
        std::filesystem::create_directories(_dir);

        // Create a minimal bundle JSON (enough for path derivation)
        std::ofstream bundleFile(_dir / "Bundle.json");
        bundleFile << "{}";
        bundleFile.close();

        if(!metaJsonContent.empty())
        {
            std::ofstream metaFile(_dir / "Bundle.meta.json");
            metaFile << metaJsonContent;
            metaFile.close();
        }
    }

    ~TempBundle()
    {
        std::filesystem::remove_all(_dir);
    }

    TempBundle(const TempBundle&) = delete;
    TempBundle& operator=(const TempBundle&) = delete;
    TempBundle(TempBundle&&) = delete;
    TempBundle& operator=(TempBundle&&) = delete;

    std::filesystem::path bundleJsonPath() const
    {
        return _dir / "Bundle.json";
    }

private:
    std::filesystem::path _dir;
};

} // namespace

// ---------------------------------------------------------------------------
// isMetaJsonFile
// ---------------------------------------------------------------------------

TEST(TestIsMetaJsonFile, DetectsMetaJson)
{
    EXPECT_TRUE(isMetaJsonFile("Small.meta.json"));
    EXPECT_TRUE(isMetaJsonFile("Large.meta.json"));
    EXPECT_TRUE(isMetaJsonFile("a.b.meta.json"));
    EXPECT_TRUE(isMetaJsonFile("/some/path/Bundle.meta.json"));
}

TEST(TestIsMetaJsonFile, RejectsBundleJson)
{
    EXPECT_FALSE(isMetaJsonFile("Small.json"));
    EXPECT_FALSE(isMetaJsonFile("Large.json"));
    EXPECT_FALSE(isMetaJsonFile("/some/path/Bundle.json"));
}

TEST(TestIsMetaJsonFile, RejectsPlainMetaJson)
{
    // "meta.json" has stem "meta" with no compound extension — not a bundle companion
    EXPECT_FALSE(isMetaJsonFile("meta.json"));
    EXPECT_FALSE(isMetaJsonFile("/some/path/meta.json"));
}

TEST(TestIsMetaJsonFile, RejectsNonJsonFiles)
{
    EXPECT_FALSE(isMetaJsonFile("Small.meta.bin"));
    EXPECT_FALSE(isMetaJsonFile("Small.tensor0.bin"));
    EXPECT_FALSE(isMetaJsonFile("README.md"));
}

// ---------------------------------------------------------------------------
// metaJsonPath
// ---------------------------------------------------------------------------

TEST(TestMetaJsonPath, DerivesMetaPathFromBundleJson)
{
    EXPECT_EQ(metaJsonPath("/dir/Small.json"), "/dir/Small.meta.json");
    EXPECT_EQ(metaJsonPath("/a/b/Large.json"), "/a/b/Large.meta.json");
    EXPECT_EQ(metaJsonPath("Bundle.json"), "Bundle.meta.json");
}

// ---------------------------------------------------------------------------
// loadBundleMetadata — reader
// ---------------------------------------------------------------------------

TEST(TestLoadBundleMetadata, LoadsValidFullMetadata)
{
    const TempBundle bundle(R"({
        "format_version": 1,
        "metadata": {
            "generator_version": "1.0.0",
            "created_at": "2026-05-04T18:00:00Z",
            "gpu_architecture": "gfx942",
            "rocm_version": "6.4.0",
            "reference_executor": "cpu",
            "reference_executor_hash": "a3f8c2e1",
            "operation": "conv_fwd",
            "seed": 42,
            "minimum_vram_mb": 8192
        }
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->formatVersion, 1);
    EXPECT_EQ(meta->generatorVersion, "1.0.0");
    EXPECT_EQ(meta->createdAt, "2026-05-04T18:00:00Z");
    EXPECT_EQ(meta->gpuArchitecture, "gfx942");
    EXPECT_EQ(meta->rocmVersion, "6.4.0");
    EXPECT_EQ(meta->referenceExecutor, "cpu");
    EXPECT_EQ(meta->referenceExecutorHash, "a3f8c2e1");
    EXPECT_EQ(meta->operation, "conv_fwd");
    EXPECT_EQ(meta->seed, 42);
    EXPECT_EQ(meta->minimumVramMb, 8192);
}

TEST(TestLoadBundleMetadata, LoadsMinimalMetadata)
{
    const TempBundle bundle(R"({"format_version": 1})");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->formatVersion, 1);
    EXPECT_FALSE(meta->generatorVersion.has_value());
    EXPECT_FALSE(meta->createdAt.has_value());
    EXPECT_FALSE(meta->gpuArchitecture.has_value());
    EXPECT_FALSE(meta->rocmVersion.has_value());
    EXPECT_FALSE(meta->referenceExecutor.has_value());
    EXPECT_FALSE(meta->referenceExecutorHash.has_value());
    EXPECT_FALSE(meta->operation.has_value());
    EXPECT_FALSE(meta->seed.has_value());
    EXPECT_FALSE(meta->minimumVramMb.has_value());
}

TEST(TestLoadBundleMetadata, ReturnsNulloptWhenFileNotFound)
{
    const TempBundle bundle; // no meta.json created
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    EXPECT_FALSE(meta.has_value());
}

TEST(TestLoadBundleMetadata, ReturnsNulloptOnMalformedJson)
{
    const TempBundle bundle("{not valid json");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    EXPECT_FALSE(meta.has_value());
}

TEST(TestLoadBundleMetadata, ReturnsNulloptOnMissingFormatVersion)
{
    const TempBundle bundle(R"({"metadata": {}})");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    EXPECT_FALSE(meta.has_value());
}

TEST(TestLoadBundleMetadata, ReturnsNulloptOnWrongFormatVersion)
{
    const TempBundle bundle(R"({"format_version": 99})");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    EXPECT_FALSE(meta.has_value());
}

TEST(TestLoadBundleMetadata, IgnoresUnknownFields)
{
    const TempBundle bundle(R"({
        "format_version": 1,
        "unknown_top_level": true,
        "metadata": {
            "generator_version": "1.0.0",
            "unknown_nested": "should be ignored"
        }
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->generatorVersion, "1.0.0");
}

TEST(TestLoadBundleMetadata, HandlesPartialMetadata)
{
    const TempBundle bundle(R"({
        "format_version": 1,
        "metadata": {
            "operation": "batchnorm_fwd",
            "minimum_vram_mb": 4096
        }
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->operation, "batchnorm_fwd");
    EXPECT_EQ(meta->minimumVramMb, 4096);
    EXPECT_FALSE(meta->generatorVersion.has_value());
    EXPECT_FALSE(meta->gpuArchitecture.has_value());
    EXPECT_FALSE(meta->seed.has_value());
}

TEST(TestLoadBundleMetadata, HandlesNegativeVram)
{
    const TempBundle bundle(R"({
        "format_version": 1,
        "metadata": {"minimum_vram_mb": -1}
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->minimumVramMb, -1);
}

TEST(TestLoadBundleMetadata, HandlesZeroVram)
{
    const TempBundle bundle(R"({
        "format_version": 1,
        "metadata": {"minimum_vram_mb": 0}
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->minimumVramMb, 0);
}

TEST(TestLoadBundleMetadata, IgnoresFloatWhereIntegerExpected)
{
    // minimum_vram_mb is 8.5 (float, not integer) — should be treated as absent
    const TempBundle bundle(R"({
        "format_version": 1,
        "metadata": {"minimum_vram_mb": 8.5, "seed": 3.14}
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_FALSE(meta->minimumVramMb.has_value());
    EXPECT_FALSE(meta->seed.has_value());
}

TEST(TestLoadBundleMetadata, ReturnsNulloptOnStringFormatVersion)
{
    const TempBundle bundle(R"({"format_version": "1"})");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    EXPECT_FALSE(meta.has_value());
}

TEST(TestLoadBundleMetadata, IgnoresMetadataThatIsNotAnObject)
{
    const TempBundle bundle(R"({"format_version": 1, "metadata": "not_an_object"})");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->formatVersion, 1);
    EXPECT_FALSE(meta->operation.has_value());
}

TEST(TestLoadBundleMetadata, HandlesEmptyStringFields)
{
    const TempBundle bundle(R"({
        "format_version": 1,
        "metadata": {
            "gpu_architecture": "",
            "reference_executor": ""
        }
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    // Empty strings are stored as-is — they are present but empty
    ASSERT_TRUE(meta->gpuArchitecture.has_value());
    EXPECT_EQ(*meta->gpuArchitecture, "");
    ASSERT_TRUE(meta->referenceExecutor.has_value());
    EXPECT_EQ(*meta->referenceExecutor, "");
}

// ---------------------------------------------------------------------------
// writeBundleMetadata — writer
// ---------------------------------------------------------------------------

TEST(TestWriteBundleMetadata, WritesFullMetadataRoundTrip)
{
    const TempBundle bundle;

    BundleMetadata meta;
    meta.formatVersion = 1;
    meta.generatorVersion = "2.0.0";
    meta.createdAt = "2026-05-31T12:00:00Z";
    meta.gpuArchitecture = "gfx1100";
    meta.rocmVersion = "6.5.0";
    meta.referenceExecutor = "gpu";
    meta.referenceExecutorHash = "deadbeef";
    meta.operation = "matmul";
    meta.seed = 99;
    meta.minimumVramMb = 16384;

    writeBundleMetadata(bundle.bundleJsonPath(), meta);

    auto loaded = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->formatVersion, 1);
    EXPECT_EQ(loaded->generatorVersion, "2.0.0");
    EXPECT_EQ(loaded->createdAt, "2026-05-31T12:00:00Z");
    EXPECT_EQ(loaded->gpuArchitecture, "gfx1100");
    EXPECT_EQ(loaded->rocmVersion, "6.5.0");
    EXPECT_EQ(loaded->referenceExecutor, "gpu");
    EXPECT_EQ(loaded->referenceExecutorHash, "deadbeef");
    EXPECT_EQ(loaded->operation, "matmul");
    EXPECT_EQ(loaded->seed, 99);
    EXPECT_EQ(loaded->minimumVramMb, 16384);
}

TEST(TestWriteBundleMetadata, WritesMinimalMetadata)
{
    const TempBundle bundle;

    BundleMetadata meta;
    // All optional fields are nullopt by default
    writeBundleMetadata(bundle.bundleJsonPath(), meta);

    auto loaded = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->formatVersion, 1);
    EXPECT_FALSE(loaded->generatorVersion.has_value());
    EXPECT_FALSE(loaded->operation.has_value());
    EXPECT_FALSE(loaded->minimumVramMb.has_value());
}

TEST(TestWriteBundleMetadata, OmitsNulloptFields)
{
    const TempBundle bundle;

    BundleMetadata meta;
    meta.operation = "conv_fwd";
    // All other optional fields left as nullopt

    writeBundleMetadata(bundle.bundleJsonPath(), meta);

    // Read raw JSON to verify absent keys
    auto metaPath = hipdnn_test_sdk::utilities::metaJsonPath(bundle.bundleJsonPath());
    std::ifstream file(metaPath);
    auto json = nlohmann::json::parse(file);

    EXPECT_TRUE(json.contains("format_version"));
    EXPECT_TRUE(json.contains("metadata"));
    EXPECT_TRUE(json["metadata"].contains("operation"));
    EXPECT_FALSE(json["metadata"].contains("generator_version"));
    EXPECT_FALSE(json["metadata"].contains("seed"));
    EXPECT_FALSE(json["metadata"].contains("minimum_vram_mb"));
}

TEST(TestWriteBundleMetadata, ThrowsOnInvalidPath)
{
    BundleMetadata meta;
    EXPECT_THROW(writeBundleMetadata("/nonexistent_directory/Bundle.json", meta),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// checkVramRequirement — pure guard function
// ---------------------------------------------------------------------------

TEST(TestCheckVramRequirement, PassesWhenVramNotSet)
{
    BundleMetadata meta;
    // minimumVramMb is nullopt
    EXPECT_FALSE(checkVramRequirement(meta, 8192).has_value());
}

TEST(TestCheckVramRequirement, PassesWhenVramZero)
{
    BundleMetadata meta;
    meta.minimumVramMb = 0;
    EXPECT_FALSE(checkVramRequirement(meta, 8192).has_value());
}

TEST(TestCheckVramRequirement, PassesWhenVramNegative)
{
    BundleMetadata meta;
    meta.minimumVramMb = -1;
    EXPECT_FALSE(checkVramRequirement(meta, 8192).has_value());
}

TEST(TestCheckVramRequirement, PassesWhenDeviceCannotBeQueried)
{
    BundleMetadata meta;
    meta.minimumVramMb = 16000;
    // deviceTotalVramMb = 0 means "could not query device"
    EXPECT_FALSE(checkVramRequirement(meta, 0).has_value());
}

TEST(TestCheckVramRequirement, PassesWhenDeviceHasEnoughVram)
{
    BundleMetadata meta;
    meta.minimumVramMb = 8192;
    EXPECT_FALSE(checkVramRequirement(meta, 16384).has_value());
}

TEST(TestCheckVramRequirement, PassesWhenDeviceHasExactVram)
{
    BundleMetadata meta;
    meta.minimumVramMb = 8192;
    EXPECT_FALSE(checkVramRequirement(meta, 8192).has_value());
}

TEST(TestCheckVramRequirement, SkipsWhenDeviceHasInsufficientVram)
{
    BundleMetadata meta;
    meta.minimumVramMb = 16000;
    auto result = checkVramRequirement(meta, 8192);
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("16000"), std::string::npos);
    EXPECT_NE(result->find("8192"), std::string::npos);
}

// ---------------------------------------------------------------------------
// checkArchCompatibility — pure guard function
// ---------------------------------------------------------------------------

TEST(TestCheckArchCompatibility, PassesWhenExecutorNotSet)
{
    BundleMetadata meta;
    // referenceExecutor is nullopt
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenExecutorIsCpu)
{
    BundleMetadata meta;
    meta.referenceExecutor = "cpu";
    meta.gpuArchitecture = "gfx1100";
    // CPU-generated data is arch-independent
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenArchNotSet)
{
    BundleMetadata meta;
    meta.referenceExecutor = "gpu";
    // gpuArchitecture is nullopt
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenDeviceCannotBeQueried)
{
    BundleMetadata meta;
    meta.referenceExecutor = "gpu";
    meta.gpuArchitecture = "gfx942";
    // empty currentArch means "could not query device"
    EXPECT_FALSE(checkArchCompatibility(meta, "").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenArchMatches)
{
    BundleMetadata meta;
    meta.referenceExecutor = "gpu";
    meta.gpuArchitecture = "gfx942";
    // "gfx942" is a substring of "gfx942:sramecc+:xnack-"
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, SkipsWhenArchMismatches)
{
    BundleMetadata meta;
    meta.referenceExecutor = "gpu";
    meta.gpuArchitecture = "gfx942";
    auto result = checkArchCompatibility(meta, "gfx1100");
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("gfx942"), std::string::npos);
    EXPECT_NE(result->find("gfx1100"), std::string::npos);
}

TEST(TestCheckArchCompatibility, SkipsWhenArchSubstringNotFound)
{
    BundleMetadata meta;
    meta.referenceExecutor = "gpu";
    meta.gpuArchitecture = "gfx1100";
    auto result = checkArchCompatibility(meta, "gfx942:sramecc+:xnack-");
    ASSERT_TRUE(result.has_value());
}

TEST(TestCheckArchCompatibility, SubstringMatchIsPermissive)
{
    // "gfx94" matches inside "gfx942:..." — this is by design, matching the
    // substring convention used by TestSettings::findSkip for arch matching.
    BundleMetadata meta;
    meta.referenceExecutor = "gpu";
    meta.gpuArchitecture = "gfx94";
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenArchIsEmptyString)
{
    // Empty gpu_architecture in metadata means "not recorded" — guard is disabled
    BundleMetadata meta;
    meta.referenceExecutor = "gpu";
    meta.gpuArchitecture = "";
    // currentArch.find("") always returns 0 → match → passes
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

// NOLINTEND(readability-identifier-naming)
