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

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

namespace
{

/// Helper: create a temporary directory with a fake bundle JSON and optional
/// .meta.json companion. Auto-cleans on destruction via ScopedDirectory.
class TempBundle
{
public:
    explicit TempBundle(const std::string& metaJsonContent = "")
        : _dir(std::filesystem::temp_directory_path()
               / ("test_bundle_" + std::to_string(std::rand())))
    {
        // Create a minimal bundle JSON (enough for path derivation)
        std::ofstream bundleFile(_dir.path() / "Bundle.json");
        bundleFile << "{}";
        bundleFile.close();

        if(!metaJsonContent.empty())
        {
            std::ofstream metaFile(_dir.path() / "Bundle.meta.json");
            metaFile << metaJsonContent;
            metaFile.close();
        }
    }

    std::filesystem::path bundleJsonPath() const
    {
        return _dir.path() / "Bundle.json";
    }

private:
    hipdnn_test_sdk::utilities::ScopedDirectory _dir;
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
        "generator": "scripts/generate_golden.py",
        "generator_version": "1.0.0",
        "generated_at": "2026-05-04T18:00:00Z",
        "gpu_architecture": "gfx942",
        "rocm_version": "6.4.0",
        "reference_source": "PyTorch 2.3.0",
        "reference_source_hash": "a3f8c2e1",
        "reference_strategy": "precision_uplift",
        "operation": "conv_fwd",
        "generation_command": "python generate.py --op conv_fwd",
        "notes": "baseline run",
        "seed": 42,
        "minimum_vram_mb": 8192
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->formatVersion, 1);
    EXPECT_EQ(meta->generator, "scripts/generate_golden.py");
    EXPECT_EQ(meta->generatorVersion, "1.0.0");
    EXPECT_EQ(meta->generatedAt, "2026-05-04T18:00:00Z");
    EXPECT_EQ(meta->gpuArchitecture, "gfx942");
    EXPECT_EQ(meta->rocmVersion, "6.4.0");
    EXPECT_EQ(meta->referenceSource, "PyTorch 2.3.0");
    EXPECT_EQ(meta->referenceSourceHash, "a3f8c2e1");
    EXPECT_EQ(meta->referenceStrategy, "precision_uplift");
    EXPECT_EQ(meta->operation, "conv_fwd");
    EXPECT_EQ(meta->generationCommand, "python generate.py --op conv_fwd");
    EXPECT_EQ(meta->notes, "baseline run");
    EXPECT_EQ(meta->seed, 42);
    EXPECT_EQ(meta->minimumVramMb, 8192);
}

TEST(TestLoadBundleMetadata, LoadsMinimalMetadata)
{
    const TempBundle bundle(R"({"format_version": 1})");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->formatVersion, 1);
    EXPECT_FALSE(meta->generator.has_value());
    EXPECT_FALSE(meta->generatorVersion.has_value());
    EXPECT_FALSE(meta->generatedAt.has_value());
    EXPECT_FALSE(meta->gpuArchitecture.has_value());
    EXPECT_FALSE(meta->rocmVersion.has_value());
    EXPECT_FALSE(meta->referenceSource.has_value());
    EXPECT_FALSE(meta->referenceSourceHash.has_value());
    EXPECT_FALSE(meta->referenceStrategy.has_value());
    EXPECT_FALSE(meta->operation.has_value());
    EXPECT_FALSE(meta->generationCommand.has_value());
    EXPECT_FALSE(meta->notes.has_value());
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
    const TempBundle bundle(R"({"operation": "conv_fwd"})");
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
        "unknown_field": true,
        "generator_version": "1.0.0",
        "another_unknown": "should be ignored"
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->generatorVersion, "1.0.0");
}

TEST(TestLoadBundleMetadata, HandlesPartialMetadata)
{
    const TempBundle bundle(R"({
        "format_version": 1,
        "operation": "batchnorm_fwd",
        "minimum_vram_mb": 4096
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
        "minimum_vram_mb": -1
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->minimumVramMb, -1);
}

TEST(TestLoadBundleMetadata, HandlesZeroVram)
{
    const TempBundle bundle(R"({
        "format_version": 1,
        "minimum_vram_mb": 0
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
        "minimum_vram_mb": 8.5,
        "seed": 3.14
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

TEST(TestLoadBundleMetadata, ReturnsNulloptOnFloatFormatVersion)
{
    // 1.0 is a float, not an integer — is_number_integer() returns false
    const TempBundle bundle(R"({"format_version": 1.0})");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    EXPECT_FALSE(meta.has_value());
}

TEST(TestLoadBundleMetadata, HandlesEmptyStringFields)
{
    const TempBundle bundle(R"({
        "format_version": 1,
        "gpu_architecture": "",
        "reference_source": ""
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    // Empty strings are stored as-is — they are present but empty
    ASSERT_TRUE(meta->gpuArchitecture.has_value());
    EXPECT_EQ(*meta->gpuArchitecture, "");
    ASSERT_TRUE(meta->referenceSource.has_value());
    EXPECT_EQ(*meta->referenceSource, "");
}

// ---------------------------------------------------------------------------
// checkVramRequirement — pure guard function
// ---------------------------------------------------------------------------

TEST(TestCheckVramRequirement, PassesWhenVramNotSet)
{
    const BundleMetadata meta;
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

TEST(TestCheckArchCompatibility, PassesWhenArchNotSet)
{
    // No gpu_architecture → data is portable → pass on any device
    const BundleMetadata meta;
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenArchNotSetEvenWithReferenceSource)
{
    // reference_source is informational — guard only looks at gpu_architecture
    BundleMetadata meta;
    meta.referenceSource = "PyTorch 2.3.0";
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenArchIsEmptyString)
{
    // Empty gpu_architecture is treated as "not recorded" — data is portable
    BundleMetadata meta;
    meta.gpuArchitecture = "";
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenDeviceCannotBeQueried)
{
    BundleMetadata meta;
    meta.gpuArchitecture = "gfx942";
    // empty currentArch means "could not query device" — skip disabled
    EXPECT_FALSE(checkArchCompatibility(meta, "").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenArchMatches)
{
    BundleMetadata meta;
    meta.gpuArchitecture = "gfx942";
    // "gfx942" is a prefix of "gfx942:sramecc+:xnack-" (followed by ':')
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, SkipsWhenArchMismatches)
{
    BundleMetadata meta;
    meta.gpuArchitecture = "gfx942";
    auto result = checkArchCompatibility(meta, "gfx1100");
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("gfx942"), std::string::npos);
    EXPECT_NE(result->find("gfx1100"), std::string::npos);
}

TEST(TestCheckArchCompatibility, SkipsWhenArchDoesNotMatch)
{
    BundleMetadata meta;
    meta.gpuArchitecture = "gfx1100";
    auto result = checkArchCompatibility(meta, "gfx942:sramecc+:xnack-");
    ASSERT_TRUE(result.has_value());
}

TEST(TestCheckArchCompatibility, RejectsPartialArchName)
{
    // "gfx94" is a prefix of "gfx942" but not a complete base arch.
    // The guard requires an exact base-arch match (up to the ':' delimiter).
    BundleMetadata meta;
    meta.gpuArchitecture = "gfx94";
    EXPECT_TRUE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, ReferenceSourceDoesNotAffectGuard)
{
    // reference_source is purely informational — the guard ignores it.
    // Even with a GPU-sounding source, no gpu_architecture means portable.
    BundleMetadata meta;
    meta.referenceSource = "AITER 0.1.13";
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenBareArchMatchesExactly)
{
    // Device reports bare arch without feature flags
    BundleMetadata meta;
    meta.gpuArchitecture = "gfx942";
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942").has_value());
}

TEST(TestCheckArchCompatibility, PassesWhenFullArchStringMatchesDevice)
{
    // If metadata stores the full arch string (with feature flags),
    // it matches the same full string from the device.
    BundleMetadata meta;
    meta.gpuArchitecture = "gfx942:sramecc+:xnack-";
    EXPECT_FALSE(checkArchCompatibility(meta, "gfx942:sramecc+:xnack-").has_value());
}

TEST(TestCheckArchCompatibility, RejectsWhenFeatureFlagsDiffer)
{
    // Full arch string in metadata with different feature flags on device
    BundleMetadata meta;
    meta.gpuArchitecture = "gfx942:sramecc+:xnack-";
    EXPECT_TRUE(checkArchCompatibility(meta, "gfx942:sramecc-:xnack-").has_value());
}

// ---------------------------------------------------------------------------
// loadBundleMetadata — additional corner cases
// ---------------------------------------------------------------------------

TEST(TestLoadBundleMetadata, IgnoresIntegerWhereStringExpected)
{
    // gpu_architecture is 42 (integer, not string) — readString skips it
    const TempBundle bundle(R"({
        "format_version": 1,
        "gpu_architecture": 42,
        "operation": 100
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_FALSE(meta->gpuArchitecture.has_value());
    EXPECT_FALSE(meta->operation.has_value());
}

TEST(TestLoadBundleMetadata, IgnoresNullFieldValues)
{
    // JSON null is not a string or integer — fields become nullopt
    const TempBundle bundle(R"({
        "format_version": 1,
        "seed": null,
        "gpu_architecture": null
    })");

    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_FALSE(meta->seed.has_value());
    EXPECT_FALSE(meta->gpuArchitecture.has_value());
}

TEST(TestLoadBundleMetadata, ReturnsNulloptOnEmptyFile)
{
    const TempBundle bundle(" ");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    EXPECT_FALSE(meta.has_value());
}

TEST(TestLoadBundleMetadata, ReturnsNulloptOnJsonArray)
{
    // Valid JSON but not an object — format_version check fails
    const TempBundle bundle("[1, 2, 3]");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    EXPECT_FALSE(meta.has_value());
}

TEST(TestLoadBundleMetadata, RejectsFormatVersionZero)
{
    const TempBundle bundle(R"({"format_version": 0})");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    EXPECT_FALSE(meta.has_value());
}

// NOLINTEND(readability-identifier-naming)
