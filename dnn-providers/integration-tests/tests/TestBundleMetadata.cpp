// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "harness/BundleMetadata.hpp"

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

using hipdnn_integration_tests::BundleMetadata;
using hipdnn_integration_tests::BundleMetadataError;
using hipdnn_integration_tests::checkArchCompatibility;
using hipdnn_integration_tests::checkVramRequirement;
using hipdnn_integration_tests::EnforcementLevel;
using hipdnn_integration_tests::loadBundleMetadata;
using hipdnn_integration_tests::metaJsonPath;
using hipdnn_integration_tests::parseBundleMetadataJson;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::isMetaJsonFile;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

namespace
{

/// Helper: create a temporary directory with a fake bundle JSON and optional
/// .meta.json companion. Auto-cleans on destruction via ScopedDirectory.
///
/// The directory name must be unique per process AND per construction, or two test
/// binaries running concurrently -- ctest -j N, or an install-tree and build-tree run
/// at once -- collide and the second dies with "ScopedDirectory: Directory already
/// exists" before reaching an assertion. claimScratchDirectory() owns that; see
/// ScratchDirectory.hpp.
class TempBundle
{
public:
    explicit TempBundle(const std::string& metaJsonContent = "")
        : _dir(claimScratchDirectory("bundle_metadata"))
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

TEST(TestLoadBundleMetadata, ThrowsOnMalformedJson)
{
    const TempBundle bundle("{not valid json");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, ThrowsOnMissingFormatVersion)
{
    const TempBundle bundle(R"({"operation": "conv_fwd"})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, ThrowsOnWrongFormatVersion)
{
    const TempBundle bundle(R"({"format_version": 99})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
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

TEST(TestLoadBundleMetadata, ThrowsOnStringFormatVersion)
{
    const TempBundle bundle(R"({"format_version": "1"})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, ThrowsOnFloatFormatVersion)
{
    // 1.0 is a float, not an integer — is_number_integer() returns false
    const TempBundle bundle(R"({"format_version": 1.0})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
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

TEST(TestLoadBundleMetadata, ThrowsOnEmptyFile)
{
    const TempBundle bundle(" ");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, ThrowsOnJsonArray)
{
    // Valid JSON but not an object
    const TempBundle bundle("[1, 2, 3]");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, ThrowsOnFormatVersionZero)
{
    const TempBundle bundle(R"({"format_version": 0})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, ThrowsOnFormatVersionThatTruncatesToOne)
{
    // 2^32 + 1 narrows to 1 as a 32-bit int; it must not be read as version 1.
    const TempBundle bundle(R"({"format_version": 4294967297})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

// ---------------------------------------------------------------------------
// loadBundleMetadata — enforcement_level
// ---------------------------------------------------------------------------

TEST(TestLoadBundleMetadata, EnforcementLevelDefaultsToFullWhenAbsent)
{
    const TempBundle bundle(R"({"format_version": 1})");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->enforcementLevel, EnforcementLevel::FULL);
}

TEST(TestLoadBundleMetadata, EnforcementLevelParsesApplicability)
{
    const TempBundle bundle(R"({"format_version": 1, "enforcement_level": "applicability"})");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->enforcementLevel, EnforcementLevel::APPLICABILITY);
}

TEST(TestLoadBundleMetadata, EnforcementLevelParsesBuildable)
{
    const TempBundle bundle(R"({"format_version": 1, "enforcement_level": "buildable"})");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->enforcementLevel, EnforcementLevel::BUILDABLE);
}

TEST(TestLoadBundleMetadata, EnforcementLevelParsesFull)
{
    const TempBundle bundle(R"({"format_version": 1, "enforcement_level": "full"})");
    auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->enforcementLevel, EnforcementLevel::FULL);
}

TEST(TestLoadBundleMetadata, EnforcementLevelInvalidTokenThrows)
{
    const TempBundle bundle(R"({"format_version": 1, "enforcement_level": "buildible"})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, EnforcementLevelNonStringThrows)
{
    const TempBundle bundle(R"({"format_version": 1, "enforcement_level": 1})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, ErrorNamesTheMetaJsonFile)
{
    const TempBundle bundle(R"({"format_version": 2})");
    try
    {
        loadBundleMetadata(bundle.bundleJsonPath());
        FAIL() << "expected BundleMetadataError";
    }
    catch(const BundleMetadataError& e)
    {
        EXPECT_NE(std::string(e.what()).find(metaJsonPath(bundle.bundleJsonPath()).string()),
                  std::string::npos)
            << e.what();
    }
}

// ---------------------------------------------------------------------------
// parseBundleMetadataJson — inline (sweep-case) metadata
// ---------------------------------------------------------------------------

TEST(TestParseBundleMetadataJson, ParsesValidObject)
{
    auto meta = parseBundleMetadataJson(
        nlohmann::json::parse(R"({"format_version": 1, "enforcement_level": "buildable"})"),
        "case_0");
    EXPECT_EQ(meta.enforcementLevel, EnforcementLevel::BUILDABLE);
}

TEST(TestParseBundleMetadataJson, ThrowsOnMalformedObjects)
{
    for(const auto* text : {R"([])",
                            R"("metadata")",
                            R"({})",
                            R"({"format_version": "1"})",
                            R"({"format_version": 2})",
                            R"({"format_version": 1, "enforcement_level": "FULL"})",
                            R"({"format_version": 1, "enforcement_level": null})"})
    {
        EXPECT_THROW(parseBundleMetadataJson(nlohmann::json::parse(text), "case_0"),
                     BundleMetadataError)
            << text;
    }
}

TEST(TestParseBundleMetadataJson, ErrorNamesTheSource)
{
    try
    {
        parseBundleMetadataJson(nlohmann::json::parse(R"({"format_version": 7})"),
                                "sweep.json#case_3");
        FAIL() << "expected BundleMetadataError";
    }
    catch(const BundleMetadataError& e)
    {
        EXPECT_NE(std::string(e.what()).find("sweep.json#case_3"), std::string::npos) << e.what();
    }
}

TEST(TestParseBundleMetadataJson, ParsesNumericInputsKeys)
{
    auto meta = parseBundleMetadataJson(nlohmann::json::parse(R"({"format_version": 1,
            "inputs": {"0": {"fill": "zeros"}, "12": {"fill": "ones"}}})"),
                                        "case_0");
    ASSERT_TRUE(meta.inputs.has_value());
    ASSERT_EQ(meta.inputs->size(), 2u);
    EXPECT_EQ(meta.inputs->at(0)["fill"], "zeros");
    EXPECT_EQ(meta.inputs->at(12)["fill"], "ones");
}

// inputs keys are tensor UIDs. A key that is not entirely an integer used to be
// dropped with a warning, silently losing that tensor's input spec. A numeric
// prefix ("12abc") is rejected too, rather than read as uid 12.
TEST(TestParseBundleMetadataJson, ThrowsOnNonNumericInputsKey)
{
    for(const auto* key : {"x", "", "12abc", "1.5"})
    {
        const nlohmann::json json
            = {{"format_version", 1}, {"inputs", {{key, nlohmann::json::object()}}}};
        try
        {
            parseBundleMetadataJson(json, "case_0");
            FAIL() << "expected BundleMetadataError for inputs key \"" << key << "\"";
        }
        catch(const BundleMetadataError& e)
        {
            EXPECT_NE(std::string(e.what()).find("non-numeric inputs key"), std::string::npos)
                << e.what();
        }
    }
}

// NOLINTEND(readability-identifier-naming)
