// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "harness/BundleMetadata.hpp"

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

using hipdnn_integration_tests::BundleMetadata;
using hipdnn_integration_tests::BundleMetadataError;
using hipdnn_integration_tests::CasePlacement;
using hipdnn_integration_tests::CasePurpose;
using hipdnn_integration_tests::CaseValidator;
using hipdnn_integration_tests::checkArchCompatibility;
using hipdnn_integration_tests::checkVramRequirement;
using hipdnn_integration_tests::EnforcementLevel;
using hipdnn_integration_tests::loadBundleMetadata;
using hipdnn_integration_tests::metaJsonPath;
using hipdnn_integration_tests::OutputLiveRows;
using hipdnn_integration_tests::parseBundleMetadataJson;
using hipdnn_integration_tests::parseTensorAliases;
using hipdnn_integration_tests::TensorAlias;
using hipdnn_integration_tests::TensorStartOffset;
using hipdnn_integration_tests::toJson;
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

TEST(TestLoadBundleMetadata, IgnoresAnArchsKeyLikeAnyUnknownField)
{
    const TempBundle bundle(R"({"format_version": 1, "archs": ["gfx942"]})");
    const auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->formatVersion, 1);

    for(const auto* archs :
        {R"(["gfx942"])", R"("gfx942")", "[]", "null", "{}", R"(["gfx942:sramecc+:xnack-"])"})
    {
        EXPECT_NO_THROW(parseBundleMetadataJson(
            nlohmann::json::parse(std::string(R"({"format_version": 2, "archs": )") + archs + "}")))
            << archs;
    }
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

TEST(TestLoadBundleMetadata, PurposeIsUnsetWhenAbsent)
{
    const TempBundle bundle(R"({"format_version": 1})");
    const auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_FALSE(meta->purpose.has_value());
}

TEST(TestLoadBundleMetadata, PurposeParsesEachValue)
{
    const std::array<std::pair<const char*, CasePurpose>, 3> cases
        = {{{R"({"format_version": 1, "purpose": "claim"})", CasePurpose::CLAIM},
            {R"({"format_version": 1, "purpose": "probe"})", CasePurpose::PROBE},
            {R"({"format_version": 1, "purpose": "regression"})", CasePurpose::REGRESSION}}};
    for(const auto& [text, expected] : cases)
    {
        const TempBundle bundle(text);
        const auto meta = loadBundleMetadata(bundle.bundleJsonPath());
        ASSERT_TRUE(meta.has_value()) << text;
        ASSERT_TRUE(meta->purpose.has_value()) << text;
        EXPECT_EQ(*meta->purpose, expected) << text;
    }
}

TEST(TestLoadBundleMetadata, PurposeInvalidTokenThrows)
{
    const TempBundle bundle(R"({"format_version": 1, "purpose": "probes"})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, PurposeNonStringThrows)
{
    const TempBundle bundle(R"({"format_version": 1, "purpose": 1})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, PurposeSpellingIsExact)
{
    EXPECT_STREQ(toString(CasePurpose::CLAIM), "claim");
    EXPECT_STREQ(toString(CasePurpose::PROBE), "probe");
    EXPECT_STREQ(toString(CasePurpose::REGRESSION), "regression");
}

TEST(TestParseBundleMetadataJson, PurposeIsRejectedWhenInvalidInASweepCase)
{
    for(const auto* text : {R"({"format_version": 1, "purpose": "PROBE"})",
                            R"({"format_version": 1, "purpose": null})",
                            R"({"format_version": 1, "purpose": ""})"})
    {
        EXPECT_THROW(parseBundleMetadataJson(nlohmann::json::parse(text), "case_0"),
                     BundleMetadataError)
            << text;
    }
}

TEST(TestParseBundleMetadataJson, PurposeParsesInASweepCase)
{
    const auto meta = parseBundleMetadataJson(
        nlohmann::json::parse(R"({"format_version": 1, "purpose": "regression"})"), "case_0");
    ASSERT_TRUE(meta.purpose.has_value());
    EXPECT_EQ(*meta.purpose, CasePurpose::REGRESSION);
}

// Bug caught: a typo in `validator` or `placement` silently running a case without the
// grading or the placement it was written for.
TEST(TestParseBundleMetadataJson, ValidatorAndPlacementDefaultWhenAbsent)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(R"({"format_version": 1})"));
    EXPECT_EQ(meta.validator, CaseValidator::DEFAULT);
    EXPECT_EQ(meta.placement, CasePlacement::DEFAULT);
    EXPECT_FALSE(meta.placementUid.has_value());
}

TEST(TestParseBundleMetadataJson, ValidatorAndPlacementParseEachValue)
{
    const auto nonfinite = parseBundleMetadataJson(
        nlohmann::json::parse(R"({"format_version": 1, "validator": "match_nonfinite"})"));
    EXPECT_EQ(nonfinite.validator, CaseValidator::MATCH_NONFINITE);

    const auto exact = parseBundleMetadataJson(
        nlohmann::json::parse(R"({"format_version": 1, "placement": "exact"})"));
    EXPECT_EQ(exact.placement, CasePlacement::EXACT);
    EXPECT_FALSE(exact.placementUid.has_value());

    const auto under = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 1, "placement": "under", "placement_uid": 4})"));
    EXPECT_EQ(under.placement, CasePlacement::UNDER);
    ASSERT_TRUE(under.placementUid.has_value());
    EXPECT_EQ(*under.placementUid, 4);

    const auto explicitDefaults = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 1, "validator": "default", "placement": "default"})"));
    EXPECT_EQ(explicitDefaults.validator, CaseValidator::DEFAULT);
    EXPECT_EQ(explicitDefaults.placement, CasePlacement::DEFAULT);
}

TEST(TestParseBundleMetadataJson, UnknownOrMistypedValidatorAndPlacementThrow)
{
    for(const auto* text : {R"({"format_version": 1, "validator": "match_nonfinit"})",
                            R"({"format_version": 1, "validator": "MATCH_NONFINITE"})",
                            R"({"format_version": 1, "validator": 1})",
                            R"({"format_version": 1, "validator": null})",
                            R"({"format_version": 1, "placement": "exacts"})",
                            R"({"format_version": 1, "placement": "Exact"})",
                            R"({"format_version": 1, "placement": true})",
                            R"({"format_version": 1, "placement": "exact", "placement_uid": "x"})",
                            R"({"format_version": 1, "placement": "under", "placement_uid": 1.5})"})
    {
        EXPECT_THROW(parseBundleMetadataJson(nlohmann::json::parse(text), "case_0"),
                     BundleMetadataError)
            << text;
    }
}

// Bug caught: "under" without a target tensor has nothing to misalign, and a stray uid on
// another placement would be silently ignored.
TEST(TestParseBundleMetadataJson, PlacementUidIsRequiredForUnderAndIgnoredOtherwise)
{
    EXPECT_THROW(parseBundleMetadataJson(
                     nlohmann::json::parse(R"({"format_version": 1, "placement": "under"})")),
                 BundleMetadataError);
    // A space renders the uid for every placement; only "under" acts on it.
    const auto exact = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 1, "placement": "exact", "placement_uid": 4})"));
    EXPECT_EQ(exact.placement, CasePlacement::EXACT);
    EXPECT_EQ(exact.placementUid, std::optional<int64_t>(4));
    const auto bare = parseBundleMetadataJson(
        nlohmann::json::parse(R"({"format_version": 1, "placement_uid": 4})"));
    EXPECT_EQ(bare.placement, CasePlacement::DEFAULT);
}

TEST(TestParseBundleMetadataJson, ValidatorAndPlacementSpellingIsExact)
{
    EXPECT_STREQ(toString(CaseValidator::DEFAULT), "default");
    EXPECT_STREQ(toString(CaseValidator::MATCH_NONFINITE), "match_nonfinite");
    EXPECT_STREQ(toString(CasePlacement::DEFAULT), "default");
    EXPECT_STREQ(toString(CasePlacement::EXACT), "exact");
    EXPECT_STREQ(toString(CasePlacement::UNDER), "under");
}

TEST(TestLoadBundleMetadata, ErrorNamesTheMetaJsonFile)
{
    const TempBundle bundle(R"({"format_version": 3})");
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
                            R"({"format_version": 3})",
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
            "inputs": {"0": {"seed": 1}, "12": {"seed": 2}}})"),
                                        "case_0");
    ASSERT_TRUE(meta.inputs.has_value());
    ASSERT_EQ(meta.inputs->size(), 2u);
    EXPECT_EQ(meta.inputs->at(0)["seed"], 1);
    EXPECT_EQ(meta.inputs->at(12)["seed"], 2);
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

// The reference and rejection flags are strict booleans: absent is false, a bool is
// read as written, and anything else is an authoring error. A string "false" quietly
// reading as false (or "yes" as true) would flip how the bundle is verified.
TEST(TestParseBundleMetadataJson, ReferenceAndRejectionFlagsDefaultToFalse)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(R"({"format_version": 1})"));
    EXPECT_FALSE(meta.noCpuReference);
    EXPECT_FALSE(meta.noGpuReference);
    EXPECT_FALSE(meta.expectedRejection);
}

TEST(TestParseBundleMetadataJson, ReferenceAndRejectionFlagsAreReadIndependently)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 1, "no_cpu_reference": true, "no_gpu_reference": false,
            "expected_rejection": true})"));
    EXPECT_TRUE(meta.noCpuReference);
    EXPECT_FALSE(meta.noGpuReference);
    EXPECT_TRUE(meta.expectedRejection);

    const auto other = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 1, "no_cpu_reference": false, "no_gpu_reference": true})"));
    EXPECT_FALSE(other.noCpuReference);
    EXPECT_TRUE(other.noGpuReference);
    EXPECT_FALSE(other.expectedRejection);
}

TEST(TestParseBundleMetadataJson, NonBooleanFlagsThrowNamingTheFlag)
{
    for(const char* flag : {"no_cpu_reference", "no_gpu_reference", "expected_rejection"})
    {
        for(const char* value : {R"("true")", "1", "0", "null", "[]"})
        {
            SCOPED_TRACE(std::string(flag) + " = " + value);
            const auto json = nlohmann::json::parse(std::string(R"({"format_version": 1, ")") + flag
                                                    + R"(": )" + value + "}");
            try
            {
                parseBundleMetadataJson(json, "bundle.meta.json");
                ADD_FAILURE() << "expected BundleMetadataError";
            }
            catch(const BundleMetadataError& e)
            {
                EXPECT_NE(std::string(e.what()).find(flag), std::string::npos) << e.what();
                EXPECT_NE(std::string(e.what()).find("bundle.meta.json"), std::string::npos)
                    << e.what();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// format_version 2, input fill specs, and fill recipes validated at parse time
// ---------------------------------------------------------------------------

namespace
{

nlohmann::json metadataWithInputs(const std::string& inputsJson)
{
    auto json = nlohmann::json::parse(R"({"format_version": 1})");
    json["inputs"] = nlohmann::json::parse(inputsJson);
    return json;
}

// What parseBundleMetadataJson says about `json`, or "" when it parses.
std::string parseError(const nlohmann::json& json)
{
    try
    {
        parseBundleMetadataJson(json, "case_0");
    }
    catch(const BundleMetadataError& e)
    {
        return e.what();
    }
    return {};
}

// The BundleMetadataError message for `text`, or "" when it parses.
std::string metadataError(const std::string& text)
{
    return parseError(nlohmann::json::parse(text));
}

bool mentions(const std::string& text, const std::string& part)
{
    return text.find(part) != std::string::npos;
}

// Every fill kind, plus entries carrying fields their kind does not read, as a space renders
// them, and a seed-only entry.
constexpr const char* K_ALL_KINDS_INPUTS = R"({
    "0": {"kind": "free", "lo": -1, "hi": 1, "value": 0},
    "1": {"kind": "fixed", "value": 0.5, "lo": 0, "hi": 0},
    "2": {"kind": "nonzero", "lo": -2, "hi": 2},
    "3": {"kind": "zeros", "lo": -1, "hi": 1, "value": 0, "scale": 1},
    "4": {"kind": "large", "scale": 100},
    "5": {"kind": "subnormal"},
    "6": {"kind": "special", "special": "nan", "fraction": 0.25, "lo": -1, "hi": 1},
    "7": {"kind": "special", "special": "-inf", "fraction": 1},
    "8": {"kind": "free", "lo": -3, "hi": 3, "seed": 17},
    "9": {"seed": 5}
})";

} // namespace

// Bug caught: a typo in a recipe (unknown kind, wrong field type, an unordered range, a
// missing required field) was only found when the harness ran the case, inside the gtest
// factory; it must instead fail the metadata parse, naming the uid and the field.
TEST(TestParseBundleMetadataJson, RejectsMalformedInputRecipesNamingUidAndField)
{
    struct Case
    {
        const char* inputs;
        const char* uid;
        const char* field;
    };
    const std::vector<Case> cases = {
        {R"({"3": {"kind": "constant"}})", "uid 3", "\"kind\""},
        {R"({"3": {"kind": "non_zero"}})", "uid 3", "non_zero"},
        {R"({"3": {"kind": 4}})", "uid 3", "\"kind\""},
        {R"({"3": {"kind": "free", "lo": "x"}})", "uid 3", "\"lo\""},
        {R"({"3": {"kind": "fixed", "value": null}})", "uid 3", "\"value\""},
        {R"({"3": {"kind": "free", "seed": -1}})", "uid 3", "\"seed\""},
        {R"({"4": {"kind": "nonzero", "lo": 2, "hi": -2}})", "uid 4", "lo <= hi"},
        {R"({"5": {"kind": "large"}})", "uid 5", "\"scale\""},
        {R"({"5": {"kind": "large", "scale": 0}})", "uid 5", "\"scale\""},
        {R"({"6": {"kind": "special", "fraction": 0.5}})", "uid 6", "\"special\""},
        {R"({"6": {"kind": "special", "special": "bogus", "fraction": 0.5}})",
         "uid 6",
         "\"special\""},
        {R"({"6": {"kind": "special", "special": "nan"}})", "uid 6", "\"fraction\""},
        {R"({"6": {"kind": "special", "special": "nan", "fraction": 2}})", "uid 6", "\"fraction\""},
        {R"({"6": "free"})", "uid 6", "JSON object"},
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.inputs);
        const auto message = parseError(metadataWithInputs(c.inputs));
        ASSERT_FALSE(message.empty()) << "the malformed recipe parsed";
        EXPECT_TRUE(mentions(message, c.uid)) << message;
        EXPECT_TRUE(mentions(message, c.field)) << message;
    }
}

// A bad entry beside good ones still fails the whole parse: the case must not run with some of
// its overrides silently missing.
TEST(TestParseBundleMetadataJson, OneMalformedRecipeAmongValidOnesFailsTheParse)
{
    const auto message = parseError(metadataWithInputs(
        R"({"1": {"kind": "zeros"}, "2": {"kind": "large", "scale": 8}, "9": {"kind": "typo"}})"));

    ASSERT_FALSE(message.empty());
    EXPECT_TRUE(mentions(message, "uid 9")) << message;
}

// Every kind, in the form a space renders it (extra fields included), parses, and the parsed
// map is the input JSON unchanged: validation must not rewrite or drop an entry.
TEST(TestParseBundleMetadataJson, ValidMixOfEveryFillKindParsesUnchanged)
{
    const auto json = metadataWithInputs(K_ALL_KINDS_INPUTS);

    const auto meta = parseBundleMetadataJson(json, "case_0");

    ASSERT_TRUE(meta.inputs.has_value());
    ASSERT_EQ(meta.inputs->size(), json.at("inputs").size());
    for(const auto& [key, spec] : json.at("inputs").items())
    {
        SCOPED_TRACE(key);
        EXPECT_EQ(meta.inputs->at(std::stoll(key)), spec);
    }
}

// The same two outcomes through the file reader: a malformed recipe in a .meta.json is a
// BundleMetadataError naming the uid and field (so a FailedLoad), a valid mix loads unchanged.
TEST(TestLoadBundleMetadata, MalformedInputRecipeThrowsAndValidMixLoads)
{
    const TempBundle bad(R"({"format_version": 1, "inputs": {"2": {"kind": "large"}}})");
    try
    {
        loadBundleMetadata(bad.bundleJsonPath());
        FAIL() << "expected BundleMetadataError";
    }
    catch(const BundleMetadataError& e)
    {
        EXPECT_TRUE(mentions(e.what(), "uid 2")) << e.what();
        EXPECT_TRUE(mentions(e.what(), "\"scale\"")) << e.what();
    }

    const TempBundle good(std::string(R"({"format_version": 1, "inputs": )") + K_ALL_KINDS_INPUTS
                          + "}");
    const auto meta = loadBundleMetadata(good.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    ASSERT_TRUE(meta->inputs.has_value());
    EXPECT_EQ(meta->inputs->size(), 10u);
    EXPECT_EQ(meta->inputs->at(4), nlohmann::json::parse(R"({"kind": "large", "scale": 100})"));
}

// Bug caught: a misspelled "kind" key loaded no recipe and the case silently ran on default
// fills. An inputs entry with neither "kind" nor "seed" fails the metadata parse naming the uid
// and its keys; a seed-only entry and a kind carrying a field it does not read remain valid.
TEST(TestParseBundleMetadataJson, InputsEntryWithNeitherKindNorSeedIsRejected)
{
    const auto message = parseError(
        metadataWithInputs(R"({"2": {"kind": "zeros"}, "5": {"Kind": "nonzero", "lo": -1}})"));
    ASSERT_FALSE(message.empty());
    EXPECT_TRUE(mentions(message, "uid 5")) << message;
    EXPECT_TRUE(mentions(message, "\"Kind\"")) << message;

    EXPECT_EQ(
        parseError(metadataWithInputs(R"({"2": {"seed": 5}, "3": {"kind": "free", "scale": 2}})")),
        "");
}

TEST(TestParseBundleMetadataJson, AcceptsFormatVersion2)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 2, "inputs": {"1": {"kind": "free", "lo": 0, "hi": 2},
                                           "2": {"kind": "fixed", "value": 1}}})"));
    EXPECT_EQ(meta.formatVersion, 2);
    ASSERT_TRUE(meta.inputs.has_value());
    EXPECT_EQ(meta.inputs->size(), 2u);
}

TEST(TestParseBundleMetadataJson, ThrowsOnFormatVersion3)
{
    const auto error = metadataError(R"({"format_version": 3})");
    EXPECT_NE(error.find("unsupported format_version 3"), std::string::npos) << error;
}

TEST(TestParseBundleMetadataJson, ThrowsOnUnknownFillKind)
{
    const auto error
        = metadataError(R"({"format_version": 1, "inputs": {"7": {"kind": "vaules"}}})");
    EXPECT_NE(error.find("case_0, uid 7"), std::string::npos) << error;
    EXPECT_NE(error.find("vaules"), std::string::npos) << error;
}

TEST(TestParseBundleMetadataJson, ThrowsOnNonStringFillKind)
{
    for(const auto* kind : {"1", "null", "true", R"(["free"])"})
    {
        const auto error = metadataError(
            std::string(R"({"format_version": 1, "inputs": {"7": {"kind": )") + kind + "}}}");
        EXPECT_NE(error.find("case_0, uid 7"), std::string::npos) << kind << ": " << error;
    }
}

TEST(TestParseBundleMetadataJson, ThrowsOnInputSpecThatIsNotAnObject)
{
    const auto error = metadataError(R"({"format_version": 1, "inputs": {"7": "free"}})");
    EXPECT_NE(error.find("case_0, uid 7"), std::string::npos) << error;
}

TEST(TestParseBundleMetadataJson, SeedOnlyInputStillLoads)
{
    const auto meta = parseBundleMetadataJson(
        nlohmann::json::parse(R"({"format_version": 1, "inputs": {"7": {"seed": 5}}})"));
    ASSERT_TRUE(meta.inputs.has_value());
    EXPECT_EQ(meta.inputs->at(7).at("seed").get<int>(), 5);
}

TEST(TestParseBundleMetadataJson, ThrowsOnValuesKindInFormatVersion1)
{
    const auto error = metadataError(
        R"({"format_version": 1, "inputs": {"7": {"kind": "values", "values": [64, 32]}}})");
    EXPECT_NE(error.find("kind values needs format_version 2"), std::string::npos) << error;
}

TEST(TestParseBundleMetadataJson, AcceptsValuesKindInFormatVersion2)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 2, "inputs": {"7": {"kind": "values", "values": [64, 32], "repeat": true}}})"));
    EXPECT_EQ(meta.formatVersion, 2);
    ASSERT_TRUE(meta.inputs.has_value());
    EXPECT_EQ(meta.inputs->at(7).at("kind").get<std::string>(), "values");
}

TEST(TestParseBundleMetadataJson, ThrowsOnMalformedValuesSpecInFormatVersion2)
{
    for(const auto* spec : {R"({"kind": "values", "values": []})",
                            R"({"kind": "values", "values": ["64"]})",
                            R"({"kind": "values", "values": [9223372036854775808]})",
                            R"({"kind": "values", "values": [1], "repeat": "no"})",
                            R"({"kind": "values", "values": [1], "count": 4})"})
    {
        const auto error
            = metadataError(std::string(R"({"format_version": 2, "inputs": {"7": )") + spec + "}}");
        EXPECT_NE(error.find("case_0, uid 7"), std::string::npos) << spec << ": " << error;
    }
}

TEST(TestLoadBundleMetadata, UnknownFillKindInAFileThrows)
{
    const TempBundle bundle(R"({"format_version": 1, "inputs": {"7": {"kind": "vaules"}}})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, NonBooleanFlagFromAFileThrows)
{
    const TempBundle bundle(R"({"format_version": 1, "no_gpu_reference": "yes"})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

// ---------------------------------------------------------------------------
// aliases
// ---------------------------------------------------------------------------

TEST(TestParseBundleMetadataJson, AliasesParseInFormatVersion2)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 2, "aliases": [{"uid": 3, "of": 2, "byte_offset": 0},
                                              {"uid": 5, "of": 2, "byte_offset": 16}]})"));
    ASSERT_EQ(meta.aliases.size(), 2u);
    EXPECT_EQ(meta.aliases[0], (TensorAlias{3, 2, 0}));
    EXPECT_EQ(meta.aliases[1], (TensorAlias{5, 2, 16}));
}

TEST(TestParseBundleMetadataJson, AliasesAbsentLeavesTheListEmpty)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(R"({"format_version": 2})"));
    EXPECT_TRUE(meta.aliases.empty());
}

TEST(TestParseBundleMetadataJson, AliasesRoundTripThroughJson)
{
    const std::vector<TensorAlias> aliases = {{3, 2, 0}, {5, 2, 16}};
    EXPECT_EQ(parseTensorAliases(toJson(aliases)), aliases);
}

TEST(TestParseBundleMetadataJson, ThrowsOnAliasesInFormatVersion1)
{
    const auto error = metadataError(
        R"({"format_version": 1, "aliases": [{"uid": 3, "of": 2, "byte_offset": 0}]})");
    EXPECT_NE(error.find("case_0: aliases needs format_version 2"), std::string::npos) << error;
}

// The version check wins over any problem in the value.
TEST(TestParseBundleMetadataJson, ThrowsOnMalformedAliasesInFormatVersion1)
{
    for(const auto* aliases : {"[]", "3", R"([{"uid": 3}])"})
    {
        SCOPED_TRACE(aliases);
        const auto error
            = metadataError(std::string(R"({"format_version": 1, "aliases": )") + aliases + "}");
        EXPECT_NE(error.find("case_0: aliases needs format_version 2"), std::string::npos) << error;
    }
}

TEST(TestParseBundleMetadataJson, ThrowsOnMalformedAliases)
{
    struct Case
    {
        const char* aliases;
        const char* rule;
    };
    const std::vector<Case> cases = {
        {R"({"uid": 3, "of": 2, "byte_offset": 0})", "aliases is not an array"},
        {"[]", "aliases is empty"},
        {"[3]", "aliases[0] is not an object"},
        {R"([{"uid": 3, "of": 2}])", R"(aliases[0] (view 3, owner 2) lacks "byte_offset")"},
        {R"([{"uid": 3, "byte_offset": 0}])", R"(aliases[0] (view 3) lacks "of")"},
        {R"([{"of": 2, "byte_offset": 0}])", R"(aliases[0] (owner 2) lacks "uid")"},
        {R"([{"uid": "3", "of": 2, "byte_offset": 0}])",
         R"(aliases[0] (owner 2): "uid" is not a 64-bit signed integer)"},
        {R"([{"uid": 3, "of": 2.5, "byte_offset": 0}])",
         R"(aliases[0] (view 3): "of" is not a 64-bit signed integer)"},
        {R"([{"uid": 3, "of": 2, "byte_offset": 9223372036854775808}])",
         R"(aliases[0] (view 3, owner 2): "byte_offset" is not a 64-bit signed integer)"},
        {R"([{"uid": 3, "of": 2, "byte_offset": 0, "stride": 1}])",
         R"(aliases[0] (view 3, owner 2) has unknown key "stride")"},
        {R"([{"uid": 3, "of": 2, "byte_offset": -4}])",
         "alias view 3 of owner 2: byte_offset -4 is negative"},
        {R"([{"uid": 3, "of": 3, "byte_offset": 0}])",
         "alias view 3 of owner 3: the view and the owner are one tensor"},
        {R"([{"uid": 3, "of": 2, "byte_offset": 0}, {"uid": 3, "of": 1, "byte_offset": 0}])",
         "alias view 3 of owner 1: the view appears in more than one entry"},
        {R"([{"uid": 3, "of": 2, "byte_offset": 0}, {"uid": 4, "of": 3, "byte_offset": 0}])",
         "alias view 3 of owner 2: the view is also the owner of view 4"},
    };
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.aliases);
        const auto error
            = metadataError(std::string(R"({"format_version": 2, "aliases": )") + c.aliases + "}");
        EXPECT_NE(error.find(std::string("case_0: ") + c.rule), std::string::npos) << error;
    }
}

TEST(TestParseBundleMetadataJson, ThrowsOnAFillKindForAView)
{
    const auto error = metadataError(
        R"({"format_version": 2, "aliases": [{"uid": 3, "of": 2, "byte_offset": 0}],
            "inputs": {"3": {"kind": "fixed", "value": 1}}})");
    EXPECT_NE(
        error.find("case_0: alias view 3 of owner 2: the view's inputs entry sets a fill kind"),
        std::string::npos)
        << error;
}

TEST(TestParseBundleMetadataJson, SeedOnlyInputForAViewLoads)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 2, "aliases": [{"uid": 3, "of": 2, "byte_offset": 0}],
            "inputs": {"3": {"seed": 7}, "2": {"kind": "fixed", "value": 1}}})"));
    ASSERT_EQ(meta.aliases.size(), 1u);
    ASSERT_TRUE(meta.inputs.has_value());
    EXPECT_EQ(meta.inputs->at(3).at("seed").get<int>(), 7);
}

TEST(TestLoadBundleMetadata, AliasesInAVersion1FileThrows)
{
    const TempBundle bundle(
        R"({"format_version": 1, "aliases": [{"uid": 3, "of": 2, "byte_offset": 0}]})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

// ---------------------------------------------------------------------------
// start_offsets
// ---------------------------------------------------------------------------

TEST(TestParseBundleMetadataJson, StartOffsetsParseInFormatVersion2)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 2, "start_offsets": [{"uid": 3, "start_offset": 2},
                                                    {"uid": 7, "start_offset": 255}]})"));
    ASSERT_EQ(meta.startOffsets.size(), 2u);
    EXPECT_EQ(meta.startOffsets[0], (TensorStartOffset{3, 2}));
    EXPECT_EQ(meta.startOffsets[1], (TensorStartOffset{7, 255}));
}

TEST(TestParseBundleMetadataJson, StartOffsetsAbsentLeavesTheListEmpty)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(R"({"format_version": 2})"));
    EXPECT_TRUE(meta.startOffsets.empty());
}

TEST(TestParseBundleMetadataJson, ThrowsOnStartOffsetsInFormatVersion1)
{
    const auto error = metadataError(
        R"({"format_version": 1, "start_offsets": [{"uid": 3, "start_offset": 2}]})");
    EXPECT_NE(error.find("case_0: start_offsets needs format_version 2"), std::string::npos)
        << error;
}

// The version check wins over any problem in the value.
TEST(TestParseBundleMetadataJson, ThrowsOnMalformedStartOffsetsInFormatVersion1)
{
    for(const auto* startOffsets :
        {"[]", "3", R"([{"uid": 3}])", R"([{"uid": 3, "start_offset": 0}])"})
    {
        SCOPED_TRACE(startOffsets);
        const auto error = metadataError(std::string(R"({"format_version": 1, "start_offsets": )")
                                         + startOffsets + "}");
        EXPECT_NE(error.find("case_0: start_offsets needs format_version 2"), std::string::npos)
            << error;
    }
}

TEST(TestParseBundleMetadataJson, ThrowsOnMalformedStartOffsets)
{
    struct Case
    {
        const char* startOffsets;
        const char* rule;
    };
    const std::vector<Case> cases = {
        {R"({"uid": 3, "start_offset": 2})", "start_offsets is not an array"},
        {"[]", "start_offsets is empty"},
        {"[3]", "start_offsets[0] is not an object"},
        {R"([{"uid": 3}])", R"(start_offsets[0] (tensor 3) lacks "start_offset")"},
        {R"([{"start_offset": 2}])", R"(start_offsets[0] lacks "uid")"},
        {R"([{"uid": "3", "start_offset": 2}])",
         R"(start_offsets[0]: "uid" is not a 64-bit signed integer)"},
        {R"([{"uid": 3, "start_offset": 2.5}])",
         R"(start_offsets[0] (tensor 3): "start_offset" is not a 64-bit signed integer)"},
        {R"([{"uid": 3, "start_offset": 9223372036854775808}])",
         R"(start_offsets[0] (tensor 3): "start_offset" is not a 64-bit signed integer)"},
        {R"([{"uid": 3, "start_offset": 2, "alignment": 1}])",
         R"(start_offsets[0] (tensor 3) has unknown key "alignment")"},
        {R"([{"uid": 3, "start_offset": 0}])",
         "start offset of tensor 3: start_offset 0 is not in the range 1 to 255"},
        {R"([{"uid": 3, "start_offset": -1}])",
         "start offset of tensor 3: start_offset -1 is not in the range 1 to 255"},
        {R"([{"uid": 3, "start_offset": 256}])",
         "start offset of tensor 3: start_offset 256 is not in the range 1 to 255"},
        {R"([{"uid": 3, "start_offset": 2}, {"uid": 3, "start_offset": 4}])",
         "start offset of tensor 3: the tensor appears in more than one entry"},
    };
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.startOffsets);
        const auto error = metadataError(std::string(R"({"format_version": 2, "start_offsets": )")
                                         + c.startOffsets + "}");
        EXPECT_NE(error.find(std::string("case_0: ") + c.rule), std::string::npos) << error;
    }
}

TEST(TestLoadBundleMetadata, StartOffsetsInAVersion1FileThrows)
{
    const TempBundle bundle(
        R"({"format_version": 1, "start_offsets": [{"uid": 3, "start_offset": 2}]})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, StartOffsetsInAVersion2FileLoad)
{
    const TempBundle bundle(
        R"({"format_version": 2, "start_offsets": [{"uid": 3, "start_offset": 1}]})");
    const auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->startOffsets, (std::vector<TensorStartOffset>{{3, 1}}));
}

// ---------------------------------------------------------------------------
// live_rows
// ---------------------------------------------------------------------------

TEST(TestParseBundleMetadataJson, LiveRowsParseInFormatVersion2)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(
        R"({"format_version": 2, "live_rows": [{"uid": 9, "axis": 2, "lengths": 14},
                                                {"uid": 10, "like": 1}]})"));
    ASSERT_EQ(meta.liveRows.size(), 2u);
    EXPECT_EQ(meta.liveRows[0], (OutputLiveRows{9, 2, 14, std::nullopt}));
    EXPECT_EQ(meta.liveRows[1], (OutputLiveRows{10, std::nullopt, std::nullopt, 1}));
}

TEST(TestParseBundleMetadataJson, LiveRowsAbsentLeavesTheListEmpty)
{
    const auto meta = parseBundleMetadataJson(nlohmann::json::parse(R"({"format_version": 2})"));
    EXPECT_TRUE(meta.liveRows.empty());
}

// The version check wins over any problem in the value.
TEST(TestParseBundleMetadataJson, ThrowsOnLiveRowsInFormatVersion1)
{
    for(const auto* liveRows :
        {R"([{"uid": 9, "axis": 2, "lengths": 14}])", "[]", "3", R"([{"uid": 9}])"})
    {
        SCOPED_TRACE(liveRows);
        const auto error
            = metadataError(std::string(R"({"format_version": 1, "live_rows": )") + liveRows + "}");
        EXPECT_NE(error.find("case_0: live_rows needs format_version 2"), std::string::npos)
            << error;
    }
}

TEST(TestParseBundleMetadataJson, ThrowsOnMalformedLiveRows)
{
    struct Case
    {
        const char* liveRows;
        const char* rule;
    };
    const std::vector<Case> cases = {
        {R"({"uid": 9, "like": 1})", "live_rows is not an array"},
        {"[]", "live_rows is empty"},
        {"[9]", "live_rows[0] is not an object"},
        {R"([{"axis": 2, "lengths": 14}])", R"(live_rows[0] lacks "uid")"},
        {R"([{"uid": "9", "like": 1}])", R"(live_rows[0]: "uid" is not a 64-bit signed integer)"},
        {R"([{"uid": 9, "axis": 2.0, "lengths": 14}])",
         R"(live_rows[0] (output 9): "axis" is not a 64-bit signed integer)"},
        {R"([{"uid": 9, "axis": 2, "lengths": "14"}])",
         R"(live_rows[0] (output 9): "lengths" is not a 64-bit signed integer)"},
        {R"([{"uid": 9, "like": true}])",
         R"(live_rows[0] (output 9): "like" is not a 64-bit signed integer)"},
        {R"([{"uid": 9, "like": 1, "seq_axis": 2}])",
         R"(live_rows[0] (output 9) has unknown key "seq_axis")"},
        {R"([{"uid": 9}])", R"(live_rows[0] (output 9) needs exactly one of "lengths" and "like")"},
        {R"([{"uid": 9, "axis": 2}])",
         R"(live_rows[0] (output 9) needs exactly one of "lengths" and "like")"},
        {R"([{"uid": 9, "axis": 2, "lengths": 14, "like": 1}])",
         R"(live_rows[0] (output 9) needs exactly one of "lengths" and "like")"},
        {R"([{"uid": 9, "lengths": 14}])",
         R"(live_rows[0] (output 9) has "lengths" and lacks "axis")"},
        {R"([{"uid": 9, "axis": 2, "like": 1}])",
         R"(live_rows[0] (output 9) has "like" and "axis")"},
        {R"([{"uid": 9, "like": 1}, {"uid": 9, "axis": 2, "lengths": 14}])",
         "live rows of output 9: the output appears in more than one entry"},
    };
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.liveRows);
        const auto error = metadataError(std::string(R"({"format_version": 2, "live_rows": )")
                                         + c.liveRows + "}");
        EXPECT_NE(error.find(std::string("case_0: ") + c.rule), std::string::npos) << error;
    }
}

TEST(TestLoadBundleMetadata, LiveRowsInAVersion1FileThrows)
{
    const TempBundle bundle(R"({"format_version": 1, "live_rows": [{"uid": 9, "like": 1}]})");
    EXPECT_THROW(loadBundleMetadata(bundle.bundleJsonPath()), BundleMetadataError);
}

TEST(TestLoadBundleMetadata, LiveRowsInAVersion2FileLoad)
{
    const TempBundle bundle(
        R"({"format_version": 2, "live_rows": [{"uid": 9, "axis": 1, "lengths": 3}]})");
    const auto meta = loadBundleMetadata(bundle.bundleJsonPath());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->liveRows, (std::vector<OutputLiveRows>{{9, 1, 3, std::nullopt}}));
}

// NOLINTEND(readability-identifier-naming)
