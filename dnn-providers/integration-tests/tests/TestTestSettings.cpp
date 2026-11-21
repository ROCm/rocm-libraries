// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "harness/TestSettings.hpp"

using hipdnn_integration_tests::ConvToleranceDirection;
using hipdnn_integration_tests::TestSettings;
using hipdnn_test_sdk::utilities::claimScratchDirectory;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

namespace
{

// Helper to create a temporary TOML file that is auto-deleted on destruction.
//
// The name is drawn from the shared scratch directory rather than std::rand(): an
// unseeded std::rand() yields the same sequence in every process, so two concurrent
// runs of this binary both draw the same first value and race on one file.
class TempTomlFile
{
public:
    explicit TempTomlFile(const std::string& content)
        : _dir(claimScratchDirectory("test_settings"))
        , _path(_dir.path() / "settings.toml")
    {
        std::ofstream ofs(_path);
        ofs << content;
    }

    ~TempTomlFile() = default;

    TempTomlFile(const TempTomlFile&) = delete;
    TempTomlFile& operator=(const TempTomlFile&) = delete;
    TempTomlFile(TempTomlFile&&) = delete;
    TempTomlFile& operator=(TempTomlFile&&) = delete;

    const std::filesystem::path& path() const
    {
        return _path;
    }

private:
    hipdnn_test_sdk::utilities::ScopedDirectory _dir;
    std::filesystem::path _path;
};

} // namespace

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, ParsesValidTomlWithOverrides)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 1e-3
rtol = 1e-2

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp32*"]
atol = 1e-5
rtol = 1e-4
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.toleranceOverrideCount(), 2U);
}

TEST(TestSettingsParser, ParsesValidTomlWithNoOverrides)
{
    const TempTomlFile file(R"(
[meta]
version = 1
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.toleranceOverrideCount(), 0U);
}

TEST(TestSettingsParser, ThrowsOnMissingVersion)
{
    const TempTomlFile file(R"(
[meta]
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnUnsupportedVersion)
{
    const TempTomlFile file(R"(
[meta]
version = 99
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnMissingFilters)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
atol = 1e-3
rtol = 1e-2
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnMissingAtol)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*test*"]
rtol = 1e-2
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnMissingRtol)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*test*"]
atol = 1e-3
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnNonexistentFile)
{
    EXPECT_THROW(const TestSettings settings("/nonexistent/path.toml"), std::exception);
}

// ---------------------------------------------------------------------------
// Filter matching
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, FindOverrideMatchesWildcard)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 1e-3
rtol = 1e-2
)");

    const TestSettings settings(file.path());

    auto result = settings.findToleranceOverride(
        "IntegrationGpuConvFwd2dFp16/Smoke.Correctness/NCHW_params");
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->atol, 1e-3F);
    EXPECT_FLOAT_EQ(result->rtol, 1e-2F);
}

TEST(TestSettingsParser, FindOverrideReturnsNulloptWhenNoMatch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 1e-3
rtol = 1e-2
)");

    const TestSettings settings(file.path());

    auto result
        = settings.findToleranceOverride("IntegrationGpuBatchnormFp32/Smoke.Correctness/params");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindOverrideReturnsNulloptWhenNoOverrides)
{
    const TempTomlFile file(R"(
[meta]
version = 1
)");

    const TestSettings settings(file.path());

    auto result = settings.findToleranceOverride("AnyTestName");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindOverrideMatchesMultipleFiltersInEntry)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*Fp16*", "*Half*"]
atol = 1e-3
rtol = 1e-2
)");

    const TestSettings settings(file.path());

    // Should match the first filter
    auto result1
        = settings.findToleranceOverride("IntegrationGpuConvFwd2dFp16/Smoke.Correctness/params");
    ASSERT_TRUE(result1.has_value());

    // Should match the second filter
    auto result2
        = settings.findToleranceOverride("IntegrationGpuConvFwdHalf/Smoke.Correctness/params");
    ASSERT_TRUE(result2.has_value());
}

// ---------------------------------------------------------------------------
// Precedence: later entries win
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, LaterEntriesTakePrecedence)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*"]
atol = 1e-3
rtol = 1e-2

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 5e-3
rtol = 5e-2
)");

    const TestSettings settings(file.path());

    // Matches both entries - the later (more specific) one should win
    auto result
        = settings.findToleranceOverride("IntegrationGpuConvFwd2dFp16/Smoke.Correctness/params");
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->atol, 5e-3F);
    EXPECT_FLOAT_EQ(result->rtol, 5e-2F);
}

TEST(TestSettingsParser, EarlierEntryUsedWhenLaterDoesNotMatch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*"]
atol = 1e-3
rtol = 1e-2

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 5e-3
rtol = 5e-2
)");

    const TestSettings settings(file.path());

    // Matches only the first entry (Fp32, not Fp16)
    auto result
        = settings.findToleranceOverride("IntegrationGpuConvFwd2dFp32/Smoke.Correctness/params");
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->atol, 1e-3F);
    EXPECT_FLOAT_EQ(result->rtol, 1e-2F);
}

// ---------------------------------------------------------------------------
// [[validator_overrides]]
//
// allclose is the default everywhere and this section is the only thing that can
// change it, so the parser is strict: an entry that does not say exactly what it
// means is a load error, not a silent fall-back to allclose.
// ---------------------------------------------------------------------------

namespace
{

const char* const K_RMS_CONFIG = R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*LayernormBackward*"]
tensors = ["*::DSCALE", "*::DBIAS"]
validator = "rms"
rms_threshold = 1e-4
)";

} // namespace

TEST(TestSettingsValidatorOverrides, NoOverridesMeansNoOverride)
{
    const TempTomlFile file(R"(
[meta]
version = 1
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.validatorOverrideCount(), 0U);
    EXPECT_FALSE(settings.findValidatorOverride("AnyTest", "AnyTensor").has_value());
}

TEST(TestSettingsValidatorOverrides, MatchesOnBothNameAndTensorGlob)
{
    const TempTomlFile file(K_RMS_CONFIG);
    const TestSettings settings(file.path());

    ASSERT_EQ(settings.validatorOverrideCount(), 1U);

    const auto hit = settings.findValidatorOverride(
        "Smoke/IntegrationGpuLayernormBackwardPure5DFp32.Correctness/9",
        "LayernormBackward_0::DSCALE");
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->kind, hipdnn_integration_tests::ValidatorOverrideKind::RMS);
    EXPECT_FLOAT_EQ(hit->rmsThreshold, 1e-4F);
}

// The tensor glob is what keeps dx on allclose while dscale moves to RMS. Matching
// on the test name alone would take the whole graph with it.
TEST(TestSettingsValidatorOverrides, OtherTensorsInTheSameCaseAreUnaffected)
{
    const TempTomlFile file(K_RMS_CONFIG);
    const TestSettings settings(file.path());

    EXPECT_FALSE(
        settings
            .findValidatorOverride("Smoke/IntegrationGpuLayernormBackwardPure5DFp32.Correctness/9",
                                   "LayernormBackward_0::DX")
            .has_value());
}

TEST(TestSettingsValidatorOverrides, OtherSuitesWithTheSameTensorAreUnaffected)
{
    const TempTomlFile file(K_RMS_CONFIG);
    const TestSettings settings(file.path());

    EXPECT_FALSE(
        settings
            .findValidatorOverride("Smoke/IntegrationGpuRMSNormBackwardPureFp32.Correctness/0",
                                   "RMSNormBackward_0::DSCALE")
            .has_value());
}

TEST(TestSettingsValidatorOverrides, LaterEntriesTakePrecedence)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*LayernormBackward*"]
tensors = ["*::DSCALE"]
validator = "rms"
rms_threshold = 1e-4

[[validator_overrides]]
filters = ["*LayernormBackward*Bfp16*"]
tensors = ["*::DSCALE"]
validator = "allclose"
)");

    const TestSettings settings(file.path());

    const auto fp32
        = settings.findValidatorOverride("Smoke/LayernormBackwardFp32.C/0", "x::DSCALE");
    ASSERT_TRUE(fp32.has_value());
    EXPECT_EQ(fp32->kind, hipdnn_integration_tests::ValidatorOverrideKind::RMS);

    const auto bf16
        = settings.findValidatorOverride("Smoke/LayernormBackwardBfp16.C/0", "x::DSCALE");
    ASSERT_TRUE(bf16.has_value());
    EXPECT_EQ(bf16->kind, hipdnn_integration_tests::ValidatorOverrideKind::ALLCLOSE);
}

TEST(TestSettingsValidatorOverrides, ParsesAllcloseMatchingInfinities)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*SdpaFwd*"]
tensors = ["*::LSE"]
validator = "allclose_matching_infinities"
)");

    const TestSettings settings(file.path());
    ASSERT_EQ(settings.validatorOverrideCount(), 1U);

    const auto hit
        = settings.findValidatorOverride("Smoke/SdpaFwdPure4D.Correctness/0", "Sdpa_0::LSE");
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->kind,
              hipdnn_integration_tests::ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES);
    // Nothing else on the entry carries a number: this kind takes its atol/rtol from
    // the tolerance the harness resolved, so rms_threshold stays zero.
    EXPECT_FLOAT_EQ(hit->rmsThreshold, 0.0F);

    EXPECT_FALSE(settings.findValidatorOverride("Smoke/SdpaFwdPure4D.Correctness/0", "Sdpa_0::O")
                     .has_value())
        << "the 'tensors' glob is what keeps the other outputs of the same graph on allclose";
    EXPECT_FALSE(settings.findValidatorOverride("Smoke/ConvFwdPure4D.Correctness/0", "Conv_0::LSE")
                     .has_value())
        << "the 'filters' glob is what keeps other tests out";
}

TEST(TestSettingsValidatorOverrides, AllcloseMatchingInfinitiesLaterEntryTakesPrecedence)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*SdpaFwd*"]
tensors = ["*::LSE"]
validator = "rms"
rms_threshold = 1e-4

[[validator_overrides]]
filters = ["*SdpaFwd*Masked*"]
tensors = ["*::LSE"]
validator = "allclose_matching_infinities"
)");

    const TestSettings settings(file.path());

    const auto unmasked = settings.findValidatorOverride("Smoke/SdpaFwdPure4D.C/0", "Sdpa_0::LSE");
    ASSERT_TRUE(unmasked.has_value());
    EXPECT_EQ(unmasked->kind, hipdnn_integration_tests::ValidatorOverrideKind::RMS);

    const auto masked
        = settings.findValidatorOverride("Smoke/SdpaFwdMaskedPure4D.C/0", "Sdpa_0::LSE");
    ASSERT_TRUE(masked.has_value());
    EXPECT_EQ(masked->kind,
              hipdnn_integration_tests::ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES);
}

TEST(TestSettingsValidatorOverrides, ThrowsOnMissingTensors)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*LayernormBackward*"]
validator = "rms"
rms_threshold = 1e-4
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

// This entry is wrong twice over: the name is not one the parser knows, and it carries
// a threshold no non-rms validator may carry. The name is the one to report: an
// operator told to delete 'rms_threshold' would be sent to the line that is not the
// mistake.
TEST(TestSettingsValidatorOverrides, ThrowsOnUnknownValidatorKind)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*LayernormBackward*"]
tensors = ["*::DSCALE"]
validator = "relative-rms"
rms_threshold = 1e-4
)");

    try
    {
        const TestSettings settings(file.path());
        FAIL() << "an unknown validator name must be a load error";
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("unknown validator"), std::string::npos)
            << "reported the wrong line: " << e.what();
    }
}

TEST(TestSettingsValidatorOverrides, ThrowsOnRmsWithoutThreshold)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*LayernormBackward*"]
tensors = ["*::DSCALE"]
validator = "rms"
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsValidatorOverrides, ThrowsOnNonPositiveRmsThreshold)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*LayernormBackward*"]
tensors = ["*::DSCALE"]
validator = "rms"
rms_threshold = 0.0
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

// An allclose entry carrying a threshold means the file no longer says what its
// author meant — 'validator' was edited and 'rms_threshold' left behind, or the
// reverse. Either way, the file is wrong about which check runs.
TEST(TestSettingsValidatorOverrides, ThrowsOnAllcloseWithRmsThreshold)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*LayernormBackward*"]
tensors = ["*::DSCALE"]
validator = "allclose"
rms_threshold = 1e-4
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

// The same "does not say what it means" rule covers the matching-infinities spelling:
// it takes its atol/rtol from the resolved tolerance and has no threshold of its own.
TEST(TestSettingsValidatorOverrides, ThrowsOnAllcloseMatchingInfinitiesWithRmsThreshold)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*SdpaFwd*"]
tensors = ["*::LSE"]
validator = "allclose_matching_infinities"
rms_threshold = 1e-4
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

// An entry with 'tensors' but no 'filters' reads as "apply to every test" to a human,
// and is one of the likelier hand-edit mistakes. It is a load error instead, because
// a validator override is scoped to the tests whose numerics justified it.
TEST(TestSettingsValidatorOverrides, ThrowsOnMissingFilters)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
tensors = ["*::DSCALE"]
validator = "rms"
rms_threshold = 1e-4
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsValidatorOverrides, ThrowsOnEmptyFilters)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = []
tensors = ["*::DSCALE"]
validator = "rms"
rms_threshold = 1e-4
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

// A bare value where a table belongs — the shape a mistyped array-of-tables takes.
// Reading it as "no overrides" would drop the entry and quietly grade with allclose.
// The key sits above [meta] so it lands at the document root: written below the
// header it would be a key of [meta] instead, which is a different mistake.
TEST(TestSettingsValidatorOverrides, ThrowsOnNonTableEntry)
{
    const TempTomlFile file(R"(
validator_overrides = ["*LayernormBackward*"]

[meta]
version = 1
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

// ---------------------------------------------------------------------------
// [[test_skips]] parsing
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, ParsesValidTomlWithSkips)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx1100", "gfx1101"]
filters = ["*Activation*Fp16*"]
reason  = "missing activation kernel on RDNA3"

[[test_skips]]
archs   = ["gfx942"]
filters = ["*ConvFwd3d*"]
reason  = "3d conv unimplemented on MI300"
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.skipEntryCount(), 2U);
}

TEST(TestSettingsParser, SkipsCoexistWithOverrides)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*Fp16*"]
atol = 1e-3
rtol = 1e-2

[[test_skips]]
archs   = ["gfx942"]
filters = ["*Conv*"]
reason  = "test"
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.toleranceOverrideCount(), 1U);
    EXPECT_EQ(settings.skipEntryCount(), 1U);
}

TEST(TestSettingsParser, ParsesSkipWithoutArchsAsGlobal)
{
    // 'archs' omitted => global skip on every arch.
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
filters = ["*KnownBroken*"]
reason  = "tracked in #1234"
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.skipEntryCount(), 1U);
}

TEST(TestSettingsParser, ParsesSkipWithEmptyArchsAsGlobal)
{
    // 'archs = []' is also a global skip.
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = []
filters = ["*KnownBroken*"]
reason  = "tracked in #1234"
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.skipEntryCount(), 1U);
}

TEST(TestSettingsParser, ThrowsOnMissingSkipFilters)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs  = ["gfx942"]
reason = "r"
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnMissingSkipReason)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*test*"]
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnEmptySkipReason)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*test*"]
reason  = ""
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

// ---------------------------------------------------------------------------
// [[test_skips]] matching
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, FindSkipMatchesArchSubstring)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*ConvFwd*"]
reason  = "no kernel"
)");

    const TestSettings settings(file.path());

    // Bare arch in TOML must match against the full ROCm-formatted device string.
    auto result
        = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942:sramecc+:xnack-", "linux");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "no kernel");
}

TEST(TestSettingsParser, FindSkipMatchesExactArchString)
{
    // A more-qualified arch string in TOML still matches when the device
    // string is identical (the substring check is symmetric for equality).
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942:xnack-"]
filters = ["*Conv*"]
reason  = "xnack-only issue"
)");

    const TestSettings settings(file.path());

    auto matched
        = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942:sramecc+:xnack-", "linux");
    ASSERT_FALSE(matched.has_value())
        << "literal substring 'gfx942:xnack-' should not appear in 'gfx942:sramecc+:xnack-'";

    auto matchedExact
        = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942:xnack-", "linux");
    ASSERT_TRUE(matchedExact.has_value());
}

TEST(TestSettingsParser, FindSkipReturnsNulloptOnArchMismatch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx1100"]
filters = ["*ConvFwd*"]
reason  = "rdna3 only"
)");

    const TestSettings settings(file.path());

    auto result
        = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942:sramecc+:xnack-", "linux");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindSkipReturnsNulloptOnFilterMismatch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*Batchnorm*"]
reason  = "batchnorm only"
)");

    const TestSettings settings(file.path());

    auto result
        = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942:sramecc+:xnack-", "linux");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindSkipArchScopedReturnsNulloptOnEmptyDeviceArch)
{
    // Arch-scoped rule + empty device arch => substring search fails, no skip.
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*ConvFwd*"]
reason  = "r"
)");

    const TestSettings settings(file.path());

    auto result = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "", "linux");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindSkipGlobalRuleMatchesAnyArch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
filters = ["*KnownBroken*"]
reason  = "global"
)");

    const TestSettings settings(file.path());

    // Global rule fires on any arch.
    EXPECT_TRUE(
        settings.findSkip("KnownBroken.Test", "gfx942:sramecc+:xnack-", "linux").has_value());
    EXPECT_TRUE(settings.findSkip("KnownBroken.Test", "gfx1100", "linux").has_value());
    EXPECT_TRUE(settings.findSkip("KnownBroken.Test", "anything", "linux").has_value());
    // ...even when the device arch could not be detected.
    EXPECT_TRUE(settings.findSkip("KnownBroken.Test", "", "linux").has_value());

    // But filter still has to match.
    EXPECT_FALSE(settings.findSkip("Other.Test", "gfx942", "linux").has_value());
}

TEST(TestSettingsParser, FindSkipReturnsNulloptWhenNoSkipsConfigured)
{
    const TempTomlFile file(R"(
[meta]
version = 1
)");

    const TestSettings settings(file.path());

    auto result = settings.findSkip("AnyTest.Name", "gfx942", "linux");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindSkipFirstMatchWins)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*ConvFwd*"]
reason  = "first"

[[test_skips]]
archs   = ["gfx942"]
filters = ["*ConvFwd*Fp16*"]
reason  = "second more specific"
)");

    const TestSettings settings(file.path());

    auto result = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942", "linux");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "first");
}

TEST(TestSettingsParser, FindSkipMatchesAnyArchInList)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx1100", "gfx1101", "gfx942"]
filters = ["*ConvFwd*"]
reason  = "broad arch list"
)");

    const TestSettings settings(file.path());

    EXPECT_TRUE(
        settings.findSkip("IntegrationGpuConvFwd.Test", "gfx942:xnack-", "linux").has_value());
    EXPECT_TRUE(settings.findSkip("IntegrationGpuConvFwd.Test", "gfx1100", "linux").has_value());
    EXPECT_FALSE(settings.findSkip("IntegrationGpuConvFwd.Test", "gfx906", "linux").has_value());
}

// ---------------------------------------------------------------------------
// [[test_skips]] platforms
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, ParsesSkipWithPlatforms)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs     = ["gfx1100"]
platforms = ["windows"]
filters   = ["*Foo*"]
reason    = "windows + RDNA3 only"
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.skipEntryCount(), 1U);
}

TEST(TestSettingsParser, FindSkipMatchesPlatform)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
platforms = ["windows"]
filters   = ["*Foo*"]
reason    = "windows-only"
)");

    const TestSettings settings(file.path());

    auto winMatch = settings.findSkip("Foo.Test", "gfx942", "windows");
    ASSERT_TRUE(winMatch.has_value());
    EXPECT_EQ(*winMatch, "windows-only");

    auto linuxMiss = settings.findSkip("Foo.Test", "gfx942", "linux");
    EXPECT_FALSE(linuxMiss.has_value());
}

TEST(TestSettingsParser, FindSkipMatchesAnyPlatformInList)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
platforms = ["windows", "linux"]
filters   = ["*Foo*"]
reason    = "broken everywhere"
)");

    const TestSettings settings(file.path());

    EXPECT_TRUE(settings.findSkip("Foo.Test", "gfx942", "windows").has_value());
    EXPECT_TRUE(settings.findSkip("Foo.Test", "gfx942", "linux").has_value());
}

TEST(TestSettingsParser, FindSkipPlatformIsExactNotSubstring)
{
    // Unlike archs, platform matching is exact equality. "win" must not match
    // "windows".
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
platforms = ["win"]
filters   = ["*Foo*"]
reason    = "should not fire"
)");

    const TestSettings settings(file.path());
    auto result = settings.findSkip("Foo.Test", "gfx942", "windows");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindSkipOmittedPlatformsMatchesAny)
{
    // No 'platforms' field => entry matches any platform (the existing
    // arch-only behavior is unchanged).
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*Foo*"]
reason  = "arch-only rule"
)");

    const TestSettings settings(file.path());

    EXPECT_TRUE(settings.findSkip("Foo.Test", "gfx942", "windows").has_value());
    EXPECT_TRUE(settings.findSkip("Foo.Test", "gfx942", "linux").has_value());
}

TEST(TestSettingsParser, FindSkipArchAndPlatformBothRequired)
{
    // When both 'archs' and 'platforms' are set, BOTH must match.
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs     = ["gfx1100"]
platforms = ["windows"]
filters   = ["*Foo*"]
reason    = "windows + gfx1100 only"
)");

    const TestSettings settings(file.path());

    // Both match
    EXPECT_TRUE(settings.findSkip("Foo.Test", "gfx1100", "windows").has_value());
    // Wrong arch
    EXPECT_FALSE(settings.findSkip("Foo.Test", "gfx942", "windows").has_value());
    // Wrong platform
    EXPECT_FALSE(settings.findSkip("Foo.Test", "gfx1100", "linux").has_value());
}

// ---------------------------------------------------------------------------
// [conv_fwd_tolerance], [conv_dgrad_tolerance], [conv_wrw_tolerance]
// ---------------------------------------------------------------------------

namespace
{

constexpr std::array<ConvToleranceDirection, 3> CONV_DIRECTIONS
    = {ConvToleranceDirection::FWD, ConvToleranceDirection::DGRAD, ConvToleranceDirection::WRW};
constexpr std::array<const char*, 3> CONV_TABLES
    = {"conv_fwd_tolerance", "conv_dgrad_tolerance", "conv_wrw_tolerance"};
constexpr std::array<const char*, 3> CONV_TYPES = {"float", "half", "bfloat16"};
// A device that no arch table names.
constexpr const char* NO_ARCH = "";

// The text of the error a settings file with this body is rejected with; fails the test when it
// loads.
std::string loadErrorOf(const std::string& body)
{
    const TempTomlFile file("[meta]\nversion = 1\n\n" + body);
    try
    {
        const TestSettings settings(file.path());
    }
    catch(const std::runtime_error& e)
    {
        return e.what();
    }
    ADD_FAILURE() << "loaded without an error:\n" << body;
    return {};
}

} // namespace

TEST(TestSettingsParser, ConvToleranceCoefficientsAreUnsetWhenTheFileSaysNothing)
{
    const TempTomlFile file(R"(
[meta]
version = 1
)");
    const TestSettings settings(file.path());
    for(const auto direction : CONV_DIRECTIONS)
    {
        for(const char* type : CONV_TYPES)
        {
            const auto c = settings.convToleranceCoefficients(direction, type, NO_ARCH);
            EXPECT_FALSE(c.summationOrder.has_value()) << type;
            EXPECT_FALSE(c.lambda.has_value()) << type;
            EXPECT_FALSE(c.lambdaPerQuarticRootK.has_value()) << type;
        }
    }
}

// A file of the form that was valid before the other directions had tables loads as it did.
TEST(TestSettingsParser, ParsesTheFloatSummationOrderOfTheForwardTable)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[conv_fwd_tolerance.float]
summation_order = 9.5
)");
    const TestSettings settings(file.path());
    const auto fwd
        = settings.convToleranceCoefficients(ConvToleranceDirection::FWD, "float", NO_ARCH);
    ASSERT_TRUE(fwd.summationOrder.has_value());
    EXPECT_DOUBLE_EQ(*fwd.summationOrder, 9.5);
    EXPECT_FALSE(fwd.lambda.has_value());
    EXPECT_FALSE(fwd.lambdaPerQuarticRootK.has_value());
    // Nothing is set for another type or another direction.
    for(const char* type : {"half", "bfloat16"})
    {
        EXPECT_FALSE(settings.convToleranceCoefficients(ConvToleranceDirection::FWD, type, NO_ARCH)
                         .summationOrder.has_value());
    }
    for(const auto direction : {ConvToleranceDirection::DGRAD, ConvToleranceDirection::WRW})
    {
        EXPECT_FALSE(settings.convToleranceCoefficients(direction, "float", NO_ARCH)
                         .summationOrder.has_value());
    }
}

// Nine tables, each with a value of its own: every direction and type reads its own and none
// another's. Catches a table stored under the wrong direction or type.
TEST(TestSettingsParser, EveryDirectionAndTypeKeepsItsOwnCoefficients)
{
    std::string body;
    double next = 1.0;
    for(const auto* table : CONV_TABLES)
    {
        for(const char* type : CONV_TYPES)
        {
            body += std::string("[") + table + "." + type + "]\n";
            if(std::string(type) == "float")
            {
                body += "summation_order = " + std::to_string(next++) + "\n";
            }
            body += "lambda = " + std::to_string(next++) + "\n";
            body += "lambda_per_quartic_root_k = " + std::to_string(next++) + "\n\n";
        }
    }
    const TempTomlFile file("[meta]\nversion = 1\n\n" + body);
    const TestSettings settings(file.path());

    double expected = 1.0;
    for(const auto direction : CONV_DIRECTIONS)
    {
        for(const char* type : CONV_TYPES)
        {
            const auto c = settings.convToleranceCoefficients(direction, type, NO_ARCH);
            if(std::string(type) == "float")
            {
                ASSERT_TRUE(c.summationOrder.has_value()) << type;
                EXPECT_DOUBLE_EQ(*c.summationOrder, expected++) << type;
            }
            else
            {
                EXPECT_FALSE(c.summationOrder.has_value()) << type;
            }
            ASSERT_TRUE(c.lambda.has_value()) << type;
            EXPECT_DOUBLE_EQ(*c.lambda, expected++) << type;
            ASSERT_TRUE(c.lambdaPerQuarticRootK.has_value()) << type;
            EXPECT_DOUBLE_EQ(*c.lambdaPerQuarticRootK, expected++) << type;
        }
    }
}

TEST(TestSettingsParser, AnOutputTypeTheFileDoesNotNameKeepsTheSdkDefaults)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[conv_dgrad_tolerance.half]
lambda = 20.0
)");
    const TestSettings settings(file.path());
    EXPECT_TRUE(settings.convToleranceCoefficients(ConvToleranceDirection::DGRAD, "half", NO_ARCH)
                    .lambda.has_value());
    for(const char* other : {"bfloat16", "float", "double"})
    {
        EXPECT_FALSE(
            settings.convToleranceCoefficients(ConvToleranceDirection::DGRAD, other, NO_ARCH)
                .lambda.has_value())
            << other;
    }
}

TEST(TestSettingsParser, AnIntegerCoefficientIsANumber)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[conv_wrw_tolerance.float]
summation_order = 8
lambda = 3
lambda_per_quartic_root_k = 2
)");
    const TestSettings settings(file.path());
    const auto c
        = settings.convToleranceCoefficients(ConvToleranceDirection::WRW, "float", NO_ARCH);
    ASSERT_TRUE(c.summationOrder.has_value());
    EXPECT_DOUBLE_EQ(*c.summationOrder, 8.0);
    ASSERT_TRUE(c.lambda.has_value());
    EXPECT_DOUBLE_EQ(*c.lambda, 3.0);
    ASSERT_TRUE(c.lambdaPerQuarticRootK.has_value());
    EXPECT_DOUBLE_EQ(*c.lambdaPerQuarticRootK, 2.0);
}

// An engine may need no allowance at all: zero is a value, not an unset one.
TEST(TestSettingsParser, ALambdaOfZeroIsSet)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[conv_dgrad_tolerance.bfloat16]
lambda = 0.0
lambda_per_quartic_root_k = 0.0
)");
    const TestSettings settings(file.path());
    const auto c
        = settings.convToleranceCoefficients(ConvToleranceDirection::DGRAD, "bfloat16", NO_ARCH);
    ASSERT_TRUE(c.lambda.has_value());
    EXPECT_DOUBLE_EQ(*c.lambda, 0.0);
    ASSERT_TRUE(c.lambdaPerQuarticRootK.has_value());
    EXPECT_DOUBLE_EQ(*c.lambdaPerQuarticRootK, 0.0);
}

TEST(TestSettingsParser, RejectsASummationOrderThatIsNotFiniteAndPositive)
{
    for(const auto* table : CONV_TABLES)
    {
        for(const char* bad : {"0", "0.0", "-1.5", "inf", "-inf", "nan"})
        {
            const auto message = loadErrorOf(std::string("[") + table
                                             + ".float]\nsummation_order = " + bad + "\n");
            EXPECT_NE(message.find(std::string(table) + ".float"), std::string::npos)
                << table << " " << bad << ": " << message;
        }
    }
}

TEST(TestSettingsParser, RejectsALambdaThatIsNotFiniteAndNotNegative)
{
    for(const auto* table : CONV_TABLES)
    {
        for(const char* type : CONV_TYPES)
        {
            for(const char* bad : {"-1.5", "-0.001", "inf", "-inf", "nan"})
            {
                for(const char* key : {"lambda", "lambda_per_quartic_root_k"})
                {
                    // The per-K key needs a lambda beside it: the rejection is of its value.
                    const std::string body
                        = std::string("[") + table + "." + type + "]\n"
                          + (std::string(key) == "lambda" ? "" : "lambda = 1.0\n") + key + " = "
                          + bad + "\n";
                    const auto message = loadErrorOf(body);
                    EXPECT_NE(message.find(std::string(table) + "." + type), std::string::npos)
                        << body << ": " << message;
                }
            }
        }
    }
}

// Each key of each table is a number; a string, a boolean, an array or a table is a load error
// that names the key.
TEST(TestSettingsParser, RejectsACoefficientOfTheWrongType)
{
    for(const auto* table : CONV_TABLES)
    {
        for(const char* key : {"summation_order", "lambda", "lambda_per_quartic_root_k"})
        {
            for(const char* bad : {R"("9")", "true", "[9.0]", "{ a = 1 }"})
            {
                const std::string body
                    = std::string("[") + table + ".float]\n"
                      + (std::string(key) == "lambda_per_quartic_root_k" ? "lambda = 1.0\n" : "")
                      + key + " = " + bad + "\n";
                const auto message = loadErrorOf(body);
                EXPECT_NE(message.find(std::string("'") + key + "' must be a number"),
                          std::string::npos)
                    << body << ": " << message;
            }
        }
    }
}

// 'summation_order' is the term of a float sum; a half or bfloat16 table has no use for it.
TEST(TestSettingsParser, RejectsASummationOrderOutsideTheFloatTable)
{
    for(const auto* table : CONV_TABLES)
    {
        for(const char* type : {"half", "bfloat16"})
        {
            const auto message
                = loadErrorOf(std::string("[") + table + "." + type + "]\nsummation_order = 2.0\n");
            EXPECT_NE(message.find(std::string(table) + "." + type), std::string::npos) << message;
        }
    }
}

// The per-K coefficient adds to a lambda; alone it would silently keep the SDK's lambda and add
// to nothing.
TEST(TestSettingsParser, RejectsAPerQuarticRootKCoefficientWithoutALambda)
{
    for(const auto* table : CONV_TABLES)
    {
        for(const char* type : CONV_TYPES)
        {
            const auto message = loadErrorOf(std::string("[") + table + "." + type
                                             + "]\nlambda_per_quartic_root_k = 1.0\n");
            EXPECT_NE(message.find("lambda_per_quartic_root_k"), std::string::npos) << message;
            EXPECT_NE(message.find(std::string(table) + "." + type), std::string::npos) << message;
        }
    }
}

TEST(TestSettingsParser, RejectsAnUnknownConvToleranceTypeOrKey)
{
    for(const auto* table : CONV_TABLES)
    {
        const std::string name(table);
        const std::vector<std::string> bad = {
            // an output type the harness does not read
            "[" + name + ".double]\nlambda = 2.0\n",
            "[" + name + ".fp16]\nlambda = 2.0\n",
            // a term it does not read
            "[" + name + ".float]\npartial_result = 12.0\n",
            "[" + name + ".half]\nlambda = 2.0\nextra = 1\n",
            // not tables
            "[" + name + "]\nfloat = 9.0\n",
        };
        for(const auto& text : bad)
        {
            const auto message = loadErrorOf(text);
            EXPECT_NE(message.find(name), std::string::npos) << text << ": " << message;
        }
    }
}

TEST(TestSettingsParser, RejectsAConvToleranceThatIsNotATable)
{
    for(const auto* table : CONV_TABLES)
    {
        const TempTomlFile file(std::string(table) + " = 9.0\n\n[meta]\nversion = 1\n");
        EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error) << table;
    }
}

// A top-level key the loader does not read is an error: a misspelt section name would otherwise
// leave its setting out of the file without a word.
TEST(TestSettingsParser, RejectsATopLevelKeyThatIsNoTable)
{
    const std::vector<std::string> bad = {
        // a misspelt section of each kind
        "[conv_fwd_tolerence.float]\nsummation_order = 9.0\n",
        "[conv_dgrad_toleranc.half]\nlambda = 9.0\n",
        "[conv_wrw_tolerance_half]\nlambda = 9.0\n",
        "[[tolerance_override]]\nfilters = [\"*\"]\natol = 1.0\nrtol = 1.0\n",
        "[[validator_override]]\nfilters = [\"*\"]\ntensors = [\"*\"]\nvalidator = \"rms\"\n"
        "rms_threshold = 1e-4\n",
        "[[test_skip]]\nfilters = [\"*\"]\nreason = \"x\"\n",
    };
    for(const auto& text : bad)
    {
        const TempTomlFile file("[meta]\nversion = 1\n\n" + text);
        EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error) << text;
    }
    // a key that is no section at all, before the first table (after [meta] it would be its)
    const TempTomlFile loose("unknown = 1\n\n[meta]\nversion = 1\n");
    EXPECT_THROW(const TestSettings settings(loose.path()), std::runtime_error);
}

TEST(TestSettingsParser, AcceptsEveryTopLevelKeyOfTheSchemaInOneFile)
{
    const TempTomlFile file(R"([meta]
version = 1

[[tolerance_overrides]]
filters = ["*"]
atol = 1.0
rtol = 1.0

[[validator_overrides]]
filters = ["*"]
tensors = ["*"]
validator = "allclose"

[[test_skips]]
filters = ["*"]
reason = "known"

[conv_fwd_tolerance.float]
summation_order = 9.0

[conv_dgrad_tolerance.half]
lambda = 20.0

[conv_wrw_tolerance.bfloat16]
lambda = 30.0
lambda_per_quartic_root_k = 1.0
)");
    const TestSettings settings(file.path());
    EXPECT_EQ(settings.toleranceOverrideCount(), 1u);
    EXPECT_EQ(settings.validatorOverrideCount(), 1u);
    EXPECT_EQ(settings.skipEntryCount(), 1u);
    EXPECT_TRUE(settings.convToleranceCoefficients(ConvToleranceDirection::FWD, "float", NO_ARCH)
                    .summationOrder.has_value());
    EXPECT_TRUE(settings.convToleranceCoefficients(ConvToleranceDirection::DGRAD, "half", NO_ARCH)
                    .lambda.has_value());
    EXPECT_TRUE(settings.convToleranceCoefficients(ConvToleranceDirection::WRW, "bfloat16", NO_ARCH)
                    .lambda.has_value());
}

// ---------------------------------------------------------------------------
// [conv_<direction>_tolerance.<type>.<arch>]: the coefficients of one architecture
// ---------------------------------------------------------------------------

namespace
{

constexpr const char* GFX942 = "gfx942:sramecc+:xnack-";
constexpr const char* GFX90A = "gfx90a:sramecc+:xnack-";

} // namespace

TEST(TestSettingsParser, AnArchTableAppliesToTheDevicesItNamesAndNoOther)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[conv_wrw_tolerance.half]
lambda = 10.0

[conv_wrw_tolerance.half.gfx90a]
lambda = 40.0
lambda_per_quartic_root_k = 2.0
)");
    const TestSettings settings(file.path());
    const auto onGfx90a
        = settings.convToleranceCoefficients(ConvToleranceDirection::WRW, "half", GFX90A);
    ASSERT_TRUE(onGfx90a.lambda.has_value());
    EXPECT_DOUBLE_EQ(*onGfx90a.lambda, 40.0);
    ASSERT_TRUE(onGfx90a.lambdaPerQuarticRootK.has_value());
    EXPECT_DOUBLE_EQ(*onGfx90a.lambdaPerQuarticRootK, 2.0);

    for(const char* other : {GFX942, "", "gfx1100"})
    {
        const auto c
            = settings.convToleranceCoefficients(ConvToleranceDirection::WRW, "half", other);
        ASSERT_TRUE(c.lambda.has_value()) << other;
        EXPECT_DOUBLE_EQ(*c.lambda, 10.0) << other;
        EXPECT_FALSE(c.lambdaPerQuarticRootK.has_value()) << other;
    }
}

// The arch table is the whole of the coefficients of its device: a key it leaves unset is the
// SDK's value there, not the output type table's.
TEST(TestSettingsParser, AnArchTableReplacesTheTypeTableWholeAndDoesNotMergeWithIt)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[conv_dgrad_tolerance.float]
summation_order = 9.0
lambda = 10.0
lambda_per_quartic_root_k = 3.0

[conv_dgrad_tolerance.float.gfx90a]
lambda = 20.0

[conv_wrw_tolerance.half]
lambda = 10.0
lambda_per_quartic_root_k = 3.0

[conv_wrw_tolerance.half.gfx90a]
)");
    const TestSettings settings(file.path());

    const auto dgrad
        = settings.convToleranceCoefficients(ConvToleranceDirection::DGRAD, "float", GFX90A);
    EXPECT_FALSE(dgrad.summationOrder.has_value());
    ASSERT_TRUE(dgrad.lambda.has_value());
    EXPECT_DOUBLE_EQ(*dgrad.lambda, 20.0);
    EXPECT_FALSE(dgrad.lambdaPerQuarticRootK.has_value());

    // An arch table that sets nothing is the SDK's values on that device.
    const auto wrw
        = settings.convToleranceCoefficients(ConvToleranceDirection::WRW, "half", GFX90A);
    EXPECT_FALSE(wrw.summationOrder.has_value());
    EXPECT_FALSE(wrw.lambda.has_value());
    EXPECT_FALSE(wrw.lambdaPerQuarticRootK.has_value());

    // The other devices keep the type table whole.
    const auto other
        = settings.convToleranceCoefficients(ConvToleranceDirection::DGRAD, "float", GFX942);
    ASSERT_TRUE(other.summationOrder.has_value());
    EXPECT_DOUBLE_EQ(*other.summationOrder, 9.0);
    ASSERT_TRUE(other.lambda.has_value());
    EXPECT_DOUBLE_EQ(*other.lambda, 10.0);
    ASSERT_TRUE(other.lambdaPerQuarticRootK.has_value());
    EXPECT_DOUBLE_EQ(*other.lambdaPerQuarticRootK, 3.0);
}

// A type table may hold only arch tables: the other devices then have no coefficients of it.
TEST(TestSettingsParser, ATypeTableOfArchTablesOnlySetsNothingOnOtherDevices)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[conv_dgrad_tolerance.half.gfx90a]
lambda = 46.0
)");
    const TestSettings settings(file.path());
    EXPECT_TRUE(settings.convToleranceCoefficients(ConvToleranceDirection::DGRAD, "half", GFX90A)
                    .lambda.has_value());
    EXPECT_FALSE(settings.convToleranceCoefficients(ConvToleranceDirection::DGRAD, "half", GFX942)
                     .lambda.has_value());
}

// Nine type tables, each with an arch table of its own value: every direction and type reads its
// own. Catches an arch table kept under the wrong direction or type.
TEST(TestSettingsParser, EveryDirectionAndTypeKeepsItsOwnArchTable)
{
    std::string body;
    double next = 1.0;
    for(const auto* table : CONV_TABLES)
    {
        for(const char* type : CONV_TYPES)
        {
            body += std::string("[") + table + "." + type
                    + ".gfx90a]\nlambda = " + std::to_string(next++) + "\n\n";
        }
    }
    const TempTomlFile file("[meta]\nversion = 1\n\n" + body);
    const TestSettings settings(file.path());

    double expected = 1.0;
    for(const auto direction : CONV_DIRECTIONS)
    {
        for(const char* type : CONV_TYPES)
        {
            const auto c = settings.convToleranceCoefficients(direction, type, GFX90A);
            ASSERT_TRUE(c.lambda.has_value()) << type;
            EXPECT_DOUBLE_EQ(*c.lambda, expected++) << type;
            EXPECT_FALSE(
                settings.convToleranceCoefficients(direction, type, GFX942).lambda.has_value())
                << type;
        }
    }
}

TEST(TestSettingsParser, AFloatArchTableMayCarryASummationOrder)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[conv_fwd_tolerance.float.gfx942]
summation_order = 12.0
)");
    const TestSettings settings(file.path());
    const auto c = settings.convToleranceCoefficients(ConvToleranceDirection::FWD, "float", GFX942);
    ASSERT_TRUE(c.summationOrder.has_value());
    EXPECT_DOUBLE_EQ(*c.summationOrder, 12.0);
}

// The arch name is matched as a substring of the raw gcnArchName, as 'archs' of [[test_skips]]
// is: a family prefix names every device of the family.
TEST(TestSettingsParser, AnArchTableNameIsMatchedAsASubstringOfTheRawArchName)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[conv_wrw_tolerance.bfloat16.gfx9]
lambda = 5.0
)");
    const TestSettings settings(file.path());
    for(const char* arch : {GFX942, GFX90A})
    {
        EXPECT_TRUE(
            settings.convToleranceCoefficients(ConvToleranceDirection::WRW, "bfloat16", arch)
                .lambda.has_value())
            << arch;
    }
    EXPECT_FALSE(
        settings.convToleranceCoefficients(ConvToleranceDirection::WRW, "bfloat16", "gfx1100")
            .lambda.has_value());
}

// Two arch tables of one output type that both match the device cannot be ordered: the lookup
// that finds out names both.
TEST(TestSettingsParser, TwoArchTablesThatBothMatchTheDeviceAreAnErrorNamingBoth)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[conv_wrw_tolerance.half.gfx90a]
lambda = 40.0

[conv_wrw_tolerance.half.gfx90]
lambda = 30.0

[conv_wrw_tolerance.bfloat16.gfx90a]
lambda = 4.0

[conv_dgrad_tolerance.half.gfx90]
lambda = 3.0
)");
    const TestSettings settings(file.path());
    try
    {
        static_cast<void>(
            settings.convToleranceCoefficients(ConvToleranceDirection::WRW, "half", GFX90A));
        ADD_FAILURE() << "two matching arch tables did not throw";
    }
    catch(const std::runtime_error& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("conv_wrw_tolerance.half.gfx90a]"), std::string::npos) << message;
        EXPECT_NE(message.find("conv_wrw_tolerance.half.gfx90]"), std::string::npos) << message;
    }
    // Only the device both match is an error: a device one of them matches, an output type with
    // one matching table, and another direction are not.
    EXPECT_NO_THROW(static_cast<void>(
        settings.convToleranceCoefficients(ConvToleranceDirection::WRW, "half", GFX942)));
    EXPECT_NO_THROW(static_cast<void>(
        settings.convToleranceCoefficients(ConvToleranceDirection::WRW, "bfloat16", GFX90A)));
    EXPECT_NO_THROW(static_cast<void>(
        settings.convToleranceCoefficients(ConvToleranceDirection::DGRAD, "half", GFX90A)));
}

TEST(TestSettingsParser, RejectsAnArchTableWithAnInvalidCoefficientNamingIt)
{
    for(const auto* table : CONV_TABLES)
    {
        const std::string name = std::string(table) + ".half.gfx90a";
        const std::vector<std::string> bad = {
            "[" + name + "]\nlambda = -1.0\n",
            "[" + name + "]\nlambda = inf\n",
            "[" + name + "]\nlambda = \"9\"\n",
            "[" + name + "]\nlambda_per_quartic_root_k = 1.0\n",
            "[" + name + "]\nlambda = 1.0\nlambda_per_quartic_root_k = -1.0\n",
            "[" + name + "]\nsummation_order = 2.0\n",
            "[" + name + "]\npartial_result = 12.0\n",
            "[" + std::string(table) + ".float.gfx90a]\nsummation_order = 0.0\n",
        };
        for(const auto& text : bad)
        {
            const auto message = loadErrorOf(text);
            EXPECT_NE(message.find(std::string(table)), std::string::npos)
                << text << ": " << message;
            EXPECT_NE(message.find("gfx90a"), std::string::npos) << text << ": " << message;
        }
    }
}

// No table nests under an arch table: the arch is the last level.
TEST(TestSettingsParser, RejectsATableNestedUnderAnArchTable)
{
    for(const auto* table : CONV_TABLES)
    {
        const std::string name = std::string(table) + ".half.gfx90a";
        const auto message
            = loadErrorOf("[" + name + "]\nlambda = 1.0\n\n[" + name + ".gfx90a]\nlambda = 2.0\n");
        EXPECT_NE(message.find(name), std::string::npos) << message;
        EXPECT_NE(message.find("nested"), std::string::npos) << message;
    }
}

TEST(TestSettingsParser, RejectsAnArchTableWithAnEmptyName)
{
    const auto message = loadErrorOf("[conv_wrw_tolerance.half.\"\"]\nlambda = 1.0\n");
    EXPECT_NE(message.find("conv_wrw_tolerance.half"), std::string::npos) << message;
}

// NOLINTEND(readability-identifier-naming)
