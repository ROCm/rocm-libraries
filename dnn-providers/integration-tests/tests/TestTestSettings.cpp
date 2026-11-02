// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "harness/TestSettings.hpp"
#include "harness/TomlGuards.hpp"

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

// ---------------------------------------------------------------------------
// [[validator_overrides]] atol / rtol — one tensor's tolerance
//
// A [[tolerance_overrides]] entry relaxes every output of a test. These keys are the way to
// relax one output of a graph, so what they must not touch matters as much as what they set.
// ---------------------------------------------------------------------------

namespace
{

// The scoping the gauntlet entries of an engine TOML use: its suites carry "gauntlet" in the
// name (run_gauntlet_fuzz.py, `gauntlet_<Op>_<Space>_<tag>_cases.s<seed>`); quick-bundle
// tests are `quick_<Op>_<Variant>.<case id>` (BundleDiscovery.hpp deriveSuiteName).
const char* const K_GAUNTLET_DX_CONFIG = R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*BatchnormBackward*"]
atol = 1e-1
rtol = 2e-1

[[validator_overrides]]
filters = ["*gauntlet*BatchnormBackward*"]
tensors = ["*::DX"]
validator = "allclose"
atol = 4e-4
rtol = 2e-3
)";

const char* const K_GAUNTLET_TEST = "gauntlet_BatchnormBackward_Default_fp16_cases.s1234";
const char* const K_QUICK_TEST = "quick_BatchnormBackward_Variant2.2_32_8_8_fp16_nchw_train";

// What the parser said about a validator entry, or "" when it loaded.
std::string validatorEntryLoadError(const std::string& entryBody)
{
    const TempTomlFile file("[meta]\nversion = 1\n\n[[validator_overrides]]\n"
                            "filters = [\"*gauntlet*BatchnormBackward*\"]\n"
                            "tensors = [\"*::DX\"]\n"
                            + entryBody);
    try
    {
        const TestSettings settings(file.path());
    }
    catch(const std::runtime_error& e)
    {
        return e.what();
    }
    return "";
}

} // namespace

TEST(TestSettingsValidatorOverrides, AtolAndRtolApplyOnlyToTheMatchedTensor)
{
    const TempTomlFile file(K_GAUNTLET_DX_CONFIG);
    const TestSettings settings(file.path());

    const auto dx = settings.findValidatorOverride(K_GAUNTLET_TEST, "BatchnormBackward_0::DX");
    ASSERT_TRUE(dx.has_value());
    EXPECT_EQ(dx->kind, hipdnn_integration_tests::ValidatorOverrideKind::ALLCLOSE);
    ASSERT_TRUE(dx->atol.has_value());
    ASSERT_TRUE(dx->rtol.has_value());
    EXPECT_FLOAT_EQ(*dx->atol, 4e-4F);
    EXPECT_FLOAT_EQ(*dx->rtol, 2e-3F);

    EXPECT_FALSE(
        settings.findValidatorOverride(K_GAUNTLET_TEST, "BatchnormBackward_0::DSCALE").has_value())
        << "a sibling output of the same graph keeps the default tolerance";
    EXPECT_FALSE(
        settings.findValidatorOverride(K_GAUNTLET_TEST, "BatchnormBackward_0::DBIAS").has_value());
}

// The entries are scoped to the gauntlet so a checked-in bundle is graded as before.
TEST(TestSettingsValidatorOverrides, GauntletScopedEntryDoesNotMatchQuickBundleNames)
{
    const TempTomlFile file(K_GAUNTLET_DX_CONFIG);
    const TestSettings settings(file.path());

    EXPECT_TRUE(
        settings.findValidatorOverride(K_GAUNTLET_TEST, "BatchnormBackward_0::DX").has_value());
    EXPECT_FALSE(
        settings.findValidatorOverride(K_QUICK_TEST, "BatchnormBackward_0::DX").has_value());
    EXPECT_FALSE(settings
                     .findValidatorOverride("full_BatchnormBackward_Default.4096_8_8_8_fp32_nchw",
                                            "BatchnormBackward_0::DX")
                     .has_value());
    EXPECT_FALSE(settings
                     .findValidatorOverride("gauntlet_SdpaFwd_Default_bfp16_cases.s1",
                                            "BatchnormBackward_0::DX")
                     .has_value())
        << "another operation's gauntlet suite is not matched either";
}

TEST(TestSettingsValidatorOverrides, AtolWithoutRtolLeavesRtolEmpty)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*SdpaFwd*"]
tensors = ["*::O"]
validator = "allclose_matching_infinities"
atol = 0.5
)");
    const TestSettings settings(file.path());

    const auto hit = settings.findValidatorOverride("gauntlet_SdpaFwd_x.s1", "SdpaFwd_0::O");
    ASSERT_TRUE(hit.has_value());
    ASSERT_TRUE(hit->atol.has_value());
    EXPECT_FLOAT_EQ(*hit->atol, 0.5F);
    EXPECT_FALSE(hit->rtol.has_value());
}

// Zero is a legitimate tolerance (an exact-match output), so non-negative, not positive.
TEST(TestSettingsValidatorOverrides, ZeroToleranceIsAccepted)
{
    EXPECT_EQ(validatorEntryLoadError("validator = \"allclose\"\natol = 0.0\nrtol = 0.0\n"), "");
}

TEST(TestSettingsValidatorOverrides, AnEntryWithoutAtolOrRtolCarriesNeither)
{
    const TempTomlFile file(K_RMS_CONFIG);
    const TestSettings settings(file.path());

    const auto hit = settings.findValidatorOverride(
        "Smoke/IntegrationGpuLayernormBackwardPure5DFp32.Correctness/9",
        "LayernormBackward_0::DSCALE");
    ASSERT_TRUE(hit.has_value());
    EXPECT_FALSE(hit->atol.has_value());
    EXPECT_FALSE(hit->rtol.has_value());
}

// A later matching entry replaces the earlier one as a whole: a later "allclose" with no
// tolerance must put the tensor back on the default rather than inherit the earlier atol.
TEST(TestSettingsValidatorOverrides, LaterEntryWinsAsAWholeForAtolAndRtol)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[validator_overrides]]
filters = ["*gauntlet*BatchnormBackward*"]
tensors = ["*::DX"]
validator = "allclose"
atol = 4e-4
rtol = 2e-3

[[validator_overrides]]
filters = ["*gauntlet*BatchnormBackward*bfp16_*"]
tensors = ["*::DX"]
validator = "allclose"
rtol = 2e-2

[[validator_overrides]]
filters = ["*gauntlet*BatchnormBackward*fp32_*"]
tensors = ["*::DX"]
validator = "allclose"
)");
    const TestSettings settings(file.path());

    const auto half = settings.findValidatorOverride(
        "gauntlet_BatchnormBackward_Default_fp16_cases.s1", "BatchnormBackward_0::DX");
    ASSERT_TRUE(half.has_value());
    EXPECT_FLOAT_EQ(*half->atol, 4e-4F);
    EXPECT_FLOAT_EQ(*half->rtol, 2e-3F);

    const auto bf16 = settings.findValidatorOverride(
        "gauntlet_BatchnormBackward_Default_bfp16_cases.s1", "BatchnormBackward_0::DX");
    ASSERT_TRUE(bf16.has_value());
    EXPECT_FALSE(bf16->atol.has_value()) << "the later entry does not inherit the earlier atol";
    EXPECT_FLOAT_EQ(*bf16->rtol, 2e-2F);

    const auto fp32 = settings.findValidatorOverride(
        "gauntlet_BatchnormBackward_Default_fp32_cases.s1", "BatchnormBackward_0::DX");
    ASSERT_TRUE(fp32.has_value());
    EXPECT_FALSE(fp32->atol.has_value());
    EXPECT_FALSE(fp32->rtol.has_value());
}

TEST(TestSettingsValidatorOverrides, ThrowsOnNegativeAtol)
{
    EXPECT_NE(validatorEntryLoadError("validator = \"allclose\"\natol = -1e-3\n")
                  .find("'atol' must be finite and non-negative"),
              std::string::npos);
}

TEST(TestSettingsValidatorOverrides, ThrowsOnNegativeRtol)
{
    EXPECT_NE(
        validatorEntryLoadError("validator = \"allclose_matching_infinities\"\nrtol = -1e-3\n")
            .find("'rtol' must be finite and non-negative"),
        std::string::npos);
}

TEST(TestSettingsValidatorOverrides, ThrowsOnNanAtol)
{
    EXPECT_NE(validatorEntryLoadError("validator = \"allclose\"\natol = nan\n")
                  .find("'atol' must be finite and non-negative"),
              std::string::npos);
}

TEST(TestSettingsValidatorOverrides, ThrowsOnInfiniteRtol)
{
    EXPECT_NE(validatorEntryLoadError("validator = \"allclose\"\nrtol = inf\n")
                  .find("'rtol' must be finite and non-negative"),
              std::string::npos);
}

// 1e300 is a finite double but not a float: narrowing it would silently grade with infinity.
TEST(TestSettingsValidatorOverrides, ThrowsOnAtolBeyondFloatRange)
{
    EXPECT_NE(validatorEntryLoadError("validator = \"allclose\"\natol = 1e300\n")
                  .find("'atol' must be finite and non-negative"),
              std::string::npos);
}

TEST(TestSettingsValidatorOverrides, ThrowsOnNonNumericAtol)
{
    EXPECT_NE(validatorEntryLoadError("validator = \"allclose\"\natol = \"4e-4\"\n")
                  .find("'atol' must be a number"),
              std::string::npos);
}

// An RMS comparison reads neither tolerance, so an entry that sets one does not do what it
// says; it is a load error rather than a tolerance that is silently ignored.
TEST(TestSettingsValidatorOverrides, ThrowsOnRmsWithAtol)
{
    EXPECT_NE(validatorEntryLoadError("validator = \"rms\"\nrms_threshold = 1e-3\natol = 1e-3\n")
                  .find("also sets 'atol'"),
              std::string::npos);
}

TEST(TestSettingsValidatorOverrides, ThrowsOnRmsWithRtol)
{
    EXPECT_NE(validatorEntryLoadError("validator = \"rms\"\nrms_threshold = 1e-3\nrtol = 1e-3\n")
                  .find("also sets 'rtol'"),
              std::string::npos);
}

// ---------------------------------------------------------------------------
// What the harness grades a tensor with: the parsed entries through gradingForTensor
// ---------------------------------------------------------------------------

namespace
{

hipdnn_integration_tests::bundle::ComparisonTolerance
    gradedAs(const TestSettings& settings, const std::string& testName, const std::string& tensor)
{
    // The default the caller resolved: a stand-in for the policy value of the op and dtype.
    return hipdnn_integration_tests::gradingForTensor(
        testName,
        tensor,
        settings.findValidatorOverride(testName, tensor),
        settings.findToleranceOverride(testName),
        1e-5F,
        2e-5F);
}

} // namespace

// The point of the feature: DX takes its own atol/rtol even though a [[tolerance_overrides]]
// entry matches the test and sets different values, and DSCALE keeps that tolerance entry.
TEST(TestSettingsValidatorOverrides, ValidatorToleranceBeatsToleranceOverrideForItsTensorOnly)
{
    const TempTomlFile file(K_GAUNTLET_DX_CONFIG);
    const TestSettings settings(file.path());

    const auto dx = gradedAs(settings, K_GAUNTLET_TEST, "BatchnormBackward_0::DX");
    EXPECT_EQ(dx.kind, hipdnn_integration_tests::bundle::ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(dx.atol, 4e-4F);
    EXPECT_FLOAT_EQ(dx.rtol, 2e-3F);

    const auto dscale = gradedAs(settings, K_GAUNTLET_TEST, "BatchnormBackward_0::DSCALE");
    EXPECT_EQ(dscale.kind, hipdnn_integration_tests::bundle::ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(dscale.atol, 1e-1F);
    EXPECT_FLOAT_EQ(dscale.rtol, 2e-1F);
}

// The quick bundle of the same op still takes the tolerance entry, never the gauntlet's.
TEST(TestSettingsValidatorOverrides, QuickBundleIsGradedByTheToleranceOverrideNotTheGauntletEntry)
{
    const TempTomlFile file(K_GAUNTLET_DX_CONFIG);
    const TestSettings settings(file.path());

    const auto dx = gradedAs(settings, K_QUICK_TEST, "BatchnormBackward_0::DX");
    EXPECT_FLOAT_EQ(dx.atol, 1e-1F);
    EXPECT_FLOAT_EQ(dx.rtol, 2e-1F);
}

// Either key alone: the other one falls back to the tolerance entry, then to the default.
TEST(TestSettingsValidatorOverrides, OneKeyAloneLeavesTheOtherToTheLowerLayers)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*gauntlet_A_*"]
atol = 1e-1
rtol = 2e-1

[[validator_overrides]]
filters = ["*gauntlet_*"]
tensors = ["*::O"]
validator = "allclose_matching_infinities"
atol = 3e-3
)");
    const TestSettings settings(file.path());

    const auto withToleranceEntry = gradedAs(settings, "gauntlet_A_x.s1", "Op_0::O");
    EXPECT_EQ(withToleranceEntry.kind,
              hipdnn_integration_tests::bundle::ValidatorKind::ALLCLOSE_MATCHING_INFINITIES);
    EXPECT_FLOAT_EQ(withToleranceEntry.atol, 3e-3F);
    EXPECT_FLOAT_EQ(withToleranceEntry.rtol, 2e-1F);

    const auto withDefault = gradedAs(settings, "gauntlet_B_x.s1", "Op_0::O");
    EXPECT_FLOAT_EQ(withDefault.atol, 3e-3F);
    EXPECT_FLOAT_EQ(withDefault.rtol, 2e-5F);
}

// An entry that sets no tolerance leaves the existing tolerance path untouched.
TEST(TestSettingsValidatorOverrides, EntryWithoutToleranceLeavesTheTolerancePathUnchanged)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*gauntlet_*"]
atol = 1e-1
rtol = 2e-1

[[validator_overrides]]
filters = ["*gauntlet_*"]
tensors = ["*::O"]
validator = "allclose"
)");
    const TestSettings settings(file.path());

    const auto graded = gradedAs(settings, "gauntlet_A_x.s1", "Op_0::O");
    EXPECT_EQ(graded.kind, hipdnn_integration_tests::bundle::ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(graded.atol, 1e-1F);
    EXPECT_FLOAT_EQ(graded.rtol, 2e-1F);
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

// NOLINTEND(readability-identifier-naming)
