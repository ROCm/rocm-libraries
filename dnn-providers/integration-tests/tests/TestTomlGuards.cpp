// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <fstream>
#include <optional>
#include <string>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "harness/TestSettings.hpp"
#include "harness/TomlGuards.hpp"

using hipdnn_integration_tests::applyTomlToleranceOverride;
using hipdnn_integration_tests::checkTomlSkip;
using hipdnn_integration_tests::currentTestName;
using hipdnn_integration_tests::findTomlToleranceOverride;
using hipdnn_integration_tests::findTomlValidatorOverride;
using hipdnn_integration_tests::gradingForTensor;
using hipdnn_integration_tests::TestSettings;
using hipdnn_integration_tests::ToleranceOverride;
using hipdnn_integration_tests::ValidatorOverride;
using hipdnn_integration_tests::ValidatorOverrideKind;
using hipdnn_integration_tests::bundle::ValidatorKind;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

// ---------------------------------------------------------------------------
// currentTestName — pure gtest, no TestConfig dependency
// ---------------------------------------------------------------------------

TEST(TestTomlGuards, NameReturnsExpectedFormat)
{
    const auto name = currentTestName();
    EXPECT_EQ(name, "TestTomlGuards.NameReturnsExpectedFormat");
}

TEST(TestTomlGuards, NameContainsDot)
{
    const auto name = currentTestName();
    EXPECT_NE(name.find('.'), std::string::npos);
}

// ---------------------------------------------------------------------------
// checkTomlSkip / applyTomlToleranceOverride — empty-name early-return path
// ---------------------------------------------------------------------------

TEST(TestTomlGuards, CheckTomlSkipReturnsNulloptForEmptyName)
{
    EXPECT_EQ(checkTomlSkip(""), std::nullopt);
}

TEST(TestTomlGuards, ApplyTomlToleranceOverrideReturnsFalseForEmptyName)
{
    float atol = 1.0f;
    float rtol = 1.0f;
    EXPECT_FALSE(applyTomlToleranceOverride("", atol, rtol));
    EXPECT_FLOAT_EQ(atol, 1.0f);
    EXPECT_FLOAT_EQ(rtol, 1.0f);
}

TEST(TestTomlGuards, FindTomlValidatorOverrideReturnsNulloptForEmptyName)
{
    EXPECT_FALSE(findTomlValidatorOverride(nullptr, "", "LayernormBackward_0::DSCALE").has_value());
}

// ---------------------------------------------------------------------------
// checkTomlSkip / applyTomlToleranceOverride — no TOML loaded
//
// TestConfig is initialized (by TestConfigInitialized in TestTestConfig.cpp,
// same binary) without a settings file, so findSkipForTest / findToleranceOverride
// return nullopt for any test name.
// ---------------------------------------------------------------------------

TEST(TestTomlGuards, CheckTomlSkipReturnsNulloptWhenNoSettings)
{
    EXPECT_EQ(checkTomlSkip("SomeTest.Name"), std::nullopt);
}

TEST(TestTomlGuards, ApplyTomlToleranceOverrideReturnsFalseWhenNoSettings)
{
    float atol = 1.0f;
    float rtol = 1.0f;
    EXPECT_FALSE(applyTomlToleranceOverride("SomeTest.Name", atol, rtol));
    EXPECT_FLOAT_EQ(atol, 1.0f);
    EXPECT_FLOAT_EQ(rtol, 1.0f);
}

// The default is allclose, and it is the absence of a matching [[validator_overrides]]
// entry that expresses it — no config, no selected validator, on either harness.
TEST(TestTomlGuards, FindTomlValidatorOverrideReturnsNulloptWhenNoSettings)
{
    EXPECT_FALSE(findTomlValidatorOverride(nullptr, "SomeTest.Name", "LayernormBackward_0::DSCALE")
                     .has_value());
}

// Both harnesses grade every output tensor through gradingForTensor, so the no-config
// answer is the contract the whole suite runs under: the caller's own tolerance, graded
// by allclose.
TEST(TestTomlGuards, GradingForTensorKeepsTheCallersToleranceWhenNoSettings)
{
    const auto grading
        = gradingForTensor("SomeTest.Name", "LayernormBackward_0::DSCALE", nullptr, 1e-3f, 2e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(grading.atol, 1e-3f);
    EXPECT_FLOAT_EQ(grading.rtol, 2e-3f);
}

// ---------------------------------------------------------------------------
// gradingForTensor — overrides resolved by the caller, no TestConfig dependency
// ---------------------------------------------------------------------------

// A matching-infinities entry still grades every finite element by atol/rtol, so with no
// [[tolerance_overrides]] entry it has to keep the tolerance the harness resolved.
TEST(TestTomlGuards, GradingForTensorMatchingInfinitiesKeepsTheCallersTolerance)
{
    const ValidatorOverride matchingInfinities{ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES,
                                               0.0f};

    const auto grading = gradingForTensor(
        "SdpaFwd.Masked", "SdpaFwd_0::LSE", matchingInfinities, std::nullopt, 1e-3f, 2e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE_MATCHING_INFINITIES);
    EXPECT_FLOAT_EQ(grading.atol, 1e-3f);
    EXPECT_FLOAT_EQ(grading.rtol, 2e-3f);
}

TEST(TestTomlGuards, GradingForTensorMatchingInfinitiesAppliesTheToleranceOverride)
{
    const ValidatorOverride matchingInfinities{ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES,
                                               0.0f};
    const ToleranceOverride tomlTolerance{5e-2f, 6e-2f};

    const auto grading = gradingForTensor(
        "SdpaFwd.Masked", "SdpaFwd_0::LSE", matchingInfinities, tomlTolerance, 1e-3f, 2e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE_MATCHING_INFINITIES);
    EXPECT_FLOAT_EQ(grading.atol, 5e-2f);
    EXPECT_FLOAT_EQ(grading.rtol, 6e-2f);
}

// The tolerance the harness resolves for a half or bfloat16 convolution-forward output is the
// dynamic one, passed here as the fallback. A [[tolerance_overrides]] entry replaces it, and
// replaces it whole: neither of the dynamic atol and rtol survives.
TEST(TestTomlGuards, GradingForTensorToleranceOverrideReplacesTheCallersTolerance)
{
    const ToleranceOverride tomlTolerance{0.5f, 0.25f};

    const auto grading = gradingForTensor("ConvolutionFwd.Fp16Case",
                                          "ConvolutionFwd_0::Y",
                                          std::nullopt,
                                          tomlTolerance,
                                          2.3e-2f,
                                          1.95e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(grading.atol, 0.5f);
    EXPECT_FLOAT_EQ(grading.rtol, 0.25f);
}

// An explicit allclose entry takes the same path as no entry, so a tolerance override still
// applies to it.
TEST(TestTomlGuards, GradingForTensorExplicitAllcloseKeepsTheToleranceOverride)
{
    const ValidatorOverride allclose{ValidatorOverrideKind::ALLCLOSE, 0.0f};
    const ToleranceOverride tomlTolerance{0.5f, 0.25f};

    const auto grading = gradingForTensor("ConvolutionFwd.Fp16Case",
                                          "ConvolutionFwd_0::Y",
                                          allclose,
                                          tomlTolerance,
                                          2.3e-2f,
                                          1.95e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(grading.atol, 0.5f);
    EXPECT_FLOAT_EQ(grading.rtol, 0.25f);
}

// A [[validator_overrides]] entry outranks a [[tolerance_overrides]] one: an rms entry grades by
// its threshold and ignores both tolerances.
TEST(TestTomlGuards, GradingForTensorRmsValidatorOutranksTheToleranceOverride)
{
    const ValidatorOverride rms{ValidatorOverrideKind::RMS, 1e-4f};
    const ToleranceOverride tomlTolerance{0.5f, 0.25f};

    const auto grading = gradingForTensor(
        "ConvolutionFwd.Fp16Case", "ConvolutionFwd_0::Y", rms, tomlTolerance, 2.3e-2f, 1.95e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::RMS);
    EXPECT_FLOAT_EQ(grading.rmsThreshold, 1e-4f);
}

// ---------------------------------------------------------------------------
// gradingForTensor with settings loaded — the overload both harnesses call
// ---------------------------------------------------------------------------

namespace
{

constexpr const char* const K_OVERRIDES = R"([meta]
version = 1

[[tolerance_overrides]]
filters = ["ConvolutionFwd.Fp16*"]
atol = 0.5
rtol = 0.25

[[validator_overrides]]
filters = ["ConvolutionFwd.Fp16Rms*"]
tensors = ["*::Y"]
validator = "rms"
rms_threshold = 1e-4
)";

class TestTomlGuardsWithSettings : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::optional<TestSettings> _settings;

    void SetUp() override
    {
        _scopedDir.emplace(hipdnn_test_sdk::utilities::claimScratchDirectory("toml_guards"));
        useSettings(K_OVERRIDES);
    }

    void useSettings(const std::string& toml)
    {
        const auto path = _scopedDir->path() / "settings.toml";
        std::ofstream(path) << toml;
        _settings.emplace(path);
    }
};

} // namespace

// The tolerance the harness resolved is the fallback, and the entry that matches the test
// replaces it whole.
TEST_F(TestTomlGuardsWithSettings, ToleranceOverrideOfTheSettingsReplacesTheCallersTolerance)
{
    const auto grading = gradingForTensor(
        "ConvolutionFwd.Fp16Case", "ConvolutionFwd_0::Y", &*_settings, 2.3e-2f, 1.95e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(grading.atol, 0.5f);
    EXPECT_FLOAT_EQ(grading.rtol, 0.25f);
}

TEST_F(TestTomlGuardsWithSettings, UnmatchedNameKeepsTheCallersTolerance)
{
    const auto grading = gradingForTensor(
        "ConvolutionFwd.Bf16Case", "ConvolutionFwd_0::Y", &*_settings, 2.3e-2f, 1.95e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(grading.atol, 2.3e-2f);
    EXPECT_FLOAT_EQ(grading.rtol, 1.95e-3f);
}

TEST_F(TestTomlGuardsWithSettings, ValidatorOverrideOfTheSettingsOutranksTheToleranceOverride)
{
    const auto grading = gradingForTensor(
        "ConvolutionFwd.Fp16RmsCase", "ConvolutionFwd_0::Y", &*_settings, 2.3e-2f, 1.95e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::RMS);
    EXPECT_FLOAT_EQ(grading.rmsThreshold, 1e-4f);
}

TEST_F(TestTomlGuardsWithSettings, ValidatorOverrideIsPerTensor)
{
    const auto grading = gradingForTensor(
        "ConvolutionFwd.Fp16RmsCase", "ConvolutionFwd_0::Z", &*_settings, 2.3e-2f, 1.95e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(grading.atol, 0.5f);
}

// An entry whose glob matches everything still does not apply to a test with no name.
TEST_F(TestTomlGuardsWithSettings, EmptyNameMatchesNoEntry)
{
    useSettings(R"([meta]
version = 1

[[tolerance_overrides]]
filters = ["*"]
atol = 0.5
rtol = 0.25

[[validator_overrides]]
filters = ["*"]
tensors = ["*"]
validator = "rms"
rms_threshold = 1e-4
)");
    ASSERT_TRUE(findTomlToleranceOverride(&*_settings, "Any.Name").has_value());
    ASSERT_TRUE(findTomlValidatorOverride(&*_settings, "Any.Name", "Any_0::Y").has_value());
    EXPECT_FALSE(findTomlToleranceOverride(&*_settings, "").has_value());
    EXPECT_FALSE(findTomlValidatorOverride(&*_settings, "", "ConvolutionFwd_0::Y").has_value());
}

// NOLINTEND(readability-identifier-naming)
