// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "harness/TomlGuards.hpp"

using hipdnn_integration_tests::applyTomlToleranceOverride;
using hipdnn_integration_tests::checkTomlSkip;
using hipdnn_integration_tests::currentTestName;
using hipdnn_integration_tests::findTomlValidatorOverride;
using hipdnn_integration_tests::gradingForTensor;
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
    EXPECT_FALSE(findTomlValidatorOverride("", "LayernormBackward_0::DSCALE").has_value());
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
    EXPECT_FALSE(
        findTomlValidatorOverride("SomeTest.Name", "LayernormBackward_0::DSCALE").has_value());
}

// Both harnesses grade every output tensor through gradingForTensor, so the no-config
// answer is the contract the whole suite runs under: the caller's own tolerance, graded
// by allclose.
TEST(TestTomlGuards, GradingForTensorKeepsTheCallersToleranceWhenNoSettings)
{
    const auto grading
        = gradingForTensor("SomeTest.Name", "LayernormBackward_0::DSCALE", 1e-3f, 2e-3f);

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

// Each layer replaces the one before it: the caller's default, the test's tolerance entry, then
// the validator entry's own atol/rtol. A validator entry names a tensor, so it wins.
TEST(TestTomlGuards, GradingForTensorValidatorToleranceBeatsToleranceOverride)
{
    const ValidatorOverride allclose{ValidatorOverrideKind::ALLCLOSE, 0.0f, 4e-4f, 2e-3f};
    const ToleranceOverride tomlTolerance{5e-2f, 6e-2f};

    const auto grading
        = gradingForTensor("gauntlet.s1", "Op_0::DX", allclose, tomlTolerance, 1e-5f, 2e-5f);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(grading.atol, 4e-4f);
    EXPECT_FLOAT_EQ(grading.rtol, 2e-3f);
}

TEST(TestTomlGuards, GradingForTensorValidatorAtolAloneKeepsTheOtherTolerancesRtol)
{
    const ValidatorOverride allclose{ValidatorOverrideKind::ALLCLOSE, 0.0f, 4e-4f, std::nullopt};
    const ToleranceOverride tomlTolerance{5e-2f, 6e-2f};

    const auto withTomlTolerance
        = gradingForTensor("gauntlet.s1", "Op_0::DX", allclose, tomlTolerance, 1e-5f, 2e-5f);
    EXPECT_FLOAT_EQ(withTomlTolerance.atol, 4e-4f);
    EXPECT_FLOAT_EQ(withTomlTolerance.rtol, 6e-2f);

    const auto withDefault
        = gradingForTensor("gauntlet.s1", "Op_0::DX", allclose, std::nullopt, 1e-5f, 2e-5f);
    EXPECT_FLOAT_EQ(withDefault.atol, 4e-4f);
    EXPECT_FLOAT_EQ(withDefault.rtol, 2e-5f);
}

TEST(TestTomlGuards, GradingForTensorMatchingInfinitiesTakesTheValidatorTolerance)
{
    const ValidatorOverride matchingInfinities{
        ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES, 0.0f, std::nullopt, 7e-3f};
    const ToleranceOverride tomlTolerance{5e-2f, 6e-2f};

    const auto grading = gradingForTensor(
        "gauntlet.s1", "SdpaFwd_0::LSE", matchingInfinities, tomlTolerance, 1e-3f, 2e-3f);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE_MATCHING_INFINITIES);
    EXPECT_FLOAT_EQ(grading.atol, 5e-2f);
    EXPECT_FLOAT_EQ(grading.rtol, 7e-3f);
}

// An allclose entry with no atol/rtol is the pre-existing path: the tolerance entry still wins
// over the caller's default and the entry changes nothing else.
TEST(TestTomlGuards, GradingForTensorEntryWithoutToleranceLeavesTheTolerancePathUnchanged)
{
    const ValidatorOverride allclose{ValidatorOverrideKind::ALLCLOSE, 0.0f};
    const ToleranceOverride tomlTolerance{5e-2f, 6e-2f};

    const auto withTomlTolerance
        = gradingForTensor("gauntlet.s1", "Op_0::DX", allclose, tomlTolerance, 1e-5f, 2e-5f);
    EXPECT_FLOAT_EQ(withTomlTolerance.atol, 5e-2f);
    EXPECT_FLOAT_EQ(withTomlTolerance.rtol, 6e-2f);

    const auto withDefault
        = gradingForTensor("gauntlet.s1", "Op_0::DX", allclose, std::nullopt, 1e-5f, 2e-5f);
    EXPECT_FLOAT_EQ(withDefault.atol, 1e-5f);
    EXPECT_FLOAT_EQ(withDefault.rtol, 2e-5f);
}

// RMS reads no atol/rtol, so the threshold decides whatever else the layers hold.
TEST(TestTomlGuards, GradingForTensorRmsIgnoresTheToleranceLayers)
{
    const ValidatorOverride rms{ValidatorOverrideKind::RMS, 1e-3f};
    const ToleranceOverride tomlTolerance{5e-2f, 6e-2f};

    const auto grading
        = gradingForTensor("gauntlet.s1", "Op_0::DSCALE", rms, tomlTolerance, 1e-5f, 2e-5f);

    EXPECT_EQ(grading.kind, ValidatorKind::RMS);
    EXPECT_FLOAT_EQ(grading.rmsThreshold, 1e-3f);
}

// NOLINTEND(readability-identifier-naming)
