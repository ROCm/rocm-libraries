// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "harness/TomlGuards.hpp"

using hipdnn_integration_tests::applyTomlToleranceOverride;
using hipdnn_integration_tests::checkTomlSkip;
using hipdnn_integration_tests::currentTestName;
using hipdnn_integration_tests::findTomlValidatorOverride;
using hipdnn_integration_tests::gradingForTensor;

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

    EXPECT_EQ(grading.kind, hipdnn_integration_tests::bundle::ValidatorKind::ALLCLOSE);
    EXPECT_FLOAT_EQ(grading.atol, 1e-3f);
    EXPECT_FLOAT_EQ(grading.rtol, 2e-3f);
}

// What gradingForTensor hands back when a [[validator_overrides]] entry selects the
// matching-infinities kind. Unlike rms, which zeroes atol/rtol because a threshold
// decided the verdict instead, this kind still grades every finite element by them —
// so it has to carry the tolerance the harness resolved, not discard it.
TEST(TestTomlGuards, AllCloseMatchingInfinitiesToleranceCarriesAtolAndRtol)
{
    const auto grading
        = hipdnn_integration_tests::bundle::ComparisonTolerance::allCloseMatchingInfinities(1e-3f,
                                                                                            2e-3f);

    EXPECT_EQ(grading.kind,
              hipdnn_integration_tests::bundle::ValidatorKind::ALLCLOSE_MATCHING_INFINITIES);
    EXPECT_FLOAT_EQ(grading.atol, 1e-3f);
    EXPECT_FLOAT_EQ(grading.rtol, 2e-3f);
    EXPECT_FLOAT_EQ(grading.rmsThreshold, 0.0f);
}

// NOLINTEND(readability-identifier-naming)
