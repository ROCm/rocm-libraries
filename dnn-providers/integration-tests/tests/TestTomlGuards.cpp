// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "harness/TestSettings.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/ValidationSite.hpp"
#include "harness/bundle/OutputComparison.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

using hipdnn_integration_tests::applyTomlToleranceOverride;
using hipdnn_integration_tests::checkTomlSkip;
using hipdnn_integration_tests::currentTestName;
using hipdnn_integration_tests::findTomlValidatorOverride;
using hipdnn_integration_tests::gradingForTensor;
using hipdnn_integration_tests::TestSettings;
using hipdnn_integration_tests::ToleranceOverride;
using hipdnn_integration_tests::ValidatorOverride;
using hipdnn_integration_tests::ValidatorOverrideKind;
using hipdnn_integration_tests::bundle::ValidatorKind;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

namespace
{

// The engine TOML these tests grade under. Both overrides match only this suite's fp8
// tests, and only LSE is switched to allclose_matching_infinities.
TestSettings fp8OverrideSettings()
{
    const auto dir = hipdnn_test_sdk::utilities::claimScratchDirectory("toml_guards");
    const auto path = dir.path() / "settings.toml";
    {
        std::ofstream ofs(path);
        ofs << R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["TestTomlGuards.GradingForTensorFp8Output*"]
atol = 0.5
rtol = 0.25

[[validator_overrides]]
filters = ["TestTomlGuards.GradingForTensorFp8Output*"]
tensors = ["SdpaFwd_0::LSE"]
validator = "allclose_matching_infinities"
)";
    }
    return TestSettings(path);
}

} // namespace

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

// ---------------------------------------------------------------------------
// gradingForTensor — a [[tolerance_overrides]] entry on an fp8 output
//
// The settings are a local TestSettings, never the TestConfig singleton, so nothing
// here is visible to any other test.
// ---------------------------------------------------------------------------

TEST(TestTomlGuards, GradingForTensorFp8OutputTakesTheToleranceOverrideInOrder)
{
    using hipdnn_data_sdk::types::fp8_e4m3;
    using hipdnn_data_sdk::utilities::Tensor;
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    const auto fp8Step
        = hipdnn_integration_tests::tolerance::fp8OutputTolerance(DataType::FP8_E4M3);
    ASSERT_TRUE(fp8Step.has_value());

    const auto settings = fp8OverrideSettings();
    const auto testName = currentTestName();
    const std::string label = "SdpaFwd_0::O";

    const auto grading = gradingForTensor(testName,
                                          label,
                                          settings.findValidatorOverride(testName, label),
                                          settings.findToleranceOverride(testName),
                                          fp8Step->atol,
                                          fp8Step->rtol);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE);
    EXPECT_EQ(grading.atol, 0.5f);
    EXPECT_EQ(grading.rtol, 0.25f);

    const auto selection = hipdnn_integration_tests::bundle::makeValidator(
        DataType::FP8_E4M3, label, grading, hipdnn_integration_tests::ValidationSite::HOST);
    ASSERT_NE(selection.validator, nullptr) << selection.error;

    // Against a reference of 8, the override allows |diff| <= 0.5 + 0.25 * 8 = 2.5.
    // The fp8 default allows about 1, swapped values 4.25, and atol for both 4.5.
    const auto allCloseToEight = [&selection](float value) {
        Tensor<fp8_e4m3> reference(std::vector<int64_t>{1});
        Tensor<fp8_e4m3> actual(std::vector<int64_t>{1});
        reference.setHostValue(fp8_e4m3(8.0f), 0);
        actual.setHostValue(fp8_e4m3(value), 0);
        return selection.validator->allClose(reference, actual);
    };
    EXPECT_TRUE(allCloseToEight(10.0f));
    EXPECT_FALSE(allCloseToEight(12.0f));
}

TEST(TestTomlGuards, GradingForTensorFp8OutputMatchingInfinitiesTakesTheToleranceOverrideInOrder)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    const auto fp8Step
        = hipdnn_integration_tests::tolerance::fp8OutputTolerance(DataType::FP8_E4M3);
    ASSERT_TRUE(fp8Step.has_value());

    const auto settings = fp8OverrideSettings();
    const auto testName = currentTestName();
    const std::string label = "SdpaFwd_0::LSE";

    const auto grading = gradingForTensor(testName,
                                          label,
                                          settings.findValidatorOverride(testName, label),
                                          settings.findToleranceOverride(testName),
                                          fp8Step->atol,
                                          fp8Step->rtol);

    EXPECT_EQ(grading.kind, ValidatorKind::ALLCLOSE_MATCHING_INFINITIES);
    EXPECT_EQ(grading.atol, 0.5f);
    EXPECT_EQ(grading.rtol, 0.25f);
}

// NOLINTEND(readability-identifier-naming)
