// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

/**
 * @file TestGlobalKnobDefines.cpp
 * @brief Unit tests for GlobalKnobDefines.hpp: HIPDNN_FORCE_BENCHMARKING parsing.
 *
 * Deliberately outside tests/ingestor/ and NOT guarded on
 * HIPDNN_ENABLE_KERNEL_INGESTOR. The header under test is itself unguarded, because the
 * MIOpen provider reads the override in flag-off builds, so its tests must compile in
 * those builds too. Living inside the guard left the parser untested in exactly the
 * configuration that ships it.
 */
namespace
{

struct OverrideVariantCase
{
    std::string envValue;
    std::optional<bool> expected;
};

class TestBenchmarkingOverrideVariants : public ::testing::TestWithParam<OverrideVariantCase>
{
};

TEST_P(TestBenchmarkingOverrideVariants, ParsesAsExpected)
{
    const auto& testCase = GetParam();
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, testCase.envValue);

    EXPECT_EQ(hipdnn_plugin_sdk::benchmarkingOverrideFromEnv(), testCase.expected);
}

std::string mixedCase(const std::string& spelling)
{
    // Alternates case per character (e.g. "enabled" -> "EnAbLeD"), a deterministic
    // "mixed" transform distinct from all-lower and all-upper.
    std::string mixed = spelling;
    for(size_t index = 0; index < mixed.size(); ++index)
    {
        mixed[index] = (index % 2 == 0) ? static_cast<char>(std::toupper(mixed[index]))
                                        : static_cast<char>(std::tolower(mixed[index]));
    }
    return mixed;
}

std::string allUpper(const std::string& spelling)
{
    std::string upper = spelling;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
    });
    return upper;
}

std::vector<OverrideVariantCase> everyCasingOf(const std::string& spelling, bool expected)
{
    return {{spelling, expected},
            {allUpper(spelling), expected},
            {mixedCase(spelling), expected},
            {" " + spelling + " ", expected}};
}

std::vector<OverrideVariantCase> allAcceptedVariantCases()
{
    std::vector<OverrideVariantCase> cases;
    for(const auto& trueSpelling : {std::string("1"),
                                    std::string("true"),
                                    std::string("on"),
                                    std::string("yes"),
                                    std::string("enable"),
                                    std::string("enabled")})
    {
        for(auto& variant : everyCasingOf(trueSpelling, true))
        {
            cases.push_back(std::move(variant));
        }
    }
    for(const auto& falseSpelling : {std::string("0"),
                                     std::string("false"),
                                     std::string("off"),
                                     std::string("no"),
                                     std::string("disable"),
                                     std::string("disabled")})
    {
        for(auto& variant : everyCasingOf(falseSpelling, false))
        {
            cases.push_back(std::move(variant));
        }
    }
    // Near-misses that must resolve to nullopt, never to on.
    for(const auto& nearMiss : {std::string("2"),
                                std::string("-1"),
                                std::string("onn"),
                                std::string("tru"),
                                std::string("y"),
                                std::string("n"),
                                std::string(""),
                                std::string("   ")})
    {
        cases.push_back({nearMiss, std::nullopt});
    }
    return cases;
}

std::string variantCaseName(const ::testing::TestParamInfo<OverrideVariantCase>& info)
{
    std::string name = "Case" + std::to_string(info.index) + "_";
    for(const char character : info.param.envValue)
    {
        name += std::isalnum(static_cast<unsigned char>(character)) != 0 ? std::string(1, character)
                                                                         : std::string("_");
    }
    if(name.back() == '_' && info.param.envValue.empty())
    {
        name += "Empty";
    }
    return name;
}

INSTANTIATE_TEST_SUITE_P(EveryAcceptedSpellingAndNearMiss,
                         TestBenchmarkingOverrideVariants,
                         ::testing::ValuesIn(allAcceptedVariantCases()),
                         variantCaseName);

/// The knob names are consumed by name across provider boundaries, so a rename is an
/// interface change rather than a local edit.
TEST(TestGlobalKnobDefines, KnobAndEnvironmentNamesAreStable)
{
    EXPECT_STREQ(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, "global.benchmarking");
    EXPECT_STREQ(hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, "global.workspace_size_limit");
    EXPECT_STREQ(hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "HIPDNN_FORCE_BENCHMARKING");
}

/// An unset variable is the default path every non-benchmarking run takes.
TEST(TestGlobalKnobDefines, UnsetOverrideResolvesToNullopt)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME);

    EXPECT_EQ(hipdnn_plugin_sdk::benchmarkingOverrideFromEnv(), std::nullopt);
}

/// Read per call, never cached, so a value set after the first read still takes effect.
TEST(TestGlobalKnobDefines, TheOverrideIsReReadOnEveryCall)
{
    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "1");
    ASSERT_EQ(hipdnn_plugin_sdk::benchmarkingOverrideFromEnv(), std::optional<bool>(true));

    guard.setValue("0");
    EXPECT_EQ(hipdnn_plugin_sdk::benchmarkingOverrideFromEnv(), std::optional<bool>(false));
}

} // namespace
