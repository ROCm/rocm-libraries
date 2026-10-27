// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EnvBool.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <optional>
#include <string>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

namespace
{

constexpr const char* TEST_ENV_NAME = "HIPDNN_TEST_ENV_BOOL";

const std::vector<std::string> TRUE_TOKENS = {"1", "true", "on", "yes", "enable", "enabled"};
const std::vector<std::string> FALSE_TOKENS = {"0", "false", "off", "no", "disable", "disabled"};

} // namespace

TEST(TestEnvBool, EveryTrueTokenParsesAsTrue)
{
    for(const auto& token : TRUE_TOKENS)
    {
        const auto parsed = parseBoolToken(token);
        EXPECT_EQ(parsed.value, std::optional<bool>(true)) << token;
        EXPECT_FALSE(parsed.unrecognized) << token;
    }
}

TEST(TestEnvBool, EveryFalseTokenParsesAsFalse)
{
    for(const auto& token : FALSE_TOKENS)
    {
        const auto parsed = parseBoolToken(token);
        EXPECT_EQ(parsed.value, std::optional<bool>(false)) << token;
        EXPECT_FALSE(parsed.unrecognized) << token;
    }
}

TEST(TestEnvBool, TokensAreCaseInsensitive)
{
    EXPECT_EQ(parseBoolToken("TRUE").value, std::optional<bool>(true));
    EXPECT_EQ(parseBoolToken("EnAbLeD").value, std::optional<bool>(true));
    EXPECT_EQ(parseBoolToken("Off").value, std::optional<bool>(false));
    EXPECT_EQ(parseBoolToken("DISABLED").value, std::optional<bool>(false));
}

TEST(TestEnvBool, SurroundingWhitespaceIsTrimmedAndRawKeepsIt)
{
    const auto parsed = parseBoolToken(" \tYes\n ");
    EXPECT_EQ(parsed.value, std::optional<bool>(true));
    EXPECT_FALSE(parsed.unrecognized);
    EXPECT_EQ(parsed.raw, " \tYes\n ");
}

TEST(TestEnvBool, EmptyAndWhitespaceOnlyAreNeitherValueNorUnrecognized)
{
    for(const char* token : {"", " ", "\t \n"})
    {
        const auto parsed = parseBoolToken(token);
        EXPECT_EQ(parsed.value, std::nullopt) << "'" << token << "'";
        EXPECT_FALSE(parsed.unrecognized) << "'" << token << "'";
    }
}

TEST(TestEnvBool, ANonEmptyTokenOutsideBothSetsIsUnrecognized)
{
    for(const char* token : {"2", "tru", "truee", "y", "n", "maybe", "on off", "-1"})
    {
        const auto parsed = parseBoolToken(token);
        EXPECT_EQ(parsed.value, std::nullopt) << token;
        EXPECT_TRUE(parsed.unrecognized) << token;
        EXPECT_EQ(parsed.raw, token);
    }
}

TEST(TestEnvBool, AnUnsetVariableReadsAsEmpty)
{
    const ScopedEnvironmentVariableSetter guard(TEST_ENV_NAME);

    const auto parsed = readBoolEnv(TEST_ENV_NAME);
    EXPECT_EQ(parsed.value, std::nullopt);
    EXPECT_FALSE(parsed.unrecognized);
    EXPECT_TRUE(parsed.raw.empty());
}

TEST(TestEnvBool, ReadBoolEnvParsesTheCurrentValueOnEveryCall)
{
    ScopedEnvironmentVariableSetter guard(TEST_ENV_NAME, "on");
    ASSERT_EQ(readBoolEnv(TEST_ENV_NAME).value, std::optional<bool>(true));

    guard.setValue("0");
    EXPECT_EQ(readBoolEnv(TEST_ENV_NAME).value, std::optional<bool>(false));

    guard.setValue("garbage");
    const auto parsed = readBoolEnv(TEST_ENV_NAME);
    EXPECT_EQ(parsed.value, std::nullopt);
    EXPECT_TRUE(parsed.unrecognized);
    EXPECT_EQ(parsed.raw, "garbage");
}
