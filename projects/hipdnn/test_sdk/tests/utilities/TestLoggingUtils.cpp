// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/LoggingUtils.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

using namespace hipdnn_data_sdk::logging;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;

TEST(TestLoggingUtils, IsValidLogLevelWithValidLevels)
{
    // Test all valid log levels
    EXPECT_TRUE(isValidLogLevel("off"));
    EXPECT_TRUE(isValidLogLevel("info"));
    EXPECT_TRUE(isValidLogLevel("warn"));
    EXPECT_TRUE(isValidLogLevel("error"));
    EXPECT_TRUE(isValidLogLevel("fatal"));
}

TEST(TestLoggingUtils, IsValidLogLevelWithInvalidLevels)
{
    // Test invalid log levels
    EXPECT_FALSE(isValidLogLevel(""));
    EXPECT_FALSE(isValidLogLevel("debug"));
    EXPECT_FALSE(isValidLogLevel("trace"));
    EXPECT_FALSE(isValidLogLevel("verbose"));
    EXPECT_FALSE(isValidLogLevel("invalid"));
    EXPECT_FALSE(isValidLogLevel("123"));
}

TEST(TestLoggingUtils, IsValidLogLevelCaseInsensitive)
{
    // Log levels should be case-insensitive
    EXPECT_TRUE(isValidLogLevel("INFO"));
    EXPECT_TRUE(isValidLogLevel("Off"));
    EXPECT_TRUE(isValidLogLevel("ERROR"));
    EXPECT_TRUE(isValidLogLevel("Warn"));
    EXPECT_TRUE(isValidLogLevel("FATAL"));
}

TEST(TestLoggingUtils, IsValidLogLevelTrimsWhitespace)
{
    // Log levels should trim leading/trailing whitespace
    EXPECT_TRUE(isValidLogLevel(" info"));
    EXPECT_TRUE(isValidLogLevel("info "));
    EXPECT_TRUE(isValidLogLevel("  warn  "));
    EXPECT_TRUE(isValidLogLevel("\terror\t"));
    EXPECT_TRUE(isValidLogLevel("\n off \n"));
}

TEST(TestLoggingUtils, IsLoggingEnabledWithValidLevels)
{
    ScopedEnvironmentVariableSetter guard("HIPDNN_LOG_LEVEL");

    guard.setValue("off");
    EXPECT_FALSE(isLoggingEnabled());

    guard.setValue("info");
    EXPECT_TRUE(isLoggingEnabled());

    guard.setValue("error");
    EXPECT_TRUE(isLoggingEnabled());
}

TEST(TestLoggingUtils, IsLoggingEnabledWithInvalidOrUnsetLevels)
{
    const ScopedEnvironmentVariableSetter guard("HIPDNN_LOG_LEVEL", "invalid");

    EXPECT_FALSE(isLoggingEnabled());

    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_LEVEL");
    EXPECT_FALSE(isLoggingEnabled());
}

// ============================================================================
// Tests for stringToSeverity() - returns std::optional<hipdnnSeverity_t>
// ============================================================================

TEST(TestStringToSeverity, ValidOffReturnsOptionalWithOff)
{
    auto result = hipdnn_data_sdk::logging::detail::stringToSeverity("off");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), HIPDNN_SEV_OFF);
}

TEST(TestStringToSeverity, ValidInfoReturnsOptionalWithInfo)
{
    auto result = hipdnn_data_sdk::logging::detail::stringToSeverity("info");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), HIPDNN_SEV_INFO);
}

TEST(TestStringToSeverity, ValidWarnReturnsOptionalWithWarn)
{
    auto result = hipdnn_data_sdk::logging::detail::stringToSeverity("warn");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), HIPDNN_SEV_WARN);
}

TEST(TestStringToSeverity, ValidErrorReturnsOptionalWithError)
{
    auto result = hipdnn_data_sdk::logging::detail::stringToSeverity("error");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), HIPDNN_SEV_ERROR);
}

TEST(TestStringToSeverity, ValidFatalReturnsOptionalWithFatal)
{
    auto result = hipdnn_data_sdk::logging::detail::stringToSeverity("fatal");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), HIPDNN_SEV_FATAL);
}

TEST(TestStringToSeverity, InvalidStringReturnsNullopt)
{
    EXPECT_FALSE(hipdnn_data_sdk::logging::detail::stringToSeverity("invalid").has_value());
    EXPECT_FALSE(hipdnn_data_sdk::logging::detail::stringToSeverity("debug").has_value());
    EXPECT_FALSE(hipdnn_data_sdk::logging::detail::stringToSeverity("trace").has_value());
    EXPECT_FALSE(hipdnn_data_sdk::logging::detail::stringToSeverity("verbose").has_value());
    EXPECT_FALSE(hipdnn_data_sdk::logging::detail::stringToSeverity("123").has_value());
}

TEST(TestStringToSeverity, EmptyStringReturnsNullopt)
{
    EXPECT_FALSE(hipdnn_data_sdk::logging::detail::stringToSeverity("").has_value());
}

TEST(TestStringToSeverity, CaseInsensitiveMatching)
{
    // Uppercase
    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("OFF").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("OFF").value(), HIPDNN_SEV_OFF);

    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("INFO").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("INFO").value(), HIPDNN_SEV_INFO);

    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("WARN").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("WARN").value(), HIPDNN_SEV_WARN);

    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("ERROR").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("ERROR").value(),
              HIPDNN_SEV_ERROR);

    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("FATAL").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("FATAL").value(),
              HIPDNN_SEV_FATAL);

    // Mixed case
    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("Info").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("Info").value(), HIPDNN_SEV_INFO);

    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("WaRn").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("WaRn").value(), HIPDNN_SEV_WARN);
}

TEST(TestStringToSeverity, TrimsWhitespace)
{
    // Leading whitespace
    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("  info").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("  info").value(),
              HIPDNN_SEV_INFO);

    // Trailing whitespace
    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("warn  ").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("warn  ").value(),
              HIPDNN_SEV_WARN);

    // Both ends
    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("  error  ").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("  error  ").value(),
              HIPDNN_SEV_ERROR);

    // Tabs and newlines
    ASSERT_TRUE(hipdnn_data_sdk::logging::detail::stringToSeverity("\tfatal\n").has_value());
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverity("\tfatal\n").value(),
              HIPDNN_SEV_FATAL);

    // Whitespace-only still invalid
    EXPECT_FALSE(hipdnn_data_sdk::logging::detail::stringToSeverity("   ").has_value());
}

// ============================================================================
// Tests for stringToSeverityOrOff() - returns hipdnnSeverity_t (OFF on invalid)
// ============================================================================

TEST(TestStringToSeverityOrOff, ValidInputsReturnCorrectEnum)
{
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("off"), HIPDNN_SEV_OFF);
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("info"), HIPDNN_SEV_INFO);
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("warn"), HIPDNN_SEV_WARN);
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("error"), HIPDNN_SEV_ERROR);
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("fatal"), HIPDNN_SEV_FATAL);
}

TEST(TestStringToSeverityOrOff, InvalidInputReturnsOff)
{
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("invalid"), HIPDNN_SEV_OFF);
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("debug"), HIPDNN_SEV_OFF);
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("trace"), HIPDNN_SEV_OFF);
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("123"), HIPDNN_SEV_OFF);
}

TEST(TestStringToSeverityOrOff, EmptyStringReturnsOff)
{
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff(""), HIPDNN_SEV_OFF);
}

TEST(TestStringToSeverityOrOff, CaseInsensitiveAndTrimsWhitespace)
{
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("  INFO  "), HIPDNN_SEV_INFO);
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("\tWARN\n"), HIPDNN_SEV_WARN);
    EXPECT_EQ(hipdnn_data_sdk::logging::detail::stringToSeverityOrOff("Error"), HIPDNN_SEV_ERROR);
}
