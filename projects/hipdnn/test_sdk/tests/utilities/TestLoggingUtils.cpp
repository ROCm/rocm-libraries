// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
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
    EXPECT_FALSE(isValidLogLevel("INFO"));
    EXPECT_FALSE(isValidLogLevel("Off"));
    EXPECT_FALSE(isValidLogLevel("ERROR"));
    EXPECT_FALSE(isValidLogLevel("123"));
    EXPECT_FALSE(isValidLogLevel(" info"));
    EXPECT_FALSE(isValidLogLevel("info "));
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
    ScopedEnvironmentVariableSetter guard("HIPDNN_LOG_LEVEL", "invalid");

    EXPECT_FALSE(isLoggingEnabled());

    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_LEVEL");
    EXPECT_FALSE(isLoggingEnabled());
}

TEST(TestLoggingUtils, GeneratePatternString)
{
    // The data_sdk pattern is simplified to just "[component] %v"
    // The backend adds timestamp, thread ID, and log level when receiving the callback
    std::string pattern = generatePatternString("test_component");

    EXPECT_NE(pattern.find("[test_component]"), std::string::npos);
    EXPECT_NE(pattern.find("%v"), std::string::npos); // Message
    EXPECT_EQ(pattern, "[test_component] %v");
}

// NOTE: ComponentFormatter tests have been moved to backend/tests/TestBackendLogger.cpp
// since ComponentFormatter is an spdlog-specific class that belongs in the backend.
