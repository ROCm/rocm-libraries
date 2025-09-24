// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/logging/ComponentFormatter.hpp>
#include <hipdnn_sdk/logging/LoggingUtils.hpp>
#include <hipdnn_sdk/test_utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <spdlog/details/log_msg.h>
#include <spdlog/pattern_formatter.h>

using namespace hipdnn_sdk::logging;
using namespace hipdnn_sdk::test_utilities;

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

    hipdnn_sdk::utilities::unsetEnv("HIPDNN_LOG_LEVEL");
    EXPECT_FALSE(isLoggingEnabled());
}

TEST(TestLoggingUtils, GeneratePatternString)
{
    std::string pattern = generatePatternString("test_component");

    EXPECT_NE(pattern.find("[test_component]"), std::string::npos);
    EXPECT_NE(pattern.find("%Y-%m-%d"), std::string::npos); // Date format
    EXPECT_NE(pattern.find("%H:%M:%S"), std::string::npos); // Time format
    EXPECT_NE(pattern.find("[tid %t]"), std::string::npos); // Thread ID
    EXPECT_NE(pattern.find("[%l]"), std::string::npos); // Log level
    EXPECT_NE(pattern.find("%v"), std::string::npos); // Message
}

TEST(TestLoggingUtils, ComponentFormatterPassThrough)
{
    ComponentFormatter formatter;

    // Create a log message for "hipdnn_callback_receiver"
    spdlog::details::log_msg msg("hipdnn_callback_receiver", spdlog::level::info, "Test message");
    spdlog::memory_buf_t buf;

    formatter.format(msg, buf);

    // Should pass through without formatting (just the message)
    std::string result(buf.data(), buf.size());
    EXPECT_EQ(result, "Test message\n");
}

TEST(TestLoggingUtils, ComponentFormatterStandard)
{
    ComponentFormatter formatter;

    // Create a log message for any other logger
    spdlog::details::log_msg msg("test_logger", spdlog::level::info, "Test message");
    spdlog::memory_buf_t buf;

    formatter.format(msg, buf);

    // Should include formatting with component name
    std::string result(buf.data(), buf.size());
    EXPECT_NE(result.find("[test_logger]"), std::string::npos);
    EXPECT_NE(result.find("Test message"), std::string::npos);
}

TEST(TestLoggingUtils, ComponentFormatterClone)
{
    ComponentFormatter formatter;
    auto cloned = formatter.clone();

    EXPECT_NE(cloned, nullptr);

    spdlog::details::log_msg msg("test_logger", spdlog::level::info, "Clone test");
    spdlog::memory_buf_t buf;

    cloned->format(msg, buf);

    std::string result(buf.data(), buf.size());
    EXPECT_NE(result.find("[test_logger]"), std::string::npos);
    EXPECT_NE(result.find("Clone test"), std::string::npos);
}
