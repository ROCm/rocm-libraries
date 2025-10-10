// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "gtest/internal/gtest-port.h"
#include <fcntl.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_sdk/test_utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <logging/Logging.hpp>

class TestBackendLogger : public ::testing::Test
{
protected:
    std::string _logFile;
    std::array<int, 2> _stderrPipe;
    int _oldStderr;
    std::unique_ptr<hipdnn_sdk::test_utilities::ScopedEnvironmentVariableSetter> _logLevelGuard;
    std::unique_ptr<hipdnn_sdk::test_utilities::ScopedEnvironmentVariableSetter> _logFileGuard;

public:
    void SetUp() override
    {
        _logLevelGuard
            = std::make_unique<hipdnn_sdk::test_utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_LEVEL");
        _logFileGuard
            = std::make_unique<hipdnn_sdk::test_utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_FILE");

        hipdnn_backend::logging::cleanup();

        testing::internal::CaptureStderr();

        hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "off");
        hipdnn_sdk::utilities::unsetEnv("HIPDNN_LOG_FILE");
    }

    void TearDown() override
    {
        hipdnn_backend::logging::cleanup();

        _logLevelGuard.reset();
        _logFileGuard.reset();

        if(!_logFile.empty())
        {
            std::remove(_logFile.c_str());
        }
    }

    static std::string getStderrContent()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        return testing::internal::GetCapturedStderr();
    }

    static void verifyStderrContains(const std::string& expectedContent)
    {
        std::string logContent = getStderrContent();
        EXPECT_NE(logContent.find(expectedContent), std::string::npos)
            << std::string("Expected to find: \"") << expectedContent << "\" in stderr."
            << "\nActual stderr content:\n"
            << logContent;
    }

    static void verifyStderrNotContains(const std::string& unexpectedContent)
    {
        std::string logContent = getStderrContent();
        EXPECT_EQ(logContent.find(unexpectedContent), std::string::npos)
            << std::string("Expected NOT to find: \"") << unexpectedContent << "\" in stderr."
            << "\nActual stderr content:\n"
            << logContent;
    }
};

TEST_F(TestBackendLogger, MacrosDontLogWhenOff)
{
    HIPDNN_LOG_INFO("Initializing with info message");
    HIPDNN_LOG_WARN("Initializing with warn message");
    HIPDNN_LOG_ERROR("Initializing with error message");

    std::string logContent = getStderrContent();
    EXPECT_TRUE(logContent.empty())
        << std::string("Expected stderr to be empty, but it contained:\n") << logContent;
}

TEST_F(TestBackendLogger, MacrosRespectLogLevelInfo)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_LOG_INFO("Info test message");
    HIPDNN_LOG_WARN("Warn test message");
    HIPDNN_LOG_ERROR("Error test message");

    std::string logContent = getStderrContent();
    EXPECT_NE(logContent.find("Info test message"), std::string::npos);
    EXPECT_NE(logContent.find("Warn test message"), std::string::npos);
    EXPECT_NE(logContent.find("Error test message"), std::string::npos);
}

TEST_F(TestBackendLogger, MacrosRespectLogLevelWarn)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "warn");

    HIPDNN_LOG_INFO("Info should not appear");
    HIPDNN_LOG_WARN("Warn should appear");
    HIPDNN_LOG_ERROR("Error should appear");

    std::string logContent = getStderrContent();
    EXPECT_EQ(logContent.find("Info should not appear"), std::string::npos);
    EXPECT_NE(logContent.find("Warn should appear"), std::string::npos);
    EXPECT_NE(logContent.find("Error should appear"), std::string::npos);
}

TEST_F(TestBackendLogger, MacrosRespectLogLevelError)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "error");

    HIPDNN_LOG_INFO("Info should not appear");
    HIPDNN_LOG_WARN("Warn should not appear");
    HIPDNN_LOG_ERROR("Error should appear");

    std::string logContent = getStderrContent();
    EXPECT_EQ(logContent.find("Info should not appear"), std::string::npos);
    EXPECT_EQ(logContent.find("Warn should not appear"), std::string::npos);
    EXPECT_NE(logContent.find("Error should appear"), std::string::npos);
}

TEST_F(TestBackendLogger, LoggingCanBeReinitialized)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "off");
    HIPDNN_LOG_INFO("This should not appear");

    hipdnn_backend::logging::cleanup();

    hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
    HIPDNN_LOG_INFO("This should appear after reinitialization");

    verifyStderrContains("This should appear after reinitialization");
}

TEST_F(TestBackendLogger, LogPatternFormatIsCorrectOnStderr)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_LOG_INFO("Pattern format test message");

    std::string logContent = getStderrContent();

    // [timestamp format] [thread id] [log level] [hipdnn_backend] message
    std::regex patternRegex(
        R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[info\] \[hipdnn_backend\] Pattern format test message)");

    EXPECT_TRUE(std::regex_search(logContent, patternRegex))
        << std::string("Expected log format pattern not found. Stderr content:\n") << logContent;
}

TEST_F(TestBackendLogger, MultipleMessagesAreLoggedToStderr)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_LOG_INFO("First backend message");
    HIPDNN_LOG_INFO("Second backend message");
    HIPDNN_LOG_INFO("Third backend message");

    std::string logContent = getStderrContent();
    EXPECT_NE(logContent.find("First backend message"), std::string::npos);
    EXPECT_NE(logContent.find("Second backend message"), std::string::npos);
    EXPECT_NE(logContent.find("Third backend message"), std::string::npos);

    // Verify expected order
    size_t pos1 = logContent.find("First backend message");
    size_t pos2 = logContent.find("Second backend message");
    size_t pos3 = logContent.find("Third backend message");

    EXPECT_TRUE(pos1 < pos2 && pos2 < pos3)
        << std::string("Messages not logged in expected order to stderr");
}

TEST_F(TestBackendLogger, LogFileCanBeSpecifiedByEnvVar)
{
    _logFile = "custom_backend_test.log";
    hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_FILE", _logFile.c_str());
    hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_LOG_INFO("Logging to custom file");

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    hipdnn_backend::logging::cleanup();

    std::string logContent;
    std::ifstream logFileStream(_logFile);
    ASSERT_TRUE(logFileStream.is_open()) << std::string("Log file was not created: ") << _logFile;

    logContent.assign((std::istreambuf_iterator<char>(logFileStream)),
                      std::istreambuf_iterator<char>());
    logFileStream.close();

    EXPECT_NE(logContent.find("Logging to custom file"), std::string::npos)
        << std::string("Expected to find message in log file ") << _logFile
        << "\nActual log content:\n"
        << logContent;

    verifyStderrNotContains("Logging to custom file");
}
