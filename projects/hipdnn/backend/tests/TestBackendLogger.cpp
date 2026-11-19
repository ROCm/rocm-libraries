// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "gtest/internal/gtest-port.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <logging/Logging.hpp>

class TestBackendLogger : public ::testing::Test
{
protected:
    std::string _logFile;
    std::array<int, 2> _stderrPipe;
    int _oldStderr;
    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _logLevelGuard;
    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _logFileGuard;
    bool _stderrContentRetrieved = false;

public:
    void SetUp() override
    {
        _logLevelGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_LEVEL");
        _logFileGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_FILE");

        hipdnn_backend::logging::loggerShutdown();

        testing::internal::CaptureStderr();

        hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "off");
        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_FILE");
    }

    void TearDown() override
    {
        hipdnn_backend::logging::loggerShutdown();

        _logLevelGuard.reset();
        _logFileGuard.reset();

        if(!_logFile.empty())
        {
            std::remove(_logFile.c_str());
        }
    }

    std::string getStderrContent()
    {
        EXPECT_FALSE(_stderrContentRetrieved)
            << "getStderrContent() can only be called once per test";
        if(_stderrContentRetrieved)
        {
            return {};
        }
        _stderrContentRetrieved = true;

        hipdnn_backend::logging::loggerShutdown();

        return testing::internal::GetCapturedStderr();
    }
};

TEST_F(TestBackendLogger, MacrosDontLogWhenOff)
{
    HIPDNN_BACKEND_LOG_INFO("Initializing with info message");
    HIPDNN_BACKEND_LOG_WARN("Initializing with warn message");
    HIPDNN_BACKEND_LOG_ERROR("Initializing with error message");
    HIPDNN_BACKEND_LOG_FATAL("Initializing with fatal message");

    const std::string logContent = getStderrContent();
    EXPECT_TRUE(logContent.empty())
        << std::string("Expected stderr to be empty, but it contained:\n") << logContent;
}

TEST_F(TestBackendLogger, SetLogLevelOverridesEnvironmentVariable)
{
    // Environment variable says "off", but we'll programmatically set to "info"
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "off");

    // Programmatically set log level to WARN (before any logging)
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_WARN);

    // Log messages at various levels
    HIPDNN_BACKEND_LOG_INFO("Info message should not appear");
    HIPDNN_BACKEND_LOG_WARN("Warn message should appear");
    HIPDNN_BACKEND_LOG_ERROR("Error message should appear");

    const std::string logContent = getStderrContent();

    // All messages should appear because we set log level to INFO programmatically
    EXPECT_EQ(logContent.find("Info message should not appear"), std::string::npos)
        << "Expected info message in stderr, actual content:\n"
        << logContent;
    EXPECT_NE(logContent.find("Warn message should appear"), std::string::npos)
        << "Expected warn message in stderr, actual content:\n"
        << logContent;
    EXPECT_NE(logContent.find("Error message should appear"), std::string::npos)
        << "Expected error message in stderr, actual content:\n"
        << logContent;
}

TEST_F(TestBackendLogger, MacrosRespectLogLevelInfo)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_BACKEND_LOG_INFO("Info test message");
    HIPDNN_BACKEND_LOG_WARN("Warn test message");
    HIPDNN_BACKEND_LOG_ERROR("Error test message");
    HIPDNN_BACKEND_LOG_FATAL("Fatal test message");

    const std::string logContent = getStderrContent();
    EXPECT_NE(logContent.find("Info test message"), std::string::npos);
    EXPECT_NE(logContent.find("Warn test message"), std::string::npos);
    EXPECT_NE(logContent.find("Error test message"), std::string::npos);
    EXPECT_NE(logContent.find("Fatal test message"), std::string::npos);
}

TEST_F(TestBackendLogger, MacrosRespectLogLevelWarn)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "warn");

    HIPDNN_BACKEND_LOG_INFO("Info should not appear");
    HIPDNN_BACKEND_LOG_WARN("Warn should appear");
    HIPDNN_BACKEND_LOG_ERROR("Error should appear");
    HIPDNN_BACKEND_LOG_FATAL("Fatal should appear");

    const std::string logContent = getStderrContent();
    EXPECT_EQ(logContent.find("Info should not appear"), std::string::npos);
    EXPECT_NE(logContent.find("Warn should appear"), std::string::npos);
    EXPECT_NE(logContent.find("Error should appear"), std::string::npos);
    EXPECT_NE(logContent.find("Fatal should appear"), std::string::npos);
}

TEST_F(TestBackendLogger, MacrosRespectLogLevelError)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "error");

    HIPDNN_BACKEND_LOG_INFO("Info should not appear");
    HIPDNN_BACKEND_LOG_WARN("Warn should not appear");
    HIPDNN_BACKEND_LOG_ERROR("Error should appear");
    HIPDNN_BACKEND_LOG_ERROR("Fatal should appear");

    const std::string logContent = getStderrContent();
    EXPECT_EQ(logContent.find("Info should not appear"), std::string::npos);
    EXPECT_EQ(logContent.find("Warn should not appear"), std::string::npos);
    EXPECT_NE(logContent.find("Error should appear"), std::string::npos);
    EXPECT_NE(logContent.find("Fatal should appear"), std::string::npos);
}

TEST_F(TestBackendLogger, LoggingCanBeReinitialized)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
    HIPDNN_BACKEND_LOG_INFO("This first info log should appear");

    hipdnn_backend::logging::loggerShutdown();

    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "off");
    HIPDNN_BACKEND_LOG_INFO("This should not appear");

    hipdnn_backend::logging::loggerShutdown();

    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
    HIPDNN_BACKEND_LOG_INFO("This second info log should appear after reinitialization");

    const std::string logContent = getStderrContent();
    EXPECT_NE(logContent.find("This first info log should appear"), std::string::npos)
        << "Expected to find: \"This first info log should appear\" in stderr."
        << "\nActual stderr content:\n"
        << logContent;
    EXPECT_EQ(logContent.find("This should not appear"), std::string::npos)
        << "Expected NOT to find: \"This should not appear\" in stderr."
        << "\nActual stderr content:\n"
        << logContent;
    EXPECT_NE(logContent.find("This second info log should appear after reinitialization"),
              std::string::npos)
        << "Expected to find: \"This second info log should appear after reinitialization\" in "
           "stderr."
        << "\nActual stderr content:\n"
        << logContent;
}

TEST_F(TestBackendLogger, LogPatternFormatIsCorrectOnStderr)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_BACKEND_LOG_INFO("Pattern format test message");

    const std::string logContent = getStderrContent();

    // [timestamp format] [thread id] [log level] [hipdnn_backend] message
    const std::regex patternRegex(
        R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[info\] \[hipdnn_backend\] Pattern format test message)");

    EXPECT_TRUE(std::regex_search(logContent, patternRegex))
        << std::string("Expected log format pattern not found. Stderr content:\n") << logContent;
}

TEST_F(TestBackendLogger, MultipleMessagesAreLoggedToStderr)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_BACKEND_LOG_INFO("First backend message");
    HIPDNN_BACKEND_LOG_INFO("Second backend message");
    HIPDNN_BACKEND_LOG_INFO("Third backend message");

    const std::string logContent = getStderrContent();
    EXPECT_NE(logContent.find("First backend message"), std::string::npos);
    EXPECT_NE(logContent.find("Second backend message"), std::string::npos);
    EXPECT_NE(logContent.find("Third backend message"), std::string::npos);

    // Verify expected order
    const size_t pos1 = logContent.find("First backend message");
    const size_t pos2 = logContent.find("Second backend message");
    const size_t pos3 = logContent.find("Third backend message");

    EXPECT_TRUE(pos1 < pos2 && pos2 < pos3)
        << std::string("Messages not logged in expected order to stderr");
}

TEST_F(TestBackendLogger, LogFileCanBeSpecifiedByEnvVar)
{
    _logFile = "custom_backend_test.log";
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", _logFile.c_str());
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_BACKEND_LOG_INFO("Logging to custom file");

    hipdnn_backend::logging::loggerShutdown();

    auto stdErrContent = getStderrContent();

    std::string logFileContent;
    std::ifstream logFileStream(_logFile);
    ASSERT_TRUE(logFileStream.is_open()) << std::string("Log file was not created: ") << _logFile;

    logFileContent.assign((std::istreambuf_iterator<char>(logFileStream)),
                          std::istreambuf_iterator<char>());
    logFileStream.close();

    EXPECT_NE(logFileContent.find("Logging to custom file"), std::string::npos)
        << std::string("Expected to find message in log file ") << _logFile
        << "\nActual log content:\n"
        << logFileContent;

    EXPECT_EQ(stdErrContent.find("Logging to custom file"), std::string::npos)
        << "Expected NOT to find: \"Logging to custom file\" in stderr."
        << "\nActual stderr content:\n"
        << stdErrContent;
}

TEST_F(TestBackendLogger, ParamsAreNotExpandedIfLogLevelIsDisabled)
{
    bool wasCalledForInfo = false;
    bool wasCalledForWarn = false;
    bool wasCalledForError = false;
    bool wasCalledForFatal = false;
    const std::string infoMessage("info log message");
    const std::string warnMessage("warn log message");
    const std::string errorMessage("error log message");
    const std::string fatalMessage("fatal log message");
    auto trackingLambda = [](bool& wasCalled, std::string message) {
        wasCalled = true;
        return message;
    };

    // Set level to error so info and warn should be ignored
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "error");

    HIPDNN_BACKEND_LOG_INFO(trackingLambda(wasCalledForInfo, infoMessage));
    HIPDNN_BACKEND_LOG_WARN(trackingLambda(wasCalledForWarn, warnMessage));
    HIPDNN_BACKEND_LOG_ERROR(trackingLambda(wasCalledForError, errorMessage));
    HIPDNN_BACKEND_LOG_FATAL(trackingLambda(wasCalledForFatal, fatalMessage));

    const std::string logContent = getStderrContent();

    EXPECT_THAT(logContent, ::testing::Not(::testing::HasSubstr(infoMessage)));
    EXPECT_THAT(logContent, ::testing::Not(::testing::HasSubstr(infoMessage)));
    EXPECT_THAT(logContent, ::testing::Not(::testing::HasSubstr(warnMessage)));
    EXPECT_THAT(logContent, ::testing::Not(::testing::HasSubstr(warnMessage)));
    EXPECT_THAT(logContent, ::testing::HasSubstr(errorMessage));
    EXPECT_THAT(logContent, ::testing::HasSubstr(fatalMessage));
    EXPECT_FALSE(wasCalledForInfo);
    EXPECT_FALSE(wasCalledForWarn);
    EXPECT_TRUE(wasCalledForError);
    EXPECT_TRUE(wasCalledForFatal);
}

TEST_F(TestBackendLogger, ShutdownThenReinitializeConcurrently)
{
    _logFile = "concurrent_logging_test.log";
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", _logFile.c_str());
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    constexpr int NUM_LOGGER_THREADS = 8;
    constexpr int NUM_SHUTDOWN_THREADS = 2;
    constexpr int SHUTDOWN_ITERATIONS = 30;

    std::atomic<bool> startFlag{false};
    std::atomic<bool> stopFlag{false};
    std::vector<std::thread> threads;
    threads.reserve(NUM_LOGGER_THREADS + NUM_SHUTDOWN_THREADS);

    // Threads that continuously log (triggering lazy init if shutdown occurred)
    for(int i = 0; i < NUM_LOGGER_THREADS; ++i)
    {
        threads.emplace_back([&, threadId = i]() {
            while(!startFlag.load())
            {
                std::this_thread::yield();
            }

            int msgCount = 0;
            while(!stopFlag.load())
            {
                HIPDNN_BACKEND_LOG_INFO("Logger thread {} message {}", threadId, msgCount++);
                // Small yield to allow interleaving
                if(msgCount % 10 == 0)
                {
                    std::this_thread::yield();
                }
            }
        });
    }

    // Threads that periodically call shutdown
    for(int i = 0; i < NUM_SHUTDOWN_THREADS; ++i)
    {
        threads.emplace_back([&]() {
            while(!startFlag.load())
            {
                std::this_thread::yield();
            }

            for(int j = 0; j < SHUTDOWN_ITERATIONS && !stopFlag.load(); ++j)
            {
                hipdnn_backend::logging::loggerShutdown();
                // Small delay between shutdowns
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    // Start all threads simultaneously
    startFlag.store(true);

    // Let them race for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Signal stop
    stopFlag.store(true);

    // Join all threads
    for(auto& t : threads)
    {
        t.join();
    }

    // Consume captured stderr to clean up for the fixture
    testing::internal::GetCapturedStderr();

    // If we get here without crashing or deadlocking, the test passes
}

TEST_F(TestBackendLogger, ConcurrentLoggingToFile)
{
    _logFile = "concurrent_logging_test.log";
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", _logFile.c_str());
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    constexpr int NUM_THREADS = 4;
    constexpr int LOGS_PER_THREAD = 100;
    std::atomic<bool> startFlag{false};
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    // Create logging threads
    for(int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&, threadId = i]() {
            // Wait for start signal
            while(!startFlag.load())
            {
                std::this_thread::yield();
            }

            for(int j = 0; j < LOGS_PER_THREAD; ++j)
            {
                HIPDNN_BACKEND_LOG_INFO("Thread {} message {}", threadId, j);
            }
        });
    }

    // Start all threads simultaneously
    startFlag.store(true);

    // Join all threads
    for(auto& t : threads)
    {
        t.join();
    }

    // Flush and shutdown to ensure all messages are written to file
    hipdnn_backend::logging::loggerShutdown();

    // Read and verify the log file
    std::string logContent;
    std::ifstream logFileStream(_logFile);
    ASSERT_TRUE(logFileStream.is_open()) << std::string("Log file was not created: ") << _logFile;

    logContent.assign((std::istreambuf_iterator<char>(logFileStream)),
                      std::istreambuf_iterator<char>());
    logFileStream.close();

    // Verify that messages from all threads appear in the log file
    for(int threadId = 0; threadId < NUM_THREADS; ++threadId)
    {
        const std::string expectedMarker = "Thread " + std::to_string(threadId) + " message";
        EXPECT_NE(logContent.find(expectedMarker), std::string::npos)
            << "Expected to find messages from thread " << threadId << " in log file";
    }

    // Consume captured stderr to clean up for the fixture
    testing::internal::GetCapturedStderr();
}

namespace
{

constexpr auto K_CALL_DEADLINE = std::chrono::seconds(10);

// Runs body on a worker thread with a deadline. A worker that misses the deadline can be
// neither joined nor cancelled, so the test reports the failure and ends the process. The
// fixture captures stderr, so the report goes to stdout.
void runWithDeadline(const char* description, const std::function<void()>& body)
{
    auto worker = std::async(std::launch::async, body);
    if(worker.wait_for(K_CALL_DEADLINE) == std::future_status::timeout)
    {
        std::fprintf(stdout,
                     "FAILED: %s did not return within %lld s\n",
                     description,
                     static_cast<long long>(K_CALL_DEADLINE.count()));
        std::fflush(nullptr);
        std::_Exit(EXIT_FAILURE);
    }
    worker.get();
}

// Two HIPDNN_LOG_FILE values that cannot be opened: a file whose parent directory is missing
// and cannot be created (a regular file occupies its path), and an existing directory.
class UnopenableLogFilePaths
{
public:
    UnopenableLogFilePaths()
        : _root(std::filesystem::temp_directory_path()
                / ("hipdnn_unopenable_log_"
                   + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        std::filesystem::create_directories(_root / "existing_directory");
        std::ofstream(_root / "regular_file") << "not a directory";
    }

    ~UnopenableLogFilePaths()
    {
        std::error_code ec;
        std::filesystem::remove_all(_root, ec);
    }

    UnopenableLogFilePaths(const UnopenableLogFilePaths&) = delete;
    UnopenableLogFilePaths& operator=(const UnopenableLogFilePaths&) = delete;
    UnopenableLogFilePaths(UnopenableLogFilePaths&&) = delete;
    UnopenableLogFilePaths& operator=(UnopenableLogFilePaths&&) = delete;

    std::vector<std::string> all() const
    {
        return {(_root / "regular_file" / "missing_directory" / "hipdnn.log").string(),
                (_root / "existing_directory").string()};
    }

private:
    std::filesystem::path _root;
};

struct SelfRemovalState
{
    std::atomic<int> invocations{0};
    std::atomic<hipdnnStatus_t> removalStatus{HIPDNN_STATUS_INTERNAL_ERROR};
};

// Unregisters itself from its first message.
void selfRemovingCallback(hipdnnUserLogCallbackHandle_t userHandle,
                          hipdnnSeverity_t /*severity*/,
                          const char* /*message*/)
{
    auto* state = static_cast<SelfRemovalState*>(userHandle);
    if(state->invocations.fetch_add(1) == 0)
    {
        state->removalStatus.store(hipdnn_backend::logging::setUserLogCallback(
            selfRemovingCallback, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_SYNC, userHandle));
    }
}

struct CallbackCounter
{
    std::atomic<int> invocations{0};
};

void countingCallback(hipdnnUserLogCallbackHandle_t userHandle,
                      hipdnnSeverity_t /*severity*/,
                      const char* /*message*/)
{
    static_cast<CallbackCounter*>(userHandle)->invocations.fetch_add(1);
}

struct AddingCallbackState
{
    std::atomic<int> invocations{0};
    std::atomic<hipdnnStatus_t> addStatus{HIPDNN_STATUS_INTERNAL_ERROR};
    CallbackCounter added;
};

// Registers countingCallback (with &state->added) from its first message.
void addingCallback(hipdnnUserLogCallbackHandle_t userHandle,
                    hipdnnSeverity_t /*severity*/,
                    const char* /*message*/)
{
    auto* state = static_cast<AddingCallbackState*>(userHandle);
    if(state->invocations.fetch_add(1) == 0)
    {
        state->addStatus.store(hipdnn_backend::logging::setUserLogCallback(
            countingCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC, &state->added));
    }
}

// Runs as the statement of a threadsafe death test, in a fresh process, so the backend
// library's own logger is initialized here by the first C API call, with these variables.
// The temporary paths are removed before exiting; any failed call is a nonzero exit status.
[[noreturn]] void createHandleWithUnopenableLogFile(size_t pathIndex)
{
    int status = 0;
    {
        const UnopenableLogFilePaths unopenable;
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE",
                                           unopenable.all().at(pathIndex).c_str());

        // hipdnnCreate is only the logging trigger here. An empty absolute plugin search path
        // keeps provider plugins out of the unit binary.
        if(hipdnnSetEnginePluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE)
               != HIPDNN_STATUS_SUCCESS
           || hipdnnSetHeuristicPluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE)
                  != HIPDNN_STATUS_SUCCESS)
        {
            status = 2;
        }

        hipdnnHandle_t handle = nullptr;
        if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS || handle == nullptr)
        {
            status = 3;
        }
        else if(hipdnnDestroy(handle) != HIPDNN_STATUS_SUCCESS)
        {
            status = 4;
        }
    }
    std::exit(status);
}

} // namespace

// Port of the exploratory test Issue_A2_02_unopenable_log_file_terminates_process. An
// HIPDNN_LOG_FILE that cannot be opened neither terminates the process nor changes the API
// result: hipDNN warns on stderr, naming the variable, and logs to stderr instead.
// hipdnnCreate reaches the logger inside the backend library, which this binary cannot reset
// (TestBackendLogger resets the copy linked into the test binary), so each case runs in a
// fresh process where that logger starts uninitialized.
TEST(TestBackendLoggerFreshProcess, HandleCreationDoesNotTerminate)
{
    SKIP_IF_NO_DEVICES();

    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // EXPECT_EXIT expands to a switch over AssumeRole() carrying no default label. The
    // diagnostic is attributed to this expansion site rather than to the GoogleTest
    // header, so -isystem does not suppress it and -Werror makes it fatal.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-default"
    // A file whose parent directory cannot be created
    EXPECT_EXIT(
        createHandleWithUnopenableLogFile(0), testing::ExitedWithCode(0), "HIPDNN_LOG_FILE");
    // A path that is an existing directory
    EXPECT_EXIT(
        createHandleWithUnopenableLogFile(1), testing::ExitedWithCode(0), "HIPDNN_LOG_FILE");
#pragma clang diagnostic pop
}

// Port of the exploratory test Issue_A2_02_unopenable_log_file_terminates_process, through
// backendLoggingCallback, the function hipdnnLoggingCallback_ext forwards frontend log calls to.
TEST_F(TestBackendLogger, FrontendLogCallDoesNotTerminate)
{
    const UnopenableLogFilePaths unopenable;
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    const auto paths = unopenable.all();
    for(size_t i = 0; i < paths.size(); ++i)
    {
        hipdnn_backend::logging::loggerShutdown();
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", paths[i].c_str());

        hipdnn_backend::logging::backendLoggingCallback(
            HIPDNN_SEV_INFO, ("Frontend message " + std::to_string(i)).c_str());
    }

    const std::string logContent = getStderrContent();
    EXPECT_THAT(logContent, ::testing::HasSubstr("HIPDNN_LOG_FILE"));
    for(size_t i = 0; i < paths.size(); ++i)
    {
        EXPECT_THAT(logContent, ::testing::HasSubstr(paths[i]));
        EXPECT_THAT(logContent, ::testing::HasSubstr("Frontend message " + std::to_string(i)));
    }
}

// Port of the exploratory test Issue_C4_03_level_raised_after_first_call_leaves_stderr_silent.
// Logging initialized while the level is OFF writes to stderr once the level is raised.
TEST_F(TestBackendLogger, StderrReceivesLogsAfterLevelRaised)
{
    HIPDNN_BACKEND_LOG_INFO("Logged while the level is off");

    EXPECT_EQ(hipdnn_backend::logging::setGlobalLogLevel(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);
    HIPDNN_BACKEND_LOG_INFO("Logged after the level was raised");

    const std::string logContent = getStderrContent();
    EXPECT_THAT(logContent, ::testing::Not(::testing::HasSubstr("Logged while the level is off")));
    EXPECT_THAT(logContent, ::testing::HasSubstr("Logged after the level was raised"));
}

// As StderrReceivesLogsAfterLevelRaised, with the level raised directly in the data SDK, so
// only the first enabled message can create the console sink.
TEST_F(TestBackendLogger, StderrReceivesLogsAfterSdkLevelRaised)
{
    HIPDNN_BACKEND_LOG_INFO("Logged while the level is off");

    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);
    HIPDNN_BACKEND_LOG_INFO("Logged after the level was raised");

    const std::string logContent = getStderrContent();
    EXPECT_THAT(logContent, ::testing::HasSubstr("Logged after the level was raised"));
}

// As StderrReceivesLogsAfterLevelRaised, with HIPDNN_LOG_FILE set: no file exists while the
// level is OFF, and the file receives the logs once the level is raised.
TEST_F(TestBackendLogger, LogFileReceivesLogsAfterLevelRaised)
{
    _logFile = "level_raised_after_first_call_test.log";
    std::remove(_logFile.c_str());
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", _logFile.c_str());

    HIPDNN_BACKEND_LOG_INFO("Logged while the level is off");
    EXPECT_FALSE(std::filesystem::exists(_logFile))
        << "A log file was created while the level is OFF: " << _logFile;

    EXPECT_EQ(hipdnn_backend::logging::setGlobalLogLevel(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);
    HIPDNN_BACKEND_LOG_INFO("Logged after the level was raised");

    const std::string stdErrContent = getStderrContent();

    std::ifstream logFileStream(_logFile);
    ASSERT_TRUE(logFileStream.is_open()) << "Log file was not created: " << _logFile;
    const std::string logFileContent((std::istreambuf_iterator<char>(logFileStream)),
                                     std::istreambuf_iterator<char>());

    EXPECT_THAT(logFileContent, ::testing::HasSubstr("Logged after the level was raised"));
    EXPECT_THAT(logFileContent,
                ::testing::Not(::testing::HasSubstr("Logged while the level is off")));
    EXPECT_THAT(stdErrContent,
                ::testing::Not(::testing::HasSubstr("Logged after the level was raised")));
}

// As StderrReceivesLogsAfterLevelRaised, with logging initialized by registering a user
// callback while the level is OFF.
TEST_F(TestBackendLogger, StderrReceivesLogsAfterLevelRaisedWithCallbackRegisteredFirst)
{
    CallbackCounter counter;
    EXPECT_EQ(hipdnn_backend::logging::setUserLogCallback(
                  countingCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC, &counter),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnn_backend::logging::setGlobalLogLevel(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);
    HIPDNN_BACKEND_LOG_INFO("Logged after the level was raised");

    // Shuts the logger down, which also unregisters the callback before counter goes away
    const std::string logContent = getStderrContent();
    EXPECT_THAT(logContent, ::testing::HasSubstr("Logged after the level was raised"));
    EXPECT_GT(counter.invocations.load(), 0);
}

// Port of the exploratory test Issue_C4_05_callback_unregisters_itself_deadlocks. A SYNC
// callback that unregisters itself from inside its own invocation returns, and receives no
// later message.
TEST_F(TestBackendLogger, SelfRemovalReturns)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    SelfRemovalState state;
    EXPECT_EQ(hipdnn_backend::logging::setUserLogCallback(
                  selfRemovingCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC, &state),
              HIPDNN_STATUS_SUCCESS);

    runWithDeadline("logging to a SYNC callback that unregisters itself", [] {
        HIPDNN_BACKEND_LOG_INFO("First message, received by the callback");
        HIPDNN_BACKEND_LOG_INFO("Second message, logged after the callback unregistered");
    });

    EXPECT_EQ(state.removalStatus.load(), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(state.invocations.load(), 1);

    // Shuts the logger down, which also unregisters the callback before state goes away
    static_cast<void>(getStderrContent());
}

// Port of the exploratory test Issue_C4_05_callback_unregisters_itself_deadlocks, adding a
// callback instead of removing one: the registration returns and the added callback receives
// later messages.
TEST_F(TestBackendLogger, CallbackAddedFromCallbackReturns)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");

    AddingCallbackState state;
    EXPECT_EQ(hipdnn_backend::logging::setUserLogCallback(
                  addingCallback, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC, &state),
              HIPDNN_STATUS_SUCCESS);

    runWithDeadline("logging to a SYNC callback that registers another callback", [] {
        HIPDNN_BACKEND_LOG_INFO("First message, received by the registering callback");
        HIPDNN_BACKEND_LOG_INFO("Second message, received by both callbacks");
    });

    EXPECT_EQ(state.addStatus.load(), HIPDNN_STATUS_SUCCESS);
    EXPECT_GT(state.added.invocations.load(), 0);

    // Shuts the logger down, which also unregisters both callbacks before state goes away
    static_cast<void>(getStderrContent());
}
