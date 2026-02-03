// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/ComponentFormatter.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

// Structure to hold recorded log entries
struct RecordedLog
{
    hipdnnSeverity_t severity;
    std::string message;
};

inline hipdnnSeverity_t stringToSeverity(const std::string& levelStr)
{
    if(levelStr == "info")
    {
        return HIPDNN_SEV_INFO;
    }
    if(levelStr == "warn")
    {
        return HIPDNN_SEV_WARN;
    }
    if(levelStr == "error")
    {
        return HIPDNN_SEV_ERROR;
    }
    if(levelStr == "fatal")
    {
        return HIPDNN_SEV_FATAL;
    }
    return HIPDNN_SEV_OFF;
}

inline std::string severityToString(hipdnnSeverity_t severity)
{
    switch(severity)
    {
    case HIPDNN_SEV_INFO:
        return "info";
    case HIPDNN_SEV_WARN:
        return "warn";
    case HIPDNN_SEV_ERROR:
        return "error";
    case HIPDNN_SEV_FATAL:
        return "fatal";
    default:
        return "(unknown)";
    }
}

class LogRecording
{
public:
    static LogRecording& instance()
    {
        static LogRecording s_instance;
        return s_instance;
    }

    void startRecording()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _isRecording = true;
    }

    void stopRecording()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _isRecording = false;
    }

    void clearLogs()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _recordedLogs.clear();
    }

    void recordLog(hipdnnSeverity_t severity, const std::string& message)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if(_isRecording)
        {
            _recordedLogs.push_back({severity, message});
        }
    }

    const std::vector<RecordedLog>& getRecordedLogs() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _recordedLogs;
    }

    LogRecording(const LogRecording&) = delete;
    LogRecording& operator=(const LogRecording&) = delete;
    LogRecording(LogRecording&&) = delete;
    LogRecording& operator=(LogRecording&&) = delete;

private:
    LogRecording() = default;
    ~LogRecording() = default;

    mutable std::mutex _mutex;
    bool _isRecording{false};
    std::vector<RecordedLog> _recordedLogs;
};

// RAII helper class for scoped recording
class LogRecorder
{
public:
    LogRecorder()
    {
        LogRecording::instance().clearLogs();
        LogRecording::instance().startRecording();
    }

    ~LogRecorder()
    {
        LogRecording::instance().stopRecording();
        LogRecording::instance().clearLogs();
    }

    static bool hasLogContaining(const std::string& text)
    {
        const auto& logs = LogRecording::instance().getRecordedLogs();
        for(const auto& log : logs)
        {
            if(log.message.find(text) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    static bool hasLogContaining(hipdnnSeverity_t severity, const std::string& text)
    {
        const auto& logs = LogRecording::instance().getRecordedLogs();
        for(const auto& log : logs)
        {
            if(log.severity == severity && log.message.find(text) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    static const std::vector<RecordedLog>& getRecordedLogs()
    {
        return LogRecording::instance().getRecordedLogs();
    }

    static size_t getRecordedLogCount()
    {
        return LogRecording::instance().getRecordedLogs().size();
    }

    static std::string getRecordedLogsAsString(size_t maxLogs = 1000)
    {
        const auto& logs = LogRecording::instance().getRecordedLogs();
        if(logs.empty())
        {
            return "(no logs recorded)";
        }

        size_t logsToShow = std::min(logs.size(), maxLogs);
        std::ostringstream oss;
        if(logsToShow < logs.size())
        {
            oss << "(Showing first " << logsToShow << " logs of " << logs.size()
                << " total recorded.)\n";
        }

        for(size_t i = 0; i < logsToShow; ++i)
        {
            const auto& log = logs[i];
            oss << "[" << i << "] [" << severityToString(log.severity) << "] " << log.message
                << "\n";
        }
        return oss.str();
    }

    // Non-copyable, non-movable
    LogRecorder(const LogRecorder&) = delete;
    LogRecorder& operator=(const LogRecorder&) = delete;
    LogRecorder(LogRecorder&&) = delete;
    LogRecorder& operator=(LogRecorder&&) = delete;
};

// Static variable to hold the chained callback for chained test logging
static hipdnnCallback_t sChainedCallback = nullptr;

// Callback that records to LogRecording AND chains to another callback
inline void testChainedLoggingCallback(hipdnnSeverity_t severity, const char* message)
{
    // Record log if recording is active
    LogRecording::instance().recordLog(severity, message);

#ifndef DISABLE_TEST_LOGGING
    // Chain to the configured callback (e.g., hipdnnLoggingCallback_ext)
    if(sChainedCallback != nullptr)
    {
        sChainedCallback(severity, message);
    }
#endif
}

// Initialize test harness logging to use a logging function that will record logs
// to LogRecording (when recording is active) and then calls the chained logging
// function.
inline void initializeChainedTestLogging(const std::string& componentName,
                                         hipdnnCallback_t chainedCallback = nullptr)
{
    sChainedCallback = chainedCallback;
    hipdnn::logging::initializeCallbackLogging(componentName, testChainedLoggingCallback);
}

inline void testLoggingCallback(hipdnnSeverity_t severity, const char* message)
{
    // Record log if recording of logs for testing is active
    LogRecording::instance().recordLog(severity, message);

#ifndef DISABLE_TEST_LOGGING
    std::string logLevelStr = hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_LEVEL", "off");

    if(logLevelStr == "off")
    {
        return;
    }

    hipdnnSeverity_t configuredLevel = stringToSeverity(logLevelStr);

    if(severity >= configuredLevel)
    {
        std::cerr << message << '\n';
    }
#endif
}

inline void initializeSpdlogDefaultLogger(const std::string& componentName)
{
#ifndef DISABLE_TEST_LOGGING
    spdlog::drop_all();
    auto logger = spdlog::stdout_color_mt(componentName);
    logger->set_formatter(std::make_unique<hipdnn_data_sdk::logging::ComponentFormatter>());
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info); // Set default log level
#endif
}

} // namespace hipdnn_test_sdk::utilities
