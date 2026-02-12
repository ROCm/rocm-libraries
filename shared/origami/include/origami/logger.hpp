// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

namespace origami {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& instance();
    void log(LogLevel level, const std::string& message, const char* file, int line);
    bool is_enabled() const { return enabled_; }
    void flush();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

private:
    Logger();
    ~Logger();

    std::ofstream log_file_;
    std::mutex mutex_;
    bool enabled_;

    const char* level_to_string(LogLevel level) const;
};

class LogStream {
public:
    LogStream(LogLevel level, const char* file, int line)
        : level_(level), file_(file), line_(line) {}

    ~LogStream() {
        Logger::instance().log(level_, stream_.str(), file_, line_);
    }

    template<typename T>
    LogStream& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

private:
    std::ostringstream stream_;
    LogLevel level_;
    const char* file_;
    int line_;
};

} // namespace origami

#define ORIGAMI_LOG_DEBUG(msg) \
    if (origami::Logger::instance().is_enabled()) \
        origami::LogStream(origami::LogLevel::DEBUG, __FILE__, __LINE__) << msg

#define ORIGAMI_LOG_INFO(msg) \
    if (origami::Logger::instance().is_enabled()) \
        origami::LogStream(origami::LogLevel::INFO, __FILE__, __LINE__) << msg

#define ORIGAMI_LOG_WARNING(msg) \
    if (origami::Logger::instance().is_enabled()) \
        origami::LogStream(origami::LogLevel::WARNING, __FILE__, __LINE__) << msg

#define ORIGAMI_LOG_ERROR(msg) \
    if (origami::Logger::instance().is_enabled()) \
        origami::LogStream(origami::LogLevel::ERROR, __FILE__, __LINE__) << msg

#define OLOG_DEBUG ORIGAMI_LOG_DEBUG
#define OLOG_INFO ORIGAMI_LOG_INFO
#define OLOG_WARNING ORIGAMI_LOG_WARNING
#define OLOG_ERROR ORIGAMI_LOG_ERROR
