// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace origami {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

enum class LogFormat {
    TEXT,
    CSV
};

class Logger {
public:
    static Logger& instance();
    bool is_enabled() const { return enabled_.load(std::memory_order_acquire); }
    LogFormat format() const { return format_; }

    void log(LogLevel level, const std::string& message, const char* file, int line);
    void flush();
    void update_from_env();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

private:
    Logger();
    ~Logger();

    const char* level_to_string(LogLevel level) const;

    void process_debug_message(const std::string& message);
    void begin_row();
    void record(const std::string& key, const std::string& value);
    void end_row();
    void flush_csv_locked();
    static std::string escape_csv(const std::string& field);

    std::string file_path_;
    std::ofstream log_file_;
    std::mutex mutex_;
    std::atomic<bool> enabled_;
    LogFormat format_ = LogFormat::TEXT;

    std::vector<std::string> columns_;
    std::unordered_map<std::string, size_t> column_index_;
    std::vector<std::vector<std::string>> rows_;
    bool header_written_ = false;
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
    do { \
        if (origami::Logger::instance().is_enabled()) \
            origami::LogStream(origami::LogLevel::DEBUG, __FILE__, __LINE__) << msg; \
    } while (0)

#define ORIGAMI_LOG_INFO(msg) \
    do { \
        if (origami::Logger::instance().is_enabled()) \
            origami::LogStream(origami::LogLevel::INFO, __FILE__, __LINE__) << msg; \
    } while (0)

#define ORIGAMI_LOG_WARNING(msg) \
    do { \
        if (origami::Logger::instance().is_enabled()) \
            origami::LogStream(origami::LogLevel::WARNING, __FILE__, __LINE__) << msg; \
    } while (0)

#define ORIGAMI_LOG_ERROR(msg) \
    do { \
        if (origami::Logger::instance().is_enabled()) \
            origami::LogStream(origami::LogLevel::ERROR, __FILE__, __LINE__) << msg; \
    } while (0)

#define OLOG_DEBUG ORIGAMI_LOG_DEBUG
#define OLOG_INFO ORIGAMI_LOG_INFO
#define OLOG_WARNING ORIGAMI_LOG_WARNING
#define OLOG_ERROR ORIGAMI_LOG_ERROR
