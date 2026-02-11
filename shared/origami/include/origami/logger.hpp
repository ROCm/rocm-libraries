/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
