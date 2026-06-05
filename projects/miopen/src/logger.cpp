/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include <miopen/env.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/logger.hpp>
#include <miopen/config.h>
#include <miopen/sysinfo_utils.hpp>

#include <cstdlib>
#include <chrono>
#include <fstream>
#include <ios>
#include <iomanip>
#include <sstream>

#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h> /* For SYS_xxx definitions */
#endif

/// Enable logging of the most important function calls.
/// Name of envvar in a bit inadequate due to historical reasons.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_ENABLE_LOGGING)

/// Prints driver command lines into log.
/// Works from any application which uses the library.
/// Allows to reproduce library use cases using the driver instead of the actual application.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_ENABLE_LOGGING_CMD)

/// Prefix each log line with information which allows the user
/// to uniquiely identify log records printed from different processes
/// or threads. Useful for debugging multi-process/multi-threaded apps.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_ENABLE_LOGGING_MPMT)

/// Add timestamps to each log line.
/// Not useful  with multi-process/multi-threaded apps.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME)
/// Add timestamps to each log line.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_ENABLE_LOGGING_DATE_TIME)

/// See LoggingLevel in the header.
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_LOG_LEVEL)

/// Enable logging of function calls to ROCTX api.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_ENABLE_LOGGING_ROCTX)

/// Disable logging quieting.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_LOGGING_QUIETING_DISABLE)

MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_LOG_BUFFER_SIZE, 0);

namespace miopen {

size_t GetBufferSize() { return env::value(MIOPEN_LOG_BUFFER_SIZE); }

size_t& GetBufferIdx()
{
    static thread_local size_t log_buffer_i = 0;
    return log_buffer_i;
}

// Optimization: Thread-local stream pool to avoid repeated ostringstream construction
std::ostringstream& GetThreadLocalLogStream()
{
    static thread_local std::ostringstream stream;
    stream.str(""); // Clear contents
    stream.clear(); // Clear state flags
    return stream;
}

std::vector<std::string>& GetLogBuffer()
{
    auto log_buffer_size = GetBufferSize();
    static thread_local std::vector<std::string> log_buffer(log_buffer_size, "");
    if(log_buffer_size != log_buffer.size())
    {
        log_buffer.resize(log_buffer_size);
        auto& log_buffer_i = GetBufferIdx();
        if(log_buffer_i > 0 && log_buffer_i >= log_buffer_size)
            log_buffer_i = 0;
    }
    return log_buffer;
}

bool IsLogBufferOn() { return GetBufferSize() != 0 && !IsLogging(LoggingLevel::Info2); }

void ClearLogBuffer()
{
    auto& log_buffer   = GetLogBuffer();
    auto& log_buffer_i = GetBufferIdx();
    log_buffer         = std::vector<std::string>(GetBufferSize(), "");
    log_buffer_i       = 0;
}

void BufferLog(std::string&& line)
{
    auto& log_buffer         = GetLogBuffer();
    auto& log_buffer_i       = GetBufferIdx();
    log_buffer[log_buffer_i] = std::move(line);
    log_buffer_i             = (log_buffer_i + 1) % GetBufferSize();
}

void LogXQCustomImpl(const LoggingLevel level,
                     const bool disableQuieting,
                     const std::string_view category,
                     const std::string_view fn_name,
                     std::string message)
{
    const bool is_logging = IsLogging(level, disableQuieting);
    if(is_logging)
    {
        // Path 1: Logging to stderr is enabled - use full prefix.
        auto& miopen_log_ss = GetThreadLocalLogStream();
        miopen_log_ss << LoggingPrefix() << category << " [" << fn_name << "] " << message
                      << std::endl;
        std::cerr << miopen_log_ss.str();
        // Also buffer if buffer is enabled.
        if(IsLogBufferOn())
        {
            if(level < LoggingLevel::Trace)
            {
                BufferLog(miopen_log_ss.str());
            }
            if(level == LoggingLevel::Error)
                OutputBufferedLogs();
        }
    }
    else
    {
        // Path 2: Logging disabled, buffer-only - use minimal prefix.
        if(IsLogBufferOn())
        {
            if(level < LoggingLevel::Trace)
            {
                auto& miopen_log_ss = GetThreadLocalLogStream();
                miopen_log_ss << LoggingPrefixMinimal() << category << " [" << fn_name << "] "
                              << message << std::endl;
                BufferLog(miopen_log_ss.str());
            }
            if(level == LoggingLevel::Error)
                OutputBufferedLogs();
        }
    }
}

void OutputBufferedLogs()
{
    auto& log_buffer     = GetLogBuffer();
    auto& log_buffer_i   = GetBufferIdx();
    auto log_buffer_size = GetBufferSize();
    auto buffer_size     = (log_buffer[log_buffer_size - 1] == "") ? log_buffer_i : log_buffer_size;
    auto filename =
        fs::temp_directory_path() / ("miopen_error_" + std::to_string(::getpid()) + ".log");
    std::cerr << "Buffered " << buffer_size << " messages to file: " << sysinfo::GetSystemHostname()
              << ":" << filename.string() << std::endl;
    auto err_file = std::ofstream{filename, std::ofstream::app};
    size_t i      = log_buffer_i;
    do
    {
        if(log_buffer[i] != "")
        {
            err_file << log_buffer[i];
        }
        i = (i + 1) % log_buffer_size;
    } while(i != log_buffer_i);
    err_file << std::endl;
    ClearLogBuffer();
}

namespace debug {

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
MIOPEN_EXPORT bool LoggingQuiet = false;

} // namespace debug

namespace {

inline bool operator!=(const int& lhs, const LoggingLevel& rhs)
{
    return lhs != static_cast<int>(rhs);
}
inline bool operator==(const int& lhs, const LoggingLevel& rhs)
{
    return lhs == static_cast<int>(rhs);
}
inline bool operator>=(const int& lhs, const LoggingLevel& rhs)
{
    return lhs >= static_cast<int>(rhs);
}
inline bool operator>(const int& lhs, const LoggingLevel& rhs)
{
    return lhs > static_cast<int>(rhs);
}

/// Returns value which uniquiely identifies current process/thread
/// and can be printed into logs for MP/MT environments.
inline int GetProcessAndThreadId()
{
#ifdef __linux__
    // LWP is fine for identifying both processes and threads.
    return syscall(SYS_gettid); // NOLINT
#else
    return 0; // Not implemented.
#endif
}

inline float GetTimeDiff()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static auto prev = std::chrono::steady_clock::now();
    auto now         = std::chrono::steady_clock::now();
    auto rv =
        std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(now - prev).count();
    prev = now;
    return rv;
}

inline std::string GetDateTimeMs()
{
    auto now = std::chrono::system_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto now_t = std::chrono::system_clock::to_time_t(now);

    std::tm local_time{};
#if defined(_WIN32)
    if(localtime_s(&local_time, &now_t) != 0)
        return std::string{};
#else
    if(localtime_r(&now_t, &local_time) == nullptr)
        return std::string{};
#endif

    std::ostringstream time_s;
    time_s << std::put_time(&local_time, "%Y-%m-%dT%H:%M:%S");
    time_s << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    return time_s.str();
}

} // namespace

bool IsLoggingDebugQuiet()
{
    return debug::LoggingQuiet && !env::enabled(MIOPEN_DEBUG_LOGGING_QUIETING_DISABLE);
}

bool IsLoggingFunctionCalls()
{
    return env::enabled(MIOPEN_ENABLE_LOGGING) && !IsLoggingDebugQuiet();
}

bool IsLoggingToRoctx()
{
    return env::enabled(MIOPEN_ENABLE_LOGGING_ROCTX) && !IsLoggingDebugQuiet();
}

bool IsLogging(const LoggingLevel level, const bool disableQuieting)
{
    auto enabled_level = env::value(MIOPEN_LOG_LEVEL);
    if(IsLoggingDebugQuiet() && !disableQuieting)
    {
        // Disable all levels higher than fatal.
        if(enabled_level > LoggingLevel::DebugQuietMax || enabled_level == LoggingLevel::Default)
            enabled_level = static_cast<int>(LoggingLevel::DebugQuietMax);
    }
    if(enabled_level != LoggingLevel::Default)
        return enabled_level >= level;
#ifdef NDEBUG // Simplest way.
    return LoggingLevel::Warning >= level;
#else
    return LoggingLevel::Info >= level;
#endif
}

std::string LoggingLevelToCustomString(const LoggingLevel level, const char* custom)
{
    std::ostringstream oss;
    oss << custom << " " << LoggingLevelToCString(level);
    return oss.str();
}

const char* LoggingLevelToCString(const LoggingLevel level)
{
    switch(level)
    {
    case LoggingLevel::Default: return "Default";
    case LoggingLevel::Quiet: return "Quiet";
    case LoggingLevel::Fatal: return "Fatal";
    case LoggingLevel::Error: return "Error";
    case LoggingLevel::Warning: return "Warning";
    case LoggingLevel::Info: return "Info";
    case LoggingLevel::Info2: return "Info2";
    case LoggingLevel::Trace: return "Trace";
    default: return "<Unknown>";
    }
}

bool IsLoggingCmd() { return env::enabled(MIOPEN_ENABLE_LOGGING_CMD) && !IsLoggingDebugQuiet(); }

std::string LoggingPrefix()
{
    std::stringstream ss;
    if(env::enabled(MIOPEN_ENABLE_LOGGING_MPMT))
    {
        ss << sysinfo::GetSystemHostname() << ' ';
        ss << GetProcessAndThreadId() << ' ';
    }
    ss << "MIOpen";
#if MIOPEN_BACKEND_OPENCL
    ss << "(OpenCL)";
#elif MIOPEN_BACKEND_HIP
    ss << "(HIP)";
#endif
    if(env::enabled(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME))
    {
        ss << std::fixed << std::setprecision(3) << std::setw(8) << GetTimeDiff();
    }
    if(env::enabled(MIOPEN_ENABLE_LOGGING_DATE_TIME))
    {
        ss << " (" << GetDateTimeMs() << ")";
    }
    ss << ": ";
    return ss.str();
}

std::string LoggingPrefixMinimal()
{
    // Optimization: Cache the static prefix to avoid repeated string construction
    // This minimal prefix skips expensive syscalls (hostname, TID, timestamps)
    // and is used for buffer-only logs when actual logging is disabled.
    static const std::string prefix = []() {
        std::stringstream ss;
        ss << "MIOpen";
#if MIOPEN_BACKEND_OPENCL
        ss << "(OpenCL)";
#elif MIOPEN_BACKEND_HIP
        ss << "(HIP)";
#endif
        ss << ": ";
        return ss.str();
    }();
    return prefix;
}

} // namespace miopen
