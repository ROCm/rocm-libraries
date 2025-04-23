/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include "include/logger.hpp"

#include <string.h>
#include <time.h>
#include <unistd.h>

#include <mutex>

namespace hiptensor
{
    Logger::Logger()
        : mEnabled(true)
        , mOwnsStream(false)
        , mLogMask(0)
        , mWriteStream(stdout)
        , mCallback(nullptr)
    {
    }

    Logger::~Logger()
    {
        std::scoped_lock lock(mMutex);
        if(mOwnsStream && mWriteStream != nullptr)
        {
            fclose(mWriteStream);
        }
    }

    Logger::Status_t Logger::writeToStream(FILE* stream)
    {
        std::scoped_lock lock(mMutex);
        if(stream != nullptr)
        {
            if(mOwnsStream && mWriteStream != nullptr)
            {
                fclose(mWriteStream);
                mOwnsStream = false;
            }
            mWriteStream = stream;
            return Status_t::SUCCESS;
        }

        return Status_t::INVALID_FILE_STREAM;
    }

    Logger::Status_t Logger::openFileStream(const char* fileName)
    {
        std::scoped_lock lock(mMutex);
        if(fileName != nullptr && strcmp(fileName, "") != 0)
        {
            if(mOwnsStream && mWriteStream != nullptr)
            {
                fclose(mWriteStream);
                mOwnsStream = false;
            }
            mWriteStream = fopen(fileName, "w");
            if(mWriteStream == nullptr)
            {
                // Revert back to stdout
                mOwnsStream  = false;
                mWriteStream = stdout;
                return Status_t::FILE_OPEN_FAILED;
            }
            mOwnsStream = true;
            return Status_t::SUCCESS;
        }

        // Revert back to stdout
        mOwnsStream  = false;
        mWriteStream = stdout;
        return Status_t::INVALID_FILE_NAME;
    }

    Logger::Status_t Logger::setCallback(Callback_t callbackFunc)
    {
        std::scoped_lock lock(mMutex);

        mCallback = callbackFunc;
        return Status_t::SUCCESS;
    }

    int32_t Logger::getLogMask() const
    {
        std::scoped_lock lock(mMutex);
        return mLogMask;
    }

    Logger::Status_t Logger::setLogLevel(LogLevel_t level)
    {
        std::scoped_lock lock(mMutex);
        switch(level)
        {
        // Only accept discretized log level
        case LogLevel_t::LOG_LEVEL_OFF:
        case LogLevel_t::LOG_LEVEL_ERROR:
        case LogLevel_t::LOG_LEVEL_PERF_TRACE:
        case LogLevel_t::LOG_LEVEL_PERF_HINT:
        case LogLevel_t::LOG_LEVEL_HEURISTICS_TRACE:
        case LogLevel_t::LOG_LEVEL_API_TRACE:
        {
            mLogMask = (int32_t)level;
            return Status_t::SUCCESS;
        }
        default:
        {
            return Status_t::INVALID_LOG_LEVEL;
        }
        }
    }

    Logger::Status_t Logger::setLogMask(int32_t mask)
    {
        std::scoped_lock lock(mMutex);
        if(mask >= 0 && mask <= 0x1F)
        {
            mLogMask = mask;
            return Status_t::SUCCESS;
        }
        return Status_t::INVALID_LOG_MASK;
    }

    void Logger::disable()
    {
        std::scoped_lock lock(mMutex);
        mEnabled = false;
    }

    void Logger::enable()
    {
        std::scoped_lock lock(mMutex);
        mEnabled = true;
    }

    Logger::Status_t
        Logger::logMessage(int32_t context, const char* apiFuncName, const char* message)
    {
        std::scoped_lock lock(mMutex);
        if((context & mLogMask) > 0 && mEnabled)
        {
            // Init message
            char buff[2048];
            snprintf(buff,
                     sizeof(buff),
                     "[%d][%s][hipTensor][%s][%s] %s\n",
                     appPid(),
                     timeStamp(),
                     contextString((LogLevel_t)context),
                     apiFuncName,
                     message);

            // Invoke logger callback
            if(mCallback != nullptr)
            {
                (*mCallback)(context, apiFuncName, buff);
            }

            // Log to stream
            fprintf(mWriteStream, "%s", buff);
        }

        return Status_t::SUCCESS;
    }

    Logger::Status_t Logger::logError(const char* apiFuncName, const char* message)
    {
        return Logger::logMessage(
            static_cast<int>(LogLevel_t::LOG_LEVEL_ERROR), apiFuncName, message);
    }

    Logger::Status_t Logger::logPerformanceTrace(const char* apiFuncName, const char* message)
    {
        return Logger::logMessage(
            static_cast<int>(LogLevel_t::LOG_LEVEL_PERF_TRACE), apiFuncName, message);
    }

    Logger::Status_t Logger::logHeuristics(const char* apiFuncName, const char* message)
    {
        return Logger::logMessage(
            static_cast<int>(LogLevel_t::LOG_LEVEL_HEURISTICS_TRACE), apiFuncName, message);
    }

    Logger::Status_t Logger::logAPITrace(const char* apiFuncName, const char* message)
    {
        return Logger::logMessage(
            static_cast<int>(LogLevel_t::LOG_LEVEL_API_TRACE), apiFuncName, message);
    }

    /* static */
    const char* Logger::timeStamp()
    {
        static std::mutex tsMutex;
        std::scoped_lock  lock(tsMutex);

        // Statically allocate buffer.
        // Beware thread concurrency.
        static char buff[32];

        time_t     t;
        struct tm* tInfo;

        // Retrieve the time information
        time(&t);
        tInfo = localtime(&t);

        // Format the timestamp string
        // YYYY-MM-DD HH:MM:SS
        strftime(buff, 32, "%F %T", tInfo);
        return buff;
    }

    /* static */
    const char* Logger::statusString(Status_t status)
    {
        switch(status)
        {
        case Status_t::SUCCESS:
            return "SUCCESS";
        case Status_t::INVALID_FILE_NAME:
            return "INVALID_FILE_NAME";
        case Status_t::INVALID_FILE_STREAM:
            return "INVALID_FILE_STREAM";
        case Status_t::INVALID_CALLBACK:
            return "INVALID_CALLBACK";
        case Status_t::INVALID_LOG_MASK:
            return "INVALID_LOG_MASK";
        case Status_t::INVALID_LOG_LEVEL:
            return "INVALID_LOG_LEVEL";
        case Status_t::FILE_OPEN_FAILED:
            return "FILE_OPEN_FAILED";
        default:
            return "STATUS_UNKNOWN";
        }
    }

    /* static */
    int32_t Logger::appPid()
    {
        // App PID won't generally change
        static int pid = getpid();
        return pid;
    }

    /* static */
    const char* Logger::contextString(LogLevel_t context)
    {
        // NOTE: MUST align with hiptensorLogLevel_t
        switch(context)
        {
        case LogLevel_t::LOG_LEVEL_ERROR:
            return "Error";
        case LogLevel_t::LOG_LEVEL_PERF_TRACE:
            return "Performance";
        case LogLevel_t::LOG_LEVEL_PERF_HINT:
            return "Performance Hint";
        case LogLevel_t::LOG_LEVEL_HEURISTICS_TRACE:
            return "Heuristics Trace";
        case LogLevel_t::LOG_LEVEL_API_TRACE:
            return "API";
        default:
            return "";
        }
    }

} // namespace hiptensor
