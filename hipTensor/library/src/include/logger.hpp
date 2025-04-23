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

#ifndef HIPTENSOR_LOGGER_HPP
#define HIPTENSOR_LOGGER_HPP

#include "singleton.hpp"

#include <mutex>

namespace hiptensor
{
    class Logger : public LazySingleton<Logger>
    {
    private:
        using Callback_t = void (*)(int32_t logLevel, const char* funcName, const char* msg);

    public:
        enum struct Status_t : int32_t
        {
            SUCCESS = 0,
            INVALID_FILE_NAME,
            INVALID_FILE_STREAM,
            INVALID_CALLBACK,
            INVALID_LOG_MASK,
            INVALID_LOG_LEVEL,
            FILE_OPEN_FAILED,
        };

        enum struct LogLevel_t : int32_t
        {
            LOG_LEVEL_OFF              = 0,
            LOG_LEVEL_ERROR            = 1,
            LOG_LEVEL_PERF_TRACE       = 2,
            LOG_LEVEL_PERF_HINT        = 4,
            LOG_LEVEL_HEURISTICS_TRACE = 8,
            LOG_LEVEL_API_TRACE        = 16,
        };

        // For static initialization
        friend std::unique_ptr<Logger> std::make_unique<Logger>();

        ~Logger();

        Status_t writeToStream(FILE* stream);
        Status_t openFileStream(const char* fileName);
        Status_t setCallback(Callback_t callbackFunc);
        int32_t  getLogMask() const;
        Status_t setLogMask(int32_t mask);
        Status_t setLogLevel(LogLevel_t level);
        void     disable();
        void     enable();

        Status_t logMessage(int32_t context, const char* apiFuncName, const char* message);
        Status_t logError(const char* apiFuncName, const char* message);
        Status_t logPerformanceTrace(const char* apiFuncName, const char* message);
        Status_t logHeuristics(const char* apiFuncName, const char* message);
        Status_t logAPITrace(const char* apiFuncName, const char* message);

        static const char* statusString(Status_t status);

    private:
        Logger();
        static const char* timeStamp();
        static int32_t     appPid();
        static const char* contextString(LogLevel_t context);

    private:
        bool mEnabled;
        bool mOwnsStream;

        int32_t    mLogMask;
        FILE*      mWriteStream;
        Callback_t mCallback;

        mutable std::mutex mMutex;
    };

} // namespace hiptensor

#endif // HIPTENSOR_LOGGER_HPP
