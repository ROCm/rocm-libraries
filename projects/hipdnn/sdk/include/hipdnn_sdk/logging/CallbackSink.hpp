// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "CallbackTypes.h"
#include "LoggingUtils.hpp"
#include <functional>
#include <mutex>
#include <spdlog/async.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/details/synchronous_factory.h>
#include <spdlog/sinks/base_sink.h>
#include <string>

namespace hipdnn_sdk::logging
{

inline hipdnnSeverity_t spdlogToHipdnnSeverity(spdlog::level::level_enum level)
{
    switch(level)
    {
    case spdlog::level::critical:
        return HIPDNN_SEV_FATAL;
    case spdlog::level::err:
        return HIPDNN_SEV_ERROR;
    case spdlog::level::warn:
        return HIPDNN_SEV_WARN;
    case spdlog::level::off:
        return HIPDNN_SEV_OFF;
    default:
        return HIPDNN_SEV_INFO;
    }
}

// This warning is for cross-compiler compatibility, so it's safe to ignore since we only use Clang.
// Moreover, MSVC is likely the incompatible compiler here.
// NOLINTBEGIN(portability-template-virtual-member-function)
template <typename Mutex>
class CallbackSink final : public spdlog::sinks::base_sink<Mutex>
{
public:
    explicit CallbackSink(hipdnnCallback_t callback)
        : _callbackFn{callback}
    {
    }
    ~CallbackSink() override = default;

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        if(_callbackFn == nullptr)
        {
            return;
        }

        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        std::string formattedStr(formatted.data(), formatted.size());

        while(!formattedStr.empty() && (formattedStr.back() == '\n' || formattedStr.back() == '\r'))
        {
            formattedStr.pop_back();
        }

        hipdnnSeverity_t severity = spdlogToHipdnnSeverity(msg.level);

        _callbackFn(severity, formattedStr.c_str());
    }

    void flush_() override {}

private:
    hipdnnCallback_t _callbackFn;
};
// NOLINTEND(portability-template-virtual-member-function)

using CallbackSinkMt = CallbackSink<std::mutex>;

inline std::shared_ptr<spdlog::logger> createAsyncCallbackLoggerMt(hipdnnCallback_t callback,
                                                                   const std::string& source)
{
    auto sink = std::make_shared<hipdnn_sdk::logging::CallbackSinkMt>(callback);
    auto logger = std::make_shared<spdlog::async_logger>(
        source, sink, spdlog::thread_pool(), spdlog::async_overflow_policy::block);
    logger->set_pattern(generatePatternString(source));
    logger->flush_on(spdlog::level::info);
    return logger;
}

template <typename Factory = spdlog::synchronous_factory>
inline std::shared_ptr<spdlog::logger> createCallbackLoggerMt(hipdnnCallback_t callback,
                                                              const std::string& source)
{
    auto logger = Factory::template create<hipdnn_sdk::logging::CallbackSinkMt>(source, callback);
    logger->set_pattern(generatePatternString(source));
    return logger;
}

} // namespace hipdnn_sdk::logging
