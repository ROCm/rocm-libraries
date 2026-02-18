// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <iostream>
#include <mutex>
#include <spdlog/details/log_msg.h>
#include <spdlog/sinks/base_sink.h>
#include <string>

namespace hipdnn_backend::logging
{

/**
 * @brief Custom spdlog sink that invokes the global backend log output callback function.
 *
 * This sink is used to redirect log messages to the backend's global log output callback
 * instead of writing to console or file. It handles exceptions from the callback to
 * prevent crashes.
 */
// NOLINTNEXTLINE(portability-template-virtual-member-function)
class BackendLogOutputSink : public spdlog::sinks::base_sink<std::mutex>
{
public:
    /**
     * @brief Construct a new BackendLogOutputSink
     *
     * @param callback The backend log output callback function to invoke
     */
    explicit BackendLogOutputSink(hipdnnBackendLogOutputCallback_t callback)
        : _callback(callback)
    {
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        if(_callback == nullptr)
        {
            return;
        }

        // Format message (strip trailing newline)
        spdlog::memory_buf_t formatted;
        formatter_->format(msg, formatted);
        std::string message(formatted.data(), formatted.size());

        // Convert spdlog level to hipdnnSeverity_t
        hipdnnSeverity_t severity = fromSpdlogLevel(msg.level);

        // Call backend log output callback (catch exceptions to prevent crashes)
        try
        {
            _callback(severity, hipdnn_data_sdk::utilities::removeNewlines(message).c_str());
        }
        catch(const std::exception& e)
        {
            // Log error to stderr as fallback (don't recursively log through hipDNN)
            // Wrap stderr output in try-catch to prevent exception-during-exception
            try
            {
                std::cerr << "[hipDNN] Backend log output callback threw exception: " << e.what()
                          << '\n';
            }
            catch(...)
            {
                // If stderr also fails, disable callback
                _callback = nullptr;
            }
        }
        catch(...)
        {
            try
            {
                std::cerr << "[hipDNN] Backend log output callback threw unknown exception\n";
            }
            catch(...)
            {
                // If stderr also fails, disable callback
                _callback = nullptr;
            }
        }
    }

    void flush_() override
    {
        // Nothing to flush for callback
    }

private:
    hipdnnBackendLogOutputCallback_t _callback;

    // Convert spdlog level to hipdnnSeverity_t
    static hipdnnSeverity_t fromSpdlogLevel(spdlog::level::level_enum level)
    {
        switch(level)
        {
        case spdlog::level::critical:
            return HIPDNN_SEV_FATAL;
        case spdlog::level::err:
            return HIPDNN_SEV_ERROR;
        case spdlog::level::warn:
            return HIPDNN_SEV_WARN;
        case spdlog::level::info:
            return HIPDNN_SEV_INFO;
        case spdlog::level::off:
        default:
            return HIPDNN_SEV_OFF;
        }
    }
};

} // namespace hipdnn_backend::logging
