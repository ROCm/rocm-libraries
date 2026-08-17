// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_backend.h>
#include <spdlog/sinks/base_sink.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>

namespace hipdnn_backend
{
namespace logging
{

/**
 * Custom spdlog sink that invokes user callback with user handle.
 *
 * Uses atomic callback pointer to allow instant disable when unregistering.
 * Inherits from base_sink<std::mutex> which protects sink_it_() with a mutex.
 */
// NOLINTNEXTLINE(portability-template-virtual-member-function)
class UserCallbackSink : public spdlog::sinks::base_sink<std::mutex>
{
public:
    UserCallbackSink(std::shared_ptr<std::atomic<hipdnnUserLogCallback_t>> callbackHolder,
                     hipdnnUserLogCallbackHandle_t userHandle)
        : _callbackHolder(std::move(callbackHolder))
        , _userHandle(userHandle)
    {
    }

    // Wait until any in-progress callback invocation completes.
    // Call this after setting callback to nullptr to wait for any
    // in-progress sink_it_() call to complete.
    // Blocks on the base_sink mutex, which is held during sink_it_().
    void waitForIdle()
    {
        const std::lock_guard<std::mutex> lock(mutex_);
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        auto callback = _callbackHolder->load(std::memory_order_acquire);
        if(callback == nullptr)
        {
            return;
        }

        // Format message
        spdlog::memory_buf_t formatted;
        formatter_->format(msg, formatted);
        const std::string message(formatted.data(), formatted.size());

        // Convert spdlog level to hipdnnSeverity_t
        const hipdnnSeverity_t severity = fromSpdlogLevel(msg.level);

        // Call user callback with their handle (exception safe)
        try
        {
            callback(_userHandle, severity, message.c_str());
        }
        catch(const std::exception& e)
        {
            std::cerr << "[hipDNN] User log callback threw exception: " << e.what() << '\n';
        }
        catch(...)
        {
            std::cerr << "[hipDNN] User log callback threw unknown exception\n";
        }
    }

    void flush_() override
    {
        // Nothing to flush for callback
    }

private:
    std::shared_ptr<std::atomic<hipdnnUserLogCallback_t>> _callbackHolder;
    hipdnnUserLogCallbackHandle_t _userHandle;

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

} // namespace logging
} // namespace hipdnn_backend
