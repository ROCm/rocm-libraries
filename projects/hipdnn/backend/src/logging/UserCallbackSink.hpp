// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_backend.h>
#include <spdlog/sinks/base_sink.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace hipdnn_backend
{
namespace logging
{

/**
 * Custom spdlog sink that invokes user callback with user handle.
 *
 * Uses atomic callback pointer to allow instant disable when unregistering.
 * Uses atomic _isExecuting flag to provide synchronous guarantee that no
 * callback is in progress when unregister returns.
 */
class UserCallbackSink : public spdlog::sinks::base_sink<std::mutex>
{
public:
    UserCallbackSink(std::shared_ptr<std::atomic<hipdnnUserLogCallback_t>> callbackHolder,
                     hipdnnUserLogCallbackHandle_t userHandle)
        : _callbackHolder(std::move(callbackHolder))
        , _userHandle(userHandle)
        , _isExecuting(false)
    {
    }

    // Wait until any in-progress callback invocation completes.
    // Call this after setting callback to nullptr to wait for any
    // in-progress sink_it_() function call to complete before returning.
    void waitForIdle() const
    {
        while(_isExecuting.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        // Mark execution as started
        _isExecuting.store(true, std::memory_order_release);

        auto callback = _callbackHolder->load(std::memory_order_acquire);
        if(callback == nullptr)
        {
            _isExecuting.store(false, std::memory_order_release);
            return;
        }

        // Format message
        spdlog::memory_buf_t formatted;
        formatter_->format(msg, formatted);
        std::string message(formatted.data(), formatted.size());

        // Convert spdlog level to hipdnnSeverity_t
        hipdnnSeverity_t severity = fromSpdlogLevel(msg.level);

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

        // Mark execution as complete
        _isExecuting.store(false, std::memory_order_release);
    }

    void flush_() override
    {
        // Nothing to flush for callback
    }

private:
    std::shared_ptr<std::atomic<hipdnnUserLogCallback_t>> _callbackHolder;
    hipdnnUserLogCallbackHandle_t _userHandle;
    mutable std::atomic<bool> _isExecuting;

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
