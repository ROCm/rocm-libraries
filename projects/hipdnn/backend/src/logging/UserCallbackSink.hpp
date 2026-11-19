// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_backend.h>
#include <spdlog/details/log_msg.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/sink.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace hipdnn_backend
{
namespace logging
{

/**
 * Custom spdlog sink that invokes user callback with user handle.
 *
 * One registration never runs on two threads at once: invocations are serialized by a
 * recursive mutex of this sink, and no other lock is held while the callback runs. The
 * callback may therefore register, update or remove callbacks, including its own
 * registration. The atomic callback pointer disables the sink instantly when it is
 * unregistered, and waitForIdle() waits for invocations that are already in progress.
 */
class UserCallbackSink : public spdlog::sinks::sink
{
public:
    UserCallbackSink(std::shared_ptr<std::atomic<hipdnnUserLogCallback_t>> callbackHolder,
                     hipdnnUserLogCallbackHandle_t userHandle)
        : _callbackHolder(std::move(callbackHolder))
        , _userHandle(userHandle)
        , _formatter(std::make_unique<spdlog::pattern_formatter>())
    {
    }

    void log(const spdlog::details::log_msg& msg) override
    {
        // Recursive, so the callback's own thread may re-enter this sink without deadlock.
        const std::lock_guard<std::recursive_mutex> invocationLock(_invocationMutex);

        // Counted only once the invocation lock is held, so a callback that removes itself
        // never waits for an invocation that is blocked behind its own.
        const InvocationScope scope(*this);

        // Loaded after the invocation is counted: a waitForIdle() that follows the store of
        // nullptr either waits for this invocation or this load sees nullptr.
        auto callback = _callbackHolder->load(std::memory_order_acquire);
        if(callback == nullptr)
        {
            return;
        }

        spdlog::memory_buf_t formatted;
        {
            const std::lock_guard<std::mutex> lock(_formatterMutex);
            _formatter->format(msg, formatted);
        }
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

    void flush() override
    {
        // Nothing to flush for callback
    }

    void set_pattern(const std::string& pattern) override
    {
        set_formatter(std::make_unique<spdlog::pattern_formatter>(pattern));
    }

    void set_formatter(std::unique_ptr<spdlog::formatter> sinkFormatter) override
    {
        const std::lock_guard<std::mutex> lock(_formatterMutex);
        _formatter = std::move(sinkFormatter);
    }

    // Wait until any in-progress callback invocation completes. Call this after setting the
    // callback to nullptr; no invocation that starts afterwards calls the user callback.
    // Invocations on the calling thread are not waited for: a callback that unregisters
    // itself returns instead of waiting for its own frame.
    void waitForIdle()
    {
        const std::ptrdiff_t ownInvocations = InvocationScope::countOnThisThread(this);

        std::unique_lock<std::mutex> lock(_idleMutex);
        _idleCv.wait(lock, [&] { return _inFlight <= ownInvocations; });
    }

private:
    // Counts one invocation of log() and links it into the calling thread's list of running
    // invocations, innermost first.
    class InvocationScope
    {
    public:
        explicit InvocationScope(UserCallbackSink& sink)
            : _sink(&sink)
            , _outer(innermost())
        {
            {
                const std::lock_guard<std::mutex> lock(sink._idleMutex);
                ++sink._inFlight;
            }
            innermost() = this;
        }

        ~InvocationScope()
        {
            innermost() = _outer;
            const std::lock_guard<std::mutex> lock(_sink->_idleMutex);
            --_sink->_inFlight;
            _sink->_idleCv.notify_all();
        }

        InvocationScope(const InvocationScope&) = delete;
        InvocationScope& operator=(const InvocationScope&) = delete;
        InvocationScope(InvocationScope&&) = delete;
        InvocationScope& operator=(InvocationScope&&) = delete;

        // Number of invocations of sink running on the calling thread.
        static std::ptrdiff_t countOnThisThread(const UserCallbackSink* sink)
        {
            std::ptrdiff_t count = 0;
            for(const InvocationScope* scope = innermost(); scope != nullptr; scope = scope->_outer)
            {
                if(scope->_sink == sink)
                {
                    ++count;
                }
            }
            return count;
        }

    private:
        // A plain pointer has no destructor, so waitForIdle() stays safe while thread-local
        // and static objects are destroyed (loggerShutdown() runs from ~BackendLogState).
        static const InvocationScope*& innermost()
        {
            thread_local const InvocationScope* s_innermost = nullptr;
            return s_innermost;
        }

        UserCallbackSink* _sink;
        const InvocationScope* _outer;
    };

    std::shared_ptr<std::atomic<hipdnnUserLogCallback_t>> _callbackHolder;
    hipdnnUserLogCallbackHandle_t _userHandle;

    // Held around each invocation; see log().
    std::recursive_mutex _invocationMutex;

    std::mutex _formatterMutex;
    std::unique_ptr<spdlog::formatter> _formatter;

    std::mutex _idleMutex;
    std::condition_variable _idleCv;
    std::ptrdiff_t _inFlight = 0;

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

/**
 * Delivers each message to the registered user callback sinks without holding a lock while
 * they run. Adding or removing a sink publishes a new copy of the sink list; a message goes
 * to the list that was current when it was dispatched.
 */
class UserCallbackDispatchSink : public spdlog::sinks::sink
{
public:
    void addSink(std::shared_ptr<UserCallbackSink> sink)
    {
        const std::lock_guard<std::mutex> lock(_listMutex);
        auto sinks = std::make_shared<SinkList>(*_sinks);
        sinks->push_back(std::move(sink));
        _sinks = std::move(sinks);
    }

    void removeSink(const std::shared_ptr<UserCallbackSink>& sink)
    {
        const std::lock_guard<std::mutex> lock(_listMutex);
        auto sinks = std::make_shared<SinkList>(*_sinks);
        sinks->erase(std::remove(sinks->begin(), sinks->end(), sink), sinks->end());
        _sinks = std::move(sinks);
    }

    void log(const spdlog::details::log_msg& msg) override
    {
        const auto sinks = snapshot();
        for(const auto& sink : *sinks)
        {
            if(sink->should_log(msg.level))
            {
                sink->log(msg);
            }
        }
    }

    void flush() override
    {
        // Nothing to flush for callbacks
    }

    // Each user callback sink keeps the pattern it was given when it was registered.
    void set_pattern(const std::string& /*pattern*/) override {}

    void set_formatter(std::unique_ptr<spdlog::formatter> /*sinkFormatter*/) override {}

private:
    using SinkList = std::vector<std::shared_ptr<UserCallbackSink>>;

    std::shared_ptr<const SinkList> snapshot() const
    {
        const std::lock_guard<std::mutex> lock(_listMutex);
        return _sinks;
    }

    // Guards only the publication of _sinks; never held while a sink runs.
    mutable std::mutex _listMutex;
    std::shared_ptr<const SinkList> _sinks = std::make_shared<const SinkList>();
};

} // namespace logging
} // namespace hipdnn_backend
