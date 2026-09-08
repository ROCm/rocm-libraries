// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <mutex>
#include <thread>

namespace hipdnn_data_sdk::utilities
{

/**
 * @brief Host-released device-side stall for gap-free device timing.
 *
 * A single hipEventRecord(start) / launch / hipEventRecord(stop) bracket records the
 * start event on an idle stream, so the device sits idle while the host finishes
 * validation, dispatch, and logging. hipEventElapsedTime then reports host submission
 * cost rather than device time, which dominates the measurement for short kernels.
 *
 * arm() enqueues a wait packet on the work stream, so every later item on that stream
 * (the start event, the kernels, the stop event) stays queued but unexecuted until
 * release() writes the signal from a private control stream. Each event timestamp is
 * taken when the event executes, not when the host enqueued it, so the measured span
 * starts when the device actually begins the work.
 *
 * Usage: arm(stream), record start, enqueue work, record stop, release(), synchronize.
 *
 * The class is non-throwing and best-effort: a device without stream-wait-value support
 * leaves isUsable() false and arm() returns false, which degrades to an unstalled
 * measurement instead of failing the caller. Callers that must fail loudly read
 * lastError() and lastOperation() to report which HIP call failed.
 *
 * ## Watchdog
 *
 * release() is a host action that runs after the measured region returns. Work inside
 * that region that blocks the host on the stalled stream -- hipStreamSynchronize,
 * hipDeviceSynchronize, a blocking hipMemcpy, or an implicitly synchronizing hipMalloc
 * or hipFree -- therefore deadlocks: the host waits for a stream that only the host can
 * release. A plugin's execute() is caller code, so this is reachable through the plugin
 * SDK no matter what the in-tree providers do.
 *
 * A watchdog thread bounds that wait. If the host has not released within the timeout,
 * the watchdog writes the signal itself. That is not merely an abort: the write is
 * exactly what the blocked host is waiting on, so the stream drains, the inner
 * synchronize returns, and the caller continues. A permanent hang becomes one slow
 * iteration.
 *
 * The timeout measures *host* time between arm() and release(), which is submission
 * cost only and independent of kernel duration, so a generous value cannot produce a
 * false positive on a long-running kernel.
 *
 * A watchdog release means the measurement is worthless: it contains the timeout, and
 * the host gap it was supposed to exclude. timedOut() reports it so the caller discards
 * the sample rather than averaging it. Firing also sets a process-wide sticky flag that
 * makes every later arm() a no-op, because the cause is a property of the executed code
 * and re-arming would only buy another timeout.
 *
 * Not thread-safe for concurrent arm/release: one gate arms one stream at a time.
 */
class StallGate
{
public:
    /// Default host-side budget between arm() and release(). Submission is a
    /// microsecond-scale operation, so seconds of headroom still cannot fire on a
    /// merely slow kernel; it only bounds a genuine deadlock.
    static constexpr std::chrono::milliseconds DEFAULT_TIMEOUT{2000};

    explicit StallGate(std::chrono::milliseconds timeout = DEFAULT_TIMEOUT)
        : _timeout(timeout)
    {
        int device = 0;
        auto status = hipGetDevice(&device);
        if(status != hipSuccess)
        {
            _lastError = status;
            _lastOperation = "hipGetDevice";
            return;
        }

        int canUseStreamWaitValue = 0;
        status = hipDeviceGetAttribute(
            &canUseStreamWaitValue, hipDeviceAttributeCanUseStreamWaitValue, device);
        if(status != hipSuccess || canUseStreamWaitValue == 0)
        {
            // A zero attribute without a HIP error means the device lacks support rather
            // than a call failing, so _lastError stays hipSuccess to tell them apart.
            _lastError = status;
            _lastOperation = "hipDeviceAttributeCanUseStreamWaitValue";
            return;
        }

        // Signal memory is an 8-byte HSA signal; a smaller size is rejected with
        // hipErrorInvalidValue. The 32-bit wait/write ops act on its low word.
        status = hipExtMallocWithFlags(
            reinterpret_cast<void**>(&_signal), sizeof(uint64_t), hipMallocSignalMemory);
        if(status != hipSuccess)
        {
            _signal = nullptr;
            _lastError = status;
            _lastOperation = "hipExtMallocWithFlags";
            return;
        }

        // Non-blocking so the release write runs concurrently with a stalled work stream;
        // a blocking control stream would implicitly serialize with the legacy default
        // stream and deadlock when the gate stalls it.
        status = hipStreamCreateWithFlags(&_control, hipStreamNonBlocking);
        if(status != hipSuccess)
        {
            _control = nullptr;
            _lastError = status;
            _lastOperation = "hipStreamCreateWithFlags";
            freeResources();
            return;
        }

        status = hipStreamWriteValue32(_control, _signal, 0U, 0);
        if(status != hipSuccess)
        {
            _lastError = status;
            _lastOperation = "hipStreamWriteValue32";
            freeResources();
            return;
        }

        status = hipStreamSynchronize(_control);
        if(status != hipSuccess)
        {
            _lastError = status;
            _lastOperation = "hipStreamSynchronize";
            freeResources();
            return;
        }
    }

    ~StallGate()
    {
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            // An armed, unreleased gate stalls its work stream forever.
            if(_armed)
            {
                static_cast<void>(hipStreamWriteValue32(_control, _signal, 1U, 0));
                _armed = false;
            }
            _stop = true;
        }
        _cv.notify_all();
        if(_watchdog.joinable())
        {
            _watchdog.join();
        }

        // A pending stream-wait must not outlive the signal memory it references, so
        // drain any stream this gate ever armed before freeing.
        if(_armedStream != nullptr)
        {
            static_cast<void>(hipStreamSynchronize(_armedStream));
        }
        freeResources();
    }

    // Not copyable, and not movable: the mutex and the watchdog thread bind the object
    // to its address. Nothing needs to move one -- every holder either owns it directly
    // or constructs it in place inside a std::optional.
    StallGate(const StallGate&) = delete;
    StallGate& operator=(const StallGate&) = delete;
    StallGate(StallGate&&) = delete;
    StallGate& operator=(StallGate&&) = delete;

    /// True when construction acquired both the signal memory and the control stream.
    bool isUsable() const
    {
        return _signal != nullptr && _control != nullptr;
    }

    /// Reset the signal, then enqueue a wait packet that holds every later item on
    /// `stream` until release() or the watchdog. Returns false when the gate is unusable,
    /// when a previous timeout disabled stalling process-wide, or when a HIP call failed;
    /// the gate is then not armed and the stream runs unstalled.
    bool arm(hipStream_t stream)
    {
        if(!isUsable() || isDisabledProcessWide())
        {
            return false;
        }

        auto status = hipStreamWriteValue32(_control, _signal, 0U, 0);
        if(status != hipSuccess)
        {
            _lastError = status;
            _lastOperation = "hipStreamWriteValue32";
            return false;
        }

        status = hipStreamSynchronize(_control);
        if(status != hipSuccess)
        {
            _lastError = status;
            _lastOperation = "hipStreamSynchronize";
            return false;
        }

        status = hipStreamWaitValue32(stream, _signal, 1U, hipStreamWaitValueGte, 0xFFFFFFFFU);
        if(status != hipSuccess)
        {
            _lastError = status;
            _lastOperation = "hipStreamWaitValue32";
            return false;
        }

        {
            const std::lock_guard<std::mutex> lock(_mutex);
            _armed = true;
            _armedStream = stream;
            _timedOut = false;
            _deadline = std::chrono::steady_clock::now() + _timeout;
            if(!_watchdog.joinable())
            {
                // Started on first arm, so a gate that never arms costs no thread.
                _watchdog = std::thread(&StallGate::watchdogLoop, this);
            }
        }
        _cv.notify_all();
        return true;
    }

    /// Release the gate from the otherwise idle control stream; the work stream proceeds
    /// device-side, no host sync needed. Idempotent: a no-op when not armed, including
    /// after the watchdog already released.
    void release()
    {
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            if(!_armed)
            {
                return;
            }
            static_cast<void>(hipStreamWriteValue32(_control, _signal, 1U, 0));
            _armed = false;
        }
        _cv.notify_all();
    }

    /// True when the watchdog, not the host, released the most recent arm(). The
    /// measurement from that region contains the timeout and must be discarded.
    bool timedOut() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _timedOut;
    }

    /// Clear the process-wide disable. For tests only, so one case that deliberately
    /// trips the watchdog cannot change the behavior of the next one. Production code
    /// must not re-enable stalling after a timeout: the condition that caused it is a
    /// property of the code being measured and has not gone away.
    static void resetDisabledProcessWideForTesting()
    {
        disabledFlag().store(false, std::memory_order_relaxed);
    }

    /// True once any gate in this process has timed out. Stalling stays off afterwards:
    /// the cause is a property of the code being measured, so re-arming would only buy
    /// another timeout.
    static bool isDisabledProcessWide()
    {
        return disabledFlag().load(std::memory_order_relaxed);
    }

    /// The most recent failed HIP call in the constructor or arm(). hipSuccess when
    /// nothing failed, including the unsupported-device case.
    hipError_t lastError() const
    {
        return _lastError;
    }

    /// Static string naming the most recent failed operation, or nullptr when nothing has
    /// failed. "hipDeviceAttributeCanUseStreamWaitValue" with lastError() == hipSuccess
    /// means the device lacks support.
    const char* lastOperation() const
    {
        return _lastOperation;
    }

private:
    void watchdogLoop()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        while(!_stop)
        {
            if(!_armed)
            {
                // Idle between measurements; woken by arm() or by the destructor.
                _cv.wait(lock);
                continue;
            }

            // Re-checks _armed on every wakeup, so a spurious wake or a host release
            // cannot fire the watchdog early.
            if(_cv.wait_until(lock, _deadline) == std::cv_status::timeout && _armed)
            {
                // The blocked host is waiting on this stream draining, and this write is
                // what drains it, so the write both ends the stall and unblocks the host.
                static_cast<void>(hipStreamWriteValue32(_control, _signal, 1U, 0));
                _armed = false;
                _timedOut = true;
                disabledFlag().store(true, std::memory_order_relaxed);
            }
        }
    }

    void freeResources() noexcept
    {
        if(_control != nullptr)
        {
            static_cast<void>(hipStreamDestroy(_control));
            _control = nullptr;
        }
        if(_signal != nullptr)
        {
            static_cast<void>(hipFree(_signal));
            _signal = nullptr;
        }
    }

    // Function-local rather than a static data member: one timeout anywhere in the
    // process turns stalling off for every gate, including ones created later.
    static std::atomic<bool>& disabledFlag()
    {
        static std::atomic<bool> s_disabled{false};
        return s_disabled;
    }

    uint32_t* _signal = nullptr;
    hipStream_t _control = nullptr;
    hipStream_t _armedStream = nullptr;
    hipError_t _lastError = hipSuccess;
    const char* _lastOperation = nullptr;

    std::chrono::milliseconds _timeout;

    mutable std::mutex _mutex;
    std::condition_variable _cv;
    std::thread _watchdog;
    std::chrono::steady_clock::time_point _deadline;
    bool _armed = false;
    bool _timedOut = false;
    bool _stop = false;
};

} // namespace hipdnn_data_sdk::utilities
