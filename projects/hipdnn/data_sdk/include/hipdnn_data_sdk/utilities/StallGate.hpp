// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <mutex>
#include <system_error>
#include <thread>

namespace hipdnn_data_sdk::utilities
{

/**
 * @brief Host-released device-side stall for gap-free device timing.
 *
 * On runtimes that dispatch eagerly, a start event recorded on an idle stream can
 * complete while the host is still validating, dispatching, and logging. The resulting
 * event span then includes host submission time, which can dominate short kernels.
 *
 * arm() enqueues a wait packet on the work stream, so every later item on that stream
 * (the start event, the kernels, the stop event) stays queued but unexecuted until
 * release() writes the host-visible signal. Each event timestamp is taken when the
 * event executes, not when the host enqueued it, so the measured span starts when the
 * device actually begins the work.
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
 * The timeout measures host time between arm() and release(), not kernel duration.
 * A timeout can indicate a blocked host, a slow host, or a missing release; it does
 * not identify the cause.
 *
 * A watchdog release invalidates the measurement because device work can proceed
 * before host submission finishes. timedOut() reports this attempt only; later arms
 * remain available. A benchmarking caller must discard the affected comparison and
 * retry it unstalled rather than repeatedly arming a workload that may block the host.
 *
 * Not thread-safe for concurrent arm/release: one gate arms one stream at a time.
 *
 * Introduced in hipdnn_data_sdk 0.2.0.
 */
class StallGate
{
public:
    /// Default host-side budget between arm() and release(), not a kernel time limit.
    static constexpr std::chrono::milliseconds DEFAULT_TIMEOUT{2000};

    /// Non-positive budgets use DEFAULT_TIMEOUT rather than an immediate deadline.
    explicit StallGate(std::chrono::milliseconds timeout = DEFAULT_TIMEOUT)
        : _timeout(timeout > std::chrono::milliseconds::zero() ? timeout : DEFAULT_TIMEOUT)
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
        // hipErrorInvalidValue. The 32-bit wait acts on its low word.
        status = hipExtMallocWithFlags(
            reinterpret_cast<void**>(&_signal), sizeof(uint64_t), hipMallocSignalMemory);
        if(status != hipSuccess)
        {
            _signal = nullptr;
            _lastError = status;
            _lastOperation = "hipExtMallocWithFlags";
            return;
        }

        writeSignal(0U);
        // Recorded only now that the capability query and the signal allocation both
        // succeeded, so a partially-constructed (unusable) gate never reports a device;
        // arm() compares every stream's device against this one.
        _device = device;
    }

    ~StallGate()
    {
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            // Unconditional whenever the signal memory exists, not only when _armed: an
            // armed, unreleased gate stalls its work stream forever, and writing the
            // release value when the gate was never armed (or already released) is a
            // harmless host write that does not depend on _armed tracking every
            // enqueued wait precisely.
            if(_signal != nullptr)
            {
                writeSignal(1U);
            }
            _armed = false;
            _stop = true;
        }
        _cv.notify_all();
        if(_watchdog.joinable())
        {
            _watchdog.join();
        }

        // Satisfying the predicate does not retire its wait packet. hipFree performs
        // an implicit device synchronization before freeing the signal, covering
        // both default and explicit streams. Release above must precede that drain.
        if(_signal != nullptr)
        {
            static_cast<void>(hipFree(_signal));
        }
    }

    // Not copyable, and not movable: the mutex and the watchdog thread bind the object
    // to its address. Nothing needs to move one -- every holder either owns it directly
    // or constructs it in place inside a std::optional.
    StallGate(const StallGate&) = delete;
    StallGate& operator=(const StallGate&) = delete;
    StallGate(StallGate&&) = delete;
    StallGate& operator=(StallGate&&) = delete;

    /// True when construction acquired the signal memory.
    bool isUsable() const
    {
        return _signal != nullptr;
    }

    /// Reset the signal, then enqueue a wait packet that holds every later item on
    /// `stream` until release() or the watchdog. Returns false when the gate is unusable,
    /// already armed, bound to another device, or when a HIP/runtime resource operation
    /// fails. A declined arm does not alter an already-active wait.
    ///
    /// Before re-arming, the previously armed stream must have drained past its wait
    /// packet. release() satisfies the predicate but does not retire the waiter;
    /// resetting the signal too early can stall already-released work again.
    ///
    /// After this returns, lastError() and lastOperation() describe this attempt only,
    /// unless the gate is unusable -- then they still hold the constructor's diagnosis,
    /// because nothing in this call replaced it.
    bool arm(hipStream_t stream)
    {
        if(!isUsable())
        {
            // Returned before the per-attempt reset below, so the constructor's record of
            // which acquisition failed survives for the caller to report.
            return false;
        }

        // Reset per-attempt state and reject a second arm while the first is live.
        // Rewriting the shared signal in that state could race the watchdog's release
        // and silently unstall the first stream.
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            _lastError = hipSuccess;
            _lastOperation = nullptr;
            if(_armed)
            {
                _lastOperation = "StallGate::arm(already armed)";
                return false;
            }
            _timedOut = false;
        }

        // The signal is host memory bound to the constructor's device; a stream on a
        // different device cannot legally hipStreamWaitValue32 against it. Query before
        // the reset below, so a decline here never touches the signal. Use hipGetDevice
        // for the null/default stream because some supported runtimes reject null in
        // hipStreamGetDevice.
        int streamDevice = 0;
        const auto* deviceOperation = stream == nullptr ? "hipGetDevice" : "hipStreamGetDevice";
        const auto deviceStatus = stream == nullptr ? hipGetDevice(&streamDevice)
                                                    : hipStreamGetDevice(stream, &streamDevice);
        if(deviceStatus != hipSuccess)
        {
            _lastError = deviceStatus;
            _lastOperation = deviceOperation;
            return false;
        }
        if(streamDevice != _device)
        {
            // The query itself succeeded and simply named a different device, not a
            // failed HIP call, so _lastError stays hipSuccess.
            _lastOperation = deviceOperation;
            return false;
        }
        // Create the watchdog before enqueueing the wait. std::thread construction
        // can fail under resource pressure; in that case the documented non-throwing
        // fallback must leave the stream untouched.
        if(!_watchdog.joinable())
        {
            try
            {
                _watchdog = std::thread(&StallGate::watchdogLoop, this);
            }
            catch(const std::system_error&)
            {
                _lastOperation = "std::thread";
                return false;
            }
        }

        // Update the low word observed by hipStreamWaitValue32. A host write avoids
        // depending on a second GPU stream for gate progress.
        writeSignal(0U);

        const auto status
            = hipStreamWaitValue32(stream, _signal, 1U, hipStreamWaitValueGte, 0xFFFFFFFFU);
        if(status != hipSuccess)
        {
            _lastError = status;
            _lastOperation = "hipStreamWaitValue32";
            // The reset above left the signal at 0 with no wait enqueued to ever raise
            // it; restore the released value so the signal is never mistaken for still
            // armed.
            writeSignal(1U);
            return false;
        }

        {
            const std::lock_guard<std::mutex> lock(_mutex);
            _armed = true;
            _deadline = std::chrono::steady_clock::now() + _timeout;
        }
        _cv.notify_all();
        return true;
    }

    /// Release the gate with a host write, so forward progress does not depend on a
    /// second GPU command. Idempotent: a no-op when not armed, including after the
    /// watchdog already released.
    void release()
    {
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            if(!_armed)
            {
                return;
            }
            writeSignal(1U);
            _armed = false;
        }
        _cv.notify_all();
    }

    /// True when the watchdog, not the host, released the most recent arm(). Discard
    /// that measurement. A subsequent arm starts a new attempt with this flag cleared.
    bool timedOut() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _timedOut;
    }

    /// The most recent failed HIP call in the constructor or arm(). hipSuccess when
    /// nothing failed, including unsupported-device and cross-device declines.
    hipError_t lastError() const
    {
        return _lastError;
    }

    /// Static string naming the most recent failed or declined operation, or nullptr when
    /// nothing failed.
    /// "hipDeviceAttributeCanUseStreamWaitValue" with lastError() == hipSuccess means the
    /// device lacks support; "hipGetDevice" or "hipStreamGetDevice" with lastError() ==
    /// hipSuccess means arm() declined a stream bound to a different device.
    const char* lastOperation() const
    {
        return _lastOperation;
    }

private:
    void writeSignal(uint32_t value) noexcept
    {
        // Emit the host store to the signal word observed by the device. The fences
        // conservatively order host accesses; they do not replace waiter retirement.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *static_cast<volatile uint32_t*>(_signal) = value;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

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

            // wait_until always reacquires the lock before returning, timeout or not, so
            // a fresh arm() can run between its internal timeout and this check and move
            // _deadline forward. Re-checking both _armed and that steady_clock::now() has
            // actually reached the current (possibly newer) _deadline keeps a stale
            // timeout from firing a newer arm; otherwise the loop goes back around and
            // waits on the updated deadline.
            if(_cv.wait_until(lock, _deadline) == std::cv_status::timeout && _armed
               && std::chrono::steady_clock::now() >= _deadline)
            {
                // The blocked host is waiting on this stream draining, and this host
                // write both ends the stall and unblocks it without another GPU command.
                writeSignal(1U);
                _armed = false;
                _timedOut = true;
            }
        }
    }

    uint32_t* _signal = nullptr;
    int _device = 0;
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
