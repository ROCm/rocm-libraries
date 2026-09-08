// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <utility>

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
 * release() writes the signal from a private control stream. The measured span then
 * starts when the device actually begins the work, not when the host began submitting.
 *
 * Usage: arm(stream), record start, enqueue work, record stop, release(), synchronize.
 *
 * The class is non-throwing and best-effort: a device without stream-wait-value support
 * leaves isUsable() false and arm() returns false, which degrades to an unstalled
 * measurement instead of failing the caller. Callers that must fail loudly read
 * lastError() and lastOperation() to report which HIP call failed.
 *
 * Not thread-safe: one gate arms one stream at a time.
 */
class StallGate
{
public:
    StallGate()
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
        // An armed, unreleased gate stalls its work stream forever, so release and drain
        // before freeing: a pending stream-wait must not outlive the signal memory it
        // references.
        if(_armed)
        {
            release();
            static_cast<void>(hipStreamSynchronize(_armedStream));
        }
        freeResources();
    }

    StallGate(const StallGate&) = delete;
    StallGate& operator=(const StallGate&) = delete;

    StallGate(StallGate&& other) noexcept
        : _signal(other._signal)
        , _control(other._control)
        , _armedStream(other._armedStream)
        , _armed(other._armed)
        , _lastError(other._lastError)
        , _lastOperation(other._lastOperation)
    {
        other.clearMembers();
    }

    StallGate& operator=(StallGate&& other) noexcept
    {
        if(this != &other)
        {
            if(_armed)
            {
                release();
                static_cast<void>(hipStreamSynchronize(_armedStream));
            }
            freeResources();

            _signal = other._signal;
            _control = other._control;
            _armedStream = other._armedStream;
            _armed = other._armed;
            _lastError = other._lastError;
            _lastOperation = other._lastOperation;
            other.clearMembers();
        }
        return *this;
    }

    /// True when construction acquired both the signal memory and the control stream.
    bool isUsable() const
    {
        return _signal != nullptr && _control != nullptr;
    }

    /// Reset the signal, then enqueue a wait packet that holds every later item on
    /// `stream` until release(). Returns false when the gate is unusable or a HIP call
    /// failed; the gate is then not armed and the stream runs unstalled.
    bool arm(hipStream_t stream)
    {
        if(!isUsable())
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

        _armed = true;
        _armedStream = stream;
        return true;
    }

    /// Release the gate from the otherwise idle control stream; the work stream proceeds
    /// device-side, no host sync needed. Idempotent: a no-op when not armed.
    void release()
    {
        if(!_armed)
        {
            return;
        }
        static_cast<void>(hipStreamWriteValue32(_control, _signal, 1U, 0));
        _armed = false;
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

    void clearMembers() noexcept
    {
        _signal = nullptr;
        _control = nullptr;
        _armedStream = nullptr;
        _armed = false;
        _lastError = hipSuccess;
        _lastOperation = nullptr;
    }

    uint32_t* _signal = nullptr;
    hipStream_t _control = nullptr;
    hipStream_t _armedStream = nullptr;
    bool _armed = false;
    hipError_t _lastError = hipSuccess;
    const char* _lastOperation = nullptr;
};

} // namespace hipdnn_data_sdk::utilities
