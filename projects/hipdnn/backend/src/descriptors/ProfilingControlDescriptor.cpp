// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ProfilingControlDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "handle/Handle.hpp"
#include "logging/Logging.hpp"

#include <spdlog/fmt/fmt.h>

#include <cmath>

namespace hipdnn_backend
{

// ============================================================================
// Lifecycle
// ============================================================================

ProfilingControlDescriptor::~ProfilingControlDescriptor() = default;

void ProfilingControlDescriptor::createEvents()
{
    if(_startEvent != nullptr)
    {
        return;
    }

    hipEvent_t startEvent = nullptr;
    auto status = hipEventCreate(&startEvent);
    THROW_IF_NE(status,
                hipSuccess,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "ProfilingControlDescriptor: hipEventCreate(start) failed.");
    HipEventGuard startGuard(startEvent);

    hipEvent_t stopEvent = nullptr;
    status = hipEventCreate(&stopEvent);
    THROW_IF_NE(status,
                hipSuccess,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "ProfilingControlDescriptor: hipEventCreate(stop) failed.");

    _startEvent = std::move(startGuard);
    _stopEvent = HipEventGuard(stopEvent);
}

void ProfilingControlDescriptor::checkBinding() const
{
    if(_handle == nullptr)
    {
        return;
    }
    int device = 0;
    const auto status = hipGetDevice(&device);
    THROW_IF_NE(status,
                hipSuccess,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "ProfilingControlDescriptor: hipGetDevice failed.");
    THROW_IF_TRUE(device != _device || _handle->getStream() != _stream,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ProfilingControlDescriptor: the bound device and stream must not change.");
}

// ============================================================================
// finalize
// ============================================================================

void ProfilingControlDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ProfilingControlDescriptor::finalize() failed: Already finalized.");

    // An armed gate holds the stop event unsignalled, so any precondition check below
    // that throws would otherwise abandon the stream stalled forever (watchdog aside).
    // Released before every other check -- including "handle not set" and "start/stop
    // not recorded" -- so a caller's own lifecycle error still leaves a live stream.
    // Absent when nothing ever armed.
    if(_stallGate.has_value())
    {
        _stallGate->release();
    }
    checkBinding();
    // If finalize is premature, a later corrected attempt runs after this release and
    // therefore is not a stalled measurement.
    if(_startEvent == nullptr || !_startRecorded || !_stopRecorded)
    {
        _stallUsed = false;
    }

    THROW_IF_FALSE(_startEvent != nullptr,
                   HIPDNN_STATUS_BAD_PARAM,
                   "ProfilingControlDescriptor::finalize() failed: "
                   "Handle not set (events not created).");

    THROW_IF_FALSE(_startRecorded,
                   HIPDNN_STATUS_BAD_PARAM,
                   "ProfilingControlDescriptor::finalize() failed: "
                   "Start event was not recorded.");

    THROW_IF_FALSE(_stopRecorded,
                   HIPDNN_STATUS_BAD_PARAM,
                   "ProfilingControlDescriptor::finalize() failed: "
                   "Stop event was not recorded.");

    auto status = hipEventSynchronize(_stopEvent.get());
    THROW_IF_NE(status,
                hipSuccess,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "ProfilingControlDescriptor::finalize() failed: "
                "hipEventSynchronize(stop) failed.");

    status = hipEventElapsedTime(&_elapsedMs, _startEvent.get(), _stopEvent.get());
    THROW_IF_NE(status,
                hipSuccess,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "ProfilingControlDescriptor::finalize() failed: "
                "hipEventElapsedTime failed.");

    // Gated on _stallUsed, not a live/bare query of the gate: _stallUsed is false
    // whenever this measurement never armed (including after reset() reused a gate that
    // timed out on an earlier measurement), so a skipped arm cannot inherit a stale
    // timeout from that earlier attempt.
    _timedOut = _stallUsed && _stallGate.has_value() && _stallGate->timedOut();
    // A watchdog-broken span is already explicitly invalid and remains readable only so
    // callers can inspect STALL_TIMED_OUT_EXT. For a healthy measurement, HIP success
    // does not guarantee a sane value: NaN or Inf has no finite sentinel to carry it, so
    // that case is a hard backend error. A finite negative span is not thrown here --
    // it passes finalization as a raw invalid-measurement sentinel a C caller detects
    // by sign, without hipDNN inventing a wrapper type. Zero is valid for back-to-back
    // events; only a genuinely non-finite result is rejected.
    THROW_IF_TRUE(!_timedOut && !std::isfinite(_elapsedMs),
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "ProfilingControlDescriptor::finalize() failed: "
                  "hipEventElapsedTime returned a non-finite value.");

    if(_timedOut)
    {
        // Loud, because the number below is not a measurement: the watchdog fired
        // because the host did not release within the timeout. That alone does not say
        // why (blocked host, slow host, or a missing release all look the same), only
        // that the elapsed span is not trustworthy.
        HIPDNN_BACKEND_LOG_ERROR(
            "ProfilingControlDescriptor: stall watchdog fired for this measurement (host "
            "did not release within the timeout). Elapsed time {} ms is invalid and must "
            "be discarded (HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT).",
            _elapsedMs);
    }

    HipdnnBackendDescriptorImpl<ProfilingControlDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void ProfilingControlDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                              hipdnnBackendAttributeType_t attributeType,
                                              int64_t elementCount,
                                              const void* arrayOfElements)
{
    // The sole attribute accepted once finalized: every case in the switch below is
    // still guarded by the finalized check, since RESET_EXT returns before reaching it.
    if(attributeName == HIPDNN_ATTR_PROFILING_RESET_EXT)
    {
        checkSetArgs(HIPDNN_TYPE_BOOLEAN,
                     attributeType,
                     arrayOfElements,
                     "ProfilingControlDescriptor::setAttribute(RESET)");
        THROW_IF_NE(elementCount,
                    static_cast<int64_t>(1),
                    HIPDNN_STATUS_BAD_PARAM,
                    "ProfilingControlDescriptor::setAttribute(RESET): elementCount must be 1.");
        reset();
        return;
    }

    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "ProfilingControlDescriptor::setAttribute() failed: Already finalized.");
    if(attributeName != HIPDNN_ATTR_PROFILING_STALL_RELEASE_EXT)
    {
        checkBinding();
    }

    switch(attributeName)
    {
    case HIPDNN_ATTR_PROFILING_HANDLE_EXT:
    {
        checkSetArgs(HIPDNN_TYPE_HANDLE,
                     attributeType,
                     arrayOfElements,
                     "ProfilingControlDescriptor::setAttribute(HANDLE)");
        THROW_IF_NE(elementCount,
                    static_cast<int64_t>(1),
                    HIPDNN_STATUS_BAD_PARAM,
                    "ProfilingControlDescriptor::setAttribute(HANDLE): elementCount must be 1.");

        auto* handle = *static_cast<hipdnnHandle* const*>(arrayOfElements);
        THROW_IF_NULL(handle,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "ProfilingControlDescriptor::setAttribute(HANDLE): Handle is null.");

        const auto stream = handle->getStream();
        THROW_IF_TRUE(_handle != nullptr && _handle != handle,
                      HIPDNN_STATUS_BAD_PARAM,
                      "ProfilingControlDescriptor::setAttribute(HANDLE): "
                      "Rebinding requires a new descriptor.");
        int device = 0;
        const auto status = hipGetDevice(&device);
        THROW_IF_NE(status,
                    hipSuccess,
                    HIPDNN_STATUS_INTERNAL_ERROR,
                    "ProfilingControlDescriptor::setAttribute(HANDLE): hipGetDevice failed.");

        int streamDevice = device;
        if(stream != nullptr)
        {
            const auto streamStatus = hipStreamGetDevice(stream, &streamDevice);
            THROW_IF_NE(streamStatus,
                        hipSuccess,
                        HIPDNN_STATUS_INTERNAL_ERROR,
                        "ProfilingControlDescriptor::setAttribute(HANDLE): "
                        "hipStreamGetDevice failed.");
        }
        THROW_IF_NE(streamDevice,
                    device,
                    HIPDNN_STATUS_BAD_PARAM,
                    "ProfilingControlDescriptor::setAttribute(HANDLE): "
                    "The stream must belong to the current device.");
        _device = device;
        _handle = handle;
        _stream = stream;
        createEvents();
        break;
    }
    case HIPDNN_ATTR_PROFILING_START_EXT:
    {
        checkSetArgs(HIPDNN_TYPE_BOOLEAN,
                     attributeType,
                     arrayOfElements,
                     "ProfilingControlDescriptor::setAttribute(START)");
        THROW_IF_NE(elementCount,
                    static_cast<int64_t>(1),
                    HIPDNN_STATUS_BAD_PARAM,
                    "ProfilingControlDescriptor::setAttribute(START): elementCount must be 1.");
        THROW_IF_FALSE(_startEvent != nullptr,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ProfilingControlDescriptor::setAttribute(START): "
                       "Handle must be set before recording events.");
        THROW_IF_TRUE(_startRecorded,
                      HIPDNN_STATUS_BAD_PARAM,
                      "ProfilingControlDescriptor::setAttribute(START): "
                      "Start has already been recorded.");

        auto status = hipEventRecord(_startEvent.get(), _stream);
        THROW_IF_NE(status,
                    hipSuccess,
                    HIPDNN_STATUS_INTERNAL_ERROR,
                    "ProfilingControlDescriptor::setAttribute(START): "
                    "hipEventRecord(start) failed.");
        _startRecorded = true;
        break;
    }
    case HIPDNN_ATTR_PROFILING_STOP_EXT:
    {
        checkSetArgs(HIPDNN_TYPE_BOOLEAN,
                     attributeType,
                     arrayOfElements,
                     "ProfilingControlDescriptor::setAttribute(STOP)");
        THROW_IF_NE(elementCount,
                    static_cast<int64_t>(1),
                    HIPDNN_STATUS_BAD_PARAM,
                    "ProfilingControlDescriptor::setAttribute(STOP): elementCount must be 1.");
        THROW_IF_FALSE(_startEvent != nullptr,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ProfilingControlDescriptor::setAttribute(STOP): "
                       "Handle must be set before recording events.");
        THROW_IF_FALSE(_startRecorded,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ProfilingControlDescriptor::setAttribute(STOP): "
                       "Start must be recorded before stop.");
        THROW_IF_TRUE(_stopRecorded,
                      HIPDNN_STATUS_BAD_PARAM,
                      "ProfilingControlDescriptor::setAttribute(STOP): "
                      "Stop has already been recorded.");

        auto status = hipEventRecord(_stopEvent.get(), _stream);
        THROW_IF_NE(status,
                    hipSuccess,
                    HIPDNN_STATUS_INTERNAL_ERROR,
                    "ProfilingControlDescriptor::setAttribute(STOP): "
                    "hipEventRecord(stop) failed.");
        _stopRecorded = true;
        break;
    }
    case HIPDNN_ATTR_PROFILING_DEVICE_SYNC_EXT:
    {
        checkSetArgs(HIPDNN_TYPE_BOOLEAN,
                     attributeType,
                     arrayOfElements,
                     "ProfilingControlDescriptor::setAttribute(DEVICE_SYNC)");
        THROW_IF_NE(elementCount,
                    static_cast<int64_t>(1),
                    HIPDNN_STATUS_BAD_PARAM,
                    "ProfilingControlDescriptor::setAttribute(DEVICE_SYNC): "
                    "elementCount must be 1.");

        // The boolean value passed via arrayOfElements is intentionally unused.
        // The setAttribute call with DEVICE_SYNC acts as a trigger: the sync
        // happens unconditionally when this attribute is set. The boolean
        // parameter exists for API consistency with other boolean attributes
        // (START, STOP) where the value is also unused (the action is the
        // setAttribute call itself).
        // Device-wide sync (as opposed to stream-sync) is intentional.
        auto status = hipDeviceSynchronize();
        THROW_IF_NE(status,
                    hipSuccess,
                    HIPDNN_STATUS_INTERNAL_ERROR,
                    "ProfilingControlDescriptor::setAttribute(DEVICE_SYNC): "
                    "hipDeviceSynchronize failed.");
        break;
    }
    case HIPDNN_ATTR_PROFILING_STALL_ARM_EXT:
    {
        checkSetArgs(HIPDNN_TYPE_BOOLEAN,
                     attributeType,
                     arrayOfElements,
                     "ProfilingControlDescriptor::setAttribute(STALL_ARM)");
        THROW_IF_NE(elementCount,
                    static_cast<int64_t>(1),
                    HIPDNN_STATUS_BAD_PARAM,
                    "ProfilingControlDescriptor::setAttribute(STALL_ARM): "
                    "elementCount must be 1.");
        THROW_IF_FALSE(_startEvent != nullptr,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ProfilingControlDescriptor::setAttribute(STALL_ARM): "
                       "Handle must be set before arming the stall.");
        THROW_IF_TRUE(_stallUsed,
                      HIPDNN_STATUS_BAD_PARAM,
                      "ProfilingControlDescriptor::setAttribute(STALL_ARM): "
                      "Stall is already armed for this measurement.");
        THROW_IF_TRUE(_startRecorded,
                      HIPDNN_STATUS_BAD_PARAM,
                      "ProfilingControlDescriptor::setAttribute(STALL_ARM): "
                      "Stall must be armed before start is recorded.");

        // The boolean value passed via arrayOfElements is intentionally unused: the
        // setAttribute call itself is the trigger, as it is for DEVICE_SYNC above.
        //
        // Created here rather than with the descriptor, so a caller that only times or
        // only syncs never pays for signal memory and a watchdog thread.
        if(!_stallGate.has_value())
        {
            _stallGate.emplace();
        }

        // A false return is success, not an error. When stalling is unavailable the
        // descriptor still measures, but reports that the measurement was unstalled.
        _stallUsed = _stallGate->arm(_stream);
        if(!_stallUsed)
        {
            // Failure to acquire an optional gate still permits plain event timing.
            // An acquired gate that fails a HIP call during arm reports a real error.
            THROW_IF_TRUE(_stallGate->isUsable() && _stallGate->lastError() != hipSuccess,
                          HIPDNN_STATUS_INTERNAL_ERROR,
                          fmt::format("ProfilingControlDescriptor::setAttribute(STALL_ARM): "
                                      "{} failed.",
                                      _stallGate->lastOperation() == nullptr
                                          ? "a HIP call"
                                          : _stallGate->lastOperation()));
            HIPDNN_BACKEND_LOG_INFO(
                "ProfilingControlDescriptor: stall gate unavailable ({}); measurement is "
                "unstalled",
                _stallGate->lastOperation() == nullptr ? "unknown" : _stallGate->lastOperation());
        }
        break;
    }
    case HIPDNN_ATTR_PROFILING_STALL_RELEASE_EXT:
    {
        checkSetArgs(HIPDNN_TYPE_BOOLEAN,
                     attributeType,
                     arrayOfElements,
                     "ProfilingControlDescriptor::setAttribute(STALL_RELEASE)");
        THROW_IF_NE(elementCount,
                    static_cast<int64_t>(1),
                    HIPDNN_STATUS_BAD_PARAM,
                    "ProfilingControlDescriptor::setAttribute(STALL_RELEASE): "
                    "elementCount must be 1.");

        // Releasing a gate that was never armed -- or never created -- is a no-op, not an
        // error, so a caller need not track whether arming succeeded.
        if(_stallGate.has_value())
        {
            _stallGate->release();
        }
        break;
    }
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string(
                "ProfilingControlDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

// ============================================================================
// reset
// ============================================================================

void ProfilingControlDescriptor::reset()
{
    // Release any armed gate before draining below: an unreleased wait would otherwise
    // make the synchronize block forever (or until the watchdog).
    if(_stallGate.has_value())
    {
        _stallGate->release();
    }
    checkBinding();

    // Finalize already retired the stop event. On an error or partial measurement,
    // release alone is insufficient: retire its wait before the signal is reset.
    // The handle guard includes the default stream, whose value is nullptr.
    if(!isFinalized() && _handle != nullptr)
    {
        const auto status = hipStreamSynchronize(_stream);
        THROW_IF_NE(status,
                    hipSuccess,
                    HIPDNN_STATUS_INTERNAL_ERROR,
                    "ProfilingControlDescriptor::setAttribute(RESET): "
                    "hipStreamSynchronize failed.");
    }

    _startRecorded = false;
    _stopRecorded = false;
    _elapsedMs = 0.0F;
    _stallUsed = false;
    _timedOut = false;
    _finalized = false;
}

// ============================================================================
// getAttribute
// ============================================================================

void ProfilingControlDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                              hipdnnBackendAttributeType_t attributeType,
                                              int64_t requestedElementCount,
                                              int64_t* elementCount,
                                              void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "ProfilingControlDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT:
        getScalar<float>(_elapsedMs,
                         HIPDNN_TYPE_FLOAT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "ProfilingControlDescriptor::getAttribute(ELAPSED_MS)");
        break;
    case HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT:
    {
        // Latched at finalize() into _timedOut, not read live here: a measurement that
        // reused this gate but skipped STALL_ARM_EXT must not inherit an earlier
        // measurement's timeout from the same gate object.
        getScalar<bool>(_timedOut,
                        HIPDNN_TYPE_BOOLEAN,
                        attributeType,
                        requestedElementCount,
                        elementCount,
                        arrayOfElements,
                        "ProfilingControlDescriptor::getAttribute(STALL_TIMED_OUT)");
        break;
    }
    case HIPDNN_ATTR_PROFILING_STALL_USED_EXT:
    {
        // Whether arm() actually stalled the stream for this measurement, not the
        // current armed state (finalize() always releases first). False when
        // STALL_ARM_EXT was never set, or arming declined.
        getScalar<bool>(_stallUsed,
                        HIPDNN_TYPE_BOOLEAN,
                        attributeType,
                        requestedElementCount,
                        elementCount,
                        arrayOfElements,
                        "ProfilingControlDescriptor::getAttribute(STALL_USED)");
        break;
    }
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string(
                "ProfilingControlDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

// ============================================================================
// Other methods
// ============================================================================

hipdnnBackendDescriptorType_t ProfilingControlDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_PROFILING_CONTROL_EXT;
}

std::string ProfilingControlDescriptor::toString() const
{
    std::string str
        = fmt::format("ProfilingControlDescriptor: {{eventsCreated={}, startRecorded={}, "
                      "stopRecorded={}, finalized={}",
                      _startEvent != nullptr,
                      _startRecorded,
                      _stopRecorded,
                      isFinalized());
    if(isFinalized())
    {
        str += fmt::format(", elapsedMs={}", _elapsedMs);
    }
    str += "}";
    return str;
}

} // namespace hipdnn_backend
