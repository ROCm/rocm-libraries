// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ProfilingControlDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "handle/Handle.hpp"

#include <spdlog/fmt/fmt.h>

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

// ============================================================================
// finalize
// ============================================================================

void ProfilingControlDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ProfilingControlDescriptor::finalize() failed: Already finalized.");

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
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "ProfilingControlDescriptor::setAttribute() failed: Already finalized.");

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

        _handle = handle;
        _stream = _handle->getStream();
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
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string(
                "ProfilingControlDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
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
